#!/usr/bin/env python3
"""
Cost Benchmark: BASELINE vs TOKENOMICS
=======================================

Measures token/cost savings comparing:
- BASELINE: No optimization (cache disabled, no compression, no routing, fixed model)
- TOKENOMICS: Full pipeline (cache, compression, bandit routing, cascading)

CRITICAL CACHE RULES:
1. BASELINE: Cache completely disabled (no read, no write)
2. TOKENOMICS: Cache starts empty (cold), allowed to warm across prompts
3. Cache is cleared ONCE at benchmark start, NOT between prompts

Usage:
    python scripts/run_cost_benchmark.py --workload benchmarks/workloads_v0.json --output BENCHMARK_COST_RESULTS.md
    python scripts/run_cost_benchmark.py --two_pass  # Cold + warm pass
"""

import argparse
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import statistics

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from dotenv import load_dotenv
load_dotenv(parent_dir / '.env')

from tokenomics.core import TokenomicsPlatform
from tokenomics.config import TokenomicsConfig
import structlog

logger = structlog.get_logger()

# ============================================================================
# MODEL PRICING (per 1M tokens)
# ============================================================================
MODEL_PRICING = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "gemini-flash": {"input": 0.075, "output": 0.30},
    "gemini-pro": {"input": 1.25, "output": 5.00},
    "gemini-2.0-flash-exp": {"input": 0.075, "output": 0.30},
}

# Baseline model (what teams use without optimization)
BASELINE_MODEL = "gpt-4o"

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class PromptResult:
    """Result for a single prompt."""
    prompt_id: str
    category: str
    prompt: str
    
    # Baseline metrics
    baseline_model: str = BASELINE_MODEL
    baseline_input_tokens: int = 0
    baseline_output_tokens: int = 0
    baseline_total_tokens: int = 0
    baseline_cost_usd: float = 0.0
    baseline_latency_ms: float = 0.0
    baseline_response: str = ""
    baseline_error: Optional[str] = None
    
    # Tokenomics metrics
    tokenomics_model: str = ""
    tokenomics_input_tokens: int = 0
    tokenomics_output_tokens: int = 0
    tokenomics_total_tokens: int = 0
    tokenomics_cost_usd: float = 0.0
    tokenomics_latency_ms: float = 0.0
    tokenomics_response: str = ""
    tokenomics_error: Optional[str] = None
    
    # Tokenomics decision trace
    cache_hit: bool = False
    cache_type: str = "none"
    compression_applied: bool = False
    routed_model: str = ""
    escalated: bool = False
    complexity_label: str = ""
    strategy_selected: str = ""
    
    # Delta metrics
    token_savings_abs: int = 0
    token_savings_pct: float = 0.0
    cost_savings_abs: float = 0.0
    cost_savings_pct: float = 0.0
    latency_delta_ms: float = 0.0
    
    # Quality
    quality_score: float = 1.0
    quality_pass: bool = True
    quality_check_type: str = "none"


@dataclass
class BenchmarkSummary:
    """Aggregate benchmark summary."""
    total_prompts: int = 0
    successful_prompts: int = 0
    failed_prompts: int = 0
    
    # Token savings
    total_baseline_tokens: int = 0
    total_tokenomics_tokens: int = 0
    mean_token_savings_pct: float = 0.0
    median_token_savings_pct: float = 0.0
    
    # Cost savings
    total_baseline_cost: float = 0.0
    total_tokenomics_cost: float = 0.0
    mean_cost_savings_pct: float = 0.0
    median_cost_savings_pct: float = 0.0
    total_cost_savings: float = 0.0
    
    # Latency
    mean_latency_delta_ms: float = 0.0
    
    # Cache stats
    cache_hit_count: int = 0
    cache_hit_rate: float = 0.0
    exact_cache_hits: int = 0
    semantic_cache_hits: int = 0
    
    # Routing stats
    cheap_model_count: int = 0
    balanced_model_count: int = 0
    premium_model_count: int = 0
    
    # Quality
    quality_failures: int = 0


# ============================================================================
# COST CALCULATION
# ============================================================================

def calculate_cost(input_tokens: int, output_tokens: int, model: str) -> float:
    """Calculate cost in USD for given tokens and model."""
    pricing = MODEL_PRICING.get(model, MODEL_PRICING["gpt-4o"])
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    return input_cost + output_cost


# ============================================================================
# QUALITY CHECKS
# ============================================================================

def check_quality_minlen(response: str, baseline_response: str, min_len: int = 20) -> Tuple[float, bool]:
    """Check quality using minimum length threshold."""
    if len(response) >= min_len:
        return 1.0, True
    return len(response) / min_len, False


def check_quality_embedding(response: str, baseline_response: str, threshold: float = 0.7) -> Tuple[float, bool]:
    """Check quality using embedding similarity."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        emb1 = model.encode([response])[0]
        emb2 = model.encode([baseline_response])[0]
        
        # Cosine similarity
        similarity = float(emb1 @ emb2 / (sum(emb1**2)**0.5 * sum(emb2**2)**0.5))
        return similarity, similarity >= threshold
    except Exception as e:
        logger.warning("Embedding quality check failed", error=str(e))
        return 1.0, True


def check_quality(
    response: str,
    baseline_response: str,
    quality_check: str,
) -> Tuple[float, bool, str]:
    """Run quality check based on type."""
    if quality_check == "none":
        return 1.0, True, "none"
    elif quality_check == "minlen":
        score, passed = check_quality_minlen(response, baseline_response)
        return score, passed, "minlen"
    elif quality_check == "embedding":
        score, passed = check_quality_embedding(response, baseline_response)
        return score, passed, "embedding"
    else:
        return 1.0, True, "none"


# ============================================================================
# PLATFORM INITIALIZATION
# ============================================================================

def create_baseline_platform() -> TokenomicsPlatform:
    """Create platform configured for BASELINE mode (no optimizations)."""
    config = TokenomicsConfig.from_env()
    
    # Override for OpenAI
    config.llm.provider = "openai"
    config.llm.model = BASELINE_MODEL
    config.llm.api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    
    # DISABLE ALL OPTIMIZATIONS
    config.memory.use_exact_cache = False
    config.memory.use_semantic_cache = False
    config.memory.enable_llmlingua = False
    config.memory.persistent_cache_path = None  # No persistence
    
    config.orchestrator.enable_multi_model_routing = False
    config.cascading.enabled = False
    
    # Disable bandit state persistence for baseline
    config.bandit.state_file = None
    config.bandit.auto_save = False
    
    return TokenomicsPlatform(config=config)


def create_tokenomics_platform() -> TokenomicsPlatform:
    """Create platform configured for TOKENOMICS mode (full pipeline)."""
    config = TokenomicsConfig.from_env()
    
    # Override for OpenAI
    config.llm.provider = "openai"
    config.llm.model = "gpt-4o-mini"  # Default, bandit will route
    config.llm.api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    
    # ENABLE ALL OPTIMIZATIONS
    config.memory.use_exact_cache = True
    config.memory.use_semantic_cache = True
    config.memory.enable_llmlingua = True
    config.memory.persistent_cache_path = None  # In-memory for benchmark
    
    config.orchestrator.enable_multi_model_routing = True
    config.cascading.enabled = True
    
    # Fresh bandit state for benchmark
    config.bandit.state_file = None
    config.bandit.auto_save = False
    
    return TokenomicsPlatform(config=config)


# ============================================================================
# BENCHMARK RUNNER
# ============================================================================

def run_baseline(platform: TokenomicsPlatform, prompt: str) -> Dict[str, Any]:
    """Run baseline query (no cache, no compression, no routing)."""
    start_time = time.time()
    
    try:
        # Use query() with all features disabled
        result = platform.query(
            query=prompt,
            use_cache=False,
            use_bandit=False,
            use_compression=False,
            use_cost_aware_routing=False,
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        return {
            "success": True,
            "response": result.get("response", ""),
            "model": result.get("model", BASELINE_MODEL),
            "input_tokens": result.get("input_tokens", 0),
            "output_tokens": result.get("output_tokens", 0),
            "total_tokens": result.get("tokens_used", 0),
            "latency_ms": latency_ms,
            "error": None,
        }
    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        logger.error("Baseline query failed", error=str(e))
        return {
            "success": False,
            "response": "",
            "model": BASELINE_MODEL,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "latency_ms": latency_ms,
            "error": str(e),
        }


def run_tokenomics(platform: TokenomicsPlatform, prompt: str) -> Dict[str, Any]:
    """Run tokenomics query (full pipeline)."""
    start_time = time.time()
    
    try:
        result = platform.query(
            query=prompt,
            use_cache=True,
            use_bandit=True,
            use_compression=True,
            use_cost_aware_routing=True,
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Extract decision trace
        cache_hit = result.get("cache_hit", False)
        cache_type = result.get("cache_type", "none") or "none"
        
        # Determine if compression was applied
        compression_applied = False
        if result.get("memory_metrics"):
            mem_metrics = result["memory_metrics"]
            if mem_metrics.get("context_compressed_tokens", 0) > 0:
                compression_applied = True
        
        return {
            "success": True,
            "response": result.get("response", ""),
            "model": result.get("model", "gpt-4o-mini"),
            "input_tokens": result.get("input_tokens", 0),
            "output_tokens": result.get("output_tokens", 0),
            "total_tokens": result.get("tokens_used", 0),
            "latency_ms": latency_ms,
            "cache_hit": cache_hit,
            "cache_type": cache_type,
            "compression_applied": compression_applied,
            "strategy": result.get("strategy", ""),
            "complexity": result.get("query_type", ""),
            "error": None,
        }
    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        logger.error("Tokenomics query failed", error=str(e))
        return {
            "success": False,
            "response": "",
            "model": "gpt-4o-mini",
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "latency_ms": latency_ms,
            "cache_hit": False,
            "cache_type": "none",
            "compression_applied": False,
            "strategy": "",
            "complexity": "",
            "error": str(e),
        }


def run_single_prompt(
    baseline_platform: TokenomicsPlatform,
    tokenomics_platform: TokenomicsPlatform,
    prompt_data: Dict[str, Any],
    quality_check: str,
) -> PromptResult:
    """Run benchmark for a single prompt."""
    prompt_id = prompt_data["id"]
    category = prompt_data["category"]
    prompt = prompt_data["prompt"]
    
    print(f"  Running {prompt_id}...", end=" ", flush=True)
    
    result = PromptResult(
        prompt_id=prompt_id,
        category=category,
        prompt=prompt,
    )
    
    # Run baseline
    baseline_result = run_baseline(baseline_platform, prompt)
    result.baseline_model = baseline_result["model"]
    result.baseline_input_tokens = baseline_result["input_tokens"]
    result.baseline_output_tokens = baseline_result["output_tokens"]
    result.baseline_total_tokens = baseline_result["total_tokens"]
    result.baseline_latency_ms = baseline_result["latency_ms"]
    result.baseline_response = baseline_result["response"]
    result.baseline_error = baseline_result["error"]
    
    # Calculate baseline cost
    result.baseline_cost_usd = calculate_cost(
        result.baseline_input_tokens,
        result.baseline_output_tokens,
        result.baseline_model,
    )
    
    # Run tokenomics
    tokenomics_result = run_tokenomics(tokenomics_platform, prompt)
    result.tokenomics_model = tokenomics_result["model"]
    result.tokenomics_input_tokens = tokenomics_result["input_tokens"]
    result.tokenomics_output_tokens = tokenomics_result["output_tokens"]
    result.tokenomics_total_tokens = tokenomics_result["total_tokens"]
    result.tokenomics_latency_ms = tokenomics_result["latency_ms"]
    result.tokenomics_response = tokenomics_result["response"]
    result.tokenomics_error = tokenomics_result["error"]
    
    # Decision trace
    result.cache_hit = tokenomics_result["cache_hit"]
    result.cache_type = tokenomics_result["cache_type"]
    result.compression_applied = tokenomics_result["compression_applied"]
    result.routed_model = tokenomics_result["model"]
    result.strategy_selected = tokenomics_result["strategy"]
    result.complexity_label = tokenomics_result["complexity"]
    
    # Calculate tokenomics cost
    result.tokenomics_cost_usd = calculate_cost(
        result.tokenomics_input_tokens,
        result.tokenomics_output_tokens,
        result.tokenomics_model,
    )
    
    # Calculate deltas
    result.token_savings_abs = result.baseline_total_tokens - result.tokenomics_total_tokens
    if result.baseline_total_tokens > 0:
        result.token_savings_pct = (result.token_savings_abs / result.baseline_total_tokens) * 100
    
    result.cost_savings_abs = result.baseline_cost_usd - result.tokenomics_cost_usd
    if result.baseline_cost_usd > 0:
        result.cost_savings_pct = (result.cost_savings_abs / result.baseline_cost_usd) * 100
    
    result.latency_delta_ms = result.tokenomics_latency_ms - result.baseline_latency_ms
    
    # Quality check
    if result.tokenomics_response and result.baseline_response:
        result.quality_score, result.quality_pass, result.quality_check_type = check_quality(
            result.tokenomics_response,
            result.baseline_response,
            quality_check,
        )
    
    # Print one-line summary
    cache_info = f"cache:{result.cache_type}" if result.cache_hit else "cache:miss"
    print(f"[{result.baseline_total_tokens}→{result.tokenomics_total_tokens} tokens] "
          f"[${result.baseline_cost_usd:.6f}→${result.tokenomics_cost_usd:.6f}] "
          f"({result.cost_savings_pct:.1f}% saved) [{cache_info}]")
    
    return result


def compute_summary(results: List[PromptResult]) -> BenchmarkSummary:
    """Compute aggregate summary from results."""
    summary = BenchmarkSummary()
    summary.total_prompts = len(results)
    
    successful = [r for r in results if r.baseline_error is None and r.tokenomics_error is None]
    summary.successful_prompts = len(successful)
    summary.failed_prompts = summary.total_prompts - summary.successful_prompts
    
    if not successful:
        return summary
    
    # Token savings
    summary.total_baseline_tokens = sum(r.baseline_total_tokens for r in successful)
    summary.total_tokenomics_tokens = sum(r.tokenomics_total_tokens for r in successful)
    
    token_savings_pcts = [r.token_savings_pct for r in successful if r.baseline_total_tokens > 0]
    if token_savings_pcts:
        summary.mean_token_savings_pct = statistics.mean(token_savings_pcts)
        summary.median_token_savings_pct = statistics.median(token_savings_pcts)
    
    # Cost savings
    summary.total_baseline_cost = sum(r.baseline_cost_usd for r in successful)
    summary.total_tokenomics_cost = sum(r.tokenomics_cost_usd for r in successful)
    summary.total_cost_savings = summary.total_baseline_cost - summary.total_tokenomics_cost
    
    cost_savings_pcts = [r.cost_savings_pct for r in successful if r.baseline_cost_usd > 0]
    if cost_savings_pcts:
        summary.mean_cost_savings_pct = statistics.mean(cost_savings_pcts)
        summary.median_cost_savings_pct = statistics.median(cost_savings_pcts)
    
    # Latency
    latency_deltas = [r.latency_delta_ms for r in successful]
    summary.mean_latency_delta_ms = statistics.mean(latency_deltas) if latency_deltas else 0
    
    # Cache stats
    summary.cache_hit_count = sum(1 for r in successful if r.cache_hit)
    summary.cache_hit_rate = (summary.cache_hit_count / len(successful)) * 100 if successful else 0
    summary.exact_cache_hits = sum(1 for r in successful if r.cache_type == "exact")
    summary.semantic_cache_hits = sum(1 for r in successful if r.cache_type in ["semantic_direct", "semantic", "context"])
    
    # Routing stats
    for r in successful:
        strategy = r.strategy_selected.lower() if r.strategy_selected else ""
        if "cheap" in strategy:
            summary.cheap_model_count += 1
        elif "balanced" in strategy:
            summary.balanced_model_count += 1
        elif "premium" in strategy:
            summary.premium_model_count += 1
    
    # Quality
    summary.quality_failures = sum(1 for r in successful if not r.quality_pass)
    
    return summary


# ============================================================================
# MARKDOWN REPORT GENERATION
# ============================================================================

def get_git_hash() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=parent_dir,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def generate_markdown_report(
    results: List[PromptResult],
    summary: BenchmarkSummary,
    config: Dict[str, Any],
    pass_name: str = "",
) -> str:
    """Generate markdown report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    git_hash = get_git_hash()
    
    title_suffix = f" ({pass_name.upper()} PASS)" if pass_name else ""
    
    md = f"""# Tokenomics Cost Benchmark Results{title_suffix}

**Generated:** {timestamp}  
**Git Commit:** `{git_hash}`

---

## Benchmark Configuration

| Setting | Value |
|---------|-------|
| Baseline Model | `{BASELINE_MODEL}` |
| Tokenomics Default Model | `gpt-4o-mini` |
| Tokenomics Premium Model | `gpt-4o` |
| Temperature | 0.3 (baseline) / varies (tokenomics) |
| Max Tokens | 512 (baseline) / varies (tokenomics) |
| Cache | Disabled (baseline) / Enabled (tokenomics) |
| Compression | Disabled (baseline) / Enabled (tokenomics) |
| Routing | Disabled (baseline) / Bandit UCB (tokenomics) |

---

## Workload Summary

| Category | Count |
|----------|-------|
| Simple | {sum(1 for r in results if r.category == 'simple')} |
| Medium | {sum(1 for r in results if r.category == 'medium')} |
| Complex | {sum(1 for r in results if r.category == 'complex')} |
| **Total** | **{len(results)}** |

---

## Aggregate Summary

### Cost & Token Savings

| Metric | Value |
|--------|-------|
| Total Baseline Tokens | {summary.total_baseline_tokens:,} |
| Total Tokenomics Tokens | {summary.total_tokenomics_tokens:,} |
| **Mean Token Savings** | **{summary.mean_token_savings_pct:.1f}%** |
| Median Token Savings | {summary.median_token_savings_pct:.1f}% |
| Total Baseline Cost | ${summary.total_baseline_cost:.6f} |
| Total Tokenomics Cost | ${summary.total_tokenomics_cost:.6f} |
| **Total Cost Savings** | **${summary.total_cost_savings:.6f}** |
| **Mean Cost Savings** | **{summary.mean_cost_savings_pct:.1f}%** |
| Median Cost Savings | {summary.median_cost_savings_pct:.1f}% |

### Cache Performance

| Metric | Value |
|--------|-------|
| Cache Hit Rate | {summary.cache_hit_rate:.1f}% |
| Exact Cache Hits | {summary.exact_cache_hits} |
| Semantic Cache Hits | {summary.semantic_cache_hits} |
| Cache Misses | {summary.successful_prompts - summary.cache_hit_count} |

### Routing Distribution

| Strategy | Count | Percentage |
|----------|-------|------------|
| Cheap (gpt-4o-mini) | {summary.cheap_model_count} | {(summary.cheap_model_count/summary.successful_prompts*100) if summary.successful_prompts > 0 else 0:.1f}% |
| Balanced (gpt-4o-mini) | {summary.balanced_model_count} | {(summary.balanced_model_count/summary.successful_prompts*100) if summary.successful_prompts > 0 else 0:.1f}% |
| Premium (gpt-4o) | {summary.premium_model_count} | {(summary.premium_model_count/summary.successful_prompts*100) if summary.successful_prompts > 0 else 0:.1f}% |

### Quality & Reliability

| Metric | Value |
|--------|-------|
| Successful Prompts | {summary.successful_prompts}/{summary.total_prompts} |
| Quality Failures | {summary.quality_failures} |
| Mean Latency Delta | {summary.mean_latency_delta_ms:.0f}ms |

---

## Per-Prompt Results

| ID | Category | Baseline Tokens | Tokenomics Tokens | Token Savings | Baseline Cost | Tokenomics Cost | Cost Savings | Cache | Strategy |
|----|----------|-----------------|-------------------|---------------|---------------|-----------------|--------------|-------|----------|
"""
    
    for r in results:
        error_marker = "❌" if r.baseline_error or r.tokenomics_error else ""
        cache_str = r.cache_type if r.cache_hit else "miss"
        md += f"| {r.prompt_id} | {r.category} | {r.baseline_total_tokens} | {r.tokenomics_total_tokens} | {r.token_savings_pct:.1f}% | ${r.baseline_cost_usd:.6f} | ${r.tokenomics_cost_usd:.6f} | {r.cost_savings_pct:.1f}% | {cache_str} | {r.strategy_selected or 'N/A'} {error_marker}|\n"
    
    md += f"""
---

## Caveats

1. **Baseline Configuration**: Baseline runs with cache completely disabled (no read, no write). This ensures baseline costs are not artificially reduced by prior tokenomics runs.

2. **Cache Cold Start**: Tokenomics cache starts empty at benchmark start. The cache is NOT cleared between prompts, allowing natural cache warming.

3. **Workload Dependence**: Results are specific to this workload ({len(results)} prompts). Production savings depend on actual query patterns, repetition, and complexity distribution.

4. **Quality Metric**: Quality is measured using `{config.get('quality_check', 'minlen')}` check. This is a proxy metric, not a comprehensive quality evaluation.

5. **Latency Variance**: Latency measurements include API round-trip time and may vary based on network conditions and API load.

6. **Model Pricing**: Costs are estimated based on OpenAI's published pricing as of the benchmark date. Actual costs may vary.

---

## Reproduction Steps

```bash
# From repository root
cd /path/to/Tokenomics

# Ensure .env has OPENAI_API_KEY set
source venv/bin/activate

# Run benchmark
python scripts/run_cost_benchmark.py \\
    --workload benchmarks/workloads_v0.json \\
    --output BENCHMARK_COST_RESULTS.md \\
    --quality_check minlen \\
    --seed {config.get('seed', 42)}
```

Expected runtime: ~{len(results) * 5} seconds (varies with API latency)

---

*Report generated by `scripts/run_cost_benchmark.py`*
"""
    
    return md


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Cost Benchmark: BASELINE vs TOKENOMICS"
    )
    parser.add_argument(
        "--workload",
        type=str,
        default="benchmarks/workloads_v0.json",
        help="Path to workload JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="BENCHMARK_COST_RESULTS.md",
        help="Output markdown file",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of runs (currently only 1 supported)",
    )
    parser.add_argument(
        "--quality_check",
        type=str,
        choices=["none", "minlen", "embedding", "judge"],
        default="minlen",
        help="Quality check method",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--two_pass",
        action="store_true",
        help="Run two passes: cold (empty cache) then warm (populated cache)",
    )
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Load workload
    workload_path = parent_dir / args.workload
    if not workload_path.exists():
        print(f"Error: Workload file not found: {workload_path}")
        sys.exit(1)
    
    with open(workload_path) as f:
        workload = json.load(f)
    
    prompts = workload["prompts"]
    print(f"\n{'='*60}")
    print(f"TOKENOMICS COST BENCHMARK")
    print(f"{'='*60}")
    print(f"Workload: {args.workload} ({len(prompts)} prompts)")
    print(f"Quality Check: {args.quality_check}")
    print(f"Two-Pass Mode: {args.two_pass}")
    print(f"{'='*60}\n")
    
    # Check API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set in environment")
        sys.exit(1)
    
    # Create platforms
    print("Initializing platforms...")
    baseline_platform = create_baseline_platform()
    tokenomics_platform = create_tokenomics_platform()
    
    # Clear tokenomics cache (cold start)
    print("Clearing tokenomics cache (cold start)...")
    if tokenomics_platform.memory:
        tokenomics_platform.memory.clear()
    
    config = {
        "workload": args.workload,
        "quality_check": args.quality_check,
        "seed": args.seed,
        "two_pass": args.two_pass,
    }
    
    all_results = []
    
    # COLD PASS
    print(f"\n{'='*60}")
    print("COLD PASS (cache empty)")
    print(f"{'='*60}\n")
    
    cold_results = []
    for prompt_data in prompts:
        result = run_single_prompt(
            baseline_platform,
            tokenomics_platform,
            prompt_data,
            args.quality_check,
        )
        cold_results.append(result)
    
    cold_summary = compute_summary(cold_results)
    
    print(f"\n--- Cold Pass Summary ---")
    print(f"Total Baseline Cost:    ${cold_summary.total_baseline_cost:.6f}")
    print(f"Total Tokenomics Cost:  ${cold_summary.total_tokenomics_cost:.6f}")
    print(f"Total Savings:          ${cold_summary.total_cost_savings:.6f} ({cold_summary.mean_cost_savings_pct:.1f}%)")
    print(f"Cache Hit Rate:         {cold_summary.cache_hit_rate:.1f}%")
    
    all_results.extend(cold_results)
    
    # WARM PASS (optional)
    warm_results = []
    warm_summary = None
    
    if args.two_pass:
        print(f"\n{'='*60}")
        print("WARM PASS (cache populated from cold pass)")
        print(f"{'='*60}\n")
        
        for prompt_data in prompts:
            result = run_single_prompt(
                baseline_platform,
                tokenomics_platform,
                prompt_data,
                args.quality_check,
            )
            warm_results.append(result)
        
        warm_summary = compute_summary(warm_results)
        
        print(f"\n--- Warm Pass Summary ---")
        print(f"Total Baseline Cost:    ${warm_summary.total_baseline_cost:.6f}")
        print(f"Total Tokenomics Cost:  ${warm_summary.total_tokenomics_cost:.6f}")
        print(f"Total Savings:          ${warm_summary.total_cost_savings:.6f} ({warm_summary.mean_cost_savings_pct:.1f}%)")
        print(f"Cache Hit Rate:         {warm_summary.cache_hit_rate:.1f}%")
    
    # Generate report
    print(f"\n{'='*60}")
    print("GENERATING REPORT")
    print(f"{'='*60}\n")
    
    # Use cold results for main report (most conservative)
    report = generate_markdown_report(cold_results, cold_summary, config, pass_name="cold" if args.two_pass else "")
    
    if args.two_pass and warm_summary:
        # Append warm pass section
        warm_report = generate_markdown_report(warm_results, warm_summary, config, pass_name="warm")
        report += f"\n\n---\n\n{warm_report}"
    
    # Write report
    output_path = parent_dir / args.output
    with open(output_path, "w") as f:
        f.write(report)
    
    print(f"Report written to: {output_path}")
    
    # Final summary
    print(f"\n{'='*60}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*60}")
    print(f"Prompts: {len(prompts)}")
    print(f"Cold Pass Cost Savings: {cold_summary.mean_cost_savings_pct:.1f}%")
    if warm_summary:
        print(f"Warm Pass Cost Savings: {warm_summary.mean_cost_savings_pct:.1f}%")
        print(f"Warm Pass Cache Hit Rate: {warm_summary.cache_hit_rate:.1f}%")
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()

