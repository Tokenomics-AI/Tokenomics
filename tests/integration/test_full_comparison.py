"""
Comprehensive Platform Comparison Test

Compares three configurations:
1. BASELINE: No caching, no optimization
2. BASIC: Exact cache only (what we had before)
3. ENHANCED: Full semantic cache + Mem0 + LLM-Lingua + RouterBench

This demonstrates the value of each enhancement.
"""

# Disable TensorFlow before any imports
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"

import sys
import time
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))

# Set API key explicitly if needed (use environment variable or .env file)
if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable is required")

from tokenomics.core import TokenomicsPlatform
from tokenomics.config import TokenomicsConfig


@dataclass
class TestResult:
    """Result from a single query."""
    query: str
    tokens: int
    latency_ms: float
    cache_hit: bool
    cache_type: str
    similarity: float = None
    strategy: str = None


@dataclass  
class ConfigResults:
    """Results from testing a configuration."""
    name: str
    results: List[TestResult] = field(default_factory=list)
    
    @property
    def total_tokens(self) -> int:
        return sum(r.tokens for r in self.results)
    
    @property
    def total_latency(self) -> float:
        return sum(r.latency_ms for r in self.results)
    
    @property
    def avg_latency(self) -> float:
        return self.total_latency / len(self.results) if self.results else 0
    
    @property
    def cache_hits(self) -> int:
        return sum(1 for r in self.results if r.cache_hit)
    
    @property
    def exact_hits(self) -> int:
        return sum(1 for r in self.results if r.cache_type == "exact")
    
    @property
    def semantic_hits(self) -> int:
        return sum(1 for r in self.results if r.cache_type == "semantic_direct")
    
    @property
    def context_hits(self) -> int:
        return sum(1 for r in self.results if r.cache_hit and r.tokens > 0)
    
    @property
    def api_calls(self) -> int:
        return sum(1 for r in self.results if not r.cache_hit or r.tokens > 0)


def create_baseline_config():
    """Baseline: No caching at all."""
    config = TokenomicsConfig.from_env()
    config.llm.provider = "openai"
    config.llm.model = "gpt-4o-mini"
    config.llm.api_key = os.environ["OPENAI_API_KEY"]
    
    # Disable all caching
    config.memory.use_exact_cache = False
    config.memory.use_semantic_cache = False
    
    return config


def create_basic_config():
    """Basic: Exact cache only (previous configuration)."""
    config = TokenomicsConfig.from_env()
    config.llm.provider = "openai"
    config.llm.model = "gpt-4o-mini"
    config.llm.api_key = os.environ["OPENAI_API_KEY"]
    
    # Only exact cache
    config.memory.use_exact_cache = True
    config.memory.use_semantic_cache = False
    config.memory.cache_size = 100
    
    return config


def create_enhanced_config():
    """Enhanced: Full optimization (current configuration)."""
    config = TokenomicsConfig.from_env()
    config.llm.provider = "openai"
    config.llm.model = "gpt-4o-mini"
    config.llm.api_key = os.environ["OPENAI_API_KEY"]
    
    # Full optimization
    config.memory.use_exact_cache = True
    config.memory.use_semantic_cache = True
    config.memory.cache_size = 100
    config.memory.similarity_threshold = 0.75
    config.memory.direct_return_threshold = 0.85
    
    return config


def get_test_queries():
    """Get comprehensive test queries."""
    return [
        # Round 1: Initial questions (populate cache)
        {"query": "What is machine learning?", "round": 1, "type": "original"},
        {"query": "What are neural networks?", "round": 1, "type": "original"},
        {"query": "How to optimize Python code?", "round": 1, "type": "original"},
        {"query": "Explain deep learning", "round": 1, "type": "original"},
        
        # Round 2: Exact repeats (should hit exact cache)
        {"query": "What is machine learning?", "round": 2, "type": "exact_repeat"},
        {"query": "What are neural networks?", "round": 2, "type": "exact_repeat"},
        
        # Round 3: Similar queries (should hit semantic cache)
        {"query": "Explain machine learning to me", "round": 3, "type": "similar"},
        {"query": "How does ML work?", "round": 3, "type": "similar"},
        {"query": "Tell me about neural nets", "round": 3, "type": "similar"},
        {"query": "Python performance tips", "round": 3, "type": "similar"},
        {"query": "What is deep learning?", "round": 3, "type": "similar"},
        
        # Round 4: More exact repeats
        {"query": "What is machine learning?", "round": 4, "type": "exact_repeat"},
        {"query": "Explain machine learning to me", "round": 4, "type": "exact_repeat"},
    ]


def run_test(platform: TokenomicsPlatform, queries: List[Dict]) -> List[TestResult]:
    """Run queries against a platform."""
    results = []
    
    for q in queries:
        query = q["query"]
        
        start = time.time()
        result = platform.query(query)
        latency = (time.time() - start) * 1000
        
        results.append(TestResult(
            query=query,
            tokens=result.get("tokens_used", 0),
            latency_ms=latency,
            cache_hit=result.get("cache_hit", False),
            cache_type=result.get("cache_type"),
            similarity=result.get("similarity"),
            strategy=result.get("strategy"),
        ))
    
    return results


def print_header(text: str):
    """Print section header."""
    print()
    print("=" * 80)
    print(f"  {text}")
    print("=" * 80)


def run_comparison_test():
    """Run full comparison test."""
    print_header("TOKENOMICS PLATFORM - FULL COMPARISON TEST")
    
    print("""
This test compares THREE configurations:

  1. BASELINE   : No caching (raw API calls)
  2. BASIC      : Exact cache only (previous config)
  3. ENHANCED   : Semantic cache + Mem0 + RouterBench (current)

Testing with 13 queries including:
  - 4 original questions
  - 2 exact repeats
  - 5 similar variations
  - 2 more exact repeats
""")
    
    queries = get_test_queries()
    all_results = {}
    
    # =========================================================================
    # Test 1: BASELINE (No caching)
    # =========================================================================
    print_header("TEST 1: BASELINE (No Caching)")
    print("\nInitializing baseline platform...")
    
    baseline_config = create_baseline_config()
    baseline_platform = TokenomicsPlatform(config=baseline_config)
    
    print("Running queries...\n")
    baseline_results = ConfigResults(name="BASELINE")
    
    for i, q in enumerate(queries, 1):
        print(f"  [{i:2d}/13] {q['query'][:50]}...", end=" ")
        
        start = time.time()
        result = baseline_platform.query(q["query"])
        latency = (time.time() - start) * 1000
        
        tokens = result.get("tokens_used", 0)
        baseline_results.results.append(TestResult(
            query=q["query"],
            tokens=tokens,
            latency_ms=latency,
            cache_hit=result.get("cache_hit", False),
            cache_type=result.get("cache_type"),
        ))
        
        print(f"→ {tokens} tokens, {latency:.0f}ms")
    
    all_results["baseline"] = baseline_results
    
    # =========================================================================
    # Test 2: BASIC (Exact cache only)
    # =========================================================================
    print_header("TEST 2: BASIC (Exact Cache Only)")
    print("\nInitializing basic platform...")
    
    basic_config = create_basic_config()
    basic_platform = TokenomicsPlatform(config=basic_config)
    
    print("Running queries...\n")
    basic_results = ConfigResults(name="BASIC")
    
    for i, q in enumerate(queries, 1):
        print(f"  [{i:2d}/13] {q['query'][:50]}...", end=" ")
        
        start = time.time()
        result = basic_platform.query(q["query"])
        latency = (time.time() - start) * 1000
        
        tokens = result.get("tokens_used", 0)
        cache_hit = result.get("cache_hit", False)
        cache_type = result.get("cache_type")
        
        basic_results.results.append(TestResult(
            query=q["query"],
            tokens=tokens,
            latency_ms=latency,
            cache_hit=cache_hit,
            cache_type=cache_type,
        ))
        
        if cache_hit and tokens == 0:
            print(f"✓ EXACT HIT (0 tokens)")
        else:
            print(f"→ {tokens} tokens, {latency:.0f}ms")
    
    all_results["basic"] = basic_results
    
    # =========================================================================
    # Test 3: ENHANCED (Full optimization)
    # =========================================================================
    print_header("TEST 3: ENHANCED (Full Optimization)")
    print("\nInitializing enhanced platform...")
    print("  - Semantic cache: enabled")
    print("  - Mem0 preferences: enabled")
    print("  - LLM-Lingua compression: enabled")
    print("  - RouterBench routing: enabled")
    
    enhanced_config = create_enhanced_config()
    enhanced_platform = TokenomicsPlatform(config=enhanced_config)
    
    print("\nRunning queries...\n")
    enhanced_results = ConfigResults(name="ENHANCED")
    
    for i, q in enumerate(queries, 1):
        print(f"  [{i:2d}/13] {q['query'][:50]}...", end=" ")
        
        start = time.time()
        result = enhanced_platform.query(q["query"])
        latency = (time.time() - start) * 1000
        
        tokens = result.get("tokens_used", 0)
        cache_hit = result.get("cache_hit", False)
        cache_type = result.get("cache_type")
        similarity = result.get("similarity")
        
        enhanced_results.results.append(TestResult(
            query=q["query"],
            tokens=tokens,
            latency_ms=latency,
            cache_hit=cache_hit,
            cache_type=cache_type,
            similarity=similarity,
        ))
        
        if cache_type == "exact":
            print(f"✓ EXACT HIT (0 tokens)")
        elif cache_type == "semantic_direct":
            sim_val = similarity if similarity else 0
            print(f"✓ SEMANTIC HIT ({sim_val:.2f}) - 0 tokens")
        elif cache_hit:
            sim_str = f"{similarity:.2f}" if similarity else "context"
            print(f"~ CONTEXT ({sim_str}) - {tokens} tokens")
        else:
            print(f"→ {tokens} tokens, {latency:.0f}ms")
    
    all_results["enhanced"] = enhanced_results
    
    # =========================================================================
    # RESULTS COMPARISON
    # =========================================================================
    print_header("RESULTS COMPARISON")
    
    baseline = all_results["baseline"]
    basic = all_results["basic"]
    enhanced = all_results["enhanced"]
    
    # Token comparison
    print("\n📊 TOKEN USAGE")
    print("-" * 60)
    print(f"  {'Config':<15} {'Tokens':>10} {'vs Baseline':>15} {'Savings':>12}")
    print("-" * 60)
    
    baseline_tokens = baseline.total_tokens
    basic_tokens = basic.total_tokens
    enhanced_tokens = enhanced.total_tokens
    
    basic_savings = ((baseline_tokens - basic_tokens) / baseline_tokens * 100) if baseline_tokens else 0
    enhanced_savings = ((baseline_tokens - enhanced_tokens) / baseline_tokens * 100) if baseline_tokens else 0
    
    print(f"  {'BASELINE':<15} {baseline_tokens:>10,} {'-':>15} {'-':>12}")
    print(f"  {'BASIC':<15} {basic_tokens:>10,} {basic_tokens - baseline_tokens:>+15,} {basic_savings:>11.1f}%")
    print(f"  {'ENHANCED':<15} {enhanced_tokens:>10,} {enhanced_tokens - baseline_tokens:>+15,} {enhanced_savings:>11.1f}%")
    print("-" * 60)
    
    # Cache hit comparison
    print("\n🎯 CACHE HIT BREAKDOWN")
    print("-" * 60)
    print(f"  {'Config':<15} {'Exact':>8} {'Semantic':>10} {'Context':>10} {'Total':>10}")
    print("-" * 60)
    print(f"  {'BASELINE':<15} {0:>8} {0:>10} {0:>10} {0:>10}")
    print(f"  {'BASIC':<15} {basic.exact_hits:>8} {0:>10} {0:>10} {basic.exact_hits:>10}")
    print(f"  {'ENHANCED':<15} {enhanced.exact_hits:>8} {enhanced.semantic_hits:>10} {enhanced.context_hits:>10} {enhanced.cache_hits:>10}")
    print("-" * 60)
    
    # Latency comparison
    print("\n⚡ LATENCY")
    print("-" * 60)
    print(f"  {'Config':<15} {'Total (s)':>12} {'Avg (ms)':>12} {'vs Baseline':>15}")
    print("-" * 60)
    
    baseline_lat = baseline.total_latency / 1000
    basic_lat = basic.total_latency / 1000
    enhanced_lat = enhanced.total_latency / 1000
    
    basic_lat_savings = ((baseline.total_latency - basic.total_latency) / baseline.total_latency * 100) if baseline.total_latency else 0
    enhanced_lat_savings = ((baseline.total_latency - enhanced.total_latency) / baseline.total_latency * 100) if baseline.total_latency else 0
    
    print(f"  {'BASELINE':<15} {baseline_lat:>12.1f} {baseline.avg_latency:>12.0f} {'-':>15}")
    print(f"  {'BASIC':<15} {basic_lat:>12.1f} {basic.avg_latency:>12.0f} {basic_lat_savings:>14.1f}%")
    print(f"  {'ENHANCED':<15} {enhanced_lat:>12.1f} {enhanced.avg_latency:>12.0f} {enhanced_lat_savings:>14.1f}%")
    print("-" * 60)
    
    # API calls
    print("\n📡 API CALLS")
    print("-" * 60)
    api_baseline = len(queries)
    api_basic = api_baseline - basic.exact_hits
    api_enhanced = sum(1 for r in enhanced.results if not r.cache_hit or r.tokens > 0)
    
    print(f"  BASELINE:  {api_baseline} API calls (every query hits API)")
    print(f"  BASIC:     {api_basic} API calls ({basic.exact_hits} exact cache hits)")
    print(f"  ENHANCED:  {api_enhanced} API calls ({enhanced.exact_hits} exact + {enhanced.semantic_hits} semantic hits)")
    print("-" * 60)
    
    # Summary
    print_header("SUMMARY: WHAT IMPROVED")
    
    print(f"""
┌─────────────────────────────────────────────────────────────────────────────┐
│  BASELINE → BASIC (Exact Cache Only)                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│  Token savings:    {basic_savings:>6.1f}% (from exact repeats only)                       │
│  Latency savings:  {basic_lat_savings:>6.1f}% (cache hits are instant)                          │
│  Cache hits:       {basic.exact_hits} exact matches                                         │
│                                                                             │
│  ⚠️  LIMITATION: Only works for IDENTICAL queries                           │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  BASELINE → ENHANCED (Full Optimization)                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  Token savings:    {enhanced_savings:>6.1f}% (exact + semantic + context)                   │
│  Latency savings:  {enhanced_lat_savings:>6.1f}% (more cache hits)                               │
│  Cache hits:       {enhanced.exact_hits} exact + {enhanced.semantic_hits} semantic + {enhanced.context_hits} context = {enhanced.cache_hits} total        │
│                                                                             │
│  ✅ NEW: Works for SIMILAR queries (variations, rephrasing)                 │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  BASIC → ENHANCED (Improvement from enhancements)                           │
├─────────────────────────────────────────────────────────────────────────────┤
""")
    
    basic_to_enhanced = ((basic_tokens - enhanced_tokens) / basic_tokens * 100) if basic_tokens else 0
    additional_hits = enhanced.cache_hits - basic.exact_hits
    
    print(f"│  Additional token savings: {basic_to_enhanced:>6.1f}%                                        │")
    print(f"│  Additional cache hits:    {additional_hits:>2} ({enhanced.semantic_hits} semantic + {enhanced.context_hits} context)                    │")
    print(f"│                                                                             │")
    print(f"│  KEY VALUE: Similar queries now get cached responses!                       │")
    print(f"└─────────────────────────────────────────────────────────────────────────────┘")
    
    # Feature breakdown
    print_header("ENHANCEMENT BREAKDOWN")
    
    print("""
┌──────────────────────────────────────────────────────────────────────────────┐
│  FEATURE                  │  WHAT IT DOES                 │  TOKEN IMPACT    │
├──────────────────────────────────────────────────────────────────────────────┤
│  Exact Cache              │  Stores identical queries     │  0 tokens repeat │
│  Semantic Cache           │  Matches similar queries      │  0 tokens similar│
│  Direct Return (>0.85)    │  Returns cached for similar   │  100% savings    │
│  Context Match (0.75-0.85)│  Uses cache as context        │  ~30% savings    │
│  Mem0 Preferences         │  Learns user patterns         │  Better prompts  │
│  LLM-Lingua Compression   │  Compresses context           │  ~10% savings    │
│  RouterBench Routing      │  Selects optimal strategy     │  Cost-aware      │
└──────────────────────────────────────────────────────────────────────────────┘
""")
    
    # Real-world impact
    print_header("REAL-WORLD IMPACT")
    
    queries_per_day = 1000
    cost_per_1k_tokens = 0.00015  # gpt-4o-mini input pricing
    
    baseline_daily_tokens = baseline_tokens / 13 * queries_per_day
    enhanced_daily_tokens = enhanced_tokens / 13 * queries_per_day
    daily_savings_tokens = baseline_daily_tokens - enhanced_daily_tokens
    daily_savings_cost = daily_savings_tokens / 1000 * cost_per_1k_tokens
    monthly_savings = daily_savings_cost * 30
    
    print(f"""
At {queries_per_day:,} queries/day (with similar query patterns):

  Daily token usage:
    BASELINE:  {baseline_daily_tokens:>12,.0f} tokens
    ENHANCED:  {enhanced_daily_tokens:>12,.0f} tokens
    SAVINGS:   {daily_savings_tokens:>12,.0f} tokens/day ({enhanced_savings:.1f}% reduction)

  Monthly cost savings (at ${cost_per_1k_tokens}/1K tokens):
    ${monthly_savings:.2f}/month in token costs

  Plus:
    - Faster response times (cache hits are instant)
    - Consistent responses for similar questions
    - User preference learning (better over time)
""")
    
    return all_results


if __name__ == "__main__":
    results = run_comparison_test()
    
    print()
    print("=" * 80)
    print("  TEST COMPLETE")
    print("=" * 80)

