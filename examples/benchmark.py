"""Benchmarking script for Tokenomics platform."""

import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenomics.core import TokenomicsPlatform
from tokenomics.config import TokenomicsConfig
from usage_tracker import UsageTracker


def benchmark_queries(
    platform: TokenomicsPlatform,
    queries: List[str],
    tracker: UsageTracker,
    use_cache: bool = True,
    use_bandit: bool = True,
) -> Dict:
    """
    Benchmark platform performance.
    
    Returns:
        Dictionary with benchmark results
    """
    results = {
        "total_queries": len(queries),
        "cache_hits": 0,
        "total_tokens": 0,
        "total_latency_ms": 0,
        "responses": [],
    }
    
    start_time = time.time()
    
    for query in queries:
        result = platform.query(
            query=query,
            use_cache=use_cache,
            use_bandit=use_bandit,
        )
        
        # Track usage
        tracker.record_query(
            query=query,
            response=result["response"],
            tokens_used=result["tokens_used"],
            latency_ms=result["latency_ms"],
            cache_hit=result["cache_hit"],
            cache_type=result.get("cache_type", "none"),
            strategy=result.get("strategy", "none"),
            model=platform.config.llm.model,
        )
        
        results["total_tokens"] += result["tokens_used"]
        results["total_latency_ms"] += result["latency_ms"]
        
        if result["cache_hit"]:
            results["cache_hits"] += 1
        
        results["responses"].append(result)
    
    results["total_time_ms"] = (time.time() - start_time) * 1000
    results["avg_tokens_per_query"] = results["total_tokens"] / len(queries)
    results["avg_latency_ms"] = results["total_latency_ms"] / len(queries)
    results["cache_hit_rate"] = results["cache_hits"] / len(queries)
    
    return results


def main():
    """Run benchmark."""
    # Ensure .env is loaded
    import os
    from dotenv import load_dotenv
    from pathlib import Path
    load_dotenv(Path(__file__).parent.parent / '.env')
    
    # Set API key directly if not loaded from .env
    if not os.getenv("GEMINI_API_KEY"):
        os.environ["GEMINI_API_KEY"] = "AIzaSyCvSI80PtKuVejnkIiiNxjjN6PyRRngB1E"
    
    config = TokenomicsConfig.from_env()
    platform = TokenomicsPlatform(config=config)
    
    # Test queries
    test_queries = [
        "What is Python?",
        "Explain recursion",
        "What is Python?",  # Duplicate
        "How does HTTP work?",
        "What is Python?",  # Another duplicate
        "Explain machine learning",
        "What is recursion?",  # Semantic duplicate
    ]
    
    print("=" * 80)
    print("Tokenomics Platform - Comprehensive Benchmark")
    print("=" * 80)
    print(f"Started at: {datetime.now().isoformat()}")
    print(f"Total queries: {len(test_queries)}")
    print()
    
    # Benchmark with cache
    tracker_with = UsageTracker(output_file="usage_report_with_cache.json")
    print("[1] Benchmarking WITH cache and bandit optimization...")
    print("-" * 80)
    results_with = benchmark_queries(platform, test_queries, tracker_with, use_cache=True, use_bandit=True)
    
    tracker_with.save_report()
    tracker_with.print_summary()
    
    print("\n" + "=" * 80)
    print("[2] Benchmarking WITHOUT cache (baseline comparison)...")
    print("-" * 80)
    
    # Reset and benchmark without cache
    platform.memory.clear()
    platform.bandit.reset()
    
    tracker_without = UsageTracker(output_file="usage_report_without_cache.json")
    results_without = benchmark_queries(platform, test_queries, tracker_without, use_cache=False, use_bandit=False)
    
    tracker_without.save_report()
    tracker_without.print_summary()
    
    # Calculate savings
    summary_with = tracker_with.get_summary()
    summary_without = tracker_without.get_summary()
    
    token_savings = summary_without['total_tokens_used'] - summary_with['total_tokens_used']
    token_savings_pct = (token_savings / summary_without['total_tokens_used']) * 100 if summary_without['total_tokens_used'] > 0 else 0
    
    latency_reduction = summary_without['total_latency_ms'] - summary_with['total_latency_ms']
    
    print("\n" + "=" * 80)
    print("COMPREHENSIVE SAVINGS ANALYSIS")
    print("=" * 80)
    print(f"Token Savings: {token_savings:,} tokens ({token_savings_pct:.1f}% reduction)")
    print(f"Latency Reduction: {latency_reduction:.2f} ms ({((latency_reduction / summary_without['total_latency_ms']) * 100) if summary_without['total_latency_ms'] > 0 else 0:.1f}% faster)")
    print(f"Cache Hit Rate: {summary_with['cache_hit_rate_percent']:.1f}%")
    print()
    print("DETAILED REPORTS SAVED:")
    print(f"  - With cache: usage_report_with_cache.json")
    print(f"  - Without cache: usage_report_without_cache.json")
    print("=" * 80)


if __name__ == "__main__":
    main()

