"""Analyze diagnostic benchmark results to identify regression patterns."""

import json
import argparse
from pathlib import Path
from typing import Dict, Any, List
from collections import defaultdict
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


def analyze_regressions(results_path: str) -> Dict[str, Any]:
    """
    Analyze benchmark results to identify regression patterns.
    
    Args:
        results_path: Path to diagnostic benchmark results JSON
    
    Returns:
        Dictionary with regression analysis
    """
    # Load results
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Filter valid results (no errors)
    valid_results = [r for r in results if "error" not in r]
    
    # Filter regression queries
    token_regressions = [
        r for r in valid_results 
        if r.get("optimized_total_tokens", 0) > r.get("baseline_total_tokens", 0)
    ]
    
    latency_regressions = [
        r for r in valid_results 
        if r.get("optimized_latency_ms", 0) > r.get("baseline_latency_ms", 0)
    ]
    
    # Group token regressions by diagnostic fields
    token_by_cache_tier = defaultdict(int)
    token_by_strategy_arm = defaultdict(int)
    token_by_model_used = defaultdict(int)
    token_by_query_type = defaultdict(int)
    token_by_capsule_tokens = defaultdict(int)
    
    for r in token_regressions:
        cache_tier = r.get("cache_tier", "unknown")
        strategy_arm = r.get("strategy_arm") or "none"
        model_used = r.get("model_used", "unknown")
        query_type = r.get("query_type", "unknown")
        capsule_tokens = r.get("capsule_tokens", 0)
        
        token_by_cache_tier[cache_tier] += 1
        token_by_strategy_arm[strategy_arm] += 1
        token_by_model_used[model_used] += 1
        token_by_query_type[query_type] += 1
        
        # Group capsule_tokens into ranges
        if capsule_tokens == 0:
            token_by_capsule_tokens["0"] += 1
        elif capsule_tokens < 50:
            token_by_capsule_tokens["1-49"] += 1
        elif capsule_tokens < 100:
            token_by_capsule_tokens["50-99"] += 1
        else:
            token_by_capsule_tokens["100+"] += 1
    
    # Group latency regressions by diagnostic fields
    latency_by_cache_tier = defaultdict(int)
    latency_by_strategy_arm = defaultdict(int)
    latency_by_model_used = defaultdict(int)
    latency_by_query_type = defaultdict(int)
    
    for r in latency_regressions:
        cache_tier = r.get("cache_tier", "unknown")
        strategy_arm = r.get("strategy_arm") or "none"
        model_used = r.get("model_used", "unknown")
        query_type = r.get("query_type", "unknown")
        
        latency_by_cache_tier[cache_tier] += 1
        latency_by_strategy_arm[strategy_arm] += 1
        latency_by_model_used[model_used] += 1
        latency_by_query_type[query_type] += 1
    
    # Print analysis
    print()
    print("=" * 80)
    print("REGRESSION ANALYSIS")
    print("=" * 80)
    print()
    print(f"Total queries analyzed: {len(valid_results)}")
    print(f"Token regressions: {len(token_regressions)} ({len(token_regressions)/len(valid_results)*100:.1f}%)")
    print(f"Latency regressions: {len(latency_regressions)} ({len(latency_regressions)/len(valid_results)*100:.1f}%)")
    print()
    
    if token_regressions:
        print("=" * 80)
        print("TOKEN REGRESSIONS BY DIAGNOSTIC FIELD")
        print("=" * 80)
        print()
        
        print("Token regressions by cache_tier:")
        for tier, count in sorted(token_by_cache_tier.items(), key=lambda x: -x[1]):
            pct = (count / len(token_regressions) * 100) if token_regressions else 0
            print(f"  {tier}: {count} ({pct:.1f}%)")
        print()
        
        print("Token regressions by strategy_arm:")
        for arm, count in sorted(token_by_strategy_arm.items(), key=lambda x: -x[1]):
            pct = (count / len(token_regressions) * 100) if token_regressions else 0
            print(f"  {arm}: {count} ({pct:.1f}%)")
        print()
        
        print("Token regressions by model_used:")
        for model, count in sorted(token_by_model_used.items(), key=lambda x: -x[1]):
            pct = (count / len(token_regressions) * 100) if token_regressions else 0
            print(f"  {model}: {count} ({pct:.1f}%)")
        print()
        
        print("Token regressions by query_type:")
        for qtype, count in sorted(token_by_query_type.items(), key=lambda x: -x[1]):
            pct = (count / len(token_regressions) * 100) if token_regressions else 0
            print(f"  {qtype}: {count} ({pct:.1f}%)")
        print()
        
        print("Token regressions by capsule_tokens range:")
        for range_str, count in sorted(token_by_capsule_tokens.items(), key=lambda x: -x[1]):
            pct = (count / len(token_regressions) * 100) if token_regressions else 0
            print(f"  {range_str}: {count} ({pct:.1f}%)")
        print()
        
        # Show detailed examples
        print("Sample token regression queries:")
        for i, r in enumerate(token_regressions[:5], 1):
            print(f"  {i}. Query {r.get('query_id')}: {r.get('query_text', '')[:60]}...")
            print(f"     Baseline: {r.get('baseline_total_tokens')} tokens, Optimized: {r.get('optimized_total_tokens')} tokens (+{r.get('token_diff', 0)})")
            print(f"     cache_tier={r.get('cache_tier')}, strategy_arm={r.get('strategy_arm')}, capsule_tokens={r.get('capsule_tokens')}")
        print()
    
    if latency_regressions:
        print("=" * 80)
        print("LATENCY REGRESSIONS BY DIAGNOSTIC FIELD")
        print("=" * 80)
        print()
        
        print("Latency regressions by cache_tier:")
        for tier, count in sorted(latency_by_cache_tier.items(), key=lambda x: -x[1]):
            pct = (count / len(latency_regressions) * 100) if latency_regressions else 0
            print(f"  {tier}: {count} ({pct:.1f}%)")
        print()
        
        print("Latency regressions by strategy_arm:")
        for arm, count in sorted(latency_by_strategy_arm.items(), key=lambda x: -x[1]):
            pct = (count / len(latency_regressions) * 100) if latency_regressions else 0
            print(f"  {arm}: {count} ({pct:.1f}%)")
        print()
        
        print("Latency regressions by model_used:")
        for model, count in sorted(latency_by_model_used.items(), key=lambda x: -x[1]):
            pct = (count / len(latency_regressions) * 100) if latency_regressions else 0
            print(f"  {model}: {count} ({pct:.1f}%)")
        print()
        
        print("Latency regressions by query_type:")
        for qtype, count in sorted(latency_by_query_type.items(), key=lambda x: -x[1]):
            pct = (count / len(latency_regressions) * 100) if latency_regressions else 0
            print(f"  {qtype}: {count} ({pct:.1f}%)")
        print()
        
        # Show detailed examples
        print("Sample latency regression queries:")
        for i, r in enumerate(latency_regressions[:5], 1):
            print(f"  {i}. Query {r.get('query_id')}: {r.get('query_text', '')[:60]}...")
            print(f"     Baseline: {r.get('baseline_latency_ms')}ms, Optimized: {r.get('optimized_latency_ms')}ms (+{r.get('latency_diff', 0):.2f}ms)")
            print(f"     cache_tier={r.get('cache_tier')}, strategy_arm={r.get('strategy_arm')}, model_used={r.get('model_used')}")
        print()
    
    print("=" * 80)
    
    # Return structured analysis
    return {
        "total_queries": len(valid_results),
        "token_regressions": len(token_regressions),
        "latency_regressions": len(latency_regressions),
        "token_regressions_by": {
            "cache_tier": dict(token_by_cache_tier),
            "strategy_arm": dict(token_by_strategy_arm),
            "model_used": dict(token_by_model_used),
            "query_type": dict(token_by_query_type),
            "capsule_tokens": dict(token_by_capsule_tokens),
        },
        "latency_regressions_by": {
            "cache_tier": dict(latency_by_cache_tier),
            "strategy_arm": dict(latency_by_strategy_arm),
            "model_used": dict(latency_by_model_used),
            "query_type": dict(latency_by_query_type),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze diagnostic benchmark results")
    parser.add_argument(
        "--results", 
        default="benchmarks/results/support_benchmark_diagnostics.json",
        help="Path to diagnostic benchmark results JSON"
    )
    args = parser.parse_args()
    
    analyze_regressions(args.results)


if __name__ == "__main__":
    main()










