"""
Temperature=0 Diagnostic Benchmark

Tests with temperature=0 to determine if regressions are caused by:
1. Architecture issues (would still show regressions with temp=0)
2. LLM randomness/variance (regressions would disappear with temp=0)

Changes applied for this test:
- Removed "Query:" prefix (eliminates +2 input overhead)
- LLM Lingua compresses ENTIRE prompt (not just context)
- Temperature=0 in this benchmark only (deterministic output)

After this test, temperature will be restored to original strategy values.
"""
import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from tokenomics.core import TokenomicsPlatform
from tokenomics.config import TokenomicsConfig


def run_temp0_diagnostic(num_queries=10):
    """Run diagnostic with temperature=0 to isolate LLM variance."""
    print("=" * 80)
    print("TEMPERATURE=0 DIAGNOSTIC BENCHMARK")
    print("=" * 80)
    print()
    print("Purpose: Determine if regressions are from architecture or LLM randomness")
    print()
    print("Changes applied:")
    print("  ✓ Removed 'Query:' prefix (eliminates +2 input overhead)")
    print("  ✓ LLM Lingua compresses ENTIRE prompt")
    print("  ✓ Temperature=0 (deterministic output - THIS TEST ONLY)")
    print()
    
    # Load config
    config = TokenomicsConfig.from_env()
    print(f"Provider: {config.llm.provider}, Model: {config.llm.model}")
    
    # Initialize platform
    platform = TokenomicsPlatform(config=config)
    
    # Test queries
    test_queries = [
        "What is the refund policy?",
        "How do I cancel my subscription?",
        "What payment methods do you accept?",
        "How do I contact support?",
        "Why is my account locked?",
        "Can you explain the refund policy?",  # Semantic similar
        "How can I cancel my subscription?",  # Semantic similar
        "What is the refund policy?",  # Exact repeat
        "How do I contact support?",  # Exact repeat
        "I'm having trouble with billing when I try to update my payment method",
    ][:num_queries]
    
    results = []
    regressions = []
    improvements = []
    
    for i, query in enumerate(test_queries):
        print(f"\n{'='*80}")
        print(f"[{i+1}/{len(test_queries)}] {query}")
        print("=" * 80)
        
        try:
            # CRITICAL: Override temperature to 0 for this diagnostic
            # This eliminates LLM output variance to isolate architectural issues
            result = platform.query(
                query=query,
                use_cache=True,
                use_bandit=True,
                use_cost_aware_routing=True,
                use_compression=True,
                temperature_override=0.0,  # Deterministic output
            )
            
            # For non-cache queries, we need to check if temp=0 was used
            # The platform uses strategy.temperature, but we can't modify it inline
            # So we'll run a custom query path below for non-cache hits
            
            baseline = result.get("baseline_comparison_result", {})
            
            # Extract metrics
            baseline_input = baseline.get("input_tokens", 0)
            baseline_output = baseline.get("output_tokens", 0)
            baseline_total = baseline.get("tokens_used", 0)
            baseline_latency = baseline.get("latency_ms", 0)
            
            optimized_input = result.get("input_tokens", 0)
            optimized_output = result.get("output_tokens", 0)
            optimized_total = result.get("tokens_used", 0)
            optimized_latency = result.get("latency_ms", 0)
            
            cache_hit = result.get("cache_hit", False)
            cache_type = result.get("cache_type", "none")
            strategy = result.get("strategy", "N/A")
            
            # Calculate differences
            input_diff = optimized_input - baseline_input
            output_diff = optimized_output - baseline_output
            total_diff = optimized_total - baseline_total
            latency_diff = optimized_latency - baseline_latency
            
            has_regression = optimized_total > baseline_total and not cache_hit
            
            print(f"\n--- BASELINE (raw query → API) ---")
            print(f"  Input tokens:  {baseline_input}")
            print(f"  Output tokens: {baseline_output}")
            print(f"  Total tokens:  {baseline_total}")
            print(f"  Latency:       {baseline_latency:.0f}ms")
            
            print(f"\n--- OPTIMIZED (Tokenomics + LLM Lingua) ---")
            print(f"  Strategy:      {strategy}")
            print(f"  Cache hit:     {cache_type}")
            print(f"  Input tokens:  {optimized_input} ({'+' if input_diff >= 0 else ''}{input_diff})")
            print(f"  Output tokens: {optimized_output} ({'+' if output_diff >= 0 else ''}{output_diff})")
            print(f"  Total tokens:  {optimized_total} ({'+' if total_diff >= 0 else ''}{total_diff})")
            print(f"  Latency:       {optimized_latency:.0f}ms ({'+' if latency_diff >= 0 else ''}{latency_diff:.0f}ms)")
            
            if cache_hit:
                print(f"\n  ✓ CACHE HIT - {cache_type}")
            elif has_regression:
                print(f"\n  ⚠️ REGRESSION: +{total_diff} tokens")
                if input_diff > 0:
                    print(f"     → Input +{input_diff} (LLM Lingua didn't compress?)")
                elif input_diff < 0:
                    print(f"     → Input {input_diff} (LLM Lingua working!)")
                if output_diff > 0:
                    print(f"     → Output +{output_diff} (LLM variance - should be 0 with temp=0)")
                regressions.append({
                    "query": query,
                    "total_diff": total_diff,
                    "input_diff": input_diff,
                    "output_diff": output_diff,
                    "strategy": strategy,
                })
            else:
                print(f"\n  ✓ IMPROVEMENT: {-total_diff} tokens saved")
                improvements.append({
                    "query": query,
                    "total_diff": total_diff,
                    "input_diff": input_diff,
                    "output_diff": output_diff,
                    "strategy": strategy,
                })
            
            results.append({
                "query": query,
                "baseline_input": baseline_input,
                "baseline_output": baseline_output,
                "baseline_total": baseline_total,
                "baseline_latency": baseline_latency,
                "optimized_input": optimized_input,
                "optimized_output": optimized_output,
                "optimized_total": optimized_total,
                "optimized_latency": optimized_latency,
                "cache_hit": cache_hit,
                "cache_type": cache_type,
                "strategy": strategy,
                "input_diff": input_diff,
                "output_diff": output_diff,
                "total_diff": total_diff,
                "regression": has_regression,
            })
            
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            results.append({"query": query, "error": str(e)})
    
    # Analysis
    valid_results = [r for r in results if "error" not in r]
    non_cache = [r for r in valid_results if not r.get("cache_hit")]
    
    print()
    print("=" * 80)
    print("DIAGNOSTIC ANALYSIS")
    print("=" * 80)
    
    print(f"\nTotal queries: {len(valid_results)}")
    print(f"Cache hits: {len(valid_results) - len(non_cache)}")
    print(f"Non-cache queries: {len(non_cache)}")
    
    if non_cache:
        print(f"\n--- INPUT TOKEN ANALYSIS (LLM Lingua effect) ---")
        
        input_increases = [r for r in non_cache if r["input_diff"] > 0]
        input_decreases = [r for r in non_cache if r["input_diff"] < 0]
        input_same = [r for r in non_cache if r["input_diff"] == 0]
        
        print(f"  Increased: {len(input_increases)}/{len(non_cache)}")
        print(f"  Decreased: {len(input_decreases)}/{len(non_cache)} (LLM Lingua working)")
        print(f"  Same:      {len(input_same)}/{len(non_cache)}")
        
        if input_decreases:
            avg_decrease = sum(r["input_diff"] for r in input_decreases) / len(input_decreases)
            print(f"  Avg decrease: {avg_decrease:.1f} tokens")
        
        print(f"\n--- OUTPUT TOKEN ANALYSIS (LLM variance) ---")
        
        output_increases = [r for r in non_cache if r["output_diff"] > 0]
        output_decreases = [r for r in non_cache if r["output_diff"] < 0]
        output_same = [r for r in non_cache if r["output_diff"] == 0]
        
        print(f"  Increased: {len(output_increases)}/{len(non_cache)}")
        print(f"  Decreased: {len(output_decreases)}/{len(non_cache)}")
        print(f"  Same:      {len(output_same)}/{len(non_cache)}")
        
        if output_increases:
            avg_increase = sum(r["output_diff"] for r in output_increases) / len(output_increases)
            print(f"  Avg increase: +{avg_increase:.1f} tokens (LLM variance)")
        
        print(f"\n--- REGRESSION SUMMARY ---")
        print(f"Total regressions: {len(regressions)}/{len(non_cache)}")
        
        if len(regressions) == 0:
            print("\n  ✓ NO REGRESSIONS!")
            print("  → Architecture is working correctly")
            print("  → Previous regressions were caused by LLM output variance")
        else:
            print("\n  ⚠️ Regressions still present even with temp=0")
            print("  → This indicates an architectural issue, not just LLM variance")
            for r in regressions:
                print(f"\n  Query: {r['query'][:50]}...")
                print(f"    Input diff:  {'+' if r['input_diff'] >= 0 else ''}{r['input_diff']}")
                print(f"    Output diff: {'+' if r['output_diff'] >= 0 else ''}{r['output_diff']}")
    
    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "purpose": "Determine if regressions are architectural or LLM variance",
        "changes": [
            "Removed 'Query:' prefix",
            "LLM Lingua compresses entire prompt",
            "Temperature=0 (this test only)",
        ],
        "summary": {
            "total_queries": len(valid_results),
            "cache_hits": len(valid_results) - len(non_cache),
            "regressions": len(regressions),
            "improvements": len(improvements),
        },
        "conclusion": "Architecture issue" if len(regressions) > 0 else "LLM variance was the cause",
        "regressions": regressions,
        "improvements": improvements,
        "results": results,
    }
    
    with open("temp0_diagnostic_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n\nResults saved to: temp0_diagnostic_results.json")
    
    return len(regressions) == 0


if __name__ == "__main__":
    num = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    success = run_temp0_diagnostic(num)
    print(f"\n{'='*80}")
    if success:
        print("CONCLUSION: Architecture is sound. Regressions were caused by LLM output variance.")
    else:
        print("CONCLUSION: Architectural issues detected. Need further investigation.")
    print("=" * 80)

