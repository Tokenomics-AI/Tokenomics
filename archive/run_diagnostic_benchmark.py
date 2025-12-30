"""
Diagnostic Benchmark - Raw Architecture Analysis

Tests the platform WITHOUT artificial limits to understand:
1. Why some optimized runs use more tokens than baseline
2. Is LLM Lingua compression actually reducing input tokens?
3. Is context injection adding too many tokens?
4. What's the LLM response variance?

Architecture being tested:
- Token Orchestrator
- Smart Memory Layer (exact + semantic cache)
- Bandit Optimizer (with RouterBench)
- LLM Lingua compression (for context/prompt compression)
"""
import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from tokenomics.core import TokenomicsPlatform
from tokenomics.config import TokenomicsConfig


def run_diagnostic(num_queries=10):
    """Run diagnostic benchmark with detailed analysis."""
    print("=" * 80)
    print("DIAGNOSTIC BENCHMARK - RAW ARCHITECTURE ANALYSIS")
    print("=" * 80)
    print()
    print("Testing WITHOUT artificial limits:")
    print("  - Strategy limits: cheap=300, balanced=600, premium=1000")
    print("  - No prompt hints")
    print()
    print("Architecture being tested:")
    print("  - Token Orchestrator")
    print("  - Smart Memory Layer")
    print("  - Bandit Optimizer + RouterBench")
    print("  - LLM Lingua compression")
    print()
    
    # Load config
    config = TokenomicsConfig.from_env()
    print(f"Provider: {config.llm.provider}, Model: {config.llm.model}")
    
    # Initialize platform
    platform = TokenomicsPlatform(config=config)
    
    # Test queries - designed to test different scenarios
    test_queries = [
        # Fresh queries - no cache, pure LLM comparison
        "What is the refund policy?",
        "How do I cancel my subscription?",
        "What payment methods do you accept?",
        "How do I contact support?",
        "Why is my account locked?",
        
        # Paraphrase queries - should hit semantic cache
        "Can you explain the refund policy?",
        "How can I cancel my subscription?",
        
        # Repeat queries - should hit exact cache
        "What is the refund policy?",
        "How do I contact support?",
        
        # Complex query - might benefit from context
        "I'm having trouble with billing when I try to update my payment method",
    ][:num_queries]
    
    results = []
    
    # Detailed metrics
    regressions = []
    improvements = []
    
    for i, query in enumerate(test_queries):
        print(f"\n{'='*80}")
        print(f"[{i+1}/{len(test_queries)}] {query}")
        print("=" * 80)
        
        try:
            # Clear exact cache for fresh comparison (but keep semantic for testing)
            if hasattr(platform.memory, 'exact_cache'):
                # Don't clear - we want to test cache behavior
                pass
            
            result = platform.query(
                query=query,
                use_cache=True,
                use_bandit=True,
                use_cost_aware_routing=True,
                use_compression=True,  # Enable LLM Lingua
            )
            
            baseline = result.get("baseline_comparison_result", {})
            
            # Extract detailed metrics
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
            max_tokens_used = result.get("max_response_tokens", "N/A")
            
            # Calculate differences
            input_diff = optimized_input - baseline_input
            output_diff = optimized_output - baseline_output
            total_diff = optimized_total - baseline_total
            latency_diff = optimized_latency - baseline_latency
            
            has_regression = optimized_total > baseline_total
            
            print(f"\n--- BASELINE (raw query → API) ---")
            print(f"  Input tokens:  {baseline_input}")
            print(f"  Output tokens: {baseline_output}")
            print(f"  Total tokens:  {baseline_total}")
            print(f"  Latency:       {baseline_latency:.0f}ms")
            
            print(f"\n--- OPTIMIZED (full Tokenomics) ---")
            print(f"  Strategy:      {strategy}")
            print(f"  Cache hit:     {cache_type}")
            print(f"  Max tokens:    {max_tokens_used}")
            print(f"  Input tokens:  {optimized_input} ({'+' if input_diff >= 0 else ''}{input_diff})")
            print(f"  Output tokens: {optimized_output} ({'+' if output_diff >= 0 else ''}{output_diff})")
            print(f"  Total tokens:  {optimized_total} ({'+' if total_diff >= 0 else ''}{total_diff})")
            print(f"  Latency:       {optimized_latency:.0f}ms ({'+' if latency_diff >= 0 else ''}{latency_diff:.0f}ms)")
            
            if cache_hit:
                print(f"\n  ✓ CACHE HIT - {cache_type}")
            elif has_regression:
                print(f"\n  ⚠️ REGRESSION: +{total_diff} tokens")
                if input_diff > 0:
                    print(f"     → Input increased by {input_diff} (context injection overhead?)")
                if output_diff > 0:
                    print(f"     → Output increased by {output_diff} (LLM variance)")
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
                "max_tokens": max_tokens_used,
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
        print(f"\n--- NON-CACHE QUERY ANALYSIS ---")
        
        # Input token analysis
        input_increases = [r for r in non_cache if r["input_diff"] > 0]
        input_decreases = [r for r in non_cache if r["input_diff"] < 0]
        input_same = [r for r in non_cache if r["input_diff"] == 0]
        
        print(f"\nInput Tokens (LLM Lingua compression effect):")
        print(f"  Increased: {len(input_increases)}/{len(non_cache)}")
        print(f"  Decreased: {len(input_decreases)}/{len(non_cache)}")
        print(f"  Same:      {len(input_same)}/{len(non_cache)}")
        
        if input_increases:
            avg_increase = sum(r["input_diff"] for r in input_increases) / len(input_increases)
            print(f"  Avg increase: +{avg_increase:.1f} tokens")
            print(f"  → This suggests context injection adds more tokens than compression saves")
        
        if input_decreases:
            avg_decrease = sum(r["input_diff"] for r in input_decreases) / len(input_decreases)
            print(f"  Avg decrease: {avg_decrease:.1f} tokens")
            print(f"  → LLM Lingua compression is working")
        
        # Output token analysis
        output_increases = [r for r in non_cache if r["output_diff"] > 0]
        output_decreases = [r for r in non_cache if r["output_diff"] < 0]
        
        print(f"\nOutput Tokens (LLM response variance):")
        print(f"  Increased: {len(output_increases)}/{len(non_cache)}")
        print(f"  Decreased: {len(output_decreases)}/{len(non_cache)}")
        
        if output_increases:
            avg_increase = sum(r["output_diff"] for r in output_increases) / len(output_increases)
            print(f"  Avg increase: +{avg_increase:.1f} tokens")
            print(f"  → LLM response variance is causing regressions")
        
        # Regression analysis
        print(f"\n--- REGRESSION ROOT CAUSES ---")
        print(f"Total regressions: {len(regressions)}/{len(non_cache)}")
        
        for r in regressions:
            print(f"\n  Query: {r['query'][:50]}...")
            print(f"    Strategy: {r['strategy']}")
            print(f"    Input diff:  {'+' if r['input_diff'] >= 0 else ''}{r['input_diff']}")
            print(f"    Output diff: {'+' if r['output_diff'] >= 0 else ''}{r['output_diff']}")
            
            if r['input_diff'] > 0 and r['output_diff'] > 0:
                print(f"    → BOTH input and output increased")
            elif r['input_diff'] > 0:
                print(f"    → Input overhead > output savings")
            elif r['output_diff'] > 0:
                print(f"    → Pure LLM response variance (input was same/less)")
    
    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "provider": config.llm.provider,
            "model": config.llm.model,
            "strategy_limits": "original (300/600/1000)",
            "prompt_hints": "disabled",
        },
        "summary": {
            "total_queries": len(valid_results),
            "cache_hits": len(valid_results) - len(non_cache),
            "regressions": len(regressions),
            "improvements": len(improvements),
        },
        "regressions": regressions,
        "improvements": improvements,
        "results": results,
    }
    
    with open("diagnostic_benchmark_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n\nResults saved to: diagnostic_benchmark_results.json")
    
    return True


if __name__ == "__main__":
    num = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    run_diagnostic(num)










