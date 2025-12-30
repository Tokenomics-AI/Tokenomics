"""
True Baseline vs Optimized Comparison Benchmark

This benchmark compares:
- Baseline: Raw query → API (what teams do today, no optimization)
- Optimized: Full Tokenomics pipeline (cache, compression, bandit, RouterBench)

The comparison is now meaningful because:
1. Baseline has NO optimization (simple prompt, high max_tokens)
2. Optimized uses ALL features (cache, compression, aggressive token limits, prompt hints)
"""
import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from tokenomics.core import TokenomicsPlatform
from tokenomics.config import TokenomicsConfig


def run_true_comparison(num_queries=15):
    """Run a true baseline vs optimized comparison."""
    print("=" * 80)
    print("TRUE BASELINE VS OPTIMIZED COMPARISON")
    print("=" * 80)
    print()
    print("Baseline: Raw query → API (no optimization)")
    print("Optimized: Full Tokenomics pipeline (cache, compression, bandit, etc.)")
    print()
    
    # Load config
    config = TokenomicsConfig.from_env()
    print(f"Provider: {config.llm.provider}, Model: {config.llm.model}")
    
    # Initialize platform
    platform = TokenomicsPlatform(config=config)
    
    # Test queries - mix of types to test different scenarios
    test_queries = [
        # Simple queries - should benefit from aggressive limits
        "What is the refund policy?",
        "How do I cancel my subscription?",
        "What payment methods do you accept?",
        "How do I contact support?",
        
        # Paraphrase queries - should benefit from semantic cache
        "Can you explain the refund policy?",  # Similar to query 1
        "How can I cancel my subscription?",   # Similar to query 2
        
        # Medium complexity queries
        "Why is my account locked?",
        "Why did my upload fail?",
        "How do I enable two-factor authentication?",
        
        # Longer queries - benefit from compression
        "I'm having trouble with billing when I try to update my payment method",
        "What's the best way to handle billing for my use case?",
        "Can you help me understand billing better?",
        
        # Repeated queries - should hit exact cache
        "What is the refund policy?",  # Exact repeat of query 1
        "How do I contact support?",   # Exact repeat of query 4
        
        # Unique queries
        "What languages are supported?",
    ][:num_queries]
    
    results = []
    
    # Metrics
    total_baseline_tokens = 0
    total_optimized_tokens = 0
    total_baseline_latency = 0
    total_optimized_latency = 0
    cache_hits = 0
    token_savings_from_cache = 0
    token_regressions = 0
    
    for i, query in enumerate(test_queries):
        print(f"\n[{i+1}/{len(test_queries)}] {query[:60]}...")
        
        try:
            result = platform.query(
                query=query,
                use_cache=True,
                use_bandit=True,
                use_cost_aware_routing=True,
            )
            
            baseline = result.get("baseline_comparison_result", {})
            
            baseline_tokens = baseline.get("tokens_used", 0)
            optimized_tokens = result.get("tokens_used", 0)
            baseline_latency = baseline.get("latency_ms", 0)
            optimized_latency = result.get("latency_ms", 0)
            is_cache_hit = result.get("cache_hit", False)
            cache_type = result.get("cache_type", "none")
            strategy = result.get("strategy", "N/A")
            
            total_baseline_tokens += baseline_tokens
            total_optimized_tokens += optimized_tokens
            total_baseline_latency += baseline_latency
            total_optimized_latency += optimized_latency
            
            if is_cache_hit:
                cache_hits += 1
                token_savings_from_cache += baseline_tokens  # All baseline tokens saved
            
            has_regression = optimized_tokens > baseline_tokens
            if has_regression:
                token_regressions += 1
                status = f"⚠️  +{optimized_tokens - baseline_tokens}"
            elif is_cache_hit:
                status = f"🎯 CACHE ({cache_type})"
            else:
                savings = baseline_tokens - optimized_tokens
                status = f"✓ -{savings}" if savings > 0 else "= 0"
            
            print(f"  Strategy: {strategy}, Cache: {cache_type}")
            print(f"  Tokens: B={baseline_tokens} O={optimized_tokens} [{status}]")
            print(f"  Latency: B={baseline_latency:.0f}ms O={optimized_latency:.0f}ms")
            
            results.append({
                "query": query,
                "baseline_tokens": baseline_tokens,
                "optimized_tokens": optimized_tokens,
                "baseline_latency": baseline_latency,
                "optimized_latency": optimized_latency,
                "cache_hit": is_cache_hit,
                "cache_type": cache_type,
                "strategy": strategy,
                "regression": has_regression,
            })
            
        except Exception as e:
            print(f"  Error: {e}")
            results.append({"query": query, "error": str(e)})
    
    # Summary
    valid_results = [r for r in results if "error" not in r]
    non_cache_results = [r for r in valid_results if not r.get("cache_hit")]
    
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total queries: {len(test_queries)}")
    print(f"Successful: {len(valid_results)}")
    print()
    
    print("--- CACHE PERFORMANCE ---")
    print(f"Cache hits: {cache_hits}/{len(valid_results)} ({cache_hits/len(valid_results)*100:.1f}%)")
    print(f"Tokens saved from cache: {token_savings_from_cache}")
    print()
    
    print("--- TOKEN PERFORMANCE ---")
    print(f"Total baseline tokens: {total_baseline_tokens}")
    print(f"Total optimized tokens: {total_optimized_tokens}")
    if total_baseline_tokens > 0:
        total_savings = total_baseline_tokens - total_optimized_tokens
        savings_pct = (total_savings / total_baseline_tokens) * 100
        print(f"Total savings: {total_savings} tokens ({savings_pct:.1f}%)")
    print()
    
    print("--- NON-CACHE QUERIES ---")
    if non_cache_results:
        non_cache_baseline = sum(r["baseline_tokens"] for r in non_cache_results)
        non_cache_optimized = sum(r["optimized_tokens"] for r in non_cache_results)
        non_cache_savings = non_cache_baseline - non_cache_optimized
        non_cache_pct = (non_cache_savings / non_cache_baseline * 100) if non_cache_baseline > 0 else 0
        print(f"Non-cache queries: {len(non_cache_results)}")
        print(f"Non-cache baseline tokens: {non_cache_baseline}")
        print(f"Non-cache optimized tokens: {non_cache_optimized}")
        print(f"Non-cache savings: {non_cache_savings} tokens ({non_cache_pct:.1f}%)")
        print(f"Token regressions: {token_regressions}/{len(non_cache_results)} ({token_regressions/len(non_cache_results)*100:.1f}%)")
    print()
    
    print("--- LATENCY PERFORMANCE ---")
    print(f"Total baseline latency: {total_baseline_latency:.0f}ms")
    print(f"Total optimized latency: {total_optimized_latency:.0f}ms")
    if total_baseline_latency > 0:
        latency_savings = total_baseline_latency - total_optimized_latency
        latency_pct = (latency_savings / total_baseline_latency) * 100
        print(f"Total latency savings: {latency_savings:.0f}ms ({latency_pct:.1f}%)")
    
    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "provider": config.llm.provider,
            "model": config.llm.model,
            "num_queries": num_queries,
        },
        "summary": {
            "total_queries": len(test_queries),
            "successful": len(valid_results),
            "cache_hits": cache_hits,
            "token_regressions": token_regressions,
            "total_baseline_tokens": total_baseline_tokens,
            "total_optimized_tokens": total_optimized_tokens,
            "total_savings_tokens": total_baseline_tokens - total_optimized_tokens,
            "total_savings_percent": ((total_baseline_tokens - total_optimized_tokens) / total_baseline_tokens * 100) if total_baseline_tokens > 0 else 0,
        },
        "results": results,
    }
    
    with open("true_comparison_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: true_comparison_results.json")
    
    return len(valid_results) > 0


if __name__ == "__main__":
    num = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    success = run_true_comparison(num)
    sys.exit(0 if success else 1)










