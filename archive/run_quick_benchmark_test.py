"""Quick benchmark to verify the regression fix."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from tokenomics.core import TokenomicsPlatform
from tokenomics.config import TokenomicsConfig

def run_quick_benchmark(num_queries=15):
    """Run a quick benchmark with diverse queries."""
    print("=" * 80)
    print(f"QUICK BENCHMARK - {num_queries} queries")
    print("=" * 80)
    print()
    
    # Load config
    config = TokenomicsConfig.from_env()
    print(f"Provider: {config.llm.provider}, Model: {config.llm.model}")
    
    # Initialize platform
    platform = TokenomicsPlatform(config=config)
    
    # Diverse test queries
    test_queries = [
        "What is the refund policy?",
        "How do I cancel my subscription?",
        "Can I customize the dashboard?",
        "What payment methods do you accept?",
        "Why is my account locked?",
        "How do I contact support?",
        "What is your privacy policy?",
        "Why did my upload fail?",
        "How do I enable two-factor authentication?",
        "What languages are supported?",
        "How do I change my billing cycle?",
        "Can you help me understand billing better?",
        "What's the best way to handle billing for my use case?",
        "I'm having trouble with billing when I try to...",
        "Is there a way to customize billing settings?",
    ][:num_queries]
    
    results = []
    token_regressions = 0
    latency_regressions = 0
    total_baseline_tokens = 0
    total_optimized_tokens = 0
    
    for i, query in enumerate(test_queries):
        print(f"\n[{i+1}/{len(test_queries)}] {query[:50]}...")
        
        try:
            # Clear cache for fair comparison (if available)
            if hasattr(platform.memory, 'exact_cache'):
                platform.memory.exact_cache.clear()
            
            result = platform.query(
                query=query,
                use_cache=False,  # No caching for fair comparison
                use_bandit=True,
                use_cost_aware_routing=True,
            )
            
            baseline = result.get("baseline_comparison_result", {})
            
            baseline_tokens = baseline.get("tokens_used", 0)
            optimized_tokens = result.get("tokens_used", 0)
            baseline_latency = baseline.get("latency_ms", 0)
            optimized_latency = result.get("latency_ms", 0)
            
            total_baseline_tokens += baseline_tokens
            total_optimized_tokens += optimized_tokens
            
            has_token_regression = optimized_tokens > baseline_tokens
            has_latency_regression = optimized_latency > baseline_latency
            
            if has_token_regression:
                token_regressions += 1
                token_status = f"❌ +{optimized_tokens - baseline_tokens}"
            else:
                savings = baseline_tokens - optimized_tokens
                token_status = f"✓ -{savings}" if savings > 0 else "= 0"
                
            if has_latency_regression:
                latency_regressions += 1
            
            print(f"  Tokens: B={baseline_tokens} O={optimized_tokens} [{token_status}]")
            print(f"  Latency: B={baseline_latency:.0f}ms O={optimized_latency:.0f}ms")
            
            results.append({
                "query": query,
                "baseline_tokens": baseline_tokens,
                "optimized_tokens": optimized_tokens,
                "baseline_latency": baseline_latency,
                "optimized_latency": optimized_latency,
                "token_regression": has_token_regression,
                "latency_regression": has_latency_regression,
            })
            
        except Exception as e:
            print(f"  Error: {e}")
            results.append({"query": query, "error": str(e)})
    
    # Summary
    valid_results = [r for r in results if "error" not in r]
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total queries: {len(test_queries)}")
    print(f"Successful: {len(valid_results)}")
    print()
    print(f"Token regressions: {token_regressions}/{len(valid_results)} ({token_regressions/len(valid_results)*100:.1f}%)")
    print(f"Latency regressions: {latency_regressions}/{len(valid_results)} ({latency_regressions/len(valid_results)*100:.1f}%)")
    print()
    print(f"Total baseline tokens: {total_baseline_tokens}")
    print(f"Total optimized tokens: {total_optimized_tokens}")
    
    if total_baseline_tokens > 0:
        total_savings = total_baseline_tokens - total_optimized_tokens
        savings_pct = (total_savings / total_baseline_tokens) * 100
        print(f"Total savings: {total_savings} tokens ({savings_pct:.1f}%)")
    
    # Save results
    with open("quick_benchmark_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: quick_benchmark_test_results.json")
    
    return token_regressions == 0


if __name__ == "__main__":
    num = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    success = run_quick_benchmark(num)
    sys.exit(0 if success else 1)

