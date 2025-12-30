"""
Real Comparison Benchmark: Raw LLM vs Tokenomics

Baseline (what teams do today):
- Single model (gpt-4o-mini)
- Prompt = system + raw user query
- No memory, no semantic cache
- No LLM Lingua compression
- max_tokens=512, temperature=0.3

Tokenomics (optimized):
- Memory layer (exact + semantic cache, context injection)
- LLM Lingua compression on enriched prompts
- Bandit routing (cheap/balanced/premium strategies)
- User preference adaptation
- Realistic temperature (0.2-0.4)

The comparison: "Raw LLM vs LLM + Tokenomics brain in front of it"
"""
import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from tokenomics.core import TokenomicsPlatform
from tokenomics.config import TokenomicsConfig


def run_real_comparison(num_queries=15):
    """Run real comparison: vanilla LLM vs Tokenomics."""
    print("=" * 80)
    print("REAL COMPARISON: Raw LLM vs Tokenomics")
    print("=" * 80)
    print()
    print("BASELINE (vanilla LLM):")
    print("  - Prompt = raw user query")
    print("  - max_tokens=512, temperature=0.3")
    print("  - No memory, no cache, no compression")
    print()
    print("TOKENOMICS (optimized):")
    print("  - Memory layer (exact + semantic cache)")
    print("  - Context injection from similar queries")
    print("  - LLM Lingua compression on enriched prompts")
    print("  - Bandit routing (cheap/balanced/premium)")
    print("  - Realistic temperature (0.2-0.4)")
    print()
    
    # Load config
    config = TokenomicsConfig.from_env()
    print(f"Provider: {config.llm.provider}, Model: {config.llm.model}")
    
    # Initialize platform
    platform = TokenomicsPlatform(config=config)
    
    # Test queries designed to exercise different scenarios
    test_queries = [
        # Fresh queries (first time) - tests baseline vs optimized fresh response
        "What is the refund policy?",
        "How do I cancel my subscription?",
        "What payment methods do you accept?",
        "How do I contact support?",
        "Why is my account locked?",
        
        # Paraphrase queries - should hit semantic cache
        "Can you explain the refund policy?",
        "How can I cancel my subscription?",
        "What methods of payment are supported?",
        
        # Exact repeats - should hit exact cache (100% savings)
        "What is the refund policy?",
        "How do I contact support?",
        
        # Complex queries - may benefit from context injection
        "I'm having trouble with billing when I try to update my payment method",
        "My subscription renewal failed and I can't access my account",
        "I want to upgrade my plan but the payment is being rejected",
        
        # More paraphrases
        "Tell me about your refund process",
        "Steps to cancel my account subscription",
    ][:num_queries]
    
    results = []
    total_baseline_tokens = 0
    total_optimized_tokens = 0
    total_baseline_latency = 0
    total_optimized_latency = 0
    cache_hits = 0
    regressions = []
    improvements = []
    
    for i, query in enumerate(test_queries):
        print(f"\n{'='*80}")
        print(f"[{i+1}/{len(test_queries)}] {query}")
        print("=" * 80)
        
        try:
            result = platform.query(
                query=query,
                use_cache=True,
                use_bandit=True,
                use_cost_aware_routing=True,
                use_compression=True,
            )
            
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
            
            # Track totals (for non-cache queries)
            if not cache_hit:
                total_baseline_tokens += baseline_total
                total_optimized_tokens += optimized_total
                total_baseline_latency += baseline_latency
                total_optimized_latency += optimized_latency
            else:
                cache_hits += 1
                # For cache hits, baseline still runs but optimized is 0
                total_baseline_tokens += baseline_total
                total_baseline_latency += baseline_latency
            
            has_regression = optimized_total > baseline_total and not cache_hit
            
            print(f"\n--- BASELINE (vanilla LLM) ---")
            print(f"  Input tokens:  {baseline_input}")
            print(f"  Output tokens: {baseline_output}")
            print(f"  Total tokens:  {baseline_total}")
            print(f"  Latency:       {baseline_latency:.0f}ms")
            
            print(f"\n--- TOKENOMICS (optimized) ---")
            print(f"  Strategy:      {strategy}")
            print(f"  Cache:         {cache_type if cache_type else 'miss'}")
            print(f"  Input tokens:  {optimized_input} ({'+' if input_diff >= 0 else ''}{input_diff})")
            print(f"  Output tokens: {optimized_output} ({'+' if output_diff >= 0 else ''}{output_diff})")
            print(f"  Total tokens:  {optimized_total} ({'+' if total_diff >= 0 else ''}{total_diff})")
            print(f"  Latency:       {optimized_latency:.0f}ms ({'+' if latency_diff >= 0 else ''}{latency_diff:.0f}ms)")
            
            if cache_hit:
                print(f"\n  ✓ CACHE HIT ({cache_type}) - 100% token savings!")
            elif has_regression:
                print(f"\n  ⚠️ REGRESSION: +{total_diff} tokens")
                regressions.append({
                    "query": query,
                    "total_diff": total_diff,
                    "input_diff": input_diff,
                    "output_diff": output_diff,
                    "strategy": strategy,
                })
            else:
                savings = -total_diff
                savings_pct = (savings / baseline_total * 100) if baseline_total > 0 else 0
                print(f"\n  ✓ SAVED {savings} tokens ({savings_pct:.1f}%)")
                if total_diff < 0:
                    improvements.append({
                        "query": query,
                        "tokens_saved": savings,
                        "savings_pct": savings_pct,
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
    
    # Final Summary
    valid_results = [r for r in results if "error" not in r]
    non_cache = [r for r in valid_results if not r.get("cache_hit")]
    
    print()
    print("=" * 80)
    print("SUMMARY: Raw LLM vs Tokenomics")
    print("=" * 80)
    
    print(f"\nTotal queries: {len(valid_results)}")
    print(f"Cache hits:    {cache_hits} ({cache_hits/len(valid_results)*100:.1f}%)")
    print(f"Fresh queries: {len(non_cache)}")
    
    if total_baseline_tokens > 0:
        token_savings = total_baseline_tokens - total_optimized_tokens
        token_savings_pct = token_savings / total_baseline_tokens * 100
        print(f"\n--- TOKEN SAVINGS ---")
        print(f"  Baseline total:   {total_baseline_tokens:,} tokens")
        print(f"  Tokenomics total: {total_optimized_tokens:,} tokens")
        print(f"  SAVED:            {token_savings:,} tokens ({token_savings_pct:.1f}%)")
    
    if total_baseline_latency > 0:
        latency_savings = total_baseline_latency - total_optimized_latency
        latency_savings_pct = latency_savings / total_baseline_latency * 100
        print(f"\n--- LATENCY SAVINGS ---")
        print(f"  Baseline total:   {total_baseline_latency/1000:.1f}s")
        print(f"  Tokenomics total: {total_optimized_latency/1000:.1f}s")
        print(f"  SAVED:            {latency_savings/1000:.1f}s ({latency_savings_pct:.1f}%)")
    
    print(f"\n--- REGRESSION ANALYSIS ---")
    print(f"  Regressions: {len(regressions)}/{len(non_cache)} ({len(regressions)/max(1,len(non_cache))*100:.1f}%)")
    
    if regressions:
        print("\n  Queries with regressions:")
        for r in regressions:
            print(f"    • {r['query'][:50]}...")
            print(f"      Input: {'+' if r['input_diff'] >= 0 else ''}{r['input_diff']}, Output: {'+' if r['output_diff'] >= 0 else ''}{r['output_diff']}")
    
    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "description": "Real comparison: Raw LLM vs Tokenomics",
        "baseline_config": {
            "prompt": "system + raw query",
            "max_tokens": 512,
            "temperature": 0.3,
            "optimizations": "none",
        },
        "tokenomics_config": {
            "memory": "exact + semantic cache",
            "compression": "LLM Lingua",
            "routing": "bandit (cheap/balanced/premium)",
            "temperature": "0.2-0.4 by strategy",
        },
        "summary": {
            "total_queries": len(valid_results),
            "cache_hits": cache_hits,
            "fresh_queries": len(non_cache),
            "token_savings": token_savings if total_baseline_tokens > 0 else 0,
            "token_savings_pct": token_savings_pct if total_baseline_tokens > 0 else 0,
            "regressions": len(regressions),
            "regression_rate": len(regressions) / max(1, len(non_cache)) * 100,
        },
        "regressions": regressions,
        "improvements": improvements,
        "results": results,
    }
    
    with open("real_comparison_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n\nResults saved to: real_comparison_results.json")
    
    return True


if __name__ == "__main__":
    num = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    run_real_comparison(num)









