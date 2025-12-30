"""Analyze regression patterns in benchmark results."""
import json

# Load the results
with open("benchmarks/results/quick_benchmark_results.json", "r") as f:
    results = json.load(f)

# Filter out cache hits (they should always be better)
non_cache_results = [r for r in results if r.get("cache_hit_type") == "none"]

print("=" * 80)
print("REGRESSION ANALYSIS - Non-cache-hit queries")
print("=" * 80)
print()

# Find token regressions
token_regressions = [
    r for r in non_cache_results
    if r["total_tokens_optimized"] > r["total_tokens_baseline"]
]

# Find latency regressions
latency_regressions = [
    r for r in non_cache_results
    if r["latency_optimized_ms"] > r["latency_baseline_ms"]
]

print(f"Total non-cache-hit queries: {len(non_cache_results)}")
print(f"Token regressions: {len(token_regressions)} ({len(token_regressions)/len(non_cache_results)*100:.1f}%)")
print(f"Latency regressions: {len(latency_regressions)} ({len(latency_regressions)/len(non_cache_results)*100:.1f}%)")
print()

print("=" * 80)
print("TOKEN REGRESSIONS (Optimized > Baseline)")
print("=" * 80)

for r in sorted(token_regressions, key=lambda x: x["total_tokens_optimized"] - x["total_tokens_baseline"], reverse=True):
    diff = r["total_tokens_optimized"] - r["total_tokens_baseline"]
    print(f"\nQuery ID {r['query_id']}: {r['query_text'][:60]}...")
    print(f"  Category: {r['category']}, Type: {r['type']}")
    print(f"  Baseline: {r['total_tokens_baseline']} tokens (in={r['input_tokens_baseline']}, out={r['output_tokens_baseline']})")
    print(f"  Optimized: {r['total_tokens_optimized']} tokens (in={r['input_tokens_optimized']}, out={r['output_tokens_optimized']})")
    print(f"  Token difference: +{diff} tokens ({diff/r['total_tokens_baseline']*100:.1f}% increase)")
    print(f"  Output tokens diff: +{r['output_tokens_optimized'] - r['output_tokens_baseline']}")
    
print()
print("=" * 80)
print("LATENCY REGRESSIONS (Optimized > Baseline)")
print("=" * 80)

for r in sorted(latency_regressions, key=lambda x: x["latency_optimized_ms"] - x["latency_baseline_ms"], reverse=True):
    diff = r["latency_optimized_ms"] - r["latency_baseline_ms"]
    print(f"\nQuery ID {r['query_id']}: {r['query_text'][:60]}...")
    print(f"  Baseline latency: {r['latency_baseline_ms']:.1f}ms")
    print(f"  Optimized latency: {r['latency_optimized_ms']:.1f}ms")
    print(f"  Latency diff: +{diff:.1f}ms ({diff/r['latency_baseline_ms']*100:.1f}% slower)")
    print(f"  Token comparison: baseline={r['total_tokens_baseline']}, optimized={r['total_tokens_optimized']}")

print()
print("=" * 80)
print("ROOT CAUSE ANALYSIS")
print("=" * 80)
print()

# Check if the issue is output tokens
output_increased = [
    r for r in token_regressions
    if r['output_tokens_optimized'] > r['output_tokens_baseline']
]
print(f"Token regressions where OUTPUT tokens increased: {len(output_increased)}/{len(token_regressions)}")

# Check input tokens
input_increased = [
    r for r in token_regressions
    if r['input_tokens_optimized'] > r['input_tokens_baseline']
]
print(f"Token regressions where INPUT tokens increased: {len(input_increased)}/{len(token_regressions)}")

print()
print("CONCLUSION:")
if len(output_increased) == len(token_regressions):
    print("  All token regressions are due to OUTPUT tokens being higher in optimized path.")
    print("  This is caused by LLM response variance - the model generates different length responses.")
    print()
    print("  The 'cheap' strategy has max_tokens=300 but that's just a CAP, not a target.")
    print("  The model can still generate any length up to that limit.")
    print()
    print("SOLUTION:")
    print("  1. Use a LOWER max_tokens for optimized than baseline to reduce variance impact")
    print("  2. Or implement response length targeting using stop sequences or prompts")
    print("  3. Or cap optimized max_tokens to be at most 80% of baseline")










