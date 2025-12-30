"""Analyze benchmark results and generate summary statistics."""

import json
import argparse
import statistics
from pathlib import Path
from typing import Dict, Any, Optional, List
from collections import defaultdict
import csv


def compute_percentile(values: List[float], percentile: float) -> float:
    """Compute percentile value."""
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = int(len(sorted_values) * percentile / 100)
    return sorted_values[min(index, len(sorted_values) - 1)]


def analyze_benchmark_results(
    results_path: str = "benchmarks/results/support_benchmark_results.json",
    output_summary_path: str = "benchmarks/results/support_benchmark_summary.json",
    output_csv_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Analyze benchmark results and generate summary.
    
    Args:
        results_path: Path to benchmark results JSON
        output_summary_path: Path to save summary JSON
        output_csv_path: Optional path to save CSV summary
    
    Returns:
        Summary dictionary
    """
    # Load results
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Filter out errors
    valid_results = [r for r in results if "error" not in r]
    
    if not valid_results:
        print("No valid results found!")
        return {}
    
    print(f"Analyzing {len(valid_results)} valid results...")
    
    # Overall metrics - support both new structured format and legacy format
    total_baseline_tokens = sum(
        r.get("baseline", {}).get("total_tokens", 0) or r.get("total_tokens_baseline", 0) 
        for r in valid_results
    )
    total_optimized_tokens = sum(
        r.get("optimized", {}).get("total_tokens", 0) or r.get("total_tokens_optimized", 0)
        for r in valid_results
    )
    total_final_tokens = sum(
        r.get("final", {}).get("total_tokens", 0) or 
        (r.get("total_tokens_baseline", 0) if r.get("fallback_to_baseline", False) else r.get("total_tokens_optimized", 0))
        for r in valid_results
    )
    
    # Waste metrics
    total_wasted_tokens = sum(
        r.get("guardrail", {}).get("wasted_tokens", 0) or r.get("wasted_tokens", 0)
        for r in valid_results
    )
    
    # Gross savings (ignoring waste)
    gross_token_savings = total_baseline_tokens - total_optimized_tokens
    gross_token_savings_percent = (gross_token_savings / total_baseline_tokens * 100) if total_baseline_tokens > 0 else 0
    
    # Net savings (after subtracting wasted tokens)
    net_token_savings = total_baseline_tokens - total_final_tokens - total_wasted_tokens
    net_token_savings_percent = (net_token_savings / total_baseline_tokens * 100) if total_baseline_tokens > 0 else 0
    
    # Cost metrics
    total_baseline_cost = sum(
        r.get("baseline", {}).get("cost", 0) or r.get("cost_baseline", 0)
        for r in valid_results
    )
    total_optimized_cost = sum(
        r.get("optimized", {}).get("cost", 0) or r.get("cost_optimized", 0)
        for r in valid_results
    )
    total_final_cost = sum(
        r.get("final", {}).get("cost", 0) or
        (r.get("cost_baseline", 0) if r.get("fallback_to_baseline", False) else r.get("cost_optimized", 0))
        for r in valid_results
    )
    total_wasted_cost = sum(
        r.get("guardrail", {}).get("wasted_cost", 0) or r.get("wasted_cost", 0)
        for r in valid_results
    )
    
    # Gross cost savings (ignoring waste)
    gross_cost_savings = total_baseline_cost - total_optimized_cost
    
    # Net cost savings (after subtracting wasted cost)
    net_cost_savings = total_baseline_cost - total_final_cost - total_wasted_cost
    
    # Latency metrics
    baseline_latencies = [r.get("latency_baseline_ms", 0) for r in valid_results]
    optimized_latencies = [r.get("latency_optimized_ms", 0) for r in valid_results]
    latency_improvements = [b - o for b, o in zip(baseline_latencies, optimized_latencies)]
    
    avg_latency_baseline = statistics.mean(baseline_latencies) if baseline_latencies else 0
    avg_latency_optimized = statistics.mean(optimized_latencies) if optimized_latencies else 0
    avg_latency_improvement = statistics.mean(latency_improvements) if latency_improvements else 0
    median_latency_baseline = statistics.median(baseline_latencies) if baseline_latencies else 0
    median_latency_optimized = statistics.median(optimized_latencies) if optimized_latencies else 0
    p95_latency_baseline = compute_percentile(baseline_latencies, 95)
    p95_latency_optimized = compute_percentile(optimized_latencies, 95)
    
    # Cache breakdown
    cache_type_counts = defaultdict(int)
    for r in valid_results:
        cache_type = r.get("cache_hit_type", "none")
        cache_type_counts[cache_type] += 1
    
    total_queries = len(valid_results)
    cache_hit_rate = sum(1 for r in valid_results if r.get("cache_hit_type", "none") != "none") / total_queries * 100 if total_queries > 0 else 0
    
    # Strategy usage
    strategy_counts = defaultdict(int)
    for r in valid_results:
        strategy = r.get("strategy", "none")
        strategy_counts[strategy] += 1
    
    # Model usage
    model_counts = defaultdict(int)
    for r in valid_results:
        model = r.get("model_used", "unknown")
        model_counts[model] += 1
    
    # Guardrail statistics
    guardrail_triggers = sum(1 for r in valid_results if r.get("fallback_to_baseline", False))
    guardrail_rate = (guardrail_triggers / total_queries * 100) if total_queries > 0 else 0
    
    guardrail_reasons = defaultdict(int)
    for r in valid_results:
        if r.get("fallback_to_baseline", False):
            reason = r.get("fallback_reason", "unknown")
            guardrail_reasons[reason] += 1
    
    # Quality judge statistics
    judged_results = [r for r in valid_results if r.get("quality_judge")]
    judge_verdicts = defaultdict(int)
    judge_confidences = []
    
    for r in judged_results:
        judge = r.get("quality_judge", {})
        winner = judge.get("winner", "unknown")
        confidence = judge.get("confidence", 0)
        judge_verdicts[winner] += 1
        if confidence > 0:
            judge_confidences.append(confidence)
    
    avg_confidence = statistics.mean(judge_confidences) if judge_confidences else 0
    
    # Metrics by query type
    by_type = defaultdict(lambda: {
        "count": 0,
        "total_tokens_baseline": 0,
        "total_tokens_optimized": 0,
    })
    
    for r in valid_results:
        query_type = r.get("type", "unknown")
        by_type[query_type]["count"] += 1
        by_type[query_type]["total_tokens_baseline"] += r.get("total_tokens_baseline", 0)
        by_type[query_type]["total_tokens_optimized"] += r.get("total_tokens_optimized", 0)
    
    for query_type in by_type:
        baseline = by_type[query_type]["total_tokens_baseline"]
        optimized = by_type[query_type]["total_tokens_optimized"]
        savings = baseline - optimized
        by_type[query_type]["token_savings_percent"] = (savings / baseline * 100) if baseline > 0 else 0
    
    # Model routing by query type
    model_by_type = defaultdict(lambda: defaultdict(int))
    for r in valid_results:
        query_type = r.get("type", "unknown")
        model = r.get("model_used", "unknown")
        model_by_type[query_type][model] += 1
    
    # Build summary
    summary = {
        "overall": {
            "total_queries": total_queries,
            # Raw metrics
            "total_tokens_baseline": total_baseline_tokens,
            "total_tokens_optimized": total_optimized_tokens,
            "total_tokens_final": total_final_tokens,
            "total_cost_baseline": round(total_baseline_cost, 6),
            "total_cost_optimized": round(total_optimized_cost, 6),
            "total_cost_final": round(total_final_cost, 6),
            # Gross savings (ignoring waste)
            "gross_token_savings": gross_token_savings,
            "gross_token_savings_percent": round(gross_token_savings_percent, 2),
            "gross_cost_savings_dollars": round(gross_cost_savings, 6),
            # Net savings (after subtracting waste)
            "net_token_savings": net_token_savings,
            "net_token_savings_percent": round(net_token_savings_percent, 2),
            "net_cost_savings_dollars": round(net_cost_savings, 6),
            # Waste metrics
            "total_wasted_tokens": total_wasted_tokens,
            "total_wasted_cost": round(total_wasted_cost, 6),
            # Legacy fields for backward compatibility
            "token_savings": gross_token_savings,
            "token_savings_percent": round(gross_token_savings_percent, 2),
            "cost_savings_dollars": round(gross_cost_savings, 4),
            "avg_latency_baseline_ms": round(avg_latency_baseline, 2),
            "avg_latency_optimized_ms": round(avg_latency_optimized, 2),
            "avg_latency_improvement_ms": round(avg_latency_improvement, 2),
            "median_latency_baseline_ms": round(median_latency_baseline, 2),
            "median_latency_optimized_ms": round(median_latency_optimized, 2),
            "p95_latency_baseline_ms": round(p95_latency_baseline, 2),
            "p95_latency_optimized_ms": round(p95_latency_optimized, 2),
        },
        "by_query_type": {k: dict(v) for k, v in by_type.items()},
        "cache_breakdown": {
            "overall_hit_rate_percent": round(cache_hit_rate, 2),
            "by_type": dict(cache_type_counts),
            "hit_rates_by_type": {
                k: round(v / total_queries * 100, 2) for k, v in cache_type_counts.items()
            },
        },
        "strategy_usage": {
            "counts": dict(strategy_counts),
            "percentages": {
                k: round(v / total_queries * 100, 2) for k, v in strategy_counts.items()
            },
        },
        "model_usage": {
            "counts": dict(model_counts),
            "percentages": {
                k: round(v / total_queries * 100, 2) for k, v in model_counts.items()
            },
        },
        "model_routing_by_type": {
            k: dict(v) for k, v in model_by_type.items()
        },
        "guardrail_statistics": {
            "total_fallbacks": guardrail_triggers,
            "fallback_rate_percent": round(guardrail_rate, 2),
            "reasons": dict(guardrail_reasons),
            "total_wasted_tokens": total_wasted_tokens,
            "total_wasted_cost": round(total_wasted_cost, 6),
            "avg_wasted_tokens_per_fallback": round(total_wasted_tokens / guardrail_triggers, 2) if guardrail_triggers > 0 else 0,
        },
        "quality_judge": {
            "total_judged": len(judged_results),
            "verdicts": dict(judge_verdicts),
            "verdict_percentages": {
                k: round(v / len(judged_results) * 100, 2) if judged_results else 0
                for k, v in judge_verdicts.items()
            },
            "avg_confidence": round(avg_confidence, 3),
        },
    }
    
    # Save summary
    Path(output_summary_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary to console
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"\nOverall Performance:")
    print(f"  Total Queries: {total_queries}")
    print(f"  Baseline Tokens: {total_baseline_tokens:,}")
    print(f"  Optimized Tokens: {total_optimized_tokens:,}")
    print(f"  Final Tokens (after guardrail): {total_final_tokens:,}")
    
    print(f"\nToken Savings:")
    print(f"  Gross Savings (ignoring waste): {gross_token_savings_percent:.2f}% ({gross_token_savings:,} tokens)")
    print(f"  Net Savings (after waste): {net_token_savings_percent:.2f}% ({net_token_savings:,} tokens)")
    
    print(f"\nCost Savings:")
    print(f"  Gross Savings (ignoring waste): ${gross_cost_savings:.6f}")
    print(f"  Net Savings (after waste): ${net_cost_savings:.6f}")
    
    print(f"\nWaste Metrics:")
    print(f"  Total Wasted Tokens: {total_wasted_tokens:,}")
    print(f"  Total Wasted Cost: ${total_wasted_cost:.6f}")
    if guardrail_triggers > 0:
        avg_waste = total_wasted_tokens / guardrail_triggers
        print(f"  Avg Wasted Tokens per Fallback: {avg_waste:.2f}")
    
    print(f"\nLatency:")
    print(f"  Avg Latency Improvement: {avg_latency_improvement:.2f} ms")
    print(f"  P95 Latency Baseline: {p95_latency_baseline:.2f} ms")
    print(f"  P95 Latency Optimized: {p95_latency_optimized:.2f} ms")
    
    print(f"\nQuality Judge Results:")
    print(f"  Total Judged: {len(judged_results)}")
    for verdict, count in judge_verdicts.items():
        pct = (count / len(judged_results) * 100) if judged_results else 0
        print(f"  {verdict}: {count} ({pct:.1f}%)")
    if judge_confidences:
        print(f"  Avg Confidence: {avg_confidence:.3f}")
    
    print(f"\nGuardrail Statistics:")
    print(f"  Total Fallbacks: {guardrail_triggers} ({guardrail_rate:.1f}%)")
    print(f"  Total Wasted Tokens: {total_wasted_tokens:,}")
    print(f"  Total Wasted Cost: ${total_wasted_cost:.6f}")
    for reason, count in guardrail_reasons.items():
        print(f"  {reason}: {count}")
    
    print(f"\nModel Routing:")
    for model, count in model_counts.items():
        pct = (count / total_queries * 100) if total_queries > 0 else 0
        print(f"  {model}: {count} ({pct:.1f}%)")
    
    print(f"\nCache Breakdown:")
    print(f"  Overall Hit Rate: {cache_hit_rate:.1f}%")
    for cache_type, count in cache_type_counts.items():
        pct = (count / total_queries * 100) if total_queries > 0 else 0
        print(f"  {cache_type}: {count} ({pct:.1f}%)")
    
    print(f"\nStrategy Usage:")
    for strategy, count in strategy_counts.items():
        pct = (count / total_queries * 100) if total_queries > 0 else 0
        print(f"  {strategy}: {count} ({pct:.1f}%)")
    
    print("\n" + "=" * 80)
    print(f"Summary saved to: {output_summary_path}")
    
    # Save CSV if requested
    if output_csv_path:
        with open(output_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Metric", "Value"
            ])
            writer.writerow(["Token Savings %", f"{token_savings_percent:.2f}"])
            writer.writerow(["Cost Savings $", f"{cost_savings:.4f}"])
            writer.writerow(["Avg Latency Improvement ms", f"{avg_latency_improvement:.2f}"])
            writer.writerow(["Cache Hit Rate %", f"{cache_hit_rate:.2f}"])
            writer.writerow(["Guardrail Fallbacks", guardrail_triggers])
            writer.writerow(["Queries Judged", len(judged_results)])
            for verdict, count in judge_verdicts.items():
                writer.writerow([f"Judge: {verdict}", count])
        print(f"CSV saved to: {output_csv_path}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Analyze benchmark results")
    parser.add_argument("--results", default="benchmarks/results/support_benchmark_results.json", help="Results JSON path")
    parser.add_argument("--output", default="benchmarks/results/support_benchmark_summary.json", help="Output summary JSON path")
    parser.add_argument("--csv", help="Optional CSV output path")
    args = parser.parse_args()
    
    analyze_benchmark_results(
        results_path=args.results,
        output_summary_path=args.output,
        output_csv_path=args.csv,
    )


if __name__ == "__main__":
    main()


