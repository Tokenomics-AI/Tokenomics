"""Run A/B benchmark comparing baseline vs optimized Tokenomics pipeline."""

import json
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenomics.core import TokenomicsPlatform
from tokenomics.config import TokenomicsConfig
from tokenomics.bandit.bandit import MODEL_COSTS


def calculate_cost(tokens: int, model: str) -> float:
    """Calculate cost based on tokens and model."""
    cost_per_million = MODEL_COSTS.get(model, MODEL_COSTS["default"])
    return (tokens / 1_000_000) * cost_per_million


def run_benchmark(
    dataset_path: str = "benchmarks/data/support_dataset.json",
    output_path: str = "benchmarks/results/support_benchmark_results.json",
    judge_subset_size: int = 100,
    use_judge: bool = True,
    policy_config_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Run A/B benchmark comparing baseline vs optimized.
    
    Args:
        dataset_path: Path to dataset JSON
        output_path: Path to save results
        judge_subset_size: Number of queries to judge (if use_judge=True)
        use_judge: Whether to run quality judge
        policy_config_path: Optional path to policy config JSON
    
    Returns:
        List of benchmark results
    """
    # Load dataset
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    queries = dataset.get("queries", dataset)  # Support both formats
    
    # Initialize platform
    config = TokenomicsConfig.from_env()
    
    # Apply policy overrides if provided
    if policy_config_path and Path(policy_config_path).exists():
        from benchmarks.policy_config import PolicyConfig
        policy = PolicyConfig.from_json(policy_config_path)
        policy.apply_to_config(config)
    
    platform = TokenomicsPlatform(config=config)
    
    # Initialize judge if enabled
    judge = platform.quality_judge if use_judge and config.judge.enabled else None
    
    results = []
    guardrail_count = 0
    judge_count = 0
    
    print(f"Running benchmark on {len(queries)} queries...")
    print(f"Judge enabled: {use_judge and config.judge.enabled}")
    print(f"Judge subset size: {judge_subset_size if use_judge else 0}")
    print()
    
    for i, query_data in enumerate(queries):
        query_id = query_data.get("id", query_data.get("query_id", i))
        query_text = query_data.get("text", query_data.get("query_text", ""))
        query_type = query_data.get("type", "unique")
        query_category = query_data.get("category", "general")
        
        if not query_text:
            continue
        
        try:
            # Run optimized query (automatically runs baseline when use_bandit=True)
            # The query() method now handles baseline, guardrail, and judge internally
            optimized_result = platform.query(
                query=query_text,
                use_cache=True,
                use_bandit=True,
                use_cost_aware_routing=True,
            )
            
            # Extract baseline metrics from result (baseline was run internally)
            # The query() method runs baseline automatically when use_bandit=True
            # Extract baseline from baseline_comparison_result if available
            baseline_comparison = optimized_result.get("baseline_comparison_result")
            
            if baseline_comparison:
                # Use baseline from internal comparison
                baseline_result = {
                    "tokens_used": baseline_comparison.get("tokens_used", 0),
                    "input_tokens": baseline_comparison.get("input_tokens", 0),
                    "output_tokens": baseline_comparison.get("output_tokens", 0),
                    "latency_ms": baseline_comparison.get("latency_ms", 0),
                    "model": baseline_comparison.get("model", config.llm.model),
                    "response": baseline_comparison.get("response", ""),
                }
            else:
                # Fallback: run baseline separately if not available in result
                baseline_result = platform._run_baseline_query(query=query_text)
            
            # Get optimized metrics
            optimized_tokens = optimized_result.get("tokens_used", 0)
            optimized_input = optimized_result.get("input_tokens", 0)
            optimized_output = optimized_result.get("output_tokens", 0)
            optimized_latency = optimized_result.get("latency_ms", 0)
            optimized_cache_type = optimized_result.get("cache_type", "none")
            optimized_cache_hit = optimized_result.get("cache_hit", False)
            optimized_strategy = optimized_result.get("strategy", "none")
            optimized_model = optimized_result.get("model", config.llm.model)
            
            # Get baseline metrics
            baseline_tokens = baseline_result.get("tokens_used", 0)
            baseline_input = baseline_result.get("input_tokens", 0)
            baseline_output = baseline_result.get("output_tokens", 0)
            baseline_latency = baseline_result.get("latency_ms", 0)
            
            # Check guardrail and extract waste metrics
            fallback_to_baseline = optimized_result.get("fallback_to_baseline", False)
            fallback_reason = optimized_result.get("fallback_reason", None)
            wasted_tokens = optimized_result.get("wasted_tokens", 0)
            wasted_cost = optimized_result.get("wasted_cost", 0.0)
            
            # Get original optimized tokens before guardrail (if fallback occurred)
            original_optimized_tokens = optimized_result.get("original_optimized_tokens", optimized_tokens)
            
            if fallback_to_baseline:
                guardrail_count += 1
                # Use baseline metrics if guardrail triggered (final chosen result)
                final_tokens = baseline_tokens
                final_input = baseline_input
                final_output = baseline_output
                final_latency = baseline_latency
            else:
                # Optimized was kept
                final_tokens = optimized_tokens
                final_input = optimized_input
                final_output = optimized_output
                final_latency = optimized_latency
            
            # Calculate costs
            baseline_cost = calculate_cost(baseline_tokens, baseline_result.get("model", config.llm.model))
            optimized_cost = calculate_cost(original_optimized_tokens, optimized_model)
            final_cost = calculate_cost(final_tokens, optimized_model if not fallback_to_baseline else baseline_result.get("model", config.llm.model))
            
            # Get quality judge result if available
            quality_judge = optimized_result.get("quality_judge")
            
            # Run judge for subset if enabled and not already judged
            if judge and not quality_judge and judge_count < judge_subset_size:
                judge_result = judge.judge(
                    query=query_text,
                    baseline_answer=baseline_result["response"],
                    optimized_answer=optimized_result["response"],
                )
                if judge_result:
                    quality_judge = {
                        "winner": judge_result.winner,
                        "explanation": judge_result.explanation,
                        "confidence": judge_result.confidence,
                    }
                    judge_count += 1
            
            result = {
                "query_id": query_id,
                "query_text": query_text,
                "type": query_type,
                "category": query_category,
                # Raw baseline metrics
                "baseline": {
                    "total_tokens": baseline_tokens,
                    "input_tokens": baseline_input,
                    "output_tokens": baseline_output,
                    "latency_ms": round(baseline_latency, 2),
                    "cost": round(baseline_cost, 6),
                    "model": baseline_result.get("model", config.llm.model),
                },
                # Raw optimized metrics (before guardrail)
                "optimized": {
                    "total_tokens": original_optimized_tokens,
                    "input_tokens": optimized_input,
                    "output_tokens": optimized_output,
                    "latency_ms": round(optimized_latency, 2),
                    "cost": round(optimized_cost, 6),
                    "model": optimized_model,
                    "strategy": optimized_strategy,
                    "cache_hit": optimized_cache_hit,
                    "cache_type": optimized_cache_type if optimized_cache_hit else "none",
                },
                # Final chosen metrics (after guardrail)
                "final": {
                    "total_tokens": final_tokens,
                    "input_tokens": final_input,
                    "output_tokens": final_output,
                    "latency_ms": round(final_latency, 2),
                    "cost": round(final_cost, 6),
                },
                # Guardrail and waste metrics
                "guardrail": {
                    "fallback_triggered": fallback_to_baseline,
                    "fallback_reason": fallback_reason,
                    "wasted_tokens": wasted_tokens,
                    "wasted_cost": round(wasted_cost, 6),
                },
                # Quality judge
                "quality_judge": quality_judge,
                # Legacy fields for backward compatibility
                "total_tokens_baseline": baseline_tokens,
                "total_tokens_optimized": original_optimized_tokens,
                "input_tokens_baseline": baseline_input,
                "output_tokens_baseline": baseline_output,
                "input_tokens_optimized": optimized_input,
                "output_tokens_optimized": optimized_output,
                "latency_baseline_ms": round(baseline_latency, 2),
                "latency_optimized_ms": round(optimized_latency, 2),
                "cache_hit_type": optimized_cache_type if optimized_cache_hit else "none",
                "strategy": optimized_strategy,
                "model_used": optimized_model,
                "cost_baseline": round(baseline_cost, 4),
                "cost_optimized": round(optimized_cost, 4),
                "fallback_to_baseline": fallback_to_baseline,
                "fallback_reason": fallback_reason,
            }
            results.append(result)
            
            # Progress logging
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(queries)} queries... "
                      f"(Guardrails: {guardrail_count}, Judged: {judge_count})")
                
                # Save intermediate results
                intermediate_path = output_path.replace(".json", f"_intermediate_{i+1}.json")
                with open(intermediate_path, 'w') as f:
                    json.dump(results, f, indent=2)
        
        except Exception as e:
            print(f"Error processing query {query_id}: {e}")
            results.append({
                "query_id": query_id,
                "query_text": query_text,
                "type": query_type,
                "category": query_category,
                "error": str(e),
            })
    
    # Save final results
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nBenchmark complete!")
    print(f"Total queries: {len(results)}")
    print(f"Guardrail triggers: {guardrail_count}")
    print(f"Queries judged: {judge_count}")
    print(f"Results saved to: {output_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run A/B support benchmark")
    parser.add_argument("--dataset", default="benchmarks/data/support_dataset.json", help="Dataset JSON path")
    parser.add_argument("--output", default="benchmarks/results/support_benchmark_results.json", help="Output JSON path")
    parser.add_argument("--judge-size", type=int, default=100, help="Number of queries to judge")
    parser.add_argument("--no-judge", action="store_true", help="Disable quality judge")
    parser.add_argument("--policy", help="Path to policy config JSON")
    args = parser.parse_args()
    
    run_benchmark(
        dataset_path=args.dataset,
        output_path=args.output,
        judge_subset_size=args.judge_size,
        use_judge=not args.no_judge,
        policy_config_path=args.policy,
    )


if __name__ == "__main__":
    main()


