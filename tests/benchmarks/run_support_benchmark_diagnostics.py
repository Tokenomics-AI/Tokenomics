"""Run diagnostic benchmark with full diagnostic field capture."""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import sys
import traceback

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
sys.stderr.reconfigure(line_buffering=True) if hasattr(sys.stderr, 'reconfigure') else None

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from tokenomics.core import TokenomicsPlatform
    from tokenomics.config import TokenomicsConfig
except ImportError as e:
    print(f"ERROR: Failed to import tokenomics modules: {e}", file=sys.stderr)
    traceback.print_exc()
    sys.exit(1)


def truncate_query_text(text: str, max_length: int = 100) -> str:
    """Truncate query text for display."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def run_diagnostic_benchmark(
    dataset_path: str = "benchmarks/data/support_dataset.json",
    output_path: str = "benchmarks/results/support_benchmark_diagnostics.json",
    num_queries: int = 20,
    use_judge: bool = True,
) -> List[Dict[str, Any]]:
    """
    Run diagnostic benchmark capturing all diagnostic fields.
    
    Args:
        dataset_path: Path to dataset JSON
        output_path: Path to save results
        num_queries: Number of queries to run
        use_judge: Whether to run quality judge
    
    Returns:
        List of benchmark results with diagnostic fields
    """
    # Load dataset
    print(f"Loading dataset from {dataset_path}...", flush=True)
    try:
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Dataset file not found: {dataset_path}", file=sys.stderr, flush=True)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in dataset file: {e}", file=sys.stderr, flush=True)
        sys.exit(1)
    
    queries = dataset.get("queries", dataset)  # Support both formats
    queries = queries[:num_queries]  # Limit to num_queries
    print(f"Loaded {len(queries)} queries from dataset", flush=True)
    
    # Initialize platform
    print("Initializing Tokenomics platform...", flush=True)
    try:
        config = TokenomicsConfig.from_env()
        print(f"Config loaded: provider={config.llm.provider}, model={config.llm.model}", flush=True)
        if not config.llm.api_key:
            print("WARNING: No API key found in configuration. Benchmark may fail.", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"ERROR: Failed to load configuration: {e}", file=sys.stderr, flush=True)
        traceback.print_exc()
        sys.exit(1)
    
    try:
        platform = TokenomicsPlatform(config=config)
        print("Platform initialized successfully", flush=True)
    except Exception as e:
        print(f"ERROR: Failed to initialize platform: {e}", file=sys.stderr, flush=True)
        traceback.print_exc()
        sys.exit(1)
    
    # Initialize judge if enabled
    judge = platform.quality_judge if use_judge and config.judge.enabled else None
    
    results = []
    token_regressions = 0
    latency_regressions = 0
    
    print(f"Running diagnostic benchmark on {len(queries)} queries...", flush=True)
    print(f"Judge enabled: {use_judge and config.judge.enabled}", flush=True)
    print("", flush=True)
    
    for i, query_data in enumerate(queries):
        query_id = query_data.get("id", query_data.get("query_id", i))
        query_text = query_data.get("text", query_data.get("query_text", ""))
        
        if not query_text:
            continue
        
        try:
            print(f"[{i+1}/{len(queries)}] Processing query {query_id}...", flush=True)
            
            # Run optimized query (automatically runs baseline when use_bandit=True)
            optimized_result = platform.query(
                query=query_text,
                use_cache=True,
                use_bandit=True,
                use_cost_aware_routing=True,
            )
            
            # Extract baseline metrics from result
            baseline_comparison = optimized_result.get("baseline_comparison_result")
            
            if baseline_comparison:
                baseline_result = {
                    "tokens_used": baseline_comparison.get("tokens_used", 0),
                    "latency_ms": baseline_comparison.get("latency_ms", 0),
                    "response": baseline_comparison.get("response", ""),
                }
            else:
                # Fallback: run baseline separately if not available
                baseline_result = platform._run_baseline_query(query=query_text)
            
            # Get baseline metrics
            baseline_total_tokens = baseline_result.get("tokens_used", 0)
            baseline_latency_ms = baseline_result.get("latency_ms", 0)
            
            # Get optimized metrics
            optimized_total_tokens = optimized_result.get("tokens_used", 0)
            optimized_latency_ms = optimized_result.get("latency_ms", 0)
            
            # Extract all diagnostic fields from optimized_result
            query_type = optimized_result.get("query_type")  # From complexity analysis
            cache_tier = optimized_result.get("cache_tier", "none")
            capsule_tokens = optimized_result.get("capsule_tokens", 0)
            strategy_arm = optimized_result.get("strategy_arm")
            model_used = optimized_result.get("model_used") or optimized_result.get("model", config.llm.model)
            used_memory = optimized_result.get("used_memory", False)
            user_preference = optimized_result.get("user_preference")
            
            # Get quality judge result if available
            quality_judge = optimized_result.get("quality_judge")
            
            # Run judge if enabled and not already judged
            if judge and not quality_judge:
                try:
                    judge_result = judge.judge(
                        query=query_text,
                        baseline_answer=baseline_result.get("response", ""),
                        optimized_answer=optimized_result.get("response", ""),
                    )
                    if judge_result:
                        quality_judge = {
                            "winner": judge_result.winner,
                            "explanation": judge_result.explanation,
                            "confidence": judge_result.confidence,
                        }
                except Exception as e:
                    print(f"  Judge error: {e}")
            
            # Check for regressions
            token_regression = optimized_total_tokens > baseline_total_tokens
            latency_regression = optimized_latency_ms > baseline_latency_ms
            
            if token_regression:
                token_regressions += 1
            if latency_regression:
                latency_regressions += 1
            
            # Build result with all diagnostic fields
            result = {
                "query_id": query_id,
                "query_text": truncate_query_text(query_text),
                "query_type": query_type,
                "cache_tier": cache_tier,
                "capsule_tokens": capsule_tokens,
                "strategy_arm": strategy_arm,
                "model_used": model_used,
                "used_memory": used_memory,
                "user_preference": user_preference,
                "baseline_total_tokens": baseline_total_tokens,
                "optimized_total_tokens": optimized_total_tokens,
                "baseline_latency_ms": round(baseline_latency_ms, 2),
                "optimized_latency_ms": round(optimized_latency_ms, 2),
                "quality_judge": quality_judge,
                # Additional fields for analysis
                "token_regression": token_regression,
                "latency_regression": latency_regression,
                "token_diff": optimized_total_tokens - baseline_total_tokens,
                "latency_diff": round(optimized_latency_ms - baseline_latency_ms, 2),
            }
            
            results.append(result)
            
            # Progress indicator
            if token_regression or latency_regression:
                print(f"  ⚠️  Regression detected: tokens={token_regression}, latency={latency_regression}", flush=True)
            else:
                print(f"  ✓  OK", flush=True)
        
        except Exception as e:
            error_msg = str(e)
            print(f"  ✗  Error: {error_msg}", flush=True)
            error_traceback = traceback.format_exc()
            print(f"  Traceback: {error_traceback}", flush=True)
            results.append({
                "query_id": query_id,
                "query_text": truncate_query_text(query_text) if query_text else "",
                "error": error_msg,
                "error_traceback": error_traceback,
            })
    
    # Save results
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("", flush=True)
    print("=" * 60, flush=True)
    print("BENCHMARK SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"Total queries: {len(results)}", flush=True)
    if len(results) > 0:
        print(f"Queries with token regression: {token_regressions} ({token_regressions/len(results)*100:.1f}%)", flush=True)
        print(f"Queries with latency regression: {latency_regressions} ({latency_regressions/len(results)*100:.1f}%)", flush=True)
    print(f"Results saved to: {output_path}", flush=True)
    print("=" * 60, flush=True)
    
    return results


def main():
    # Set up logging to file
    log_file = Path(__file__).parent.parent / "benchmark_run.log"
    log_output = []
    
    def log_print(*args, **kwargs):
        msg = ' '.join(str(a) for a in args)
        log_output.append(msg)
        print(*args, **kwargs, flush=True)
        # Also write to file immediately
        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(msg + '\n')
        except:
            pass
    
    try:
        log_print("=" * 60)
        log_print("STARTING BENCHMARK")
        log_print("=" * 60)
        log_print(f"Log file: {log_file}")
        
        parser = argparse.ArgumentParser(description="Run diagnostic support benchmark")
        parser.add_argument("--dataset", default="benchmarks/data/support_dataset.json", help="Dataset JSON path")
        parser.add_argument("--output", default="benchmarks/results/support_benchmark_diagnostics.json", help="Output JSON path")
        parser.add_argument("--num-queries", type=int, default=20, help="Number of queries to run")
        parser.add_argument("--no-judge", action="store_true", help="Disable quality judge")
        args = parser.parse_args()
        
        log_print(f"Dataset: {args.dataset}")
        log_print(f"Output: {args.output}")
        log_print(f"Queries: {args.num_queries}")
        log_print(f"Judge: {not args.no_judge}")
        log_print("=" * 60)
        log_print("")
        
        run_diagnostic_benchmark(
            dataset_path=args.dataset,
            output_path=args.output,
            num_queries=args.num_queries,
            use_judge=not args.no_judge,
        )
        
        log_print("Benchmark completed successfully")
    except KeyboardInterrupt:
        log_print("\nBenchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        error_msg = f"FATAL ERROR: {e}"
        log_print(error_msg)
        log_print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()

