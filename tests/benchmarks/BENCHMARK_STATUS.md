# Diagnostic Benchmark Status

## Created Files

1. **`run_support_benchmark_diagnostics.py`** - Benchmark script that captures all diagnostic fields
2. **`analyze_support_benchmark_diagnostics.py`** - Analysis script to identify regression patterns
3. **`run_diagnostics.py`** - Wrapper script to run benchmark and analysis
4. **`DIAGNOSTIC_BENCHMARK_README.md`** - Documentation

## Diagnostic Fields Captured

Each query result includes:
- `query_id` - Query identifier
- `query_text` - Truncated query (first 100 chars)
- `query_type` - Complexity (simple/medium/complex)
- `cache_tier` - Cache tier (none/exact/semantic/capsule)
- `capsule_tokens` - Tokens added from context
- `strategy_arm` - Strategy selected (cheap/balanced/premium)
- `model_used` - Model used
- `used_memory` - Whether memory layer was used
- `user_preference` - Preference context
- `baseline_total_tokens` - Baseline token count
- `optimized_total_tokens` - Optimized token count
- `baseline_latency_ms` - Baseline latency
- `optimized_latency_ms` - Optimized latency
- `token_regression` - Boolean flag
- `latency_regression` - Boolean flag
- `token_diff` - Difference in tokens
- `latency_diff` - Difference in latency

## Running the Benchmark

The benchmark is currently running in the background. To check status:

```bash
# Check if results file exists
python -c "import os; print(os.path.exists('benchmarks/results/support_benchmark_diagnostics.json'))"

# Run benchmark manually (20 queries)
python benchmarks/run_support_benchmark_diagnostics.py --num-queries 20 --no-judge

# Run analysis after benchmark completes
python benchmarks/analyze_support_benchmark_diagnostics.py
```

## Expected Output Location

Results will be saved to: `benchmarks/results/support_benchmark_diagnostics.json`

## Next Steps

1. Wait for benchmark to complete (may take 5-10 minutes for 20 queries)
2. Check if results file exists
3. Run analysis script to see regression patterns
4. Review patterns to identify root causes










