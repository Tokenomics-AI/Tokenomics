# Diagnostic Benchmark for Regression Analysis

## Overview

This diagnostic benchmark captures detailed diagnostic fields for each query to help identify why some queries consume more tokens or have higher latency in the optimized version compared to baseline.

## Files Created

1. **`run_support_benchmark_diagnostics.py`** - Runs the benchmark and captures all diagnostic fields
2. **`analyze_support_benchmark_diagnostics.py`** - Analyzes results and groups regressions by diagnostic fields

## Diagnostic Fields Captured

For each query, the benchmark records:

- `query_id` - Query identifier
- `query_text` - Truncated query text (first 100 chars)
- `query_type` - Query complexity (simple/medium/complex)
- `cache_tier` - Cache tier used (none/exact/semantic/capsule)
- `capsule_tokens` - Number of tokens added from compressed context
- `strategy_arm` - Strategy arm selected (cheap/balanced/premium)
- `model_used` - Model used for optimized path
- `used_memory` - Whether memory layer was used (true/false)
- `user_preference` - User preference context (tone-format)
- `baseline_total_tokens` - Baseline token count
- `optimized_total_tokens` - Optimized token count
- `baseline_latency_ms` - Baseline latency
- `optimized_latency_ms` - Optimized latency
- `quality_judge` - Quality judge result (if enabled)
- `token_regression` - Boolean: optimized > baseline tokens
- `latency_regression` - Boolean: optimized > baseline latency
- `token_diff` - Difference in tokens (optimized - baseline)
- `latency_diff` - Difference in latency (optimized - baseline)

## Usage

### Run Benchmark

```bash
# Run with 20 queries (default)
python benchmarks/run_support_benchmark_diagnostics.py --num-queries 20 --no-judge

# Run with custom number of queries
python benchmarks/run_support_benchmark_diagnostics.py --num-queries 50

# Enable quality judge (slower)
python benchmarks/run_support_benchmark_diagnostics.py --num-queries 20
```

### Analyze Results

```bash
# Analyze results (default path)
python benchmarks/analyze_support_benchmark_diagnostics.py

# Analyze custom results file
python benchmarks/analyze_support_benchmark_diagnostics.py --results path/to/results.json
```

## Output

### Benchmark Output

The benchmark saves results to: `benchmarks/results/support_benchmark_diagnostics.json`

It prints a summary showing:
- Total queries processed
- Number of queries with token regressions
- Number of queries with latency regressions

### Analysis Output

The analysis script prints detailed breakdowns:

1. **Token Regressions By:**
   - `cache_tier` - Which cache tiers cause most regressions
   - `strategy_arm` - Which strategy arms cause regressions
   - `model_used` - Which models cause regressions
   - `query_type` - Which query types have regressions
   - `capsule_tokens` - Range of capsule tokens in regressions

2. **Latency Regressions By:**
   - `cache_tier`
   - `strategy_arm`
   - `model_used`
   - `query_type`

3. **Sample Regression Queries** - Shows detailed examples of regressions

## Expected Patterns to Look For

Based on the diagnostic fields, you can identify patterns like:

- **"Most bad cases have cache_tier = capsule and capsule_tokens ≈ 40–50"**
  - Indicates context injection is adding too many tokens without sufficient savings

- **"Most bad cases used strategy_arm = premium and model_used = gpt-4o"**
  - Indicates premium model selection is causing regressions

- **"Most bad cases are query_type = simple but still picked premium or added context"**
  - Indicates routing logic is over-optimizing simple queries

## Next Steps

After running the benchmark and analysis:

1. Review the regression patterns
2. Identify the most common causes of regressions
3. Adjust optimization logic based on findings:
   - If capsule tokens are the issue: improve context injection threshold
   - If premium arm is the issue: adjust strategy selection for simple queries
   - If model selection is the issue: improve cost-aware routing










