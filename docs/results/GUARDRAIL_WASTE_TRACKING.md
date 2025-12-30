# Guardrail Waste Tracking Implementation

## Overview

This document describes the implementation of wasted token and cost tracking when the guardrail triggers and falls back to baseline. The goal is to accurately measure the true cost of running baseline + optimized + guardrail, not just the final chosen path.

## Changes Made

### 1. Core Guardrail Function (`tokenomics/core.py`)

**Updated `_apply_baseline_guardrail()` method:**
- Calculates `wasted_tokens` = total tokens consumed by the optimized path when guardrail triggers
- Calculates `wasted_cost` = cost associated with those wasted tokens
- Uses the same cost calculation as benchmarks (MODEL_COSTS from bandit module)
- Adds `wasted_tokens` and `wasted_cost` fields to the result when fallback occurs
- Sets both fields to 0 when optimized is kept (no fallback)

**Key Code:**
```python
# Calculate wasted tokens and cost
wasted_tokens = optimized_tokens
optimized_model = optimized_result.get("model", self.config.llm.model)

# Calculate wasted cost using same cost calculation as benchmark
from .bandit.bandit import MODEL_COSTS
cost_per_million = MODEL_COSTS.get(optimized_model, MODEL_COSTS.get("default", 0.10))
wasted_cost = (wasted_tokens / 1_000_000) * cost_per_million

# Add to result
baseline_result["wasted_tokens"] = wasted_tokens
baseline_result["wasted_cost"] = wasted_cost
```

### 2. Benchmark Runner (`benchmarks/run_support_benchmark.py`)

**Updated result structure:**
- Added structured format with `baseline`, `optimized`, `final`, and `guardrail` sections
- `baseline`: Raw baseline metrics
- `optimized`: Raw optimized metrics (before guardrail)
- `final`: Final chosen metrics (after guardrail decision)
- `guardrail`: Guardrail and waste metrics
  - `fallback_triggered`: Boolean
  - `fallback_reason`: String
  - `wasted_tokens`: Integer
  - `wasted_cost`: Float
- Maintained legacy fields for backward compatibility

**Key Changes:**
- Extracts baseline from `baseline_comparison_result` if available (avoids double-running)
- Tracks `original_optimized_tokens` before guardrail decision
- Calculates `final_tokens` based on guardrail decision
- Includes waste metrics in JSON output

### 3. Analysis Script (`benchmarks/analyze_support_benchmark.py`)

**Added gross vs net savings calculation:**
- **Gross savings**: `baseline_total - optimized_total` (ignoring waste)
- **Net savings**: `baseline_total - final_total - wasted_total` (after subtracting waste)

**Updated summary structure:**
- `overall` section now includes:
  - Raw metrics: `total_tokens_baseline`, `total_tokens_optimized`, `total_tokens_final`
  - Gross savings: `gross_token_savings`, `gross_token_savings_percent`, `gross_cost_savings_dollars`
  - Net savings: `net_token_savings`, `net_token_savings_percent`, `net_cost_savings_dollars`
  - Waste metrics: `total_wasted_tokens`, `total_wasted_cost`
- `guardrail_statistics` section includes:
  - `total_wasted_tokens`
  - `total_wasted_cost`
  - `avg_wasted_tokens_per_fallback`

**Enhanced console output:**
- Shows both gross and net savings
- Displays waste metrics separately
- Shows average waste per fallback

### 4. Dashboard Summary (`app.py`)

**Added waste metrics to run summary:**
- `total_wasted_tokens`: Sum of wasted tokens across all queries
- `total_wasted_cost`: Sum of wasted cost across all queries
- `guardrail_trigger_count`: Number of queries where guardrail triggered
- `guardrail_trigger_rate`: Percentage of queries with guardrail triggers
- `net_token_savings`: Net savings after subtracting waste
- `net_token_savings_percent`: Net savings percentage
- `net_cost_savings`: Net cost savings after subtracting waste

**Calculation:**
```python
# Calculate waste metrics for A/B mode
total_wasted_tokens = sum of wasted_tokens from all results
total_wasted_cost = sum of wasted_cost from all results
guardrail_trigger_count = count of fallback_triggered = True

# Net savings
net_token_savings = gross_token_savings - total_wasted_tokens
net_token_savings_percent = (net_token_savings / baseline_total) * 100
```

## JSON Structure

### Per-Query Result Format

```json
{
  "query_id": "...",
  "query_text": "...",
  "baseline": {
    "total_tokens": 156,
    "input_tokens": 14,
    "output_tokens": 142,
    "latency_ms": 3740.5,
    "cost": 0.0000234,
    "model": "gpt-4o-mini"
  },
  "optimized": {
    "total_tokens": 318,
    "input_tokens": 20,
    "output_tokens": 298,
    "latency_ms": 4860.2,
    "cost": 0.0000477,
    "model": "gpt-4o-mini",
    "strategy": "cheap",
    "cache_hit": false,
    "cache_type": "none"
  },
  "final": {
    "total_tokens": 156,
    "input_tokens": 14,
    "output_tokens": 142,
    "latency_ms": 3740.5,
    "cost": 0.0000234
  },
  "guardrail": {
    "fallback_triggered": true,
    "fallback_reason": "tokens_exceeded",
    "wasted_tokens": 318,
    "wasted_cost": 0.0000477
  },
  "quality_judge": {...}
}
```

### Summary Format

```json
{
  "overall": {
    "total_queries": 100,
    "total_tokens_baseline": 15000,
    "total_tokens_optimized": 12000,
    "total_tokens_final": 11500,
    "gross_token_savings": 3000,
    "gross_token_savings_percent": 20.0,
    "net_token_savings": 2500,
    "net_token_savings_percent": 16.67,
    "total_wasted_tokens": 500,
    "total_wasted_cost": 0.000075
  },
  "guardrail_statistics": {
    "total_fallbacks": 5,
    "fallback_rate_percent": 5.0,
    "total_wasted_tokens": 500,
    "total_wasted_cost": 0.000075,
    "avg_wasted_tokens_per_fallback": 100.0
  }
}
```

## Usage

### Running Benchmark

```bash
python benchmarks/run_support_benchmark.py \
  --dataset benchmarks/data/support_dataset.json \
  --output benchmarks/results/support_benchmark_results.json
```

### Analyzing Results

```bash
python benchmarks/analyze_support_benchmark.py \
  --results benchmarks/results/support_benchmark_results.json \
  --output benchmarks/results/support_benchmark_summary.json
```

The summary will show:
- Gross savings (what we'd save if optimized always worked)
- Net savings (actual savings after accounting for waste)
- Total waste (tokens and cost wasted when guardrail triggers)

## Important Notes

1. **No Production Behavior Changes**: This is purely a measurement enhancement. The guardrail still works the same way - it just now tracks what was wasted.

2. **True Cost Reflection**: Metrics now reflect the true cost:
   - When guardrail triggers: `baseline_tokens + optimized_tokens` were consumed
   - Final result shows: `baseline_tokens` (what we actually used)
   - Waste shows: `optimized_tokens` (what we discarded)

3. **Backward Compatibility**: Legacy fields are maintained in benchmark results for compatibility with existing analysis tools.

4. **Dashboard Ready**: Summary metrics are exposed in a format ready for dashboard consumption (no frontend changes needed yet).

## Example Output

```
BENCHMARK SUMMARY
================================================================================

Overall Performance:
  Total Queries: 100
  Baseline Tokens: 15,000
  Optimized Tokens: 12,000
  Final Tokens (after guardrail): 11,500

Token Savings:
  Gross Savings (ignoring waste): 20.00% (3,000 tokens)
  Net Savings (after waste): 16.67% (2,500 tokens)

Cost Savings:
  Gross Savings (ignoring waste): $0.000450
  Net Savings (after waste): $0.000375

Waste Metrics:
  Total Wasted Tokens: 500
  Total Wasted Cost: $0.000075
  Avg Wasted Tokens per Fallback: 100.00

Guardrail Statistics:
  Total Fallbacks: 5 (5.0%)
  Total Wasted Tokens: 500
  Total Wasted Cost: $0.000075
  tokens_exceeded: 5
```
















