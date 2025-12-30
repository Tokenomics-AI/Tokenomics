# Comprehensive Diagnostic Test Suite

## Overview

The comprehensive diagnostic test suite (`test_comprehensive_diagnostics.py`) provides in-depth testing of all Tokenomics platform functionalities, measuring savings at each component level, tracking latency improvements, evaluating quality, and providing detailed diagnostics.

## What It Tests

### Component Functionality
- **Memory Layer**: Exact cache, semantic cache, compression integration
- **Orchestrator**: Complexity analysis, token allocation, query planning
- **Bandit Optimizer**: Strategy selection, learning mechanism, cost-aware routing
- **LLMLingua Compression**: Context compression, query compression, fallback behavior
- **Quality Judge**: Evaluation execution, winner detection, confidence scoring

### Use Case Scenarios
- **Query Types**: Short, medium, long, repeated, similar, complex, simple queries
- **Cache Scenarios**: Cold start, warm cache, exact hits, semantic hits, context injection
- **Strategy Scenarios**: Cheap, balanced, premium strategies, exploration vs exploitation

### Metrics Measured
- **Token Savings**: Memory layer, orchestrator, bandit, compression, total
- **Latency**: Baseline vs optimized, cache hit latency, LLM call latency
- **Quality**: Judge results, confidence scores, response relevance
- **Compression**: Context and query compression ratios, savings

## How to Run

```bash
cd Prototype
python test_comprehensive_diagnostics.py
```

## Output

The test generates three output files in `diagnostic_results/`:

1. **`diagnostic_report_{timestamp}.json`**: Complete test results in JSON format
2. **`diagnostic_metrics_{timestamp}.json`**: Aggregated metrics only
3. **`diagnostic_report_{timestamp}.html`**: Visual HTML report with charts and summaries

## Test Flow

1. **Environment Setup**: Checks API keys, verifies configuration
2. **Platform Initialization**: Initializes all components
3. **Component Health Check**: Tests each component's health status
4. **Component Functionality Tests**: Detailed tests of each component
5. **Query Tests**: Runs diverse queries covering all use cases
6. **Metrics Aggregation**: Calculates savings, latency, quality metrics
7. **Diagnostics Generation**: Identifies working/non-working features
8. **Report Generation**: Creates JSON and HTML reports

## What to Look For

### Success Indicators
- ✓ All components show "healthy" status
- ✓ Token savings > 0%
- ✓ Latency reduction > 0%
- ✓ Cache hit rate > 0% (increases with repeated queries)
- ✓ Compression savings > 0 (for long queries)
- ✓ Quality judge evaluations working (in A/B mode)

### Warning Signs
- ⚠ Components showing "degraded" status
- ⚠ Low cache hit rate (< 10%)
- ⚠ No compression savings detected
- ⚠ Query failures
- ⚠ Quality judge not executing

### Failure Indicators
- ✗ Components showing "failed" status
- ✗ Consistent query failures
- ✗ Negative savings (optimized using more tokens)
- ✗ API errors

## Interpreting Results

### Component Savings Breakdown
- **Memory Layer**: Tokens saved from cache hits (should be high for repeated queries)
- **Orchestrator**: Tokens saved from allocation/compression (should be positive)
- **Bandit**: Tokens saved from strategy selection (may be 0 if all strategies similar)
- **Compression**: Tokens saved from LLMLingua compression (should be positive for long queries)

### Cache Performance
- **Exact Hits**: Same query repeated → should be 100% hit rate
- **Semantic Hits**: Similar queries → should show semantic matching
- **Context Injection**: Medium similarity → should inject context

### Strategy Selection
- **Simple queries** → Should use "cheap" strategy
- **Complex queries** → Should use "balanced" or "premium" strategy
- **Exploration phase** → First 9 queries explore all 3 strategies
- **Exploitation phase** → After exploration, uses best performing strategy

## Troubleshooting

### If tests fail:
1. Check API keys are set in `.env` file
2. Verify LLM provider is accessible
3. Check if LLMLingua-2 is installed (optional, will use fallback)
4. Review error messages in the console output
5. Check the JSON report for detailed error information

### If metrics look wrong:
1. Verify baseline queries are running (should see baseline tokens > 0)
2. Check if cache is working (repeated queries should show cache hits)
3. Verify compression is enabled in config
4. Check if quality judge is enabled (for A/B comparisons)

## Example Output

```
================================================================================
COMPREHENSIVE DIAGNOSTIC TEST SUITE
================================================================================
Start Time: 2025-12-07T10:00:00

INITIALIZING PLATFORM
================================================================================
✓ API key found: sk-proj-...
✓ Configuration loaded
✓ Platform initialized successfully

TESTING COMPONENT HEALTH
================================================================================
✓ Memory Layer: Healthy
✓ Orchestrator: Healthy
✓ Bandit Optimizer: Healthy (3 strategies)
✓ Quality Judge: Healthy
⚠ LLMLingua: Not available (using fallback)

RUNNING QUERY TESTS - USE CASE SCENARIOS
================================================================================
[1/20] Query Test - SHORT
  Testing: What is Python?...
    Baseline: 150 tokens, 1200ms
    Optimized: 60 tokens, 450ms
    Savings: 90 tokens (60.0%), 750ms (62.5%)
    Cache: miss
    Strategy: cheap

...

AGGREGATING METRICS
================================================================================
Total Queries: 20
Successful: 20
Token Savings: 2500 tokens (35.2%)
Latency Reduction: 15000ms (42.1%)
Cache Hit Rate: 25.0% (5/20)

DIAGNOSTIC TEST SUMMARY
================================================================================
Status: HEALTHY
Success Rate: 100.0%
Token Savings: 35.2%
Latency Reduction: 42.1%
Cache Hit Rate: 25.0%
```

