# Comprehensive Diagnostic Test - Quick Start Guide

## Overview

The Comprehensive Diagnostic Test is a complete validation suite for the Tokenomics Platform. It tests **every component** and **every capability**, making it perfect for:

- ✅ Product showcase and demos
- ✅ Platform validation
- ✅ Performance benchmarking
- ✅ Component health checks
- ✅ Savings measurement

## Quick Start

### 1. Prerequisites

Ensure you have:
- Python 3.8+
- All dependencies installed (`pip install -r requirements.txt`)
- API keys configured (Gemini, OpenAI, etc.)
- Environment variables set (see `.env` file)

### 2. Run the Test

```bash
# Basic run (saves to diagnostic_results/)
python comprehensive_diagnostic_test.py

# Custom output directory
python comprehensive_diagnostic_test.py --output-dir my_results

# With custom config (if implemented)
python comprehensive_diagnostic_test.py --config config.json
```

### 3. View Results

Results are saved in two formats:

1. **JSON File**: `diagnostic_results/comprehensive_diagnostic_YYYYMMDD_HHMMSS.json`
   - Complete test results with all metrics
   - Use for detailed analysis

2. **Console Output**: Summary printed to console
   - Quick overview of results
   - Key metrics and statistics

## Test Coverage

The test validates:

### ✅ Memory Layer
- Exact cache hits
- Semantic cache (direct return)
- Context injection
- LLM Lingua compression
- User preference learning

### ✅ Orchestrator
- Query complexity analysis
- Token budget allocation
- Context compression
- Multi-model routing

### ✅ Bandit Optimizer
- Strategy selection (cheap/balanced/premium)
- RouterBench cost-aware routing
- Learning and adaptation
- Reward computation

### ✅ Quality Judge (if enabled)
- Quality comparison (optimized vs baseline)
- Quality scoring

### ✅ Component Savings
- Memory layer savings
- Orchestrator savings
- Bandit savings
- Total savings

## Test Dataset

The test includes **30+ test cases** covering:

1. **Exact Cache**: Identical queries
2. **Semantic Cache**: High similarity queries
3. **Context Injection**: Medium similarity queries
4. **LLM Lingua**: Long queries and contexts
5. **User Preferences**: Formal, casual, technical tones
6. **Query Complexity**: Simple, medium, complex
7. **Bandit Selection**: Strategy selection
8. **RouterBench**: Cost-aware routing
9. **Token Budgets**: Low, medium, high budgets
10. **Edge Cases**: Empty queries, very long queries, special characters

## Understanding Results

### Summary Metrics

After running, you'll see:

```
Test Execution:
  Total Tests: 30
  Successful: 30
  Failed: 0
  Elapsed Time: 45.23s

Cache Performance:
  Cache Hit Rate: 35.0%
  Exact Hits: 2
  Semantic Direct Hits: 3
  Context Hits: 5

Token Usage:
  Total Tokens: 12,450
  Input Tokens: 8,200
  Output Tokens: 4,250
  Avg per Query: 415

Savings:
  Total Savings: 3,750 tokens
  Memory Layer: 2,100 tokens
  Orchestrator: 1,200 tokens
  Bandit: 450 tokens
  Compression: 800 tokens
  Savings Percentage: 23.2%
```

### Key Metrics Explained

- **Cache Hit Rate**: Percentage of queries that hit cache (20-40% typical)
- **Total Savings**: Tokens saved vs baseline (15-35% typical)
- **Savings Percentage**: Overall efficiency (higher is better)
- **Strategy Distribution**: How strategies were selected

### Interpreting Results

**✅ Good Platform Health**:
- Cache hit rate >20%
- Total savings >15%
- Quality maintained
- All components working

**⚠️ Issues to Watch**:
- Cache hit rate <10% → Cache not working well
- Total savings <5% → Optimization not effective
- High error rate → Component failures

## Detailed Documentation

For complete documentation on every metric, see:
- **[COMPREHENSIVE_DIAGNOSTIC_DOCUMENTATION.md](COMPREHENSIVE_DIAGNOSTIC_DOCUMENTATION.md)**

This document explains:
- Every metric in detail
- How each component works
- What each value signifies
- Good vs bad values
- How to interpret results

## Test Structure

```
comprehensive_diagnostic_test.py
├── ComprehensiveDiagnosticTest (main class)
│   ├── _create_test_dataset()      # Creates 30+ test cases
│   ├── run_all_tests()             # Runs all tests
│   ├── _run_test_case()            # Runs single test
│   ├── _test_memory_layer()        # Component tests
│   ├── _test_orchestrator()        # Component tests
│   ├── _test_bandit_optimizer()    # Component tests
│   ├── _test_llmlingua()          # Component tests
│   ├── _calculate_summary()        # Summary statistics
│   └── save_results()              # Save to JSON
└── main()                          # Entry point
```

## Customization

### Adding Test Cases

Edit `_create_test_dataset()` to add custom test cases:

```python
{
    "id": "my_test",
    "category": "custom",
    "query": "Your test query",
    "description": "What this tests",
    "expected_cache": "miss",  # Optional
    "expected_complexity": "medium",  # Optional
}
```

### Modifying Test Parameters

Edit test case parameters:

```python
{
    "token_budget": 2000,  # Custom budget
    "use_cost_aware_routing": True,  # Enable RouterBench
}
```

## Troubleshooting

### Common Issues

**1. API Key Errors**
- Check `.env` file has correct API keys
- Verify keys are valid and have quota

**2. LLM Lingua Not Available**
- Install: `pip install llmlingua`
- Check logs for initialization errors
- Test will fall back to simple compression

**3. Low Cache Hit Rate**
- Normal for first run (cache is empty)
- Run test twice to see cache hits
- Check similarity thresholds in config

**4. High Error Rate**
- Check API quotas
- Verify network connectivity
- Review logs for specific errors

## Performance Tips

1. **First Run**: Cache will be empty, expect lower hit rate
2. **Subsequent Runs**: Cache will be populated, higher hit rate
3. **API Costs**: Test uses real API calls (costs apply)
4. **Duration**: Test takes 30-60 seconds depending on API speed

## Next Steps

1. **Run the Test**: `python comprehensive_diagnostic_test.py`
2. **Review Results**: Check JSON file and console output
3. **Read Documentation**: See detailed metric explanations
4. **Customize**: Add your own test cases
5. **Iterate**: Run multiple times to see learning effects

## Support

For questions or issues:
- Check [COMPREHENSIVE_DIAGNOSTIC_DOCUMENTATION.md](COMPREHENSIVE_DIAGNOSTIC_DOCUMENTATION.md)
- Review test logs
- Check component-specific documentation

---

**This test is designed for product showcase - use it to demonstrate the full capabilities of the Tokenomics Platform!**

