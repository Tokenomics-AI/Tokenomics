# Testing Overview

The Tokenomics Platform uses a comprehensive testing strategy to ensure quality and performance.

## Test Structure

### Unit Tests

**Location:** `tests/unit/`

Tests individual components in isolation:
- `test_memory.py` - Memory layer tests
- `test_orchestrator.py` - Orchestrator tests
- `test_bandit.py` - Bandit optimizer tests

**Run:**
```bash
pytest tests/unit/ -v
```

### Integration Tests

**Location:** `tests/integration/`

Tests component interactions:
- Platform flow tests
- Comprehensive platform tests
- Enhanced memory system tests
- Semantic cache tests

**Run:**
```bash
pytest tests/integration/ -v
```

### Diagnostic Tests

**Location:** `tests/diagnostic/`

Comprehensive validation tests:
- Extensive diagnostic test (32 queries, 10 phases)
- Quick fix validation
- Setup verification

**Run:**
```bash
python tests/diagnostic/extensive_diagnostic_test.py
```

### Benchmark Tests

**Location:** `tests/benchmarks/`

Performance evaluation tests:
- Support use case benchmarks
- Diagnostic benchmarks
- Regression analysis

**Run:**
```bash
python tests/benchmarks/run_support_benchmark.py
```

## Test Evolution

See [test-evolution.md](test-evolution.md) for the complete story of how testing evolved and improved the platform.

## Test Results

- [Diagnostic Results](../results/diagnostic-results.md)
- [Performance Analysis](../results/performance-analysis.md)
- [Quality Analysis](../results/quality-analysis.md)

## Running All Tests

```bash
# Unit tests
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v

# Diagnostic tests
python tests/diagnostic/extensive_diagnostic_test.py

# Benchmarks
python tests/benchmarks/run_support_benchmark.py
```








