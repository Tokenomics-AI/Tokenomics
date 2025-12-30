# Benchmark Tests

This directory contains benchmark tests for evaluating platform performance.

## Tests

### `run_support_benchmark.py`

Runs support use case benchmark with configurable query count.

**Usage:**
```bash
python tests/benchmarks/run_support_benchmark.py
```

### `run_support_benchmark_diagnostics.py`

Diagnostic benchmark runner with regression analysis.

**Usage:**
```bash
python tests/benchmarks/run_support_benchmark_diagnostics.py
```

### `analyze_support_benchmark.py`

Analyzes benchmark results and generates reports.

**Usage:**
```bash
python tests/benchmarks/analyze_support_benchmark.py
```

### `analyze_support_benchmark_diagnostics.py`

Analyzes diagnostic benchmark results and identifies regressions.

**Usage:**
```bash
python tests/benchmarks/analyze_support_benchmark_diagnostics.py
```

## Data

Benchmark datasets are stored in `data/`:
- `quick_dataset.json` - Quick validation dataset
- `support_dataset.json` - Support use case dataset
- `test_dataset.json` - Test dataset

## Results

Benchmark results are stored in `results/`:
- `quick_benchmark_results.json` - Quick benchmark results
- `support_benchmark_results.json` - Support benchmark results
- `test_benchmark_results.json` - Test benchmark results

## Documentation

See [docs/testing/benchmark-tests.md](../../docs/testing/benchmark-tests.md) for detailed benchmark documentation.








