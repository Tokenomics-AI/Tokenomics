"""Wrapper to run diagnostic benchmark and show results."""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from tests.benchmarks.run_support_benchmark_diagnostics import run_diagnostic_benchmark
from tests.benchmarks.analyze_support_benchmark_diagnostics import analyze_regressions

if __name__ == "__main__":
    print("=" * 80)
    print("RUNNING DIAGNOSTIC BENCHMARK")
    print("=" * 80)
    print()
    
    # Run benchmark
    results = run_diagnostic_benchmark(num_queries=20, use_judge=False)
    
    print()
    print("=" * 80)
    print("RUNNING ANALYSIS")
    print("=" * 80)
    print()
    
    # Run analysis
    results_path = "benchmarks/results/support_benchmark_diagnostics.json"
    if Path(results_path).exists():
        analyze_regressions(results_path)
    else:
        print(f"ERROR: Results file not found at {results_path}")










