#!/usr/bin/env python
"""Direct benchmark runner."""
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import and run
from tests.benchmarks.run_support_benchmark_diagnostics import run_diagnostic_benchmark

if __name__ == "__main__":
    print("Starting 20-query benchmark...")
    print("=" * 60)
    
    results = run_diagnostic_benchmark(
        dataset_path="benchmarks/data/support_dataset.json",
        output_path="benchmarks/results/support_benchmark_diagnostics.json",
        num_queries=20,
        use_judge=False,
    )
    
    print("=" * 60)
    print(f"Benchmark completed! Processed {len(results)} queries.")
    print("Results saved to: benchmarks/results/support_benchmark_diagnostics.json")









