"""Test script to capture benchmark errors."""

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    from benchmarks.run_support_benchmark_diagnostics import run_diagnostic_benchmark
    
    print("Starting benchmark with 2 queries...")
    results = run_diagnostic_benchmark(num_queries=2, use_judge=False)
    print(f"Completed! Got {len(results)} results")
    
    if results:
        print(f"First result keys: {list(results[0].keys())}")
        if 'error' in results[0]:
            print(f"Error in first result: {results[0]['error']}")
except Exception as e:
    print(f"ERROR: {e}")
    traceback.print_exc()










