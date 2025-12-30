"""Wrapper script to run benchmark and capture output."""
import sys
import os
import subprocess
from pathlib import Path

# Change to Prototype directory
prototype_dir = Path(__file__).parent
os.chdir(prototype_dir)

# Run the benchmark
result = subprocess.run(
    [sys.executable, "benchmarks/run_support_benchmark_diagnostics.py", "--num-queries", "20", "--no-judge"],
    capture_output=True,
    text=True,
    cwd=prototype_dir
)

# Write output to file
output_file = prototype_dir / "benchmark_run_output.txt"
with open(output_file, "w") as f:
    f.write("STDOUT:\n")
    f.write(result.stdout)
    f.write("\nSTDERR:\n")
    f.write(result.stderr)
    f.write(f"\nExit code: {result.returncode}\n")
    
    # Check if results file exists
    results_file = prototype_dir / "benchmarks" / "results" / "support_benchmark_diagnostics.json"
    f.write(f"\nResults file exists: {results_file.exists()}\n")
    if results_file.exists():
        f.write(f"Results file size: {results_file.stat().st_size} bytes\n")

print(f"Output written to: {output_file}")
print(f"Exit code: {result.returncode}")
print(f"STDOUT length: {len(result.stdout)}")
print(f"STDERR length: {len(result.stderr)}")









