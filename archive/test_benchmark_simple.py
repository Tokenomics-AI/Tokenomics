#!/usr/bin/env python
"""Simple test to see if benchmark can run."""
import sys
import os
from pathlib import Path

# Change to script directory
script_dir = Path(__file__).parent
os.chdir(script_dir)
sys.path.insert(0, str(script_dir))

# Set up output file
output_file = script_dir / "test_output.txt"
output_lines = []

def log(msg):
    output_lines.append(msg)
    log(msg)
    sys.stdout.flush()

log("Current directory: " + str(os.getcwd()))
log("Python version: " + sys.version)
log("")

# Test imports
log("Testing imports...")
try:
    from tokenomics.core import TokenomicsPlatform
    from tokenomics.config import TokenomicsConfig
    log("✓ Imports successful")
except Exception as e:
    log(f"✗ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test config
log("\nTesting config...")
try:
    config = TokenomicsConfig.from_env()
    log(f"✓ Config: provider={config.llm.provider}, model={config.llm.model}")
    if not config.llm.api_key:
        log("⚠ WARNING: No API key!")
except Exception as e:
    log(f"✗ Config failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test dataset
log("\nTesting dataset...")
dataset_path = script_dir / "benchmarks" / "data" / "support_dataset.json"
log(f"Dataset path: {dataset_path}")
log(f"Exists: {dataset_path.exists()}")

if dataset_path.exists():
    import json
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    queries = dataset.get("queries", dataset)
    log(f"✓ Loaded {len(queries)} queries")
else:
    log("✗ Dataset not found!")
    sys.exit(1)

# Test platform init
log("\nTesting platform initialization...")
try:
    platform = TokenomicsPlatform(config=config)
    log("✓ Platform initialized")
except Exception as e:
    log(f"✗ Platform init failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test single query
log("\nTesting single query...")
try:
    test_query = queries[0].get("text", queries[0].get("query_text", "test"))
    log(f"Query: {test_query[:50]}...")
    result = platform.query(
        query=test_query,
        use_cache=True,
        use_bandit=True,
        use_cost_aware_routing=True,
    )
    log(f"✓ Query successful: {result.get('tokens_used', 'N/A')} tokens")
except Exception as e:
    log(f"✗ Query failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

log("\n" + "="*60)
log("ALL TESTS PASSED - Benchmark should work!")
log("="*60)









