#!/usr/bin/env python
"""Test benchmark setup."""
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

output_lines = []

def log(msg):
    output_lines.append(msg)
    print(msg)
    sys.stdout.flush()

try:
    log("1. Testing imports...")
    from tokenomics.core import TokenomicsPlatform
    from tokenomics.config import TokenomicsConfig
    log("   ✓ Imports successful")
    
    log("2. Testing config...")
    config = TokenomicsConfig.from_env()
    log(f"   ✓ Config loaded: provider={config.llm.provider}, model={config.llm.model}")
    
    log("3. Testing platform initialization...")
    platform = TokenomicsPlatform(config=config)
    log("   ✓ Platform initialized")
    
    log("4. Testing dataset loading...")
    import json
    with open("benchmarks/data/support_dataset.json", 'r') as f:
        dataset = json.load(f)
    queries = dataset.get("queries", dataset)
    log(f"   ✓ Dataset loaded: {len(queries)} queries available")
    
    log("\n✓ All setup checks passed! Benchmark should work.")
    
except Exception as e:
    log(f"\n✗ Error: {e}")
    traceback.print_exc()
    sys.exit(1)
finally:
    # Write output to file
    with open("test_setup_output.txt", "w") as f:
        f.write("\n".join(output_lines))









