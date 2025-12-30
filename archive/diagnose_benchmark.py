#!/usr/bin/env python
"""Diagnose benchmark issues."""
import sys
import traceback
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

output_file = Path(__file__).parent / "diagnostic_output.txt"
output_lines = []

def log(msg):
    output_lines.append(msg)
    log(msg)
    sys.stdout.flush()

log("=" * 80)
log("BENCHMARK DIAGNOSTIC TOOL")
log("=" * 80)
log("")

# Check 1: Python version
log(f"1. Python version: {sys.version}")
log("")

# Check 2: Check if .env file exists
log("2. Checking for .env file...")
env_file = Path(__file__).parent / ".env"
if env_file.exists():
    log(f"   ✓ .env file found: {env_file}")
    # Check if it has content
    with open(env_file, 'r') as f:
        content = f.read().strip()
        if content:
            log(f"   ✓ .env file has content ({len(content)} chars)")
            # Show first few lines (without sensitive data)
            lines = content.split('\n')[:5]
            for line in lines:
                if '=' in line:
                    key = line.split('=')[0]
                    log(f"     - {key}=***")
        else:
            log("   ⚠ .env file is empty")
else:
    log(f"   ⚠ .env file not found at {env_file}")
log()

# Check 3: Environment variables
log("3. Checking environment variables...")
api_keys = {
    "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY"),
    "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
    "GOOGLE_AI_API_KEY": os.getenv("GOOGLE_AI_API_KEY"),
}
found_keys = {k: v[:10] + "..." if v and len(v) > 10 else (v or "NOT SET") for k, v in api_keys.items()}
for key, value in found_keys.items():
    status = "✓" if value != "NOT SET" else "✗"
    log(f"   {status} {key}: {value}")

provider = os.getenv("LLM_PROVIDER", "gemini")
log(f"   LLM_PROVIDER: {provider}")
log()

# Check 4: Try importing modules
log("4. Testing imports...")
try:
    from tokenomics.core import TokenomicsPlatform
    from tokenomics.config import TokenomicsConfig
    log("   ✓ Core imports successful")
except Exception as e:
    log(f"   ✗ Import failed: {e}")
    traceback.print_exc()
    sys.exit(1)
log()

# Check 5: Try loading config
log("5. Testing configuration loading...")
try:
    config = TokenomicsConfig.from_env()
    log(f"   ✓ Config loaded successfully")
    log(f"     Provider: {config.llm.provider}")
    log(f"     Model: {config.llm.model}")
    log(f"     API Key: {'SET' if config.llm.api_key else 'NOT SET'}")
    if not config.llm.api_key:
        log("     ⚠ WARNING: No API key found! Benchmark may fail.")
except Exception as e:
    log(f"   ✗ Config loading failed: {e}")
    traceback.print_exc()
    sys.exit(1)
log()

# Check 6: Try initializing platform
log("6. Testing platform initialization...")
try:
    platform = TokenomicsPlatform(config=config)
    log("   ✓ Platform initialized successfully")
except Exception as e:
    log(f"   ✗ Platform initialization failed: {e}")
    traceback.print_exc()
    sys.exit(1)
log()

# Check 7: Check dataset file
log("7. Checking dataset file...")
dataset_path = Path(__file__).parent / "benchmarks" / "data" / "support_dataset.json"
if dataset_path.exists():
    log(f"   ✓ Dataset file found: {dataset_path}")
    try:
        import json
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
        queries = dataset.get("queries", dataset)
        log(f"   ✓ Dataset loaded: {len(queries)} queries available")
    except Exception as e:
        log(f"   ✗ Dataset loading failed: {e}")
        traceback.print_exc()
else:
    log(f"   ✗ Dataset file not found: {dataset_path}")
    sys.exit(1)
log()

# Check 8: Try running a single query
log("8. Testing single query execution...")
try:
    test_query = queries[0].get("text", queries[0].get("query_text", "What is machine learning?"))
    log(f"   Running test query: {test_query[:50]}...")
    
    result = platform.query(
        query=test_query,
        use_cache=True,
        use_bandit=True,
        use_cost_aware_routing=True,
    )
    
    log(f"   ✓ Query executed successfully")
    log(f"     Tokens used: {result.get('tokens_used', 'N/A')}")
    log(f"     Cache hit: {result.get('cache_hit', False)}")
    log(f"     Strategy: {result.get('strategy', 'N/A')}")
except Exception as e:
    log(f"   ✗ Query execution failed: {e}")
    traceback.print_exc()
    log()
    log("   This is likely why the benchmark is failing!")
    sys.exit(1)
log()

# Check 9: Check benchmark script
log("9. Checking benchmark script...")
benchmark_script = Path(__file__).parent / "benchmarks" / "run_support_benchmark_diagnostics.py"
if benchmark_script.exists():
    log(f"   ✓ Benchmark script found: {benchmark_script}")
else:
    log(f"   ✗ Benchmark script not found: {benchmark_script}")
log()

log("=" * 80)
log("DIAGNOSTIC COMPLETE")
log("=" * 80)
log()
log("If all checks passed, the benchmark should work.")
log("Run: python benchmarks/run_support_benchmark_diagnostics.py --num-queries 20 --no-judge")









