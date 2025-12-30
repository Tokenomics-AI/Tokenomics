"""Quick test to verify diagnostic benchmark works."""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from tokenomics.core import TokenomicsPlatform
from tokenomics.config import TokenomicsConfig

print("Testing platform query with diagnostics...")

config = TokenomicsConfig.from_env()
platform = TokenomicsPlatform(config=config)

# Test single query
query = "What is machine learning?"
print(f"\nQuery: {query}")

try:
    result = platform.query(
        query=query,
        use_cache=True,
        use_bandit=True,
        use_cost_aware_routing=True,
    )
    
    print("\n✓ Query successful!")
    print(f"Diagnostic fields:")
    print(f"  query_type: {result.get('query_type')}")
    print(f"  cache_tier: {result.get('cache_tier')}")
    print(f"  capsule_tokens: {result.get('capsule_tokens')}")
    print(f"  strategy_arm: {result.get('strategy_arm')}")
    print(f"  model_used: {result.get('model_used')}")
    print(f"  used_memory: {result.get('used_memory')}")
    print(f"  user_preference: {result.get('user_preference')}")
    print(f"  tokens_used: {result.get('tokens_used')}")
    
    baseline = result.get('baseline_comparison_result')
    if baseline:
        print(f"\nBaseline comparison:")
        print(f"  baseline_tokens: {baseline.get('tokens_used')}")
        print(f"  optimized_tokens: {result.get('tokens_used')}")
        print(f"  Regression: {result.get('tokens_used', 0) > baseline.get('tokens_used', 0)}")
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()










