"""Test complete platform flow to verify all components work together."""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from tokenomics.core import TokenomicsPlatform
from tokenomics.config import TokenomicsConfig

def test_platform():
    """Test complete platform flow."""
    print("=" * 60)
    print("TESTING TOKENOMICS PLATFORM")
    print("=" * 60)
    
    # Load environment
    load_dotenv()
    
    # Initialize config
    print("\n1. Initializing configuration...")
    try:
        config = TokenomicsConfig.from_env()
        print(f"   ✓ Config loaded: provider={config.llm.provider}, model={config.llm.model}")
        print(f"   ✓ Judge enabled: {config.judge.enabled}")
    except Exception as e:
        print(f"   ✗ Config failed: {e}")
        return False
    
    # Initialize platform
    print("\n2. Initializing platform...")
    try:
        platform = TokenomicsPlatform(config)
        print("   ✓ Platform initialized")
        print(f"   ✓ Memory layer: {type(platform.memory).__name__}")
        print(f"   ✓ Orchestrator: {type(platform.orchestrator).__name__}")
        print(f"   ✓ Bandit: {type(platform.bandit).__name__}")
        print(f"   ✓ Quality judge: {platform.quality_judge is not None}")
    except Exception as e:
        print(f"   ✗ Platform init failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test simple query (no cache, no bandit)
    print("\n3. Testing baseline query (no cache, no bandit)...")
    try:
        result = platform.query("What is artificial intelligence?", use_cache=False, use_bandit=False)
        print(f"   ✓ Query successful")
        print(f"   - Response length: {len(result.get('response', ''))}")
        print(f"   - Tokens used: {result.get('tokens_used', 0)}")
        print(f"   - Latency: {result.get('latency_ms', 0):.2f} ms")
    except Exception as e:
        print(f"   ✗ Baseline query failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test optimized query (with cache and bandit)
    print("\n4. Testing optimized query (with cache and bandit)...")
    try:
        result = platform.query("What is artificial intelligence?", use_cache=True, use_bandit=True)
        print(f"   ✓ Query successful")
        print(f"   - Response length: {len(result.get('response', ''))}")
        print(f"   - Tokens used: {result.get('tokens_used', 0)}")
        print(f"   - Cache hit: {result.get('cache_hit', False)}")
        print(f"   - Cache type: {result.get('cache_type', 'none')}")
        print(f"   - Strategy: {result.get('strategy', 'none')}")
        print(f"   - Model: {result.get('model', 'unknown')}")
        print(f"   - Fallback to baseline: {result.get('fallback_to_baseline', False)}")
        if result.get('baseline_comparison_result'):
            baseline = result['baseline_comparison_result']
            print(f"   - Baseline tokens: {baseline.get('tokens_used', 0)}")
            print(f"   - Token savings: {baseline.get('tokens_used', 0) - result.get('tokens_used', 0)}")
    except Exception as e:
        print(f"   ✗ Optimized query failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test cache hit
    print("\n5. Testing cache hit (same query again)...")
    try:
        result = platform.query("What is artificial intelligence?", use_cache=True, use_bandit=True)
        print(f"   ✓ Query successful")
        print(f"   - Cache hit: {result.get('cache_hit', False)}")
        print(f"   - Cache type: {result.get('cache_type', 'none')}")
        print(f"   - Tokens used: {result.get('tokens_used', 0)} (should be 0 for cache hit)")
    except Exception as e:
        print(f"   ✗ Cache hit test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test stats
    print("\n6. Testing platform stats...")
    try:
        stats = platform.get_stats()
        print(f"   ✓ Stats retrieved")
        print(f"   - Memory cache size: {stats.get('memory', {}).get('size', 0)}")
        print(f"   - Bandit pulls: {stats.get('bandit', {}).get('total_pulls', 0)}")
    except Exception as e:
        print(f"   ✗ Stats failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_platform()
    sys.exit(0 if success else 1)
















