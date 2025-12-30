"""Test the token regression fix."""
import json
import os
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from tokenomics.core import TokenomicsPlatform
from tokenomics.config import TokenomicsConfig

def test_regression_fix():
    """Test that the fix prevents token regressions."""
    print("=" * 80)
    print("TESTING TOKEN REGRESSION FIX")
    print("=" * 80)
    print()
    
    # Load config
    config = TokenomicsConfig.from_env()
    print(f"Using provider: {config.llm.provider}, model: {config.llm.model}")
    
    # Initialize platform
    platform = TokenomicsPlatform(config=config)
    
    # Test queries that previously had regressions
    test_queries = [
        "What is the refund policy?",
        "Can I customize the dashboard?",
        "What is your privacy policy?",
        "Why did my upload fail?",
        "How do I contact support?",
    ]
    
    results = []
    token_regressions = 0
    
    for i, query in enumerate(test_queries):
        print(f"\n[{i+1}/{len(test_queries)}] Testing: {query[:50]}...")
        
        try:
            # Run optimized query (includes baseline comparison)
            result = platform.query(
                query=query,
                use_cache=False,  # Disable cache to test LLM path
                use_bandit=True,
                use_cost_aware_routing=True,
            )
            
            # Get baseline comparison
            baseline = result.get("baseline_comparison_result", {})
            
            baseline_tokens = baseline.get("tokens_used", 0)
            optimized_tokens = result.get("tokens_used", 0)
            baseline_output = baseline.get("output_tokens", 0)
            optimized_output = result.get("output_tokens", 0)
            
            has_regression = optimized_tokens > baseline_tokens
            if has_regression:
                token_regressions += 1
                status = "❌ REGRESSION"
            else:
                savings = baseline_tokens - optimized_tokens
                status = f"✓ SAVED {savings} tokens"
            
            print(f"  Baseline: {baseline_tokens} tokens (output={baseline_output})")
            print(f"  Optimized: {optimized_tokens} tokens (output={optimized_output})")
            print(f"  Max tokens used: {result.get('max_response_tokens', 'N/A')}")
            print(f"  Status: {status}")
            
            results.append({
                "query": query,
                "baseline_tokens": baseline_tokens,
                "optimized_tokens": optimized_tokens,
                "baseline_output": baseline_output,
                "optimized_output": optimized_output,
                "has_regression": has_regression,
                "max_response_tokens": result.get("max_response_tokens"),
            })
            
        except Exception as e:
            print(f"  Error: {e}")
            results.append({"query": query, "error": str(e)})
    
    # Summary
    valid_results = [r for r in results if "error" not in r]
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total queries: {len(test_queries)}")
    print(f"Successful: {len(valid_results)}")
    print(f"Token regressions: {token_regressions}/{len(valid_results)}")
    
    if token_regressions == 0:
        print("\n✓ SUCCESS: No token regressions detected!")
        print("The fix is working correctly.")
    else:
        print(f"\n❌ ISSUE: {token_regressions} regressions still present")
        print("Further investigation needed.")
    
    # Save results
    with open("test_regression_fix_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: test_regression_fix_results.json")
    
    return token_regressions == 0


if __name__ == "__main__":
    success = test_regression_fix()
    sys.exit(0 if success else 1)










