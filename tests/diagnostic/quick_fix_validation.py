"""
Quick validation test to compare before/after fixes.

Tests key improvements:
1. Complexity classification (should detect medium/complex)
2. Strategy selection (should use balanced/premium for complex)
3. Tone detection (should detect formal/technical/casual)
4. Compression (should trigger for 500+ char queries)
"""

import os
import sys
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / '.env')

from tokenomics.core import TokenomicsPlatform
from tokenomics.config import TokenomicsConfig

print("=" * 80)
print("QUICK FIX VALIDATION TEST")
print("=" * 80)

# Initialize platform
config = TokenomicsConfig.from_env()
config.llm.provider = "openai"
config.llm.model = "gpt-4o-mini"
config.llm.api_key = os.getenv("OPENAI_API_KEY")

platform = TokenomicsPlatform(config=config)

# Test cases
test_cases = [
    {
        "name": "Complexity: Simple Query",
        "query": "What is JSON?",
        "expected_complexity": "simple",
        "expected_strategy": "cheap",
    },
    {
        "name": "Complexity: Medium Query",
        "query": "How does JSON parsing work in Python with error handling?",
        "expected_complexity": "medium",
        "expected_strategy": "balanced",
    },
    {
        "name": "Complexity: Complex Query",
        "query": "Design a comprehensive microservices architecture for an e-commerce platform with authentication, payment processing, and order management",
        "expected_complexity": "complex",
        "expected_strategy": "premium",
    },
    {
        "name": "Tone: Formal",
        "query": "Could you please kindly explain Docker containers?",
        "expected_tone": "formal",
    },
    {
        "name": "Tone: Technical",
        "query": "Explain the Docker container runtime architecture and implementation",
        "expected_tone": "technical",
    },
    {
        "name": "Tone: Casual",
        "query": "Hey, what's the deal with Docker?",
        "expected_tone": "casual",
    },
    {
        "name": "Compression: Long Query",
        "query": "I need a comprehensive and detailed explanation of how modern machine learning algorithms work in production systems. Please cover the following topics in depth: 1) Neural network architectures including CNNs, RNNs, LSTMs, and Transformers, 2) The backpropagation algorithm and how gradients flow through the network, 3) Optimization techniques like SGD, Adam, and learning rate scheduling, 4) Regularization methods including dropout, batch normalization, and L2 regularization, 5) How these models are deployed in production with considerations for latency and throughput, 6) Best practices for model monitoring and retraining in production environments. Please provide specific examples and code snippets where appropriate.",
        "expected_compression": True,
    },
]

results = {
    "complexity_tests": [],
    "tone_tests": [],
    "compression_tests": [],
    "strategy_tests": [],
}

print("\nRunning validation tests...\n")

for test in test_cases:
    print(f"Testing: {test['name']}")
    
    try:
        result = platform.query(
            query=test["query"],
            use_cache=True,
            use_bandit=True,
            use_compression=True,
            use_cost_aware_routing=True,
        )
        
        # Check complexity
        if "expected_complexity" in test:
            actual = result.get("query_type")
            expected = test["expected_complexity"]
            passed = actual == expected
            results["complexity_tests"].append({
                "name": test["name"],
                "expected": expected,
                "actual": actual,
                "passed": passed,
            })
            status = "PASS" if passed else "FAIL"
            print(f"  Complexity: {status} (expected: {expected}, got: {actual})")
        
        # Check strategy
        if "expected_strategy" in test:
            actual = result.get("strategy")
            expected = test["expected_strategy"]
            passed = actual == expected
            results["strategy_tests"].append({
                "name": test["name"],
                "expected": expected,
                "actual": actual,
                "passed": passed,
            })
            status = "PASS" if passed else "FAIL"
            print(f"  Strategy: {status} (expected: {expected}, got: {actual})")
        
        # Check tone
        if "expected_tone" in test:
            pref_context = result.get("preference_context", {})
            actual = pref_context.get("tone", "neutral")
            expected = test["expected_tone"]
            passed = actual == expected
            results["tone_tests"].append({
                "name": test["name"],
                "expected": expected,
                "actual": actual,
                "passed": passed,
            })
            status = "PASS" if passed else "FAIL"
            print(f"  Tone: {status} (expected: {expected}, got: {actual})")
        
        # Check compression
        if "expected_compression" in test:
            compression = result.get("compression_metrics", {})
            actual = compression.get("query_compressed", False)
            expected = test["expected_compression"]
            passed = actual == expected
            results["compression_tests"].append({
                "name": test["name"],
                "expected": expected,
                "actual": actual,
                "passed": passed,
            })
            status = "PASS" if passed else "FAIL"
            print(f"  Compression: {status} (expected: {expected}, got: {actual})")
        
        print(f"  Tokens: {result.get('tokens_used', 0)}, Cache: {result.get('cache_type', 'none')}")
        print()
        
    except Exception as e:
        print(f"  ERROR: {e}\n")

# Summary
print("=" * 80)
print("VALIDATION SUMMARY")
print("=" * 80)

# Complexity
complexity_passed = sum(1 for t in results["complexity_tests"] if t["passed"])
complexity_total = len(results["complexity_tests"])
print(f"\nComplexity Classification: {complexity_passed}/{complexity_total} passed")
for t in results["complexity_tests"]:
    status = "PASS" if t["passed"] else "FAIL"
    print(f"  [{status}] {t['name']}: {t['expected']} -> {t['actual']}")

# Strategy
strategy_passed = sum(1 for t in results["strategy_tests"] if t["passed"])
strategy_total = len(results["strategy_tests"])
print(f"\nStrategy Selection: {strategy_passed}/{strategy_total} passed")
for t in results["strategy_tests"]:
    status = "PASS" if t["passed"] else "FAIL"
    print(f"  [{status}] {t['name']}: {t['expected']} -> {t['actual']}")

# Tone
tone_passed = sum(1 for t in results["tone_tests"] if t["passed"])
tone_total = len(results["tone_tests"])
print(f"\nTone Detection: {tone_passed}/{tone_total} passed")
for t in results["tone_tests"]:
    status = "PASS" if t["passed"] else "FAIL"
    print(f"  [{status}] {t['name']}: {t['expected']} -> {t['actual']}")

# Compression
compression_passed = sum(1 for t in results["compression_tests"] if t["passed"])
compression_total = len(results["compression_tests"])
print(f"\nCompression: {compression_passed}/{compression_total} passed")
for t in results["compression_tests"]:
    status = "PASS" if t["passed"] else "FAIL"
    print(f"  [{status}] {t['name']}: {t['expected']} -> {t['actual']}")

# Overall
total_passed = complexity_passed + strategy_passed + tone_passed + compression_passed
total_tests = complexity_total + strategy_total + tone_total + compression_total
print(f"\n{'=' * 80}")
print(f"OVERALL: {total_passed}/{total_tests} tests passed ({total_passed/total_tests*100:.1f}%)")
print("=" * 80)

