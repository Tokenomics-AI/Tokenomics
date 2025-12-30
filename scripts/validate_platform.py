"""Comprehensive platform validation suite."""

import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import json

sys.path.insert(0, str(Path(__file__).parent))

from tokenomics.core import TokenomicsPlatform
from tokenomics.config import TokenomicsConfig
from tokenomics.bandit import Strategy
from usage_tracker import UsageTracker


class PlatformValidator:
    """Comprehensive validation of all platform components."""
    
    def __init__(self):
        """Initialize validator."""
        self.results = {
            "validation_start": datetime.now().isoformat(),
            "tests": {},
            "overall_status": "PENDING",
        }
        self.tracker = UsageTracker(output_file="validation_report.json")
    
    def log_test(self, test_name: str, status: str, details: Dict):
        """Log test result."""
        self.results["tests"][test_name] = {
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "details": details,
        }
        print(f"\n{'='*80}")
        print(f"TEST: {test_name}")
        print(f"STATUS: {status}")
        print(f"{'='*80}")
        for key, value in details.items():
            print(f"  {key}: {value}")
    
    def test_memory_cache(self, platform: TokenomicsPlatform) -> bool:
        """Test 1: Memory cache functionality."""
        print("\n" + "="*80)
        print("TEST 1: MEMORY CACHE FUNCTIONALITY")
        print("="*80)
        
        details = {}
        
        # Test exact cache
        query1 = "What is artificial intelligence?"
        result1 = platform.query(query1, use_cache=True)
        details["first_call_tokens"] = result1["tokens_used"]
        details["first_call_latency"] = result1["latency_ms"]
        details["first_cache_hit"] = result1["cache_hit"]
        
        # Second call should hit cache
        result2 = platform.query(query1, use_cache=True)
        details["second_call_tokens"] = result2["tokens_used"]
        details["second_call_latency"] = result2["latency_ms"]
        details["second_cache_hit"] = result2["cache_hit"]
        
        # Validate
        cache_working = (
            not result1["cache_hit"] and  # First call misses
            result2["cache_hit"] and  # Second call hits
            result2["tokens_used"] == 0 and  # Zero tokens for cached
            result2["latency_ms"] < 100  # Very fast
        )
        
        details["cache_validated"] = cache_working
        details["tokens_saved"] = result1["tokens_used"]
        
        self.log_test("Memory Cache", "PASS" if cache_working else "FAIL", details)
        return cache_working
    
    def test_token_orchestrator(self, platform: TokenomicsPlatform) -> bool:
        """Test 2: Token orchestrator functionality."""
        print("\n" + "="*80)
        print("TEST 2: TOKEN ORCHESTRATOR FUNCTIONALITY")
        print("="*80)
        
        details = {}
        
        # Test with different query complexities
        queries = [
            ("Short query", "What is AI?", "simple"),
            ("Medium query", "Explain how neural networks learn from data", "medium"),
            ("Complex query", "Compare and contrast supervised learning, unsupervised learning, and reinforcement learning. Include examples of each and explain when to use which approach.", "complex"),
        ]
        
        orchestrator_results = []
        
        for name, query, expected_complexity in queries:
            result = platform.query(query, use_cache=False)
            plan = result.get("plan")
            
            if plan:
                orchestrator_results.append({
                    "query_name": name,
                    "complexity": plan.complexity.value,
                    "expected": expected_complexity,
                    "token_budget": plan.token_budget,
                    "allocations": len(plan.allocations),
                    "model": plan.model,
                })
        
        # Validate orchestrator is working
        orchestrator_working = (
            len(orchestrator_results) == 3 and
            all(r["allocations"] > 0 for r in orchestrator_results) and
            all(r["token_budget"] > 0 for r in orchestrator_results)
        )
        
        details["queries_tested"] = len(orchestrator_results)
        details["all_have_allocations"] = all(r["allocations"] > 0 for r in orchestrator_results)
        details["all_have_budgets"] = all(r["token_budget"] > 0 for r in orchestrator_results)
        details["results"] = orchestrator_results
        
        self.log_test("Token Orchestrator", "PASS" if orchestrator_working else "FAIL", details)
        return orchestrator_working
    
    def test_bandit_optimizer(self, platform: TokenomicsPlatform) -> bool:
        """Test 3: Bandit optimizer functionality."""
        print("\n" + "="*80)
        print("TEST 3: BANDIT OPTIMIZER FUNCTIONALITY")
        print("="*80)
        
        details = {}
        
        # Get initial bandit state
        initial_stats = platform.bandit.stats()
        details["initial_arms"] = len(initial_stats["arms"])
        details["initial_pulls"] = initial_stats["total_pulls"]
        
        # Run multiple queries to let bandit learn
        test_queries = [
            "What is machine learning?",
            "Explain deep learning",
            "What is natural language processing?",
            "How do transformers work?",
        ]
        
        strategies_used = []
        for query in test_queries:
            result = platform.query(query, use_cache=False)
            strategy = result.get("strategy")
            if strategy:
                strategies_used.append(strategy)
        
        # Check bandit learned
        final_stats = platform.bandit.stats()
        details["final_pulls"] = final_stats["total_pulls"]
        details["strategies_used"] = list(set(strategies_used))
        details["unique_strategies"] = len(set(strategies_used))
        
        # Get best strategy
        best_strategy = platform.bandit.get_best_strategy()
        details["best_strategy"] = best_strategy.arm_id if best_strategy else None
        
        # Validate bandit is working
        bandit_working = (
            final_stats["total_pulls"] > initial_stats["total_pulls"] and
            len(set(strategies_used)) > 0 and
            best_strategy is not None
        )
        
        details["bandit_learning"] = final_stats["total_pulls"] > initial_stats["total_pulls"]
        details["strategy_selection"] = len(set(strategies_used)) > 0
        
        self.log_test("Bandit Optimizer", "PASS" if bandit_working else "FAIL", details)
        return bandit_working
    
    def test_integration(self, platform: TokenomicsPlatform) -> bool:
        """Test 4: Full platform integration."""
        print("\n" + "="*80)
        print("TEST 4: FULL PLATFORM INTEGRATION")
        print("="*80)
        
        details = {}
        
        # Test complex scenario with all components
        queries = [
            "What is Python programming?",
            "Explain object-oriented programming",
            "What is Python programming?",  # Should hit cache
            "How does inheritance work?",
            "What is Python programming?",  # Should hit cache again
        ]
        
        integration_results = []
        cache_hits = 0
        total_tokens = 0
        
        for i, query in enumerate(queries, 1):
            result = platform.query(query, use_cache=True, use_bandit=True)
            
            integration_results.append({
                "query_num": i,
                "query": query[:50],
                "tokens": result["tokens_used"],
                "cache_hit": result["cache_hit"],
                "strategy": result.get("strategy"),
                "latency": result["latency_ms"],
            })
            
            if result["cache_hit"]:
                cache_hits += 1
            total_tokens += result["tokens_used"]
        
        # Validate integration
        integration_working = (
            cache_hits >= 2 and  # At least 2 cache hits
            total_tokens > 0 and  # Some tokens used
            all(r["strategy"] or r["cache_hit"] for r in integration_results)  # Strategy or cache
        )
        
        details["total_queries"] = len(queries)
        details["cache_hits"] = cache_hits
        details["total_tokens"] = total_tokens
        details["cache_hit_rate"] = f"{(cache_hits/len(queries)*100):.1f}%"
        details["results"] = integration_results
        
        self.log_test("Platform Integration", "PASS" if integration_working else "FAIL", details)
        return integration_working
    
    def test_token_allocation(self, platform: TokenomicsPlatform) -> bool:
        """Test 5: Token allocation and budgeting."""
        print("\n" + "="*80)
        print("TEST 5: TOKEN ALLOCATION AND BUDGETING")
        print("="*80)
        
        details = {}
        
        # Test with different budgets
        budgets = [1000, 2000, 4000]
        allocation_results = []
        
        for budget in budgets:
            query = "Explain quantum computing in detail"
            result = platform.query(query, token_budget=budget, use_cache=False)
            plan = result.get("plan")
            
            if plan:
                total_allocated = sum(a.tokens for a in plan.allocations)
                allocation_results.append({
                    "requested_budget": budget,
                    "allocated_tokens": total_allocated,
                    "within_budget": total_allocated <= budget,
                    "allocations_count": len(plan.allocations),
                })
        
        # Validate allocation
        allocation_working = (
            len(allocation_results) == len(budgets) and
            all(r["within_budget"] for r in allocation_results) and
            all(r["allocations_count"] > 0 for r in allocation_results)
        )
        
        details["budgets_tested"] = budgets
        details["all_within_budget"] = all(r["within_budget"] for r in allocation_results)
        details["allocation_results"] = allocation_results
        
        self.log_test("Token Allocation", "PASS" if allocation_working else "FAIL", details)
        return allocation_working
    
    def test_bandit_learning(self, platform: TokenomicsPlatform) -> bool:
        """Test 6: Bandit learning and adaptation."""
        print("\n" + "="*80)
        print("TEST 6: BANDIT LEARNING AND ADAPTATION")
        print("="*80)
        
        details = {}
        
        # Reset bandit for clean test
        platform.bandit.reset()
        initial_stats = platform.bandit.stats()
        
        # Run queries and track strategy performance
        queries = [
            "What is AI?",
            "What is machine learning?",
            "What is deep learning?",
            "What is neural networks?",
        ]
        
        strategy_performance = {}
        
        for query in queries:
            result = platform.query(query, use_cache=False)
            strategy = result.get("strategy")
            reward = result.get("reward", 0)
            
            if strategy:
                if strategy not in strategy_performance:
                    strategy_performance[strategy] = {"count": 0, "total_reward": 0}
                strategy_performance[strategy]["count"] += 1
                strategy_performance[strategy]["total_reward"] += reward
        
        # Check bandit learned
        final_stats = platform.bandit.stats()
        best_strategy = platform.bandit.get_best_strategy()
        
        details["initial_pulls"] = initial_stats["total_pulls"]
        details["final_pulls"] = final_stats["total_pulls"]
        details["strategy_performance"] = strategy_performance
        details["best_strategy"] = best_strategy.arm_id if best_strategy else None
        details["arms_tested"] = len(strategy_performance)
        
        # Validate learning
        learning_working = (
            final_stats["total_pulls"] > initial_stats["total_pulls"] and
            len(strategy_performance) > 0 and
            best_strategy is not None
        )
        
        details["bandit_learned"] = final_stats["total_pulls"] > initial_stats["total_pulls"]
        details["strategies_explored"] = len(strategy_performance) > 0
        
        self.log_test("Bandit Learning", "PASS" if learning_working else "FAIL", details)
        return learning_working
    
    def test_end_to_end(self, platform: TokenomicsPlatform) -> bool:
        """Test 7: End-to-end realistic scenario."""
        print("\n" + "="*80)
        print("TEST 7: END-TO-END REALISTIC SCENARIO")
        print("="*80)
        
        details = {}
        
        # Simulate realistic usage pattern
        scenario_queries = [
            ("User asks about Python", "What is Python?"),
            ("User asks about recursion", "Explain recursion"),
            ("User asks about Python again", "What is Python?"),  # Cache hit
            ("User asks about HTTP", "How does HTTP work?"),
            ("User asks about Python third time", "What is Python?"),  # Cache hit
            ("User asks about machine learning", "Explain machine learning"),
        ]
        
        scenario_results = []
        total_tokens_without_cache = 0
        total_tokens_with_cache = 0
        
        for scenario, query in scenario_queries:
            result = platform.query(query, use_cache=True, use_bandit=True)
            
            # Estimate what tokens would be without cache
            if result["cache_hit"]:
                # Use average from non-cached queries
                avg_tokens = sum(r["tokens_used"] for r in scenario_results if not r["cache_hit"]) / max(1, sum(1 for r in scenario_results if not r["cache_hit"]))
                total_tokens_without_cache += avg_tokens if avg_tokens > 0 else 1000
            else:
                total_tokens_without_cache += result["tokens_used"]
            
            total_tokens_with_cache += result["tokens_used"]
            
            scenario_results.append({
                "scenario": scenario,
                "query": query,
                "tokens": result["tokens_used"],
                "cache_hit": result["cache_hit"],
                "strategy": result.get("strategy"),
                "latency": result["latency_ms"],
            })
        
        tokens_saved = total_tokens_without_cache - total_tokens_with_cache
        savings_rate = (tokens_saved / total_tokens_without_cache * 100) if total_tokens_without_cache > 0 else 0
        
        details["scenarios_tested"] = len(scenario_queries)
        details["cache_hits"] = sum(1 for r in scenario_results if r["cache_hit"])
        details["tokens_without_cache"] = total_tokens_without_cache
        details["tokens_with_cache"] = total_tokens_with_cache
        details["tokens_saved"] = tokens_saved
        details["savings_rate"] = f"{savings_rate:.1f}%"
        details["scenario_results"] = scenario_results
        
        # Validate end-to-end
        e2e_working = (
            len(scenario_results) == len(scenario_queries) and
            tokens_saved > 0 and
            savings_rate > 0
        )
        
        self.log_test("End-to-End Scenario", "PASS" if e2e_working else "FAIL", details)
        return e2e_working
    
    def run_all_tests(self) -> Dict:
        """Run all validation tests."""
        print("\n" + "="*80)
        print("TOKENOMICS PLATFORM - COMPREHENSIVE VALIDATION")
        print("="*80)
        print(f"Started: {datetime.now().isoformat()}")
        
        # Setup
        import os
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).parent / '.env')
        
        if not os.getenv("GEMINI_API_KEY"):
            os.environ["GEMINI_API_KEY"] = "AIzaSyCvSI80PtKuVejnkIiiNxjjN6PyRRngB1E"
        
        config = TokenomicsConfig.from_env()
        config.memory.use_semantic_cache = False  # Disable to avoid TensorFlow issues
        config.memory.cache_size = 100
        
        platform = TokenomicsPlatform(config=config)
        
        # Run all tests
        test_results = {
            "memory_cache": self.test_memory_cache(platform),
            "token_orchestrator": self.test_token_orchestrator(platform),
            "bandit_optimizer": self.test_bandit_optimizer(platform),
            "platform_integration": self.test_integration(platform),
            "token_allocation": self.test_token_allocation(platform),
            "bandit_learning": self.test_bandit_learning(platform),
            "end_to_end": self.test_end_to_end(platform),
        }
        
        # Calculate overall status
        passed = sum(1 for v in test_results.values() if v)
        total = len(test_results)
        overall_status = "PASS" if passed == total else "PARTIAL" if passed > 0 else "FAIL"
        
        self.results["overall_status"] = overall_status
        self.results["validation_end"] = datetime.now().isoformat()
        self.results["tests_passed"] = passed
        self.results["tests_total"] = total
        self.results["test_results"] = test_results
        
        # Print summary
        print("\n" + "="*80)
        print("VALIDATION SUMMARY")
        print("="*80)
        print(f"Overall Status: {overall_status}")
        print(f"Tests Passed: {passed}/{total}")
        print()
        print("Test Results:")
        for test_name, result in test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"  {test_name}: {status}")
        print("="*80)
        
        # Save results
        with open("validation_results.json", "w") as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nDetailed results saved to: validation_results.json")
        
        return self.results


if __name__ == "__main__":
    validator = PlatformValidator()
    results = validator.run_all_tests()

