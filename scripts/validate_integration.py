"""Integration validation using existing data and mock responses."""

import sys
import json
from pathlib import Path
from typing import Dict
sys.path.insert(0, str(Path(__file__).parent))

from tokenomics.memory import SmartMemoryLayer
from tokenomics.orchestrator import TokenAwareOrchestrator
from tokenomics.bandit import BanditOptimizer, Strategy
from tokenomics.config import TokenomicsConfig


class IntegrationValidator:
    """Validate platform integration."""
    
    def __init__(self):
        self.results = {}
    
    def test_full_workflow(self):
        """Test complete workflow with mock data."""
        print("\n" + "="*80)
        print("TEST: Full Platform Workflow")
        print("="*80)
        
        # Initialize components
        config = TokenomicsConfig()
        config.memory.use_semantic_cache = False
        
        memory = SmartMemoryLayer(
            use_exact_cache=True,
            use_semantic_cache=False,
            cache_size=100,
        )
        
        orchestrator = TokenAwareOrchestrator(default_token_budget=2000)
        
        bandit = BanditOptimizer(algorithm="ucb")
        strategies = [
            Strategy(arm_id="fast", model="model1", max_tokens=500, temperature=0.7),
            Strategy(arm_id="balanced", model="model2", max_tokens=1000, temperature=0.7),
            Strategy(arm_id="powerful", model="model3", max_tokens=2000, temperature=0.8),
        ]
        bandit.add_strategies(strategies)
        
        # Simulate workflow
        queries = [
            "What is Python?",
            "Explain recursion",
            "What is Python?",  # Should hit cache
        ]
        
        workflow_results = []
        
        for query in queries:
            # Step 1: Check cache
            exact_match, semantic_matches = memory.retrieve(query)
            
            if exact_match:
                # Cache hit - return immediately
                workflow_results.append({
                    "query": query,
                    "cache_hit": True,
                    "tokens": 0,
                    "latency": 0,
                    "response": exact_match.response,
                })
            else:
                # Cache miss - use orchestrator and bandit
                # Step 2: Select strategy
                strategy = bandit.select_strategy()
                
                # Step 3: Create plan
                plan = orchestrator.plan_query(query, token_budget=2000)
                
                # Step 4: Simulate LLM call (mock)
                mock_tokens = plan.token_budget // 2
                mock_latency = 3000
                mock_response = f"Mock response for: {query}"
                
                # Step 5: Store in cache
                memory.store(query, mock_response, tokens_used=mock_tokens)
                
                # Step 6: Update bandit
                reward = bandit.compute_reward(quality_score=0.9, tokens_used=mock_tokens)
                bandit.update(strategy.arm_id, reward)
                
                workflow_results.append({
                    "query": query,
                    "cache_hit": False,
                    "tokens": mock_tokens,
                    "latency": mock_latency,
                    "strategy": strategy.arm_id,
                    "plan_budget": plan.token_budget,
                    "allocations": len(plan.allocations),
                })
        
        # Validate workflow
        cache_hits = sum(1 for r in workflow_results if r["cache_hit"])
        total_tokens = sum(r["tokens"] for r in workflow_results)
        
        workflow_working = (
            cache_hits == 1 and  # One cache hit expected
            total_tokens > 0 and  # Some tokens used
            all("strategy" in r or r["cache_hit"] for r in workflow_results)  # Strategy or cache
        )
        
        print(f"  Queries processed: {len(queries)}")
        print(f"  Cache hits: {cache_hits}")
        print(f"  Total tokens: {total_tokens}")
        print(f"  Bandit pulls: {bandit.stats()['total_pulls']}")
        
        status = "[PASS]" if workflow_working else "[FAIL]"
        print(f"\n{status}")
        return workflow_working
    
    def test_token_allocation_scenarios(self):
        """Test token allocation in various scenarios."""
        print("\n" + "="*80)
        print("TEST: Token Allocation Scenarios")
        print("="*80)
        
        orchestrator = TokenAwareOrchestrator(default_token_budget=2000)
        
        scenarios = [
            {
                "name": "Simple query, no context",
                "query": "What is AI?",
                "context": None,
                "budget": 1000,
            },
            {
                "name": "Complex query, with context",
                "query": "Explain quantum computing",
                "context": ["Quantum computing uses qubits", "Qubits can be in superposition"],
                "budget": 3000,
            },
            {
                "name": "Medium query, limited budget",
                "query": "How does HTTP work?",
                "context": None,
                "budget": 500,
            },
        ]
        
        allocation_results = []
        
        for scenario in scenarios:
            plan = orchestrator.plan_query(
                scenario["query"],
                token_budget=scenario["budget"],
                retrieved_context=scenario["context"],
            )
            
            total_allocated = sum(a.tokens for a in plan.allocations)
            
            allocation_results.append({
                "scenario": scenario["name"],
                "requested_budget": scenario["budget"],
                "allocated": total_allocated,
                "within_budget": total_allocated <= scenario["budget"],
                "has_allocations": len(plan.allocations) > 0,
            })
        
        # Validate
        all_valid = all(r["within_budget"] for r in allocation_results)
        all_have_allocations = all(r["has_allocations"] for r in allocation_results)
        
        print(f"  Scenarios tested: {len(scenarios)}")
        print(f"  All within budget: {all_valid}")
        print(f"  All have allocations: {all_have_allocations}")
        
        for r in allocation_results:
            print(f"    {r['scenario']}: {r['allocated']}/{r['requested_budget']} tokens")
        
        status = "[PASS]" if (all_valid and all_have_allocations) else "[FAIL]"
        print(f"\n{status}")
        return all_valid and all_have_allocations
    
    def test_bandit_learning_scenarios(self):
        """Test bandit learning in various scenarios."""
        print("\n" + "="*80)
        print("TEST: Bandit Learning Scenarios")
        print("="*80)
        
        bandit = BanditOptimizer(algorithm="ucb")
        
        strategies = [
            Strategy(arm_id="cheap", model="m1", max_tokens=300, temperature=0.5),
            Strategy(arm_id="standard", model="m2", max_tokens=1000, temperature=0.7),
            Strategy(arm_id="premium", model="m3", max_tokens=2000, temperature=0.9),
        ]
        bandit.add_strategies(strategies)
        
        # Simulate learning scenarios
        scenarios = [
            {"quality": 0.9, "tokens": 300, "expected": "cheap"},  # Good quality, low tokens
            {"quality": 0.8, "tokens": 1000, "expected": "standard"},  # Decent
            {"quality": 0.7, "tokens": 2000, "expected": "premium"},  # Lower quality, high tokens
        ]
        
        for scenario in scenarios:
            strategy = bandit.select_strategy()
            reward = bandit.compute_reward(
                quality_score=scenario["quality"],
                tokens_used=scenario["tokens"]
            )
            bandit.update(strategy.arm_id, reward)
        
        # Check learning
        stats = bandit.stats()
        best_strategy = bandit.get_best_strategy()
        
        print(f"  Total pulls: {stats['total_pulls']}")
        print(f"  Best strategy: {best_strategy.arm_id if best_strategy else 'None'}")
        print(f"  Arms tested: {len(stats['arms'])}")
        
        for arm_id, arm_stats in stats["arms"].items():
            print(f"    {arm_id}: {arm_stats['pulls']} pulls, avg reward: {arm_stats['average_reward']:.3f}")
        
        learning_working = (
            stats["total_pulls"] == len(scenarios) and
            best_strategy is not None and
            len(stats["arms"]) == len(strategies)
        )
        
        status = "[PASS]" if learning_working else "[FAIL]"
        print(f"\n{status}")
        return learning_working
    
    def test_cache_orchestrator_integration(self):
        """Test cache and orchestrator working together."""
        print("\n" + "="*80)
        print("TEST: Cache-Orchestrator Integration")
        print("="*80)
        
        memory = SmartMemoryLayer(use_exact_cache=True, use_semantic_cache=False)
        orchestrator = TokenAwareOrchestrator(default_token_budget=2000)
        
        # Store some responses
        memory.store("query1", "response1", tokens_used=500)
        memory.store("query2", "response2", tokens_used=800)
        
        # Test retrieval and planning
        exact, semantic = memory.retrieve("query1")
        
        if exact:
            # Use retrieved context in plan
            plan = orchestrator.plan_query(
                "new query",
                retrieved_context=[exact.response]
            )
            
            assert plan.use_retrieval, "Plan should use retrieved context"
            assert len(plan.retrieved_context) > 0, "Plan should have context"
            
            print("  Cache retrieval: [OK]")
            print("  Context integration: [OK]")
            print("  Plan creation: [OK]")
            print("\n[PASS]")
            return True
        
        print("\n[FAIL]")
        return False
    
    def run_all_tests(self):
        """Run all integration tests."""
        print("\n" + "="*80)
        print("INTEGRATION VALIDATION")
        print("="*80)
        
        tests = {
            "full_workflow": self.test_full_workflow,
            "token_allocation": self.test_token_allocation_scenarios,
            "bandit_learning": self.test_bandit_learning_scenarios,
            "cache_orchestrator": self.test_cache_orchestrator_integration,
        }
        
        results = {}
        for name, test_func in tests.items():
            try:
                results[name] = test_func()
            except Exception as e:
                print(f"❌ {name}: FAIL - {e}")
                import traceback
                traceback.print_exc()
                results[name] = False
        
        # Summary
        passed = sum(1 for v in results.values() if v)
        total = len(results)
        
        print("\n" + "="*80)
        print("INTEGRATION VALIDATION SUMMARY")
        print("="*80)
        print(f"Tests Passed: {passed}/{total}")
        print()
        for name, result in results.items():
            status = "[PASS]" if result else "[FAIL]"
            print(f"  {name}: {status}")
        print("="*80)
        
        return results


if __name__ == "__main__":
    validator = IntegrationValidator()
    results = validator.run_all_tests()

