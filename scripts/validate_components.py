"""Component-level validation without API calls."""

import sys
from pathlib import Path
from datetime import datetime
sys.path.insert(0, str(Path(__file__).parent))

from tokenomics.memory import MemoryCache, SmartMemoryLayer
from tokenomics.orchestrator import TokenAwareOrchestrator, QueryComplexity
from tokenomics.bandit import BanditOptimizer, Strategy
from tokenomics.config import TokenomicsConfig
import json


class ComponentValidator:
    """Validate individual components."""
    
    def __init__(self):
        self.results = {}
        self.test_details = {}
    
    def test_memory_cache_logic(self):
        """Test memory cache logic."""
        print("\n" + "="*80)
        print("TEST: Memory Cache Logic")
        print("="*80)
        
        details = {
            'test_name': 'Memory Cache Logic',
            'tests': []
        }
        
        try:
            cache = MemoryCache(max_size=10, eviction_policy="lru")
            
            # Test exact match
            cache.put("test query", "test response", tokens_used=100)
            entry = cache.get_exact("test query")
            
            assert entry is not None, "Cache should return entry"
            details['tests'].append({'name': 'Exact match retrieval', 'status': 'PASS'})
            
            assert entry.response == "test response", "Response should match"
            details['tests'].append({'name': 'Response matching', 'status': 'PASS'})
            
            assert entry.tokens_used == 100, "Tokens should be tracked"
            details['tests'].append({'name': 'Token tracking', 'status': 'PASS'})
            
            # Test cache miss
            assert cache.get_exact("different query") is None, "Different query should miss"
            details['tests'].append({'name': 'Cache miss handling', 'status': 'PASS'})
            
            # Test eviction
            for i in range(15):
                cache.put(f"query {i}", f"response {i}")
            assert len(cache._exact_cache) == 10, "Cache should respect max_size"
            details['tests'].append({'name': 'LRU eviction', 'status': 'PASS'})
            
            details['status'] = 'PASS'
            details['cache_size'] = len(cache._exact_cache)
            print("[PASS] Memory cache logic")
            return True
            
        except AssertionError as e:
            details['status'] = 'FAIL'
            details['error'] = str(e)
            details['tests'].append({'name': 'Test failed', 'status': 'FAIL', 'error': str(e)})
            print(f"[FAIL] Memory cache logic: {e}")
            return False
        except Exception as e:
            details['status'] = 'ERROR'
            details['error'] = str(e)
            print(f"[ERROR] Memory cache logic: {e}")
            return False
        finally:
            self.test_details['memory_cache'] = details
    
    def test_orchestrator_logic(self):
        """Test orchestrator logic."""
        print("\n" + "="*80)
        print("TEST: Token Orchestrator Logic")
        print("="*80)
        
        details = {
            'test_name': 'Token Orchestrator Logic',
            'tests': []
        }
        
        try:
            orchestrator = TokenAwareOrchestrator(default_token_budget=2000)
            
            # Test complexity analysis
            simple = orchestrator.analyze_complexity("short")
            medium = orchestrator.analyze_complexity("a" * 200)
            # Use a query with complex indicators to ensure it's classified as COMPLEX
            # A very long query with complex keywords should trigger COMPLEX classification
            complex_q = orchestrator.analyze_complexity("Design and implement a comprehensive system architecture for a scalable microservices pipeline that can handle enterprise-level production workloads. Compare different approaches and analyze the trade-offs.")
            
            assert simple == QueryComplexity.SIMPLE, "Should detect simple query"
            details['tests'].append({'name': 'Simple complexity detection', 'status': 'PASS'})
            
            assert medium == QueryComplexity.MEDIUM, "Should detect medium query"
            details['tests'].append({'name': 'Medium complexity detection', 'status': 'PASS'})
            
            assert complex_q == QueryComplexity.COMPLEX, f"Should detect complex query (got {complex_q.value})"
            details['tests'].append({'name': 'Complex complexity detection', 'status': 'PASS'})
            
            # Test token allocation
            components = {
                "system": {"cost": 100, "utility": 1.0},
                "query": {"cost": 50, "utility": 1.0},
                "context": {"cost": 500, "utility": 0.8},
            }
            allocations = orchestrator.allocate_tokens_greedy(components, budget=1000)
            
            assert len(allocations) > 0, "Should allocate tokens"
            details['tests'].append({'name': 'Token allocation', 'status': 'PASS', 'allocations_count': len(allocations)})
            
            total_allocated = sum(a.tokens for a in allocations)
            assert total_allocated <= 1000, "Should respect budget"
            details['tests'].append({'name': 'Budget respect', 'status': 'PASS', 'total_allocated': total_allocated, 'budget': 1000})
            
            # Test query planning
            plan = orchestrator.plan_query("test query", token_budget=2000)
            assert plan.query == "test query", "Plan should contain query"
            assert plan.token_budget == 2000, "Plan should respect budget"
            assert len(plan.allocations) > 0, "Plan should have allocations"
            details['tests'].append({'name': 'Query planning', 'status': 'PASS', 'plan_allocations': len(plan.allocations)})
            
            details['status'] = 'PASS'
            print("[PASS] Token orchestrator logic")
            return True
            
        except AssertionError as e:
            details['status'] = 'FAIL'
            details['error'] = str(e)
            details['tests'].append({'name': 'Test failed', 'status': 'FAIL', 'error': str(e)})
            print(f"[FAIL] Token orchestrator logic: {e}")
            return False
        except Exception as e:
            details['status'] = 'ERROR'
            details['error'] = str(e)
            print(f"[ERROR] Token orchestrator logic: {e}")
            return False
        finally:
            self.test_details['orchestrator'] = details
    
    def test_bandit_logic(self):
        """Test bandit optimizer logic."""
        print("\n" + "="*80)
        print("TEST: Bandit Optimizer Logic")
        print("="*80)
        
        details = {
            'test_name': 'Bandit Optimizer Logic',
            'tests': []
        }
        
        try:
            bandit = BanditOptimizer(algorithm="ucb")
            
            # Add strategies
            strategies = [
                Strategy(arm_id="s1", model="m1", max_tokens=100),
                Strategy(arm_id="s2", model="m2", max_tokens=200),
            ]
            bandit.add_strategies(strategies)
            
            assert len(bandit.arms) == 2, "Should have 2 arms"
            details['tests'].append({'name': 'Strategy addition', 'status': 'PASS', 'arms_count': len(bandit.arms)})
            
            # Test selection
            strategy = bandit.select_strategy()
            assert strategy is not None, "Should select a strategy"
            assert strategy.arm_id in ["s1", "s2"], "Should select valid strategy"
            details['tests'].append({'name': 'Strategy selection', 'status': 'PASS', 'selected_arm': strategy.arm_id})
            
            # Test update
            reward = bandit.compute_reward(quality_score=0.9, tokens_used=100)
            bandit.update(strategy.arm_id, reward)
            
            stats = bandit.stats()
            assert stats["total_pulls"] > 0, "Should track pulls"
            assert strategy.arm_id in stats["arms"], "Should track arm stats"
            details['tests'].append({'name': 'Reward update', 'status': 'PASS', 'total_pulls': stats["total_pulls"]})
            
            # Test learning
            initial_pulls = stats["total_pulls"]
            for _ in range(5):
                s = bandit.select_strategy()
                r = bandit.compute_reward(0.8, 150)
                bandit.update(s.arm_id, r)
            
            final_stats = bandit.stats()
            assert final_stats["total_pulls"] > initial_pulls, "Should learn from pulls"
            details['tests'].append({'name': 'Learning mechanism', 'status': 'PASS', 'initial_pulls': initial_pulls, 'final_pulls': final_stats["total_pulls"]})
            
            best = bandit.get_best_strategy()
            assert best is not None, "Should identify best strategy"
            details['tests'].append({'name': 'Best strategy identification', 'status': 'PASS', 'best_arm': best.arm_id if best else None})
            
            details['status'] = 'PASS'
            print("[PASS] Bandit optimizer logic")
            return True
            
        except AssertionError as e:
            details['status'] = 'FAIL'
            details['error'] = str(e)
            details['tests'].append({'name': 'Test failed', 'status': 'FAIL', 'error': str(e)})
            print(f"[FAIL] Bandit optimizer logic: {e}")
            return False
        except Exception as e:
            details['status'] = 'ERROR'
            details['error'] = str(e)
            print(f"[ERROR] Bandit optimizer logic: {e}")
            return False
        finally:
            self.test_details['bandit'] = details
    
    def test_integration_logic(self):
        """Test component integration."""
        print("\n" + "="*80)
        print("TEST: Component Integration Logic")
        print("="*80)
        
        config = TokenomicsConfig()
        config.memory.use_semantic_cache = False
        
        # Test that components can be initialized together
        from tokenomics.memory import SmartMemoryLayer
        from tokenomics.orchestrator import TokenAwareOrchestrator
        from tokenomics.bandit import BanditOptimizer
        
        memory = SmartMemoryLayer(
            use_exact_cache=True,
            use_semantic_cache=False,
            cache_size=100,
        )
        
        orchestrator = TokenAwareOrchestrator(default_token_budget=2000)
        
        bandit = BanditOptimizer(algorithm="ucb")
        strategies = [
            Strategy(arm_id="test", model="test-model", max_tokens=1000),
        ]
        bandit.add_strategies(strategies)
        
        # Test integration: orchestrator + bandit
        query = "test query"
        plan = orchestrator.plan_query(query)
        strategy = bandit.select_strategy()
        
        assert plan is not None, "Orchestrator should create plan"
        assert strategy is not None, "Bandit should select strategy"
        
        # Test integration: memory + orchestrator
        memory.store("test query", "test response", tokens_used=100)
        exact, semantic = memory.retrieve("test query")
        
        assert exact is not None, "Memory should retrieve cached entry"
        assert exact.response == "test response", "Memory should return correct response"
        
        # If we had retrieved context, orchestrator could use it
        if exact:
            plan_with_context = orchestrator.plan_query(
                query,
                retrieved_context=[exact.response]
            )
            assert plan_with_context.use_retrieval, "Plan should use retrieved context"
        
        print("[PASS] Component integration logic")
        return True
    
    def test_token_counting(self):
        """Test token counting."""
        print("\n" + "="*80)
        print("TEST: Token Counting")
        print("="*80)
        
        orchestrator = TokenAwareOrchestrator()
        
        # Test token counting
        text1 = "Hello world"
        text2 = "This is a longer text that should have more tokens"
        
        count1 = orchestrator.count_tokens(text1)
        count2 = orchestrator.count_tokens(text2)
        
        assert count1 > 0, "Should count tokens"
        assert count2 > count1, "Longer text should have more tokens"
        
        # Test compression
        long_text = "a" * 1000
        compressed = orchestrator.compress_text(long_text, target_tokens=50)
        compressed_tokens = orchestrator.count_tokens(compressed)
        
        assert compressed_tokens <= 50, "Compressed text should respect target"
        
        print("[PASS] Token counting")
        return True
    
    def test_bandit_algorithms(self):
        """Test different bandit algorithms."""
        print("\n" + "="*80)
        print("TEST: Bandit Algorithms")
        print("="*80)
        
        strategies = [
            Strategy(arm_id="s1", model="m1", max_tokens=100),
            Strategy(arm_id="s2", model="m2", max_tokens=200),
        ]
        
        # Test UCB
        bandit_ucb = BanditOptimizer(algorithm="ucb")
        bandit_ucb.add_strategies(strategies)
        strategy_ucb = bandit_ucb.select_strategy()
        assert strategy_ucb is not None, "UCB should select strategy"
        
        # Test epsilon-greedy
        bandit_eg = BanditOptimizer(algorithm="epsilon_greedy", exploration_rate=0.2)
        bandit_eg.add_strategies(strategies)
        strategy_eg = bandit_eg.select_strategy()
        assert strategy_eg is not None, "Epsilon-greedy should select strategy"
        
        print("[PASS] Bandit algorithms")
        return True
    
    def save_results(self, output_file: Path):
        """Save validation results to JSON file."""
        output_data = {
            'validation_timestamp': datetime.now().isoformat(),
            'test_results': self.results,
            'test_details': self.test_details,
            'summary': {
                'total_tests': len(self.results),
                'passed': sum(1 for v in self.results.values() if v),
                'failed': sum(1 for v in self.results.values() if not v),
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        print(f"\n✓ Component validation results saved to: {output_file}")
    
    def run_all_tests(self):
        """Run all component tests."""
        print("\n" + "="*80)
        print("COMPONENT VALIDATION - LOGIC TESTS")
        print("="*80)
        
        tests = {
            "memory_cache": self.test_memory_cache_logic,
            "orchestrator": self.test_orchestrator_logic,
            "bandit": self.test_bandit_logic,
            "integration": self.test_integration_logic,
            "token_counting": self.test_token_counting,
            "bandit_algorithms": self.test_bandit_algorithms,
        }
        
        results = {}
        for name, test_func in tests.items():
            try:
                results[name] = test_func()
            except Exception as e:
                print(f"❌ {name}: FAIL - {e}")
                results[name] = False
                if name not in self.test_details:
                    self.test_details[name] = {
                        'test_name': name,
                        'status': 'ERROR',
                        'error': str(e),
                        'tests': []
                    }
        
        self.results = results
        
        # Summary
        passed = sum(1 for v in results.values() if v)
        total = len(results)
        
        print("\n" + "="*80)
        print("VALIDATION SUMMARY")
        print("="*80)
        print(f"Tests Passed: {passed}/{total}")
        print()
        for name, result in results.items():
            status = "[PASS]" if result else "[FAIL]"
            print(f"  {name}: {status}")
        print("="*80)
        
        # Save results to file
        output_dir = Path(__file__).parent.parent / "training_data"
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / "component_validation_results.json"
        self.save_results(output_file)
        
        return results


if __name__ == "__main__":
    validator = ComponentValidator()
    results = validator.run_all_tests()

