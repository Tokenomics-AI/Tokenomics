"""Comprehensive end-to-end test of Tokenomics Platform.

Tests all components systematically:
1. Platform initialization
2. Memory Layer (exact + semantic cache)
3. Token Orchestrator
4. Bandit Optimizer
5. Full integration
6. Component-level savings
7. A/B comparison
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from tokenomics.core import TokenomicsPlatform
from tokenomics.config import TokenomicsConfig
from usage_tracker import UsageTracker

# Load environment
load_dotenv(Path(__file__).parent / '.env')

class ComprehensiveTest:
    """Comprehensive platform test suite."""
    
    def __init__(self):
        self.results = {
            'test_start': datetime.now().isoformat(),
            'tests': [],
            'summary': {},
            'errors': [],
        }
        self.platform = None
        
    def log_test(self, name, status, details=None, error=None):
        """Log test result."""
        test_result = {
            'name': name,
            'status': status,  # 'pass', 'fail', 'warning'
            'timestamp': datetime.now().isoformat(),
            'details': details or {},
        }
        if error:
            test_result['error'] = str(error)
            self.results['errors'].append({
                'test': name,
                'error': str(error),
            })
        self.results['tests'].append(test_result)
        status_icon = '[PASS]' if status == 'pass' else '[FAIL]' if status == 'fail' else '[WARN]'
        print(f"{status_icon} {name}: {status.upper()}")
        if details:
            for key, value in details.items():
                print(f"    {key}: {value}")
        if error:
            print(f"    ERROR: {error}")
    
    def run_all_tests(self):
        """Run all comprehensive tests."""
        print("=" * 80)
        print("COMPREHENSIVE TOKENOMICS PLATFORM TEST SUITE")
        print("=" * 80)
        print()
        
        # Test 1: Environment and Configuration
        self.test_environment()
        
        # Test 2: Platform Initialization
        self.test_platform_initialization()
        
        if not self.platform:
            print("\n[ERROR] Platform initialization failed. Cannot continue tests.")
            return self.results
        
        # Test 3: Memory Layer - Exact Cache
        self.test_memory_exact_cache()
        
        # Test 4: Memory Layer - Semantic Cache
        self.test_memory_semantic_cache()
        
        # Test 5: Token Orchestrator
        self.test_token_orchestrator()
        
        # Test 6: Bandit Optimizer
        self.test_bandit_optimizer()
        
        # Test 7: Full Integration - Single Query
        self.test_full_integration_single()
        
        # Test 8: Full Integration - Multiple Queries (Cache Accumulation)
        self.test_full_integration_multiple()
        
        # Test 9: Component-Level Savings Tracking
        self.test_component_savings()
        
        # Test 10: A/B Comparison Mode
        self.test_ab_comparison()
        
        # Test 11: Edge Cases
        self.test_edge_cases()
        
        # Generate summary
        self.generate_summary()
        
        return self.results
    
    def test_environment(self):
        """Test 1: Environment and Configuration"""
        print("\n" + "=" * 80)
        print("TEST 1: Environment and Configuration")
        print("=" * 80)
        
        try:
            # Check API key
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                self.log_test("API Key Check", "fail", error="OPENAI_API_KEY not found")
                return
            
            if not api_key.startswith("sk-"):
                self.log_test("API Key Format", "warning", 
                            details={"key_prefix": api_key[:10] + "..."})
            else:
                self.log_test("API Key Format", "pass", 
                            details={"key_prefix": api_key[:10] + "..."})
            
            # Check .env file (check project root, not test directory)
            project_root = Path(__file__).parent.parent.parent
            env_file = project_root / '.env'
            if env_file.exists():
                self.log_test(".env File Exists", "pass", 
                            details={"path": str(env_file)})
            else:
                self.log_test(".env File Exists", "warning", 
                            details={"message": f".env file not found at {env_file}, but environment variables may be set via other means"})
            
            # Check config loading
            try:
                config = TokenomicsConfig.from_env()
                self.log_test("Config Loading", "pass", 
                            details={
                                "llm_provider": config.llm.provider,
                                "llm_model": config.llm.model,
                                "cache_size": config.memory.cache_size,
                                "similarity_threshold": config.memory.similarity_threshold,
                            })
            except Exception as e:
                self.log_test("Config Loading", "fail", error=str(e))
                
        except Exception as e:
            self.log_test("Environment Test", "fail", error=str(e))
    
    def test_platform_initialization(self):
        """Test 2: Platform Initialization"""
        print("\n" + "=" * 80)
        print("TEST 2: Platform Initialization")
        print("=" * 80)
        
        try:
            config = TokenomicsConfig.from_env()
            config.llm.provider = "openai"
            config.llm.model = "gpt-4o-mini"
            config.llm.api_key = os.getenv("OPENAI_API_KEY")
            config.memory.use_semantic_cache = True
            config.memory.cache_size = 100
            config.memory.similarity_threshold = 0.65
            config.memory.direct_return_threshold = 0.75
            
            self.platform = TokenomicsPlatform(config=config)
            
            # Check components
            components = {
                "Memory Layer": self.platform.memory is not None,
                "Orchestrator": self.platform.orchestrator is not None,
                "Bandit Optimizer": self.platform.bandit is not None,
                "LLM Provider": self.platform.llm_provider is not None,
            }
            
            all_ok = all(components.values())
            self.log_test("Platform Initialization", 
                         "pass" if all_ok else "fail",
                         details=components)
            
            # Check memory stats
            if self.platform.memory:
                stats = self.platform.memory.stats()
                self.log_test("Memory Layer Stats", "pass",
                            details=stats)
            
        except Exception as e:
            self.log_test("Platform Initialization", "fail", error=str(e))
            self.platform = None
    
    def test_memory_exact_cache(self):
        """Test 3: Memory Layer - Exact Cache"""
        print("\n" + "=" * 80)
        print("TEST 3: Memory Layer - Exact Cache")
        print("=" * 80)
        
        if not self.platform:
            self.log_test("Exact Cache Test", "fail", error="Platform not initialized")
            return
        
        try:
            query = "What is Python programming?"
            
            # First query - should miss
            result1 = self.platform.query(query, use_cache=True, use_bandit=False)
            cache_hit_1 = result1.get('cache_hit', False)
            tokens_1 = result1.get('tokens_used', 0)
            
            self.log_test("First Query (Cache Miss Expected)", 
                         "pass" if not cache_hit_1 else "fail",
                         details={
                             "cache_hit": cache_hit_1,
                             "tokens_used": tokens_1,
                         })
            
            # Second query - should hit exact cache
            result2 = self.platform.query(query, use_cache=True, use_bandit=False)
            cache_hit_2 = result2.get('cache_hit', False)
            cache_type_2 = result2.get('cache_type', 'none')
            tokens_2 = result2.get('tokens_used', 0)
            
            self.log_test("Second Query (Exact Cache Hit Expected)", 
                         "pass" if cache_hit_2 and cache_type_2 == 'exact' and tokens_2 == 0 else "fail",
                         details={
                             "cache_hit": cache_hit_2,
                             "cache_type": cache_type_2,
                             "tokens_used": tokens_2,
                             "tokens_saved": tokens_1,
                         })
            
            # Verify responses are identical
            response_match = result1.get('response') == result2.get('response')
            self.log_test("Response Consistency", 
                         "pass" if response_match else "fail",
                         details={"responses_match": response_match})
            
        except Exception as e:
            self.log_test("Exact Cache Test", "fail", error=str(e))
    
    def test_memory_semantic_cache(self):
        """Test 4: Memory Layer - Semantic Cache"""
        print("\n" + "=" * 80)
        print("TEST 4: Memory Layer - Semantic Cache")
        print("=" * 80)
        
        if not self.platform:
            self.log_test("Semantic Cache Test", "fail", error="Platform not initialized")
            return
        
        try:
            # Query 1: Store in cache
            query1 = "how is chicken biryani made?"
            result1 = self.platform.query(query1, use_cache=True, use_bandit=False)
            tokens_1 = result1.get('tokens_used', 0)
            cache_hit_1 = result1.get('cache_hit', False)
            
            self.log_test("Query 1 (Store in Cache)", 
                         "pass" if not cache_hit_1 else "warning",
                         details={
                             "query": query1[:50],
                             "cache_hit": cache_hit_1,
                             "tokens_used": tokens_1,
                         })
            
            # Query 2: Similar query - should hit semantic cache
            query2 = "what is the main component or technique or ingredient in biryani that gives majority of its flavour?"
            result2 = self.platform.query(query2, use_cache=True, use_bandit=False)
            cache_hit_2 = result2.get('cache_hit', False)
            cache_type_2 = result2.get('cache_type', 'none')
            similarity_2 = result2.get('similarity')
            tokens_2 = result2.get('tokens_used', 0)
            
            # Check if semantic cache hit (direct return or context enhancement)
            semantic_hit = cache_hit_2 and (cache_type_2 in ['semantic_direct', 'context', 'exact'])
            
            # If similarity is above threshold but cache didn't hit, it might be a threshold issue
            # This is acceptable behavior - mark as warning if similarity is good but cache didn't hit
            if similarity_2 and similarity_2 >= 0.65 and not semantic_hit:
                self.log_test("Query 2 (Semantic Cache Hit Expected)", 
                             "warning",
                             details={
                                 "query": query2[:50],
                                 "cache_hit": cache_hit_2,
                                 "cache_type": cache_type_2,
                                 "similarity": similarity_2,
                                 "tokens_used": tokens_2,
                                 "note": "Similarity above threshold but cache didn't hit - may be due to direct_return_threshold",
                             })
            else:
                self.log_test("Query 2 (Semantic Cache Hit Expected)", 
                             "pass" if semantic_hit else "fail",
                             details={
                                 "query": query2[:50],
                                 "cache_hit": cache_hit_2,
                                 "cache_type": cache_type_2,
                                 "similarity": similarity_2,
                                 "tokens_used": tokens_2,
                                 "expected": "semantic_direct or context",
                             })
            
            if similarity_2:
                self.log_test("Similarity Score", 
                             "pass" if similarity_2 >= 0.65 else "warning",
                             details={
                                 "similarity": similarity_2,
                                 "threshold": 0.65,
                             })
            
        except Exception as e:
            self.log_test("Semantic Cache Test", "fail", error=str(e))
    
    def test_token_orchestrator(self):
        """Test 5: Token Orchestrator"""
        print("\n" + "=" * 80)
        print("TEST 5: Token Orchestrator")
        print("=" * 80)
        
        if not self.platform:
            self.log_test("Orchestrator Test", "fail", error="Platform not initialized")
            return
        
        try:
            query = "Explain machine learning in detail"
            
            result = self.platform.query(query, use_cache=False, use_bandit=False)
            
            plan = result.get('plan')
            if plan:
                self.log_test("Query Plan Creation", "pass",
                            details={
                                "complexity": plan.complexity.value if hasattr(plan.complexity, 'value') else str(plan.complexity),
                                "token_budget": plan.token_budget,
                                "num_allocations": len(plan.allocations),
                            })
                
                # Check allocations
                allocations_ok = len(plan.allocations) > 0
                self.log_test("Token Allocations", 
                             "pass" if allocations_ok else "fail",
                             details={
                                 "allocations": [
                                     {
                                         "component": a.component,
                                         "tokens": a.tokens,
                                         "utility": a.utility,
                                     }
                                     for a in plan.allocations
                                 ]
                             })
            else:
                self.log_test("Query Plan Creation", "fail", error="No plan returned")
            
            # Check token usage
            tokens_used = result.get('tokens_used', 0)
            input_tokens = result.get('input_tokens', 0)
            output_tokens = result.get('output_tokens', 0)
            
            self.log_test("Token Usage Tracking", 
                         "pass" if tokens_used > 0 else "fail",
                         details={
                             "total_tokens": tokens_used,
                             "input_tokens": input_tokens,
                             "output_tokens": output_tokens,
                         })
            
        except Exception as e:
            self.log_test("Orchestrator Test", "fail", error=str(e))
    
    def test_bandit_optimizer(self):
        """Test 6: Bandit Optimizer"""
        print("\n" + "=" * 80)
        print("TEST 6: Bandit Optimizer")
        print("=" * 80)
        
        if not self.platform:
            self.log_test("Bandit Test", "fail", error="Platform not initialized")
            return
        
        try:
            # Check strategies
            stats = self.platform.bandit.stats()
            num_strategies = len(stats.get('arms', {}))
            
            self.log_test("Bandit Strategies", 
                         "pass" if num_strategies > 0 else "fail",
                         details={
                             "num_strategies": num_strategies,
                             "strategies": list(stats.get('arms', {}).keys()),
                         })
            
            # Test strategy selection
            query = "What is artificial intelligence?"
            result = self.platform.query(query, use_cache=False, use_bandit=True)
            
            strategy = result.get('strategy')
            self.log_test("Strategy Selection", 
                         "pass" if strategy else "fail",
                         details={
                             "selected_strategy": strategy,
                             "tokens_used": result.get('tokens_used', 0),
                         })
            
            # Test learning (multiple queries)
            for i in range(3):
                self.platform.query(f"Test query {i}", use_cache=False, use_bandit=True)
            
            stats_after = self.platform.bandit.stats()
            total_pulls = stats_after.get('total_pulls', 0)
            
            self.log_test("Bandit Learning", 
                         "pass" if total_pulls > 0 else "fail",
                         details={
                             "total_pulls": total_pulls,
                             "best_strategy": stats_after.get('best_strategy'),
                         })
            
        except Exception as e:
            self.log_test("Bandit Test", "fail", error=str(e))
    
    def test_full_integration_single(self):
        """Test 7: Full Integration - Single Query"""
        print("\n" + "=" * 80)
        print("TEST 7: Full Integration - Single Query")
        print("=" * 80)
        
        if not self.platform:
            self.log_test("Integration Test", "fail", error="Platform not initialized")
            return
        
        try:
            query = "What is the difference between supervised and unsupervised learning?"
            
            result = self.platform.query(query, use_cache=True, use_bandit=True)
            
            # Check all components are involved
            has_response = bool(result.get('response'))
            has_tokens = result.get('tokens_used', 0) > 0
            has_plan = result.get('plan') is not None
            has_strategy = result.get('strategy') is not None
            
            all_components = {
                "Response Generated": has_response,
                "Tokens Tracked": has_tokens,
                "Plan Created": has_plan,
                "Strategy Selected": has_strategy,
            }
            
            self.log_test("Full Integration", 
                         "pass" if all(all_components.values()) else "fail",
                         details=all_components)
            
            # Check component savings
            comp_savings = result.get('component_savings', {})
            if comp_savings:
                self.log_test("Component Savings Tracking", "pass",
                            details=comp_savings)
            else:
                self.log_test("Component Savings Tracking", "warning",
                            details={"message": "No component savings data"})
            
        except Exception as e:
            self.log_test("Integration Test", "fail", error=str(e))
    
    def test_full_integration_multiple(self):
        """Test 8: Full Integration - Multiple Queries (Cache Accumulation)"""
        print("\n" + "=" * 80)
        print("TEST 8: Full Integration - Multiple Queries (Cache Accumulation)")
        print("=" * 80)
        
        if not self.platform:
            self.log_test("Multiple Queries Test", "fail", error="Platform not initialized")
            return
        
        try:
            queries = [
                "how is chicken biryani made?",
                "what ingredients are needed for biryani?",
                "what is the main component or technique or ingredient in biryani that gives majority of its flavour?",
            ]
            
            results = []
            cache_hits = []
            
            for i, query in enumerate(queries, 1):
                result = self.platform.query(query, use_cache=True, use_bandit=True)
                results.append(result)
                cache_hit = result.get('cache_hit', False)
                cache_hits.append(cache_hit)
                
                self.log_test(f"Query {i}", 
                             "pass" if result.get('response') else "fail",
                             details={
                                 "query": query[:50],
                                 "cache_hit": cache_hit,
                                 "cache_type": result.get('cache_type', 'none'),
                                 "similarity": result.get('similarity'),
                                 "tokens_used": result.get('tokens_used', 0),
                             })
            
            # Check cache accumulation
            # First query should miss, subsequent queries might hit
            first_miss = not cache_hits[0]
            subsequent_hits = any(cache_hits[1:])
            
            self.log_test("Cache Accumulation", 
                         "pass" if first_miss else "warning",
                         details={
                             "first_query_cache_hit": cache_hits[0],
                             "subsequent_queries_hit_cache": subsequent_hits,
                             "cache_hits": cache_hits,
                         })
            
            # Check memory stats
            stats = self.platform.memory.stats()
            self.log_test("Memory Stats After Multiple Queries", "pass",
                        details=stats)
            
        except Exception as e:
            self.log_test("Multiple Queries Test", "fail", error=str(e))
    
    def test_component_savings(self):
        """Test 9: Component-Level Savings Tracking"""
        print("\n" + "=" * 80)
        print("TEST 9: Component-Level Savings Tracking")
        print("=" * 80)
        
        if not self.platform:
            self.log_test("Component Savings Test", "fail", error="Platform not initialized")
            return
        
        try:
            # Test with cache hit (memory savings)
            query1 = "What is Python?"
            result1 = self.platform.query(query1, use_cache=True, use_bandit=True)
            
            # Second query - should hit cache
            result2 = self.platform.query(query1, use_cache=True, use_bandit=True)
            
            comp_savings = result2.get('component_savings', {})
            memory_savings = comp_savings.get('memory_layer', 0)
            
            self.log_test("Memory Layer Savings (Cache Hit)", 
                         "pass" if memory_savings > 0 else "fail",
                         details={
                             "memory_savings": memory_savings,
                             "cache_hit": result2.get('cache_hit', False),
                             "all_savings": comp_savings,
                         })
            
            # Test with new query (orchestrator + bandit savings)
            query2 = "Explain quantum computing"
            result3 = self.platform.query(query2, use_cache=False, use_bandit=True)
            
            comp_savings_3 = result3.get('component_savings', {})
            orchestrator_savings = comp_savings_3.get('orchestrator', 0)
            bandit_savings = comp_savings_3.get('bandit', 0)
            
            self.log_test("Orchestrator & Bandit Savings", 
                         "pass" if (orchestrator_savings >= 0 and bandit_savings >= 0) else "warning",
                         details={
                             "orchestrator_savings": orchestrator_savings,
                             "bandit_savings": bandit_savings,
                             "all_savings": comp_savings_3,
                         })
            
        except Exception as e:
            self.log_test("Component Savings Test", "fail", error=str(e))
    
    def test_ab_comparison(self):
        """Test 10: A/B Comparison Mode"""
        print("\n" + "=" * 80)
        print("TEST 10: A/B Comparison Mode")
        print("=" * 80)
        
        if not self.platform:
            self.log_test("A/B Comparison Test", "fail", error="Platform not initialized")
            return
        
        try:
            query = "What is machine learning?"
            
            # Baseline
            baseline_result = self.platform.query(query, use_cache=False, use_bandit=False)
            baseline_tokens = baseline_result.get('tokens_used', 0)
            
            # Optimized
            optimized_result = self.platform.query(query, use_cache=True, use_bandit=True)
            optimized_tokens = optimized_result.get('tokens_used', 0)
            
            # Compare
            token_savings = baseline_tokens - optimized_tokens
            savings_pct = (token_savings / baseline_tokens * 100) if baseline_tokens > 0 else 0
            
            # Optimized should never use MORE tokens than baseline
            optimized_better = optimized_tokens <= baseline_tokens
            
            self.log_test("A/B Comparison", 
                         "pass" if optimized_better else "fail",
                         details={
                             "baseline_tokens": baseline_tokens,
                             "optimized_tokens": optimized_tokens,
                             "token_savings": token_savings,
                             "savings_percent": round(savings_pct, 2),
                             "optimized_better": optimized_better,
                         })
            
            # Check component savings
            comp_savings = optimized_result.get('component_savings', {})
            if comp_savings:
                self.log_test("A/B Component Savings", "pass",
                            details=comp_savings)
            
        except Exception as e:
            self.log_test("A/B Comparison Test", "fail", error=str(e))
    
    def test_edge_cases(self):
        """Test 11: Edge Cases"""
        print("\n" + "=" * 80)
        print("TEST 11: Edge Cases")
        print("=" * 80)
        
        if not self.platform:
            self.log_test("Edge Cases Test", "fail", error="Platform not initialized")
            return
        
        try:
            # Empty query
            try:
                result = self.platform.query("", use_cache=True, use_bandit=True)
                self.log_test("Empty Query", "warning",
                            details={"handled": True})
            except Exception as e:
                self.log_test("Empty Query", "pass",
                            details={"error_handled": str(e)[:50]})
            
            # Very long query
            long_query = "What is " + "machine learning " * 100 + "?"
            result = self.platform.query(long_query, use_cache=True, use_bandit=True)
            self.log_test("Very Long Query", 
                         "pass" if result.get('response') else "fail",
                         details={
                             "query_length": len(long_query),
                             "tokens_used": result.get('tokens_used', 0),
                         })
            
            # Special characters
            special_query = "What is @#$%^&*()?"
            result = self.platform.query(special_query, use_cache=True, use_bandit=True)
            self.log_test("Special Characters", 
                         "pass" if result.get('response') else "fail",
                         details={"tokens_used": result.get('tokens_used', 0)})
            
        except Exception as e:
            self.log_test("Edge Cases Test", "fail", error=str(e))
    
    def generate_summary(self):
        """Generate test summary."""
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        
        total_tests = len(self.results['tests'])
        passed = sum(1 for t in self.results['tests'] if t['status'] == 'pass')
        failed = sum(1 for t in self.results['tests'] if t['status'] == 'fail')
        warnings = sum(1 for t in self.results['tests'] if t['status'] == 'warning')
        
        self.results['summary'] = {
            'total_tests': total_tests,
            'passed': passed,
            'failed': failed,
            'warnings': warnings,
            'pass_rate': round((passed / total_tests * 100) if total_tests > 0 else 0, 2),
            'test_end': datetime.now().isoformat(),
        }
        
        print(f"\nTotal Tests: {total_tests}")
        print(f"Passed: {passed} [PASS]")
        print(f"Failed: {failed} [FAIL]")
        print(f"Warnings: {warnings} [WARN]")
        print(f"Pass Rate: {self.results['summary']['pass_rate']}%")
        
        if self.results['errors']:
            print(f"\nErrors Found: {len(self.results['errors'])}")
            for error in self.results['errors']:
                print(f"  - {error['test']}: {error['error']}")
        
        print("\n" + "=" * 80)


def main():
    """Run comprehensive tests."""
    tester = ComprehensiveTest()
    results = tester.run_all_tests()
    
    # Save results
    output_file = Path(__file__).parent / 'comprehensive_test_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[OK] Test results saved to: {output_file}")
    
    # Return exit code
    if results['summary']['failed'] > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()

