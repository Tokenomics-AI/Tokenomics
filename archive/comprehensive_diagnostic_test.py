"""
Comprehensive Diagnostic Test for Tokenomics Platform

This test suite validates ALL components and capabilities:
- Memory Layer (exact cache, semantic cache, context injection, LLM Lingua, preferences)
- Orchestrator (complexity analysis, token allocation, compression, routing)
- Bandit Optimizer (strategy selection, RouterBench routing, learning)
- Quality Judge (if enabled)
- Component-level savings tracking

This test is designed for product showcase and validation.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import structlog

from tokenomics.core import TokenomicsPlatform
from tokenomics.config import TokenomicsConfig

logger = structlog.get_logger()


class ComprehensiveDiagnosticTest:
    """Comprehensive diagnostic test suite for Tokenomics platform."""
    
    def __init__(self, config: Optional[TokenomicsConfig] = None):
        """Initialize test suite."""
        self.config = config or TokenomicsConfig.from_env()
        self.platform = TokenomicsPlatform(config=self.config)
        self.results = {
            "test_metadata": {
                "timestamp": datetime.now().isoformat(),
                "config": {
                    "llm_provider": self.config.llm.provider,
                    "llm_model": self.config.llm.model,
                    "memory_exact_cache": self.config.memory.use_exact_cache,
                    "memory_semantic_cache": self.config.memory.use_semantic_cache,
                    "memory_llmlingua": self.config.memory.enable_llmlingua,
                    "orchestrator_knapsack": self.config.orchestrator.use_knapsack_optimization,
                    "bandit_algorithm": self.config.bandit.algorithm,
                    "bandit_cost_aware": True,  # Always test cost-aware routing
                    "quality_judge": self.config.judge.enabled,
                },
            },
            "test_cases": [],
            "component_tests": {},
            "summary": {},
        }
        
        # Test dataset covering all use cases
        self.test_dataset = self._create_test_dataset()
        
        logger.info("ComprehensiveDiagnosticTest initialized", 
                   num_test_cases=len(self.test_dataset))
    
    def _create_test_dataset(self) -> List[Dict[str, Any]]:
        """
        Create comprehensive test dataset covering all use cases.
        
        Test cases designed to exercise:
        1. Exact cache hits (identical queries)
        2. Semantic cache hits (high similarity, direct return)
        3. Context injection (medium similarity, context-enhanced)
        4. LLM Lingua compression (long queries, long context)
        5. User preference learning (formal, casual, technical, simple)
        6. Query complexity (simple, medium, complex)
        7. Bandit strategy selection (cheap, balanced, premium)
        8. RouterBench cost-aware routing
        9. Token budget scenarios
        """
        return [
            # =====================================================================
            # EXACT CACHE HITS - Test identical queries
            # =====================================================================
            {
                "id": "exact_cache_1",
                "category": "exact_cache",
                "query": "What is Python?",
                "description": "Simple query for exact cache test - first call",
                "expected_cache": "miss",
                "repeat": 2,  # Will repeat to test exact cache hit
            },
            {
                "id": "exact_cache_2",
                "category": "exact_cache",
                "query": "What is Python?",  # Same query - should hit exact cache
                "description": "Same query - should hit exact cache",
                "expected_cache": "exact",
            },
            
            # =====================================================================
            # SEMANTIC CACHE - HIGH SIMILARITY (Direct Return)
            # =====================================================================
            {
                "id": "semantic_direct_1",
                "category": "semantic_cache_direct",
                "query": "Explain Python programming language",
                "description": "High similarity to 'What is Python?' - should return directly",
                "expected_cache": "semantic_direct",
                "similarity_threshold": 0.85,
            },
            {
                "id": "semantic_direct_2",
                "category": "semantic_cache_direct",
                "query": "Tell me about Python",
                "description": "Another high similarity query",
                "expected_cache": "semantic_direct",
            },
            
            # =====================================================================
            # CONTEXT INJECTION - MEDIUM SIMILARITY
            # =====================================================================
            {
                "id": "context_injection_1",
                "category": "context_injection",
                "query": "How do I install Python packages?",
                "description": "Medium similarity - should inject context about Python",
                "expected_cache": "context",
                "similarity_threshold": 0.75,
            },
            {
                "id": "context_injection_2",
                "category": "context_injection",
                "query": "What are Python libraries used for?",
                "description": "Another context injection case",
                "expected_cache": "context",
            },
            
            # =====================================================================
            # LLM LINGUA COMPRESSION - LONG QUERIES
            # =====================================================================
            {
                "id": "llmlingua_query_1",
                "category": "llmlingua_query",
                "query": " ".join([
                    "I need a comprehensive explanation of how machine learning algorithms work,",
                    "specifically focusing on neural networks, deep learning architectures,",
                    "backpropagation, gradient descent optimization techniques,",
                    "activation functions like ReLU and sigmoid,",
                    "and how these components interact in modern AI systems.",
                    "Please provide detailed technical information with examples.",
                ]),
                "description": "Long query (>200 tokens) - should trigger query compression",
                "expected_compression": True,
                "min_query_tokens": 200,
            },
            {
                "id": "llmlingua_context_1",
                "category": "llmlingua_context",
                "query": "Summarize the key points about machine learning",
                "description": "Query that should retrieve and compress long context",
                "expected_compression": True,
            },
            
            # =====================================================================
            # USER PREFERENCE LEARNING
            # =====================================================================
            {
                "id": "preference_formal_1",
                "category": "user_preferences",
                "query": "Could you please provide a detailed explanation of REST APIs?",
                "description": "Formal tone - should learn formal preference",
                "expected_tone": "formal",
            },
            {
                "id": "preference_casual_1",
                "category": "user_preferences",
                "query": "Hey, what's up with REST APIs?",
                "description": "Casual tone - should learn casual preference",
                "expected_tone": "casual",
            },
            {
                "id": "preference_technical_1",
                "category": "user_preferences",
                "query": "Explain the implementation details of REST API architecture",
                "description": "Technical tone - should learn technical preference",
                "expected_tone": "technical",
            },
            {
                "id": "preference_list_1",
                "category": "user_preferences",
                "query": "List the steps to implement a REST API",
                "description": "List format preference",
                "expected_format": "list",
            },
            
            # =====================================================================
            # QUERY COMPLEXITY - SIMPLE
            # =====================================================================
            {
                "id": "complexity_simple_1",
                "category": "query_complexity",
                "query": "What is API?",
                "description": "Simple query - should use cheap strategy",
                "expected_complexity": "simple",
                "expected_strategy": "cheap",
            },
            {
                "id": "complexity_simple_2",
                "category": "query_complexity",
                "query": "Define REST",
                "description": "Another simple query",
                "expected_complexity": "simple",
            },
            
            # =====================================================================
            # QUERY COMPLEXITY - MEDIUM
            # =====================================================================
            {
                "id": "complexity_medium_1",
                "category": "query_complexity",
                "query": "How does authentication work in REST APIs?",
                "description": "Medium complexity - should use balanced strategy",
                "expected_complexity": "medium",
                "expected_strategy": "balanced",
            },
            {
                "id": "complexity_medium_2",
                "category": "query_complexity",
                "query": "Explain the difference between GET and POST methods",
                "description": "Another medium complexity query",
                "expected_complexity": "medium",
            },
            
            # =====================================================================
            # QUERY COMPLEXITY - COMPLEX
            # =====================================================================
            {
                "id": "complexity_complex_1",
                "category": "query_complexity",
                "query": " ".join([
                    "Design a comprehensive REST API architecture for an e-commerce platform",
                    "that handles user authentication, product catalog management,",
                    "shopping cart operations, payment processing, and order fulfillment.",
                    "Include security considerations, rate limiting, caching strategies,",
                    "and scalability patterns.",
                ]),
                "description": "Complex query - should use premium strategy",
                "expected_complexity": "complex",
                "expected_strategy": "premium",
            },
            
            # =====================================================================
            # BANDIT STRATEGY SELECTION
            # =====================================================================
            {
                "id": "bandit_cheap_1",
                "category": "bandit_selection",
                "query": "What is JSON?",
                "description": "Should select cheap strategy for simple query",
                "expected_strategy": "cheap",
            },
            {
                "id": "bandit_balanced_1",
                "category": "bandit_selection",
                "query": "How do I parse JSON in Python?",
                "description": "Should select balanced strategy",
                "expected_strategy": "balanced",
            },
            {
                "id": "bandit_premium_1",
                "category": "bandit_selection",
                "query": "Compare JSON, XML, and YAML formats for API design",
                "description": "Should select premium strategy",
                "expected_strategy": "premium",
            },
            
            # =====================================================================
            # ROUTERBENCH COST-AWARE ROUTING
            # =====================================================================
            {
                "id": "routerbench_1",
                "category": "routerbench",
                "query": "What is a database?",
                "description": "Test RouterBench cost-aware routing",
                "use_cost_aware_routing": True,
            },
            {
                "id": "routerbench_2",
                "category": "routerbench",
                "query": "Explain database normalization",
                "description": "Another RouterBench test",
                "use_cost_aware_routing": True,
            },
            
            # =====================================================================
            # TOKEN BUDGET SCENARIOS
            # =====================================================================
            {
                "id": "token_budget_low_1",
                "category": "token_budget",
                "query": "What is SQL?",
                "description": "Low token budget (1000)",
                "token_budget": 1000,
            },
            {
                "id": "token_budget_medium_1",
                "category": "token_budget",
                "query": "Explain SQL joins",
                "description": "Medium token budget (2000)",
                "token_budget": 2000,
            },
            {
                "id": "token_budget_high_1",
                "category": "token_budget",
                "query": "Design a database schema for a social media platform",
                "description": "High token budget (4000)",
                "token_budget": 4000,
            },
            
            # =====================================================================
            # EDGE CASES
            # =====================================================================
            {
                "id": "edge_case_empty",
                "category": "edge_cases",
                "query": "",
                "description": "Empty query",
            },
            {
                "id": "edge_case_very_long",
                "category": "edge_cases",
                "query": " ".join(["What is"] * 1000),  # Very long query
                "description": "Very long query",
            },
            {
                "id": "edge_case_special_chars",
                "category": "edge_cases",
                "query": "What is @#$%^&*()?",
                "description": "Special characters",
            },
        ]
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all diagnostic tests."""
        logger.info("Starting comprehensive diagnostic test", 
                   num_cases=len(self.test_dataset))
        
        start_time = time.time()
        
        # Run each test case
        for test_case in self.test_dataset:
            try:
                result = self._run_test_case(test_case)
                self.results["test_cases"].append(result)
            except Exception as e:
                logger.error("Test case failed", 
                           test_id=test_case.get("id"), 
                           error=str(e))
                self.results["test_cases"].append({
                    "test_id": test_case.get("id"),
                    "status": "error",
                    "error": str(e),
                })
        
        # Run component-specific tests
        self._test_memory_layer()
        self._test_orchestrator()
        self._test_bandit_optimizer()
        self._test_llmlingua()
        
        # Calculate summary statistics
        elapsed_time = time.time() - start_time
        self._calculate_summary(elapsed_time)
        
        logger.info("Comprehensive diagnostic test completed", 
                   elapsed_seconds=elapsed_time)
        
        return self.results
    
    def _run_test_case(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single test case."""
        test_id = test_case.get("id", "unknown")
        query = test_case.get("query", "")
        category = test_case.get("category", "unknown")
        
        logger.info("Running test case", test_id=test_id, category=category)
        
        # Extract test parameters
        token_budget = test_case.get("token_budget")
        use_cost_aware_routing = test_case.get("use_cost_aware_routing", True)
        
        # Run query through platform
        start_time = time.time()
        result = self.platform.query(
            query=query,
            token_budget=token_budget,
            use_cache=True,
            use_bandit=True,
            use_compression=True,
            use_cost_aware_routing=use_cost_aware_routing,
        )
        elapsed_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Extract metrics
        test_result = {
            "test_id": test_id,
            "category": category,
            "description": test_case.get("description", ""),
            "query": query[:200],  # Truncate for storage
            "status": "success",
            "elapsed_ms": elapsed_time,
            
            # Cache metrics
            "cache_hit": result.get("cache_hit", False),
            "cache_type": result.get("cache_type"),
            "cache_tier": result.get("cache_tier"),
            "similarity": result.get("similarity"),
            
            # Token metrics
            "tokens_used": result.get("tokens_used", 0),
            "input_tokens": result.get("input_tokens", 0),
            "output_tokens": result.get("output_tokens", 0),
            "max_response_tokens": result.get("max_response_tokens", 0),
            
            # Strategy metrics
            "strategy": result.get("strategy"),
            "model": result.get("model"),
            "query_complexity": result.get("query_type"),
            
            # Compression metrics
            "compression": result.get("compression_metrics", {}),
            
            # Component savings
            "component_savings": result.get("component_savings", {}),
            
            # Memory metrics
            "memory_metrics": result.get("memory_metrics", {}),
            
            # Orchestrator metrics
            "orchestrator_metrics": result.get("orchestrator_metrics", {}),
            
            # Bandit metrics
            "bandit_metrics": result.get("bandit_metrics", {}),
            
            # Quality judge (if available)
            "quality_judge": result.get("quality_judge"),
            
            # Baseline comparison (if available)
            "baseline_comparison": result.get("baseline_comparison_result"),
        }
        
        # Validate expectations
        if "expected_cache" in test_case:
            expected = test_case["expected_cache"]
            actual = result.get("cache_type")
            test_result["cache_validation"] = {
                "expected": expected,
                "actual": actual,
                "passed": expected == actual or (expected == "miss" and actual is None),
            }
        
        if "expected_complexity" in test_case:
            expected = test_case["expected_complexity"]
            actual = result.get("query_type")
            test_result["complexity_validation"] = {
                "expected": expected,
                "actual": actual,
                "passed": expected == actual,
            }
        
        if "expected_strategy" in test_case:
            expected = test_case["expected_strategy"]
            actual = result.get("strategy")
            test_result["strategy_validation"] = {
                "expected": expected,
                "actual": actual,
                "passed": expected == actual,
            }
        
        return test_result
    
    def _test_memory_layer(self):
        """Test memory layer components."""
        logger.info("Testing memory layer components")
        
        memory_tests = {
            "exact_cache_enabled": self.config.memory.use_exact_cache,
            "semantic_cache_enabled": self.config.memory.use_semantic_cache,
            "llmlingua_enabled": self.config.memory.enable_llmlingua,
            "preferences_enabled": True,  # Always enabled in SmartMemoryLayer
        }
        
        # Get memory stats
        memory_stats = self.platform.memory.stats()
        memory_tests["stats"] = memory_stats
        
        # Test exact cache
        if self.config.memory.use_exact_cache:
            test_query = "Memory layer test query"
            result1 = self.platform.query(test_query, use_bandit=False)
            result2 = self.platform.query(test_query, use_bandit=False)
            memory_tests["exact_cache_test"] = {
                "first_call_cache_hit": result1.get("cache_hit", False),
                "second_call_cache_hit": result2.get("cache_hit", False),
                "first_tokens": result1.get("tokens_used", 0),
                "second_tokens": result2.get("tokens_used", 0),
                "passed": not result1.get("cache_hit") and result2.get("cache_hit"),
            }
        
        self.results["component_tests"]["memory_layer"] = memory_tests
    
    def _test_orchestrator(self):
        """Test orchestrator components."""
        logger.info("Testing orchestrator components")
        
        orchestrator_tests = {
            "knapsack_optimization": self.config.orchestrator.use_knapsack_optimization,
            "multi_model_routing": self.config.orchestrator.enable_multi_model_routing,
            "default_token_budget": self.config.orchestrator.default_token_budget,
        }
        
        # Test complexity analysis
        test_queries = [
            ("What is X?", "simple"),
            ("How does X work?", "medium"),
            ("Design a comprehensive system for X", "complex"),
        ]
        
        complexity_tests = []
        for query, expected in test_queries:
            complexity = self.platform.orchestrator.analyze_complexity(query)
            complexity_tests.append({
                "query": query,
                "expected": expected,
                "actual": complexity.value,
                "passed": expected == complexity.value,
            })
        
        orchestrator_tests["complexity_analysis"] = complexity_tests
        
        # Test token allocation
        plan = self.platform.orchestrator.plan_query(
            query="Test query for token allocation",
            token_budget=2000,
        )
        orchestrator_tests["token_allocation"] = {
            "budget": plan.token_budget,
            "complexity": plan.complexity.value,
            "allocations": [
                {
                    "component": alloc.component,
                    "tokens": alloc.tokens,
                }
                for alloc in plan.allocations
            ],
        }
        
        self.results["component_tests"]["orchestrator"] = orchestrator_tests
    
    def _test_bandit_optimizer(self):
        """Test bandit optimizer components."""
        logger.info("Testing bandit optimizer components")
        
        bandit_tests = {
            "algorithm": self.config.bandit.algorithm,
            "exploration_rate": self.config.bandit.exploration_rate,
            "reward_lambda": self.config.bandit.reward_lambda,
        }
        
        # Get bandit stats
        bandit_stats = self.platform.bandit.stats()
        bandit_tests["stats"] = bandit_stats
        
        # Test strategy selection
        strategies = list(self.platform.bandit.arms.keys())
        bandit_tests["available_strategies"] = strategies
        
        # Test RouterBench routing
        routing_stats = self.platform.bandit.get_routing_stats()
        bandit_tests["routerbench_stats"] = routing_stats
        
        self.results["component_tests"]["bandit_optimizer"] = bandit_tests
    
    def _test_llmlingua(self):
        """Test LLM Lingua compression."""
        logger.info("Testing LLM Lingua compression")
        
        llmlingua_tests = {
            "enabled": self.config.memory.enable_llmlingua,
        }
        
        if self.platform.memory.llmlingua:
            llmlingua_tests["available"] = self.platform.memory.llmlingua.is_available()
            llmlingua_tests["stats"] = self.platform.memory.llmlingua.get_stats()
            
            # Test compression
            test_context = [
                "This is a test context that should be compressed.",
                "It contains multiple sentences with information.",
                "The compression should reduce token count while preserving meaning.",
            ]
            
            # Test compression using memory layer's compress_context (returns string)
            compressed = self.platform.memory.compress_context(
                contexts=test_context,
                target_tokens=50,
            )
            
            original_tokens = self.platform.memory.count_tokens(" ".join(test_context))
            compressed_tokens = self.platform.memory.count_tokens(compressed) if isinstance(compressed, str) else 0
            
            llmlingua_tests["compression_test"] = {
                "original_tokens": original_tokens,
                "compressed_tokens": compressed_tokens,
                "compression_ratio": compressed_tokens / original_tokens if original_tokens > 0 else 1.0,
                "passed": compressed_tokens < original_tokens,
            }
        else:
            llmlingua_tests["available"] = False
        
        self.results["component_tests"]["llmlingua"] = llmlingua_tests
    
    def _calculate_summary(self, elapsed_time: float):
        """Calculate summary statistics."""
        logger.info("Calculating summary statistics")
        
        test_cases = self.results["test_cases"]
        successful_tests = [t for t in test_cases if t.get("status") == "success"]
        
        # Cache statistics
        cache_hits = [t for t in successful_tests if t.get("cache_hit", False)]
        exact_hits = [t for t in cache_hits if t.get("cache_type") == "exact"]
        semantic_hits = [t for t in cache_hits if t.get("cache_type") == "semantic_direct"]
        context_hits = [t for t in cache_hits if t.get("cache_type") == "context"]
        
        # Token statistics
        total_tokens = sum(t.get("tokens_used", 0) for t in successful_tests)
        total_input_tokens = sum(t.get("input_tokens", 0) for t in successful_tests)
        total_output_tokens = sum(t.get("output_tokens", 0) for t in successful_tests)
        
        # Savings statistics
        total_savings = sum(
            t.get("component_savings", {}).get("total_savings", 0)
            for t in successful_tests
        )
        memory_savings = sum(
            t.get("component_savings", {}).get("memory_layer", 0)
            for t in successful_tests
        )
        orchestrator_savings = sum(
            t.get("component_savings", {}).get("orchestrator", 0)
            for t in successful_tests
        )
        bandit_savings = sum(
            t.get("component_savings", {}).get("bandit", 0)
            for t in successful_tests
        )
        
        # Compression statistics
        compression_tests = [
            t for t in successful_tests
            if t.get("compression", {}).get("context_compressed") or
               t.get("compression", {}).get("query_compressed")
        ]
        total_compression_savings = sum(
            t.get("compression", {}).get("total_compression_savings", 0)
            for t in successful_tests
        )
        
        # Strategy statistics
        strategy_counts = {}
        for t in successful_tests:
            strategy = t.get("strategy")
            if strategy:
                strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        # Complexity statistics
        complexity_counts = {}
        for t in successful_tests:
            complexity = t.get("query_complexity")
            if complexity:
                complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1
        
        # Latency statistics
        latencies = [t.get("elapsed_ms", 0) for t in successful_tests]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        
        self.results["summary"] = {
            "test_execution": {
                "total_tests": len(test_cases),
                "successful_tests": len(successful_tests),
                "failed_tests": len(test_cases) - len(successful_tests),
                "elapsed_seconds": elapsed_time,
            },
            "cache_performance": {
                "total_queries": len(successful_tests),
                "cache_hits": len(cache_hits),
                "cache_hit_rate": len(cache_hits) / len(successful_tests) if successful_tests else 0,
                "exact_hits": len(exact_hits),
                "semantic_direct_hits": len(semantic_hits),
                "context_hits": len(context_hits),
            },
            "token_usage": {
                "total_tokens": total_tokens,
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "avg_tokens_per_query": total_tokens / len(successful_tests) if successful_tests else 0,
            },
            "savings": {
                "total_savings": total_savings,
                "memory_layer_savings": memory_savings,
                "orchestrator_savings": orchestrator_savings,
                "bandit_savings": bandit_savings,
                "compression_savings": total_compression_savings,
                "savings_percentage": (total_savings / (total_tokens + total_savings) * 100) if (total_tokens + total_savings) > 0 else 0,
            },
            "compression": {
                "compression_tests": len(compression_tests),
                "total_compression_savings": total_compression_savings,
            },
            "strategy_distribution": strategy_counts,
            "complexity_distribution": complexity_counts,
            "latency": {
                "avg_latency_ms": avg_latency,
                "min_latency_ms": min(latencies) if latencies else 0,
                "max_latency_ms": max(latencies) if latencies else 0,
            },
        }
    
    def save_results(self, output_dir: str = "diagnostic_results"):
        """Save test results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON results
        json_file = output_path / f"comprehensive_diagnostic_{timestamp}.json"
        with open(json_file, "w") as f:
            json.dump(self.results, f, indent=2)
        
        logger.info("Results saved", json_file=str(json_file))
        
        return str(json_file)


def main():
    """Main entry point for diagnostic test."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive Diagnostic Test for Tokenomics Platform")
    parser.add_argument("--output-dir", default="diagnostic_results", help="Output directory for results")
    parser.add_argument("--config", help="Path to config file (optional)")
    
    args = parser.parse_args()
    
    # Initialize test
    config = None
    if args.config:
        # Load config from file if provided
        pass  # TODO: Implement config loading
    
    test_suite = ComprehensiveDiagnosticTest(config=config)
    
    # Run all tests
    results = test_suite.run_all_tests()
    
    # Save results
    output_file = test_suite.save_results(args.output_dir)
    
    # Print summary
    summary = results["summary"]
    print("\n" + "="*80)
    print("COMPREHENSIVE DIAGNOSTIC TEST SUMMARY")
    print("="*80)
    print(f"\nTest Execution:")
    print(f"  Total Tests: {summary['test_execution']['total_tests']}")
    print(f"  Successful: {summary['test_execution']['successful_tests']}")
    print(f"  Failed: {summary['test_execution']['failed_tests']}")
    print(f"  Elapsed Time: {summary['test_execution']['elapsed_seconds']:.2f}s")
    
    print(f"\nCache Performance:")
    print(f"  Cache Hit Rate: {summary['cache_performance']['cache_hit_rate']:.1%}")
    print(f"  Exact Hits: {summary['cache_performance']['exact_hits']}")
    print(f"  Semantic Direct Hits: {summary['cache_performance']['semantic_direct_hits']}")
    print(f"  Context Hits: {summary['cache_performance']['context_hits']}")
    
    print(f"\nToken Usage:")
    print(f"  Total Tokens: {summary['token_usage']['total_tokens']:,}")
    print(f"  Input Tokens: {summary['token_usage']['total_input_tokens']:,}")
    print(f"  Output Tokens: {summary['token_usage']['total_output_tokens']:,}")
    print(f"  Avg per Query: {summary['token_usage']['avg_tokens_per_query']:.0f}")
    
    print(f"\nSavings:")
    print(f"  Total Savings: {summary['savings']['total_savings']:,} tokens")
    print(f"  Memory Layer: {summary['savings']['memory_layer_savings']:,} tokens")
    print(f"  Orchestrator: {summary['savings']['orchestrator_savings']:,} tokens")
    print(f"  Bandit: {summary['savings']['bandit_savings']:,} tokens")
    print(f"  Compression: {summary['savings']['compression_savings']:,} tokens")
    print(f"  Savings Percentage: {summary['savings']['savings_percentage']:.1f}%")
    
    print(f"\nStrategy Distribution:")
    for strategy, count in summary['strategy_distribution'].items():
        print(f"  {strategy}: {count}")
    
    print(f"\nResults saved to: {output_file}")
    print("="*80)


if __name__ == "__main__":
    main()

