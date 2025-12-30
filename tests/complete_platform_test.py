"""
Complete End-to-End Platform Test
Tests all components: Cascading Inference, Token Prediction, Active Retrieval
Compares against baseline and trains regression model
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import sys
import os

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Import directly - same pattern as comprehensive_playground_test.py
from tokenomics.core import TokenomicsPlatform
from tokenomics.config import TokenomicsConfig
import structlog

logger = structlog.get_logger()

# Test queries covering different complexities - 50 queries for comprehensive testing
TEST_QUERIES = [
    # Simple queries (1-10)
    "What is 2+2?",
    "What is the capital of France?",
    "Explain what Python is in one sentence.",
    "What is the largest planet in our solar system?",
    "Define artificial intelligence briefly.",
    "What is a variable in programming?",
    "Name three primary colors.",
    "What is the speed of light?",
    "Explain what a function is in one sentence.",
    "What is the capital of Japan?",
    
    # Medium complexity (11-25)
    "Explain how neural networks work, including the basic concepts of neurons, layers, and activation functions.",
    "What are the main differences between supervised and unsupervised learning? Give examples of each.",
    "Describe the process of photosynthesis in plants, including the key steps and inputs/outputs.",
    "Explain the concept of recursion in programming with a simple example.",
    "What is the difference between REST and GraphQL APIs? When would you use each?",
    "Describe how a database index improves query performance.",
    "Explain the difference between HTTP and HTTPS protocols.",
    "What are the main components of a computer's CPU?",
    "Explain how version control systems like Git work.",
    "What is the difference between synchronous and asynchronous programming?",
    "Describe the basic principles of object-oriented programming.",
    "Explain how a web browser renders a webpage.",
    "What is the difference between a stack and a queue data structure?",
    "Explain how DNS (Domain Name System) works.",
    "Describe the process of compiling source code into machine code.",
    
    # Complex queries (26-40)
    "Write a detailed explanation of quantum computing, including qubits, superposition, entanglement, quantum gates, quantum algorithms like Shor's algorithm and Grover's algorithm, quantum error correction, and applications in cryptography and optimization problems. Explain the differences between classical and quantum computing paradigms.",
    "Provide a comprehensive analysis of the Transformer architecture in deep learning, including attention mechanisms, self-attention, multi-head attention, positional encoding, encoder-decoder structure, and how it revolutionized natural language processing. Compare it to previous architectures like RNNs and LSTMs.",
    "Explain the complete lifecycle of a software development project from requirements gathering to deployment, including methodologies (Agile, Waterfall, DevOps), version control, testing strategies, CI/CD pipelines, and monitoring. Include best practices for each phase.",
    "Write a comprehensive guide to building a web application using React, including setting up the development environment, component architecture, state management with Redux, routing with React Router, API integration, authentication, error handling, testing with Jest, deployment strategies, performance optimization techniques, and best practices for maintainable code.",
    "Provide a detailed explanation of the TCP/IP protocol stack, including all layers (Application, Transport, Network, Data Link, Physical), protocols at each layer, how data flows through the stack, packet structure, connection establishment, error handling, flow control, congestion control, and how it compares to the OSI model.",
    "Explain the complete process of machine learning model deployment, including model serialization, containerization with Docker, API design, load balancing, monitoring, A/B testing, model versioning, rollback strategies, and scaling considerations for production environments.",
    "Describe the architecture and implementation of a distributed system, covering concepts like consensus algorithms (Raft, Paxos), distributed transactions, CAP theorem, eventual consistency, microservices communication patterns, service discovery, and fault tolerance mechanisms.",
    "Provide a comprehensive overview of cloud computing architectures, including Infrastructure as a Service (IaaS), Platform as a Service (PaaS), Software as a Service (SaaS), container orchestration with Kubernetes, serverless computing, auto-scaling, cost optimization strategies, and security best practices.",
    "Explain the complete data science workflow from data collection and cleaning to model training and evaluation, including exploratory data analysis, feature engineering, model selection, hyperparameter tuning, cross-validation, bias-variance tradeoff, and model interpretability techniques.",
    "Describe the implementation of a real-time recommendation system, including collaborative filtering, content-based filtering, hybrid approaches, matrix factorization techniques, handling cold start problems, scalability considerations, and evaluation metrics.",
    
    # Cache testing queries - semantic similarity (41-45)
    "What is machine learning?",  # First occurrence
    "Tell me about machine learning",  # Should trigger semantic cache
    "Explain machine learning",  # Should trigger semantic cache
    "What is machine learning?",  # Exact match - should be instant cache hit
    "Can you describe machine learning?",  # Another semantic variation
    
    # Additional diverse queries (46-50)
    "Write a short story about an AI that learns to paint.",
    "Design a marketing strategy for a new SaaS product targeting small businesses.",
    "Create a workout plan for someone who wants to build muscle and improve cardiovascular health.",
    "Explain the economic principles behind inflation and how central banks control it.",
    "Describe the process of how a search engine indexes and ranks web pages.",
    "Explain the concept of blockchain technology and how it ensures data integrity.",
    "Describe the architecture of a microservices-based application and its advantages.",
    "What are the key differences between SQL and NoSQL databases? When should each be used?",
    "Explain how load balancing works in distributed systems and common algorithms used.",
    "Describe the principles of responsive web design and how to implement it effectively.",
]


class CompletePlatformTest:
    """Complete end-to-end platform test with critical analysis."""
    
    def __init__(self):
        """Initialize test environment."""
        # Initialize platform with all features enabled
        config = TokenomicsConfig.from_env()
        
        # Enable all advanced features
        config.cascading.enabled = True
        config.cascading.quality_threshold = 0.85
        config.memory.use_active_retrieval = False  # Test without active retrieval first
        config.memory.use_semantic_cache = True
        config.memory.use_exact_cache = True
        
        self.platform = TokenomicsPlatform(config=config)
        self.results: List[Dict[str, Any]] = []
        self.summary: Dict[str, Any] = {
            "total_queries": 0,
            "optimized_results": [],
            "baseline_results": [],
            "comparisons": [],
            "cascading_stats": {},
            "token_prediction_stats": {},
            "escalation_prediction_stats": {},
            "complexity_classifier_stats": {},
            "cache_stats": {},
            "component_savings": {},
            "quality_analysis": {},
        }
        
        logger.info("Complete Platform Test initialized", config=config.dict())
    
    def run_query(self, query: str, query_index: int) -> Dict[str, Any]:
        """Run a single query through both optimized and baseline paths."""
        logger.info("=" * 80)
        logger.info(f"TEST QUERY {query_index + 1}/{len(TEST_QUERIES)}")
        logger.info("=" * 80)
        logger.info("Query", query=query)
        
        result = {
            "query": query,
            "query_index": query_index,
            "timestamp": datetime.now().isoformat(),
        }
        
        # Run optimized path
        logger.info("--- Running OPTIMIZED path ---")
        optimized_start = time.time()
        try:
            optimized_result = self.platform.query(
                query=query,
                use_cache=True,
                use_compression=True,
                use_bandit=True,
                use_cost_aware_routing=True,
            )
            optimized_latency = (time.time() - optimized_start) * 1000
            
            result["optimized"] = {
                "success": True,
                "response": optimized_result.get("response", "")[:200] + "..." if len(optimized_result.get("response", "")) > 200 else optimized_result.get("response", ""),
                "response_length": len(optimized_result.get("response", "")),
                "tokens_used": optimized_result.get("tokens_used", 0),
                "input_tokens": optimized_result.get("input_tokens", 0),
                "output_tokens": optimized_result.get("output_tokens", 0),
                "latency_ms": optimized_result.get("latency_ms", optimized_latency),
                "cache_hit": optimized_result.get("cache_hit", False),
                "cache_type": optimized_result.get("cache_type", "none"),
                "similarity": optimized_result.get("similarity"),
                "strategy": optimized_result.get("strategy"),
                "model": optimized_result.get("model"),
                "cascading_escalated": optimized_result.get("cascading_escalated", False),
                "predicted_max_tokens": optimized_result.get("predicted_max_tokens"),
                "max_response_tokens": optimized_result.get("max_response_tokens"),
                "component_savings": optimized_result.get("component_savings", {}),
                "compression_metrics": optimized_result.get("compression_metrics", {}),
                "orchestrator_metrics": optimized_result.get("orchestrator_metrics", {}),
                "bandit_metrics": optimized_result.get("bandit_metrics", {}),
            }
            
            logger.info(
                "Optimized result",
                tokens=optimized_result.get("tokens_used", 0),
                latency_ms=optimized_result.get("latency_ms", optimized_latency),
                cache_hit=optimized_result.get("cache_hit", False),
                model=optimized_result.get("model"),
                cascading_escalated=optimized_result.get("cascading_escalated", False),
            )
        except Exception as e:
            logger.error("Optimized path failed", error=str(e))
            result["optimized"] = {
                "success": False,
                "error": str(e),
            }
        
        # Run baseline path (for comparison only)
        logger.info("--- Running BASELINE path (for comparison) ---")
        baseline_start = time.time()
        try:
            comparison_result = self.platform.compare_with_baseline(
                query=query,
            )
            baseline_latency = (time.time() - baseline_start) * 1000
            
            # Extract baseline data from compare_with_baseline result
            # compare_with_baseline returns optimized_result with baseline_comparison_result nested
            baseline_data = comparison_result.get("baseline_comparison_result", {})
            
            # Debug logging to understand structure
            logger.debug("Comparison result keys", keys=list(comparison_result.keys()))
            logger.debug("Baseline data extracted", baseline_data=baseline_data)
            
            if baseline_data:
                # Extract from nested structure
                baseline_response = baseline_data.get("response", "")
                baseline_tokens = baseline_data.get("tokens_used", 0)
                baseline_input = baseline_data.get("input_tokens", 0)
                baseline_output = baseline_data.get("output_tokens", 0)
                baseline_latency = baseline_data.get("latency_ms", baseline_latency)
                baseline_model = baseline_data.get("model", "gpt-4o")
            else:
                # Fallback: try direct keys (shouldn't happen but just in case)
                baseline_response = comparison_result.get("response", "")
                baseline_tokens = comparison_result.get("tokens_used", 0)
                baseline_input = comparison_result.get("input_tokens", 0)
                baseline_output = comparison_result.get("output_tokens", 0)
                baseline_latency = comparison_result.get("latency_ms", baseline_latency)
                baseline_model = comparison_result.get("model", "gpt-4o")
            
            # Validate baseline data - if tokens are 0, log warning
            if baseline_tokens == 0 and baseline_response:
                logger.warning(
                    "Baseline tokens are 0 but response exists",
                    response_length=len(baseline_response),
                    baseline_data_keys=list(baseline_data.keys()) if baseline_data else [],
                )
            
            result["baseline"] = {
                "success": True,
                "response": baseline_response[:200] + "..." if len(baseline_response) > 200 else baseline_response,
                "response_length": len(baseline_response),
                "tokens_used": baseline_tokens,
                "input_tokens": baseline_input,
                "output_tokens": baseline_output,
                "latency_ms": baseline_latency,
                "model": baseline_model,
            }
            
            logger.info(
                "Baseline result",
                tokens=baseline_tokens,  # Use extracted variable
                latency_ms=baseline_latency,  # Use extracted variable
                response_length=len(baseline_response),
            )
            
            # Quality comparison
            if "quality_judge" in comparison_result:
                judge_result = comparison_result["quality_judge"]
                result["quality_judge"] = {
                    "winner": judge_result.get("winner"),
                    "confidence": judge_result.get("confidence"),
                    "explanation": judge_result.get("explanation"),
                }
                logger.info(
                    "Quality judge",
                    winner=judge_result.get("winner"),
                    confidence=judge_result.get("confidence"),
                )
            
            # Calculate savings
            if result["optimized"].get("success") and result["baseline"].get("success"):
                opt_tokens = result["optimized"]["tokens_used"]
                base_tokens = result["baseline"]["tokens_used"]
                token_savings = base_tokens - opt_tokens
                token_savings_pct = (token_savings / base_tokens * 100) if base_tokens > 0 else 0
                
                opt_latency = result["optimized"]["latency_ms"]
                base_latency = result["baseline"]["latency_ms"]
                latency_reduction = base_latency - opt_latency
                latency_reduction_pct = (latency_reduction / base_latency * 100) if base_latency > 0 else 0
                
                result["comparison"] = {
                    "token_savings": token_savings,
                    "token_savings_percent": round(token_savings_pct, 2),
                    "latency_reduction_ms": round(latency_reduction, 2),
                    "latency_reduction_percent": round(latency_reduction_pct, 2),
                    "cost_savings": self._estimate_cost_savings(opt_tokens, base_tokens),
                }
                
                logger.info(
                    "Comparison",
                    token_savings=token_savings,
                    token_savings_pct=f"{token_savings_pct:.2f}%",
                    latency_reduction_ms=f"{latency_reduction:.2f}ms",
                    latency_reduction_pct=f"{latency_reduction_pct:.2f}%",
                )
        
        except Exception as e:
            logger.error("Baseline path failed", error=str(e))
            result["baseline"] = {
                "success": False,
                "error": str(e),
            }
        
        self.results.append(result)
        return result
    
    def _estimate_cost_savings(self, opt_tokens: int, base_tokens: int) -> float:
        """Estimate cost savings in dollars."""
        # Rough estimate: $0.01 per 1K tokens (average)
        cost_per_1k = 0.01
        savings = (base_tokens - opt_tokens) / 1000 * cost_per_1k
        return round(savings, 4)
    
    def run_all_tests(self):
        """Run all test queries."""
        logger.info("Starting complete platform test", num_queries=len(TEST_QUERIES))
        
        for i, query in enumerate(TEST_QUERIES):
            try:
                logger.info(f"Processing query {i+1}/{len(TEST_QUERIES)}", query=query[:50])
                result = self.run_query(query, i)
                self.summary["total_queries"] += 1
                
                if result.get("optimized", {}).get("success"):
                    self.summary["optimized_results"].append(result["optimized"])
                
                if result.get("baseline", {}).get("success"):
                    self.summary["baseline_results"].append(result["baseline"])
                
                if "comparison" in result:
                    self.summary["comparisons"].append(result["comparison"])
                
                # Save intermediate results
                if (i + 1) % 5 == 0:
                    self.save_results(intermediate=True)
                    logger.info(f"Saved intermediate results after {i+1} queries")
            
            except KeyboardInterrupt:
                logger.warning("Test interrupted by user")
                raise  # Re-raise to allow graceful shutdown
            except Exception as e:
                logger.error(
                    "Query failed",
                    query_index=i,
                    query=query[:50] if query else "unknown",
                    error=str(e),
                    exc_info=True,
                )
                # Add failed query to results for tracking
                self.results.append({
                    "query": query if query else "unknown",
                    "query_index": i,
                    "timestamp": datetime.now().isoformat(),
                    "error": str(e),
                    "success": False,
                })
                # Continue to next query
                continue
        
        # Log completion status
        logger.info(
            "Test loop completed",
            total_queries=len(TEST_QUERIES),
            completed_queries=self.summary["total_queries"],
            successful_optimized=len(self.summary["optimized_results"]),
            successful_baseline=len(self.summary["baseline_results"]),
        )
        
        # Collect platform stats
        self.collect_platform_stats()
        
        # Train regression model
        self.train_regression_model()
        
        # Generate final report
        self.generate_report()
        
        # Final summary log
        logger.info(
            "All test phases completed",
            queries_processed=self.summary["total_queries"],
            comparisons=len(self.summary["comparisons"]),
            has_cascading_stats=bool(self.summary.get("cascading_stats")),
            has_token_prediction_stats=bool(self.summary.get("token_prediction_stats")),
        )
    
    def collect_platform_stats(self):
        """Collect platform-wide statistics."""
        logger.info("Collecting platform-wide statistics...")
        try:
            stats = self.platform.get_stats()
            logger.debug("Platform stats retrieved", stats_keys=list(stats.keys()) if stats else [])
            
            # Cascading stats - initialize with defaults if empty
            if "cascading" in stats:
                cascading_data = stats["cascading"] or {}
                self.summary["cascading_stats"] = {
                    "total_queries": cascading_data.get("total_queries", 0),
                    "cheap_model_used": cascading_data.get("cheap_model_used", 0),
                    "escalations": cascading_data.get("escalations", 0),
                    "escalation_rate": cascading_data.get("escalation_rate", 0.0),
                    "cost_savings": cascading_data.get("cost_savings", 0.0),
                }
                logger.info("Cascading stats collected", stats=self.summary["cascading_stats"])
            else:
                self.summary["cascading_stats"] = {
                    "total_queries": 0,
                    "cheap_model_used": 0,
                    "escalations": 0,
                    "escalation_rate": 0.0,
                    "cost_savings": 0.0,
                }
            
            # Token predictor stats - initialize with defaults if empty
            if "token_predictor" in stats:
                tp_data = stats["token_predictor"] or {}
                self.summary["token_prediction_stats"] = {
                    "model_trained": tp_data.get("model_trained", False),
                    "model_type": tp_data.get("model_type", "Heuristic"),
                    "total_samples": tp_data.get("total_samples", 0),
                    "avg_output_tokens": tp_data.get("avg_output_tokens", 0),
                    "min_output_tokens": tp_data.get("min_output_tokens", 0),
                    "max_output_tokens": tp_data.get("max_output_tokens", 0),
                }
                logger.info("Token prediction stats collected", stats=self.summary["token_prediction_stats"])
            else:
                self.summary["token_prediction_stats"] = {
                    "model_trained": False,
                    "model_type": "Heuristic",
                    "total_samples": 0,
                }
            
            # Escalation predictor stats - initialize with defaults if empty
            if "escalation_predictor" in stats:
                ep_data = stats["escalation_predictor"] or {}
                self.summary["escalation_prediction_stats"] = {
                    "model_trained": ep_data.get("model_trained", False),
                    "model_type": ep_data.get("model_type", "Heuristic"),
                    "total_samples": ep_data.get("total_samples", 0),
                    "escalation_rate": ep_data.get("escalation_rate", 0.0),
                    "escalated_count": ep_data.get("escalated_count", 0),
                    "avg_context_quality": ep_data.get("avg_context_quality", 0.0),
                }
                logger.info("Escalation prediction stats collected", stats=self.summary["escalation_prediction_stats"])
            else:
                self.summary["escalation_prediction_stats"] = {
                    "model_trained": False,
                    "model_type": "Heuristic",
                    "total_samples": 0,
                    "escalation_rate": 0.0,
                }
            
            # Complexity classifier stats - initialize with defaults if empty
            if "complexity_classifier" in stats:
                cc_data = stats["complexity_classifier"] or {}
                self.summary["complexity_classifier_stats"] = {
                    "model_trained": cc_data.get("model_trained", False),
                    "model_type": cc_data.get("model_type", "Heuristic"),
                    "total_samples": cc_data.get("total_samples", 0),
                    "simple_count": cc_data.get("simple_count", 0),
                    "medium_count": cc_data.get("medium_count", 0),
                    "complex_count": cc_data.get("complex_count", 0),
                    "labeled_samples": cc_data.get("labeled_samples", 0),
                }
                logger.info("Complexity classifier stats collected", stats=self.summary["complexity_classifier_stats"])
            else:
                self.summary["complexity_classifier_stats"] = {
                    "model_trained": False,
                    "model_type": "Heuristic",
                    "total_samples": 0,
                    "simple_count": 0,
                    "medium_count": 0,
                    "complex_count": 0,
                }
            
            # Cache stats
            if "memory" in stats and stats["memory"]:
                self.summary["cache_stats"] = stats["memory"]
                logger.info("Cache stats collected", cache_keys=list(stats["memory"].keys()) if isinstance(stats["memory"], dict) else "non-dict")
            else:
                self.summary["cache_stats"] = {}
            
            # Bandit stats
            if "bandit" in stats and stats["bandit"]:
                self.summary["bandit_stats"] = stats["bandit"]
            
            logger.info("Platform stats collected", stats_keys=list(stats.keys()))
        except Exception as e:
            logger.error("Failed to collect platform stats", error=str(e))
            import traceback
            logger.error("Traceback", traceback=traceback.format_exc())
            # Initialize empty stats to prevent KeyError
            self.summary["cascading_stats"] = {}
            self.summary["token_prediction_stats"] = {}
            self.summary["cache_stats"] = {}
    
    def train_regression_model(self):
        """Train the regression model with collected data."""
        logger.info("Training regression model with collected data...")
        
        # Check current data count
        if hasattr(self.platform, 'token_predictor') and self.platform.token_predictor:
            if hasattr(self.platform.token_predictor, 'data_collector') and self.platform.token_predictor.data_collector:
                stats = self.platform.token_predictor.data_collector.get_stats()
                logger.info("Current training data stats", stats=stats)
        
        try:
            # Use lower threshold for testing (10 samples) but prefer 50+ for better model
            success = self.platform.train_token_predictor(min_samples=10)  # Lower threshold for testing
            if success:
                logger.info("Regression model trained successfully!")
                self.summary["regression_model_trained"] = True
            else:
                logger.info("Not enough data to train model yet", 
                          required_samples=10,
                          current_samples=self.summary.get("token_prediction_stats", {}).get("total_samples", 0))
                self.summary["regression_model_trained"] = False
        except Exception as e:
            logger.error("Failed to train regression model", error=str(e))
            self.summary["regression_model_trained"] = False
            self.summary["regression_model_error"] = str(e)
        
        # Train escalation predictor
        if self.platform.escalation_predictor:
            try:
                # Use lower threshold for testing (10 samples)
                success = self.platform.train_escalation_predictor(min_samples=10)
                if success:
                    logger.info("Escalation prediction model trained successfully!")
                    self.summary["escalation_model_trained"] = True
                else:
                    logger.info("Not enough data to train escalation model yet",
                              required_samples=10,
                              current_samples=self.summary.get("escalation_prediction_stats", {}).get("total_samples", 0))
                    self.summary["escalation_model_trained"] = False
            except Exception as e:
                logger.error("Failed to train escalation prediction model", error=str(e))
                self.summary["escalation_model_trained"] = False
                self.summary["escalation_model_error"] = str(e)
        
        # Train complexity classifier
        if self.platform.complexity_classifier:
            try:
                # Use lower threshold for testing (10 samples)
                success = self.platform.train_complexity_classifier(min_samples=10)
                if success:
                    logger.info("Complexity classification model trained successfully!")
                    self.summary["complexity_model_trained"] = True
                else:
                    logger.info("Not enough data to train complexity model yet",
                              required_samples=10,
                              current_samples=self.summary.get("complexity_classifier_stats", {}).get("total_samples", 0))
                    self.summary["complexity_model_trained"] = False
            except Exception as e:
                logger.error("Failed to train complexity classification model", error=str(e))
                self.summary["complexity_model_trained"] = False
                self.summary["complexity_model_error"] = str(e)
    
    def generate_report(self):
        """Generate comprehensive test report."""
        logger.info("Generating comprehensive test report...")
        
        # Calculate aggregate metrics
        if self.summary["optimized_results"]:
            opt_results = self.summary["optimized_results"]
            self.summary["aggregate_metrics"] = {
                "avg_tokens": sum(r["tokens_used"] for r in opt_results) / len(opt_results),
                "avg_latency_ms": sum(r["latency_ms"] for r in opt_results) / len(opt_results),
                "cache_hit_rate": sum(1 for r in opt_results if r.get("cache_hit", False)) / len(opt_results) * 100,
                "cascading_escalation_rate": sum(1 for r in opt_results if r.get("cascading_escalated", False)) / len(opt_results) * 100,
            }
        
        if self.summary["baseline_results"]:
            base_results = self.summary["baseline_results"]
            self.summary["baseline_aggregate"] = {
                "avg_tokens": sum(r["tokens_used"] for r in base_results) / len(base_results),
                "avg_latency_ms": sum(r["latency_ms"] for r in base_results) / len(base_results),
            }
        
        if self.summary["comparisons"]:
            comparisons = self.summary["comparisons"]
            self.summary["savings_summary"] = {
                "avg_token_savings": sum(c["token_savings"] for c in comparisons) / len(comparisons),
                "avg_token_savings_pct": sum(c["token_savings_percent"] for c in comparisons) / len(comparisons),
                "avg_latency_reduction_ms": sum(c["latency_reduction_ms"] for c in comparisons) / len(comparisons),
                "total_cost_savings": sum(c["cost_savings"] for c in comparisons),
            }
        
        # Critical analysis
        self.summary["critical_analysis"] = self._perform_critical_analysis()
    
    def _perform_critical_analysis(self) -> Dict[str, Any]:
        """Perform critical analysis of results."""
        analysis = {
            "cascading_inference": {},
            "token_prediction": {},
            "cache_performance": {},
            "overall_assessment": {},
        }
        
        # Cascading analysis
        if self.summary.get("cascading_stats"):
            cascading = self.summary["cascading_stats"]
            total = cascading.get("total_queries", 0)
            escalations = cascading.get("escalations", 0)
            
            if total > 0:
                escalation_rate = (escalations / total) * 100
                analysis["cascading_inference"] = {
                    "total_queries": total,
                    "escalations": escalations,
                    "escalation_rate_pct": round(escalation_rate, 2),
                    "assessment": "GOOD" if escalation_rate < 30 else "NEEDS_TUNING",
                    "notes": f"Escalation rate of {escalation_rate:.1f}% is {'acceptable' if escalation_rate < 30 else 'high - consider lowering quality threshold'}",
                }
        
        # Token prediction analysis
        if self.summary.get("token_prediction_stats"):
            tp_stats = self.summary["token_prediction_stats"]
            analysis["token_prediction"] = {
                "model_trained": tp_stats.get("model_trained", False),
                "model_type": tp_stats.get("model_type", "Heuristic"),
                "total_samples": tp_stats.get("total_samples", 0),
                "assessment": "READY_FOR_ML" if tp_stats.get("total_samples", 0) >= 500 else "COLLECTING_DATA",
                "notes": f"Using {tp_stats.get('model_type', 'Heuristic')} model. {'Ready for ML training' if tp_stats.get('total_samples', 0) >= 500 else 'Collecting data for ML training'}",
            }
        
        # Cache performance
        if self.summary.get("aggregate_metrics"):
            cache_hit_rate = self.summary["aggregate_metrics"].get("cache_hit_rate", 0)
            analysis["cache_performance"] = {
                "cache_hit_rate_pct": round(cache_hit_rate, 2),
                "assessment": "EXCELLENT" if cache_hit_rate > 50 else "GOOD" if cache_hit_rate > 20 else "NEEDS_IMPROVEMENT",
                "notes": f"Cache hit rate of {cache_hit_rate:.1f}% is {'excellent' if cache_hit_rate > 50 else 'good' if cache_hit_rate > 20 else 'low - consider more diverse queries'}",
            }
        
        # Overall assessment
        if self.summary.get("savings_summary"):
            savings = self.summary["savings_summary"]
            avg_savings_pct = savings.get("avg_token_savings_pct", 0)
            
            analysis["overall_assessment"] = {
                "avg_token_savings_pct": round(avg_savings_pct, 2),
                "total_cost_savings": savings.get("total_cost_savings", 0),
                "assessment": "EXCELLENT" if avg_savings_pct > 30 else "GOOD" if avg_savings_pct > 15 else "NEEDS_IMPROVEMENT",
                "notes": f"Platform achieves {avg_savings_pct:.1f}% average token savings. {'Excellent performance!' if avg_savings_pct > 30 else 'Good performance' if avg_savings_pct > 15 else 'Consider tuning optimization strategies'}",
            }
        
        return analysis
    
    def save_results(self, intermediate: bool = False):
        """Save test results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = "_intermediate" if intermediate else ""
        filename = f"complete_platform_test_results{suffix}_{timestamp}.json"
        filepath = Path(__file__).parent.parent / filename
        
        # Ensure summary is populated before saving
        if not intermediate:
            # Final save - ensure all stats are collected
            if not self.summary.get("cascading_stats"):
                self.collect_platform_stats()
            
            # Add training data stats to summary
            if hasattr(self.platform, 'token_predictor') and self.platform.token_predictor:
                if hasattr(self.platform.token_predictor, 'data_collector') and self.platform.token_predictor.data_collector:
                    training_stats = self.platform.token_predictor.data_collector.get_stats()
                    self.summary["training_data_stats"] = training_stats
                    logger.info("Training data stats added to summary", stats=training_stats)
            
            # Add escalation training data stats to summary
            if hasattr(self.platform, 'escalation_predictor') and self.platform.escalation_predictor:
                if hasattr(self.platform.escalation_predictor, 'data_collector') and self.platform.escalation_predictor.data_collector:
                    escalation_training_stats = self.platform.escalation_predictor.data_collector.get_stats("escalation")
                    self.summary["escalation_training_data_stats"] = escalation_training_stats
                    logger.info("Escalation training data stats added to summary", stats=escalation_training_stats)
            
            # Add complexity training data stats to summary
            if hasattr(self.platform, 'complexity_classifier') and self.platform.complexity_classifier:
                if hasattr(self.platform.complexity_classifier, 'data_collector') and self.platform.complexity_classifier.data_collector:
                    complexity_training_stats = self.platform.complexity_classifier.data_collector.get_stats("complexity")
                    self.summary["complexity_training_data_stats"] = complexity_training_stats
                    logger.info("Complexity training data stats added to summary", stats=complexity_training_stats)
        
        output = {
            "test_metadata": {
                "timestamp": timestamp,
                "total_queries": len(self.results),
                "intermediate": intermediate,
            },
            "results": self.results,
            "summary": self.summary,
        }
        
        with open(filepath, "w") as f:
            json.dump(output, f, indent=2, default=str)
        
        logger.info("Results saved", filepath=str(filepath))
        return filepath
    
    def print_summary(self):
        """Print test summary to console."""
        print("\n" + "=" * 80)
        print("COMPLETE PLATFORM TEST SUMMARY")
        print("=" * 80)
        
        print(f"\nTotal Queries Expected: {len(TEST_QUERIES)}")
        print(f"Total Queries Completed: {self.summary['total_queries']}")
        if self.summary['total_queries'] < len(TEST_QUERIES):
            print(f"⚠️  WARNING: Only {self.summary['total_queries']}/{len(TEST_QUERIES)} queries completed")
        
        if self.summary.get("aggregate_metrics"):
            print("\n--- OPTIMIZED PATH METRICS ---")
            metrics = self.summary["aggregate_metrics"]
            print(f"  Average Tokens: {metrics['avg_tokens']:.0f}")
            print(f"  Average Latency: {metrics['avg_latency_ms']:.2f} ms")
            print(f"  Cache Hit Rate: {metrics['cache_hit_rate']:.2f}%")
            print(f"  Cascading Escalation Rate: {metrics['cascading_escalation_rate']:.2f}%")
        
        if self.summary.get("savings_summary"):
            print("\n--- SAVINGS vs BASELINE ---")
            savings = self.summary["savings_summary"]
            print(f"  Average Token Savings: {savings['avg_token_savings']:.0f} tokens ({savings['avg_token_savings_pct']:.2f}%)")
            print(f"  Average Latency Reduction: {savings['avg_latency_reduction_ms']:.2f} ms")
            print(f"  Total Cost Savings: ${savings['total_cost_savings']:.4f}")
        
        if self.summary.get("critical_analysis"):
            print("\n--- CRITICAL ANALYSIS ---")
            analysis = self.summary["critical_analysis"]
            
            if "cascading_inference" in analysis:
                casc = analysis["cascading_inference"]
                print(f"\n  Cascading Inference: {casc.get('assessment', 'N/A')}")
                print(f"    {casc.get('notes', '')}")
            
            if "token_prediction" in analysis:
                tp = analysis["token_prediction"]
                print(f"\n  Token Prediction: {tp.get('assessment', 'N/A')}")
                print(f"    {tp.get('notes', '')}")
            
            if "cache_performance" in analysis:
                cache = analysis["cache_performance"]
                print(f"\n  Cache Performance: {cache.get('assessment', 'N/A')}")
                print(f"    {cache.get('notes', '')}")
            
            if "overall_assessment" in analysis:
                overall = analysis["overall_assessment"]
                print(f"\n  Overall Assessment: {overall.get('assessment', 'N/A')}")
                print(f"    {overall.get('notes', '')}")
        
        print("\n" + "=" * 80)


def main():
    """Run complete platform test."""
    test = CompletePlatformTest()
    
    try:
        test.run_all_tests()
        test.save_results()
        test.print_summary()
        
        # Check if all queries completed
        if test.summary['total_queries'] < len(TEST_QUERIES):
            print(f"\n⚠️  WARNING: Test completed but only {test.summary['total_queries']}/{len(TEST_QUERIES)} queries were processed")
            print("Check logs for errors on failed queries")
        else:
            print("\n✅ Complete platform test finished successfully!")
        print(f"Results saved to: complete_platform_test_results_*.json")
        
    except KeyboardInterrupt:
        logger.warning("Test interrupted by user")
        test.save_results(intermediate=True)
        print("\n⚠️  Test interrupted. Intermediate results saved.")
        print(f"Completed {test.summary['total_queries']}/{len(TEST_QUERIES)} queries before interruption")
    except Exception as e:
        logger.error("Test failed", error=str(e), exc_info=True)
        test.save_results(intermediate=True)
        print(f"\n❌ Test failed with error: {str(e)}")
        print(f"Completed {test.summary['total_queries']}/{len(TEST_QUERIES)} queries before failure")
        print("Check logs for detailed error information")
        raise


if __name__ == "__main__":
    main()






