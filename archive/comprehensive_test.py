"""Comprehensive test with detailed step-by-step documentation."""

import os
import sys
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

from tokenomics.core import TokenomicsPlatform
from tokenomics.config import TokenomicsConfig
from tokenomics.bandit import Strategy
from usage_tracker import UsageTracker


class ComprehensiveTest:
    """Comprehensive test with detailed documentation."""
    
    def __init__(self):
        """Initialize test."""
        self.documentation = {
            "test_start": datetime.now().isoformat(),
            "steps": [],
            "token_allocations": [],
            "marginal_utility": [],
            "quality_metrics": [],
            "bandit_performance": [],
            "cache_performance": [],
            "optimization_results": [],
        }
        self.tracker = UsageTracker(output_file="comprehensive_test_report.json")
    
    def log_step(self, step_num: int, step_name: str, details: Dict):
        """Log a test step."""
        step = {
            "step_number": step_num,
            "step_name": step_name,
            "timestamp": datetime.now().isoformat(),
            "details": details,
        }
        self.documentation["steps"].append(step)
        print(f"\n{'='*80}")
        print(f"STEP {step_num}: {step_name}")
        print(f"{'='*80}")
        for key, value in details.items():
            print(f"  {key}: {value}")
    
    def analyze_token_allocation(self, plan, step_num: int):
        """Analyze and document token allocation."""
        allocation_details = {
            "step": step_num,
            "total_budget": plan.token_budget,
            "allocations": [],
            "total_allocated": 0,
            "budget_utilization": 0.0,
        }
        
        for alloc in plan.allocations:
            allocation_details["allocations"].append({
                "component": alloc.component,
                "tokens": alloc.tokens,
                "priority": alloc.priority,
                "utility": alloc.utility,
                "utility_density": alloc.utility / alloc.tokens if alloc.tokens > 0 else 0,
            })
            allocation_details["total_allocated"] += alloc.tokens
        
        allocation_details["budget_utilization"] = (
            allocation_details["total_allocated"] / plan.token_budget * 100
            if plan.token_budget > 0 else 0
        )
        
        self.documentation["token_allocations"].append(allocation_details)
        
        print(f"\n  Token Allocation Analysis:")
        print(f"    Total Budget: {plan.token_budget} tokens")
        print(f"    Total Allocated: {allocation_details['total_allocated']} tokens")
        print(f"    Budget Utilization: {allocation_details['budget_utilization']:.1f}%")
        print(f"    Components Allocated:")
        for alloc in allocation_details["allocations"]:
            print(f"      - {alloc['component']}: {alloc['tokens']} tokens "
                  f"(utility: {alloc['utility']:.2f}, density: {alloc['utility_density']:.3f})")
    
    def analyze_marginal_utility(self, allocations: List, step_num: int):
        """Analyze marginal utility of token allocation."""
        # Sort by utility density (marginal utility per token)
        sorted_allocations = sorted(
            allocations,
            key=lambda a: a.utility / a.tokens if a.tokens > 0 else 0,
            reverse=True
        )
        
        utility_analysis = {
            "step": step_num,
            "total_utility": sum(a.utility for a in allocations),
            "average_utility": sum(a.utility for a in allocations) / len(allocations) if allocations else 0,
            "utility_by_component": {},
            "marginal_utility_ranking": [],
        }
        
        for alloc in sorted_allocations:
            utility_analysis["utility_by_component"][alloc.component] = {
                "utility": alloc.utility,
                "tokens": alloc.tokens,
                "utility_density": alloc.utility / alloc.tokens if alloc.tokens > 0 else 0,
            }
            utility_analysis["marginal_utility_ranking"].append({
                "component": alloc.component,
                "utility_density": alloc.utility / alloc.tokens if alloc.tokens > 0 else 0,
                "rank": len(utility_analysis["marginal_utility_ranking"]) + 1,
            })
        
        self.documentation["marginal_utility"].append(utility_analysis)
        
        print(f"\n  Marginal Utility Analysis:")
        print(f"    Total Utility: {utility_analysis['total_utility']:.3f}")
        print(f"    Average Utility: {utility_analysis['average_utility']:.3f}")
        print(f"    Utility Ranking (by density):")
        for rank, item in enumerate(utility_analysis["marginal_utility_ranking"], 1):
            print(f"      {rank}. {item['component']}: {item['utility_density']:.4f} utility/token")
    
    def analyze_quality_improvement(self, query: str, plan, result: Dict, step_num: int):
        """Analyze quality improvement from orchestrator."""
        quality_metrics = {
            "step": step_num,
            "query": query,
            "complexity": plan.complexity.value,
            "response_length": len(result["response"]),
            "tokens_used": result["tokens_used"],
            "quality_indicators": {},
        }
        
        response = result["response"]
        
        # Quality indicators
        has_definition = any(word in response.lower() for word in ["is", "are", "means", "refers", "defined"])
        has_explanation = any(word in response.lower() for word in ["because", "since", "due to", "explain", "works"])
        has_examples = any(word in response.lower() for word in ["example", "instance", "such as", "like", "e.g."])
        has_structure = any(marker in response for marker in ["**", "##", "-", "*", "1.", "2.", "\n\n"])
        
        quality_metrics["quality_indicators"] = {
            "has_definition": has_definition,
            "has_explanation": has_explanation,
            "has_examples": has_examples,
            "has_structure": has_structure,
            "quality_score": sum([has_definition, has_explanation, has_examples, has_structure]) / 4.0,
        }
        
        # Calculate quality per token (efficiency)
        quality_metrics["quality_per_token"] = (
            quality_metrics["quality_indicators"]["quality_score"] / result["tokens_used"]
            if result["tokens_used"] > 0 else 0
        )
        
        self.documentation["quality_metrics"].append(quality_metrics)
        
        print(f"\n  Quality Analysis:")
        print(f"    Complexity: {plan.complexity.value}")
        print(f"    Response Length: {len(result['response'])} characters")
        print(f"    Quality Score: {quality_metrics['quality_indicators']['quality_score']:.2f}/1.0")
        print(f"    Quality Indicators:")
        for indicator, value in quality_metrics["quality_indicators"].items():
            if indicator != "quality_score":
                status = "[YES]" if value else "[NO]"
                print(f"      - {indicator}: {status}")
        print(f"    Quality per Token: {quality_metrics['quality_per_token']:.6f}")
    
    def analyze_bandit_performance(self, platform, step_num: int, strategy_used: str, reward: float):
        """Analyze bandit optimizer performance."""
        stats = platform.bandit.stats()
        best_strategy = platform.bandit.get_best_strategy()
        
        bandit_analysis = {
            "step": step_num,
            "strategy_used": strategy_used,
            "reward": reward,
            "total_pulls": stats["total_pulls"],
            "arms_performance": {},
            "best_strategy": best_strategy.arm_id if best_strategy else None,
        }
        
        for arm_id, arm_stats in stats["arms"].items():
            bandit_analysis["arms_performance"][arm_id] = {
                "pulls": arm_stats["pulls"],
                "average_reward": arm_stats["average_reward"],
                "total_reward": arm_stats["total_reward"],
                "exploration_vs_exploitation": "exploration" if arm_stats["pulls"] <= 2 else "exploitation",
            }
        
        self.documentation["bandit_performance"].append(bandit_analysis)
        
        print(f"\n  Bandit Performance:")
        print(f"    Strategy Used: {strategy_used}")
        print(f"    Reward: {reward:.4f}")
        print(f"    Total Pulls: {stats['total_pulls']}")
        print(f"    Best Strategy: {bandit_analysis['best_strategy']}")
        print(f"    Arms Performance:")
        for arm_id, perf in bandit_analysis["arms_performance"].items():
            print(f"      - {arm_id}:")
            print(f"        Pulls: {perf['pulls']}")
            print(f"        Avg Reward: {perf['average_reward']:.4f}")
            print(f"        Mode: {perf['exploration_vs_exploitation']}")
    
    def analyze_cache_performance(self, platform, step_num: int, cache_hit: bool, tokens_saved: int):
        """Analyze cache performance."""
        cache_stats = platform.memory.stats()
        
        cache_analysis = {
            "step": step_num,
            "cache_hit": cache_hit,
            "tokens_saved_this_step": tokens_saved,
            "cache_size": cache_stats.get("size", 0),
            "total_tokens_saved": cache_stats.get("total_tokens_saved", 0),
            "cache_efficiency": 0.0,
        }
        
        # Calculate cache efficiency (hits / total queries so far)
        total_queries = len(self.documentation["steps"])
        cache_hits = sum(1 for s in self.documentation["steps"] 
                        if s.get("details", {}).get("cache_hit", False))
        cache_analysis["cache_efficiency"] = (cache_hits / total_queries * 100) if total_queries > 0 else 0
        
        self.documentation["cache_performance"].append(cache_analysis)
        
        print(f"\n  Cache Performance:")
        print(f"    Cache Hit: {cache_hit}")
        print(f"    Tokens Saved (this step): {tokens_saved}")
        print(f"    Cache Size: {cache_analysis['cache_size']} entries")
        print(f"    Total Tokens Saved: {cache_analysis['total_tokens_saved']}")
        print(f"    Cache Efficiency: {cache_analysis['cache_efficiency']:.1f}%")
    
    def calculate_optimization(self, step_num: int, baseline_tokens: int, optimized_tokens: int, 
                              baseline_latency: float, optimized_latency: float):
        """Calculate optimization improvements."""
        token_savings = baseline_tokens - optimized_tokens
        token_savings_pct = (token_savings / baseline_tokens * 100) if baseline_tokens > 0 else 0
        
        latency_reduction = baseline_latency - optimized_latency
        latency_reduction_pct = (latency_reduction / baseline_latency * 100) if baseline_latency > 0 else 0
        
        optimization = {
            "step": step_num,
            "baseline_tokens": baseline_tokens,
            "optimized_tokens": optimized_tokens,
            "token_savings": token_savings,
            "token_savings_percent": token_savings_pct,
            "baseline_latency_ms": baseline_latency,
            "optimized_latency_ms": optimized_latency,
            "latency_reduction_ms": latency_reduction,
            "latency_reduction_percent": latency_reduction_pct,
            "overall_improvement": (token_savings_pct + latency_reduction_pct) / 2,
        }
        
        self.documentation["optimization_results"].append(optimization)
        
        print(f"\n  Optimization Results:")
        print(f"    Token Savings: {token_savings} tokens ({token_savings_pct:.1f}%)")
        print(f"    Latency Reduction: {latency_reduction:.2f} ms ({latency_reduction_pct:.1f}%)")
        print(f"    Overall Improvement: {optimization['overall_improvement']:.1f}%")
    
    def run_comprehensive_test(self):
        """Run comprehensive test with detailed documentation."""
        print("\n" + "="*80)
        print("COMPREHENSIVE PLATFORM TEST")
        print("="*80)
        print(f"Started: {datetime.now().isoformat()}")
        
        # Setup
        import os
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).parent / '.env')
        
        if not os.getenv("GEMINI_API_KEY"):
            os.environ["GEMINI_API_KEY"] = "AIzaSyCvSI80PtKuVejnkIiiNxjjN6PyRRngB1E"
        
        config = TokenomicsConfig.from_env()
        config.memory.use_semantic_cache = False
        config.memory.cache_size = 50
        config.orchestrator.default_token_budget = 3000
        
        platform = TokenomicsPlatform(config=config)
        
        # Complex test scenario
        test_queries = [
            {
                "query": "Explain the concept of machine learning, including supervised learning, unsupervised learning, and reinforcement learning. Provide examples of each and explain when to use which approach.",
                "complexity": "complex",
                "expected_tokens": 2000,
            },
            {
                "query": "What is Python programming language?",
                "complexity": "simple",
                "expected_tokens": 500,
            },
            {
                "query": "Explain the concept of machine learning, including supervised learning, unsupervised learning, and reinforcement learning. Provide examples of each and explain when to use which approach.",
                "complexity": "complex",
                "expected_tokens": 0,  # Should hit cache
            },
            {
                "query": "Compare and contrast neural networks and decision trees. Include their advantages, disadvantages, and use cases.",
                "complexity": "complex",
                "expected_tokens": 1500,
            },
            {
                "query": "What is Python programming language?",
                "complexity": "simple",
                "expected_tokens": 0,  # Should hit cache
            },
        ]
        
        baseline_tokens_total = 0
        optimized_tokens_total = 0
        baseline_latency_total = 0
        optimized_latency_total = 0
        
        step_num = 0
        
        for i, test_case in enumerate(test_queries, 1):
            step_num += 1
            query = test_case["query"]
            
            print(f"\n{'#'*80}")
            print(f"QUERY {i}: {query[:60]}...")
            print(f"{'#'*80}")
            
            # Step 1: Cache Check
            step_num += 1
            exact_match, semantic_matches = platform.memory.retrieve(query)
            cache_hit = exact_match is not None
            
            self.log_step(
                step_num,
                "Cache Check",
                {
                    "query": query[:100],
                    "cache_hit": cache_hit,
                    "cache_type": "exact" if cache_hit else "miss",
                    "tokens_saved_if_hit": exact_match.tokens_used if exact_match else 0,
                }
            )
            
            if cache_hit:
                # Cache hit - analyze cache performance
                tokens_saved = exact_match.tokens_used
                self.analyze_cache_performance(platform, step_num, True, tokens_saved)
                
                baseline_tokens_total += test_case["expected_tokens"] if test_case["expected_tokens"] > 0 else 1000
                optimized_tokens_total += 0
                baseline_latency_total += 5000  # Estimated baseline
                optimized_latency_total += 0  # Instant
                
                continue
            
            # Step 2: Bandit Strategy Selection
            step_num += 1
            strategy = platform.bandit.select_strategy()
            
            self.log_step(
                step_num,
                "Bandit Strategy Selection",
                {
                    "strategy_selected": strategy.arm_id if strategy else "none",
                    "strategy_config": {
                        "model": strategy.model if strategy else "none",
                        "max_tokens": strategy.max_tokens if strategy else 0,
                        "temperature": strategy.temperature if strategy else 0,
                    } if strategy else {},
                }
            )
            
            # Step 3: Orchestrator Planning
            step_num += 1
            plan = platform.orchestrator.plan_query(
                query,
                token_budget=config.orchestrator.default_token_budget,
            )
            
            self.log_step(
                step_num,
                "Orchestrator Query Planning",
                {
                    "query_complexity": plan.complexity.value,
                    "token_budget": plan.token_budget,
                    "model_selected": plan.model,
                    "use_retrieval": plan.use_retrieval,
                }
            )
            
            # Analyze token allocation
            self.analyze_token_allocation(plan, step_num)
            self.analyze_marginal_utility(plan.allocations, step_num)
            
            # Step 4: Execute Query
            step_num += 1
            start_time = time.time()
            result = platform.query(query, use_cache=True, use_bandit=True)
            elapsed = (time.time() - start_time) * 1000
            
            self.log_step(
                step_num,
                "Query Execution",
                {
                    "tokens_used": result["tokens_used"],
                    "latency_ms": result["latency_ms"],
                    "response_length": len(result["response"]),
                    "cache_hit": result["cache_hit"],
                    "strategy_used": result.get("strategy", "none"),
                }
            )
            
            # Analyze quality
            self.analyze_quality_improvement(query, plan, result, step_num)
            
            # Analyze bandit performance
            reward = result.get("reward", 0)
            self.analyze_bandit_performance(platform, step_num, result.get("strategy", "none"), reward)
            
            # Analyze cache performance
            tokens_saved = 0  # Will be saved for future queries
            self.analyze_cache_performance(platform, step_num, False, tokens_saved)
            
            # Calculate optimization (vs baseline)
            baseline_tokens = test_case["expected_tokens"] if test_case["expected_tokens"] > 0 else 2000
            baseline_latency = 8000  # Estimated baseline latency
            
            self.calculate_optimization(
                step_num,
                baseline_tokens,
                result["tokens_used"],
                baseline_latency,
                result["latency_ms"],
            )
            
            baseline_tokens_total += baseline_tokens
            optimized_tokens_total += result["tokens_used"]
            baseline_latency_total += baseline_latency
            optimized_latency_total += result["latency_ms"]
            
            # Track usage
            self.tracker.record_query(
                query=query,
                response=result["response"],
                tokens_used=result["tokens_used"],
                latency_ms=result["latency_ms"],
                cache_hit=result["cache_hit"],
                cache_type=result.get("cache_type", "none"),
                strategy=result.get("strategy", "none"),
                model=config.llm.model,
            )
        
        # Final Summary
        step_num += 1
        total_token_savings = baseline_tokens_total - optimized_tokens_total
        total_token_savings_pct = (total_token_savings / baseline_tokens_total * 100) if baseline_tokens_total > 0 else 0
        
        total_latency_reduction = baseline_latency_total - optimized_latency_total
        total_latency_reduction_pct = (total_latency_reduction / baseline_latency_total * 100) if baseline_latency_total > 0 else 0
        
        final_stats = platform.get_stats()
        bandit_final = platform.bandit.stats()
        
        self.log_step(
            step_num,
            "Final Summary",
            {
                "total_queries": len(test_queries),
                "cache_hits": sum(1 for q in test_queries if platform.memory.exact_cache.get_exact(q["query"])),
                "baseline_tokens": baseline_tokens_total,
                "optimized_tokens": optimized_tokens_total,
                "token_savings": total_token_savings,
                "token_savings_percent": total_token_savings_pct,
                "baseline_latency_ms": baseline_latency_total,
                "optimized_latency_ms": optimized_latency_total,
                "latency_reduction_ms": total_latency_reduction,
                "latency_reduction_percent": total_latency_reduction_pct,
                "cache_size": final_stats["memory"]["size"],
                "bandit_total_pulls": bandit_final["total_pulls"],
                "best_strategy": platform.bandit.get_best_strategy().arm_id if platform.bandit.get_best_strategy() else None,
            }
        )
        
        # Save documentation
        self.documentation["test_end"] = datetime.now().isoformat()
        self.documentation["final_summary"] = {
            "total_queries": len(test_queries),
            "total_steps": step_num,
            "token_savings": total_token_savings,
            "token_savings_percent": total_token_savings_pct,
            "latency_reduction": total_latency_reduction,
            "latency_reduction_percent": total_latency_reduction_pct,
        }
        
        with open("comprehensive_test_documentation.json", "w") as f:
            json.dump(self.documentation, f, indent=2)
        
        self.tracker.save_report()
        
        print(f"\n{'='*80}")
        print("COMPREHENSIVE TEST COMPLETE")
        print(f"{'='*80}")
        print(f"Documentation saved to: comprehensive_test_documentation.json")
        print(f"Usage report saved to: comprehensive_test_report.json")
        
        return self.documentation


if __name__ == "__main__":
    test = ComprehensiveTest()
    results = test.run_comprehensive_test()

