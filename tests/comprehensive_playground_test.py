#!/usr/bin/env python3
"""
Comprehensive Playground Test with Detailed Documentation
Tests all queries and generates detailed report with cost separation.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenomics.core import TokenomicsPlatform
from tokenomics.config import TokenomicsConfig

# Model pricing per 1M tokens
MODEL_PRICING = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "gemini-flash": {"input": 0.075, "output": 0.30},
    "gemini-pro": {"input": 1.25, "output": 5.00},
}


def calculate_cost(tokens_used: int, input_tokens: int, output_tokens: int, model: str) -> Dict[str, float]:
    """Calculate cost breakdown for a query."""
    pricing = MODEL_PRICING.get(model, {"input": 0.15, "output": 0.60})
    
    # If we have input/output breakdown, use it
    if input_tokens > 0 or output_tokens > 0:
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        total_cost = input_cost + output_cost
    else:
        # Estimate 50/50 split if not available
        estimated_input = tokens_used // 2
        estimated_output = tokens_used - estimated_input
        input_cost = (estimated_input / 1_000_000) * pricing["input"]
        output_cost = (estimated_output / 1_000_000) * pricing["output"]
        total_cost = input_cost + output_cost
    
    return {
        "total_cost": total_cost,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "input_tokens": input_tokens if input_tokens > 0 else tokens_used // 2,
        "output_tokens": output_tokens if output_tokens > 0 else tokens_used - (tokens_used // 2),
    }


def extract_judge_costs(platform: TokenomicsPlatform, result: Dict, baseline_result: Dict) -> Dict[str, Any]:
    """Extract judge costs separately from operating costs."""
    judge_costs = {
        "judge_tokens_used": 0,
        "judge_input_tokens": 0,
        "judge_output_tokens": 0,
        "judge_cost": 0.0,
        "judge_model": None,
    }
    
    # Check if quality judge was used
    if result.get("quality_judge"):
        # Try to get judge provider to determine model
        if hasattr(platform, 'quality_judge') and platform.quality_judge:
            judge_config = platform.quality_judge.config
            judge_model = judge_config.model
            judge_provider = judge_config.provider
            
            # Try to get actual token usage from judge provider if available
            # The judge provider should track its usage
            judge_input_tokens = 0
            judge_output_tokens = 0
            
            # Check if judge provider has usage tracking
            if hasattr(platform.quality_judge, 'judge_provider') and platform.quality_judge.judge_provider:
                # Try to get usage from provider's last call
                # Most providers track this in their response
                pass  # Will estimate if not available
            
            # Estimate judge tokens if not available
            if judge_input_tokens == 0:
                baseline_response = baseline_result.get("response", "")
                optimized_response = result.get("response", "")
                query = result.get("query", "")
                
                # Rough estimate based on content length
                # Judge prompt includes: query + baseline + optimized + instructions (~200 tokens)
                estimated_judge_input = (
                    len(query) // 4 + 
                    len(baseline_response[:500]) // 4 +  # Truncated to 500 chars
                    len(optimized_response[:500]) // 4 +  # Truncated to 500 chars
                    200  # Instructions
                )
                estimated_judge_output = 100  # Typical judge response
                
                judge_input_tokens = estimated_judge_input
                judge_output_tokens = estimated_judge_output
            
            judge_costs["judge_model"] = judge_model
            judge_costs["judge_input_tokens"] = judge_input_tokens
            judge_costs["judge_output_tokens"] = judge_output_tokens
            judge_costs["judge_tokens_used"] = judge_input_tokens + judge_output_tokens
            
            # Calculate judge cost
            if judge_provider == "openai":
                # Use OpenAI pricing
                judge_pricing = MODEL_PRICING.get(judge_model, {"input": 2.50, "output": 10.00})
            else:
                # Use Gemini pricing
                judge_pricing = MODEL_PRICING.get(judge_model, {"input": 1.25, "output": 5.00})
            
            judge_costs["judge_cost"] = (
                (judge_input_tokens / 1_000_000) * judge_pricing["input"] +
                (judge_output_tokens / 1_000_000) * judge_pricing["output"]
            )
    
    return judge_costs


def run_comprehensive_test():
    """Run comprehensive test with all queries."""
    
    queries = [
        "What is 2+2?",
        "What is the capital of France?",
        "Explain what Python is in one sentence.",
        "What time is it?",
        "Define machine learning briefly.",
        "Explain how neural networks work, including the basic concepts of neurons, layers, and activation functions.",
        "What are the main differences between supervised and unsupervised learning? Give examples of each.",
        "Describe the process of photosynthesis in plants, including the key steps and inputs/outputs.",
        "Explain the concept of recursion in programming with a simple example.",
        "What is the difference between REST and GraphQL APIs? When would you use each?",
        "Write a detailed explanation of quantum computing, including qubits, superposition, entanglement, quantum gates, quantum algorithms like Shor's algorithm and Grover's algorithm, quantum error correction, and applications in cryptography and optimization problems. Explain the differences between classical and quantum computing paradigms.",
        "Provide a comprehensive analysis of the Transformer architecture in deep learning, including attention mechanisms, self-attention, multi-head attention, positional encoding, encoder-decoder structure, and how it revolutionized natural language processing. Compare it to previous architectures like RNNs and LSTMs.",
        "Explain the complete lifecycle of a software development project from requirements gathering to deployment, including methodologies (Agile, Waterfall, DevOps), version control, testing strategies, CI/CD pipelines, and monitoring. Include best practices for each phase.",
        "What is machine learning?",  # First run
        "Tell me about machine learning",  # Should trigger semantic cache
        "Explain machine learning",  # Should trigger semantic cache
        "What is machine learning?",  # Exact match - should be instant cache hit
        "Write a comprehensive guide to building a web application using React, including setting up the development environment, component architecture, state management with Redux, routing with React Router, API integration, authentication, error handling, testing with Jest, deployment strategies, performance optimization techniques, and best practices for maintainable code.",
        "Provide a detailed explanation of the TCP/IP protocol stack, including all layers (Application, Transport, Network, Data Link, Physical), protocols at each layer, how data flows through the stack, packet structure, connection establishment, error handling, flow control, congestion control, and how it compares to the OSI model.",
        "Write a short story about an AI that learns to paint.",
        "Design a marketing strategy for a new SaaS product targeting small businesses.",
        "Create a workout plan for someone who wants to build muscle and improve cardiovascular health.",
    ]
    
    print("=" * 80)
    print("COMPREHENSIVE PLAYGROUND TEST")
    print("=" * 80)
    print(f"Total queries: {len(queries)}")
    print(f"Start time: {datetime.now().isoformat()}")
    print()
    
    # Initialize platform
    try:
        platform = TokenomicsPlatform()
        print("✅ Platform initialized")
    except Exception as e:
        print(f"❌ Failed to initialize platform: {e}")
        return
    
    results = []
    total_operating_cost = 0.0
    total_evaluation_cost = 0.0
    total_baseline_cost = 0.0
    
    # Create output directory early
    output_dir = Path("test_results")
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_file = output_dir / f"comprehensive_playground_test_{timestamp}.json"
    
    for i, query in enumerate(queries, 1):
        print(f"\n[{i}/{len(queries)}] Processing: {query[:60]}...")
        print(f"Progress: {i-1}/{len(queries)} completed")
        
        try:
            # Run baseline
            baseline_start = time.time()
            baseline_result = platform.query(
                query=query,
                use_cache=False,
                use_bandit=False,
                use_compression=False,
                use_cost_aware_routing=False,
            )
            baseline_latency = (time.time() - baseline_start) * 1000
            
            # Run optimized
            optimized_start = time.time()
            optimized_result = platform.query(
                query=query,
                use_cache=True,
                use_bandit=True,
                use_compression=True,
                use_cost_aware_routing=True,
            )
            optimized_latency = (time.time() - optimized_start) * 1000
            
            # Calculate costs
            baseline_costs = calculate_cost(
                baseline_result.get("tokens_used", 0),
                baseline_result.get("input_tokens", 0),
                baseline_result.get("output_tokens", 0),
                baseline_result.get("model", "gpt-4o"),
            )
            
            optimized_costs = calculate_cost(
                optimized_result.get("tokens_used", 0),
                optimized_result.get("input_tokens", 0),
                optimized_result.get("output_tokens", 0),
                optimized_result.get("model", "gpt-4o-mini"),
            )
            
            # Extract judge costs (separate from operating costs)
            judge_costs = extract_judge_costs(platform, optimized_result, baseline_result)
            
            # Operating cost = optimized query cost (what platform spent to answer)
            operating_cost = optimized_costs["total_cost"]
            
            # Evaluation cost = judge cost (what we spent to grade it)
            evaluation_cost = judge_costs["judge_cost"]
            
            # Calculate savings
            token_savings = baseline_result.get("tokens_used", 0) - optimized_result.get("tokens_used", 0)
            token_savings_percent = (token_savings / baseline_result.get("tokens_used", 1) * 100) if baseline_result.get("tokens_used", 0) > 0 else 0
            
            cost_savings = baseline_costs["total_cost"] - operating_cost
            cost_savings_percent = (cost_savings / baseline_costs["total_cost"] * 100) if baseline_costs["total_cost"] > 0 else 0
            
            latency_reduction = baseline_latency - optimized_latency
            latency_reduction_percent = (latency_reduction / baseline_latency * 100) if baseline_latency > 0 else 0
            
            # Extract component savings
            component_savings = optimized_result.get("component_savings", {})
            
            # Extract compression metrics
            compression_metrics = optimized_result.get("compression_metrics", {})
            
            # Extract quality judge result
            quality_judge = optimized_result.get("quality_judge")
            
            # Build result
            result = {
                "query_num": i,
                "query": query,
                "baseline": {
                    "response": baseline_result.get("response", ""),
                    "tokens_used": baseline_result.get("tokens_used", 0),
                    "input_tokens": baseline_result.get("input_tokens", 0),
                    "output_tokens": baseline_result.get("output_tokens", 0),
                    "latency_ms": baseline_latency,
                    "model": baseline_result.get("model", "gpt-4o"),
                    "cost": baseline_costs["total_cost"],
                    "cost_breakdown": {
                        "input_cost": baseline_costs["input_cost"],
                        "output_cost": baseline_costs["output_cost"],
                    },
                },
                "tokenomics": {
                    "response": optimized_result.get("response", ""),
                    "tokens_used": optimized_result.get("tokens_used", 0),
                    "input_tokens": optimized_result.get("input_tokens", 0),
                    "output_tokens": optimized_result.get("output_tokens", 0),
                    "latency_ms": optimized_latency,
                    "model": optimized_result.get("model", "gpt-4o-mini"),
                    "cache_hit": optimized_result.get("cache_hit", False),
                    "cache_type": optimized_result.get("cache_type", "none"),
                    "similarity": optimized_result.get("similarity"),
                    "strategy": optimized_result.get("strategy", "none"),
                    "complexity": optimized_result.get("query_type", "unknown"),
                    "cost": operating_cost,
                    "cost_breakdown": {
                        "input_cost": optimized_costs["input_cost"],
                        "output_cost": optimized_costs["output_cost"],
                    },
                    "component_savings": component_savings,
                    "compression_metrics": compression_metrics,
                    "plan": {
                        "complexity": optimized_result.get("plan", {}).complexity.value if optimized_result.get("plan") and hasattr(optimized_result.get("plan"), "complexity") else "unknown",
                        "token_budget": optimized_result.get("plan", {}).token_budget if optimized_result.get("plan") else 0,
                        "model": optimized_result.get("plan", {}).model if optimized_result.get("plan") else "unknown",
                    } if optimized_result.get("plan") else None,
                },
                "judge": {
                    "enabled": quality_judge is not None,
                    "cost": evaluation_cost,
                    "cost_breakdown": {
                        "tokens_used": judge_costs["judge_tokens_used"],
                        "input_tokens": judge_costs["judge_input_tokens"],
                        "output_tokens": judge_costs["judge_output_tokens"],
                        "model": judge_costs["judge_model"],
                    },
                    "result": quality_judge if quality_judge else None,
                },
                "comparison": {
                    "token_savings": token_savings,
                    "token_savings_percent": round(token_savings_percent, 2),
                    "cost_savings": round(cost_savings, 6),
                    "cost_savings_percent": round(cost_savings_percent, 2),
                    "latency_reduction": round(latency_reduction, 2),
                    "latency_reduction_percent": round(latency_reduction_percent, 2),
                },
                "costs": {
                    "operating_cost": round(operating_cost, 6),  # What platform spent to answer
                    "evaluation_cost": round(evaluation_cost, 6),  # What we spent to grade it
                    "total_cost": round(operating_cost + evaluation_cost, 6),  # Total (for reference)
                    "baseline_cost": round(baseline_costs["total_cost"], 6),
                },
            }
            
            results.append(result)
            
            # Accumulate costs
            total_operating_cost += operating_cost
            total_evaluation_cost += evaluation_cost
            total_baseline_cost += baseline_costs["total_cost"]
            
            print(f"  ✅ Baseline: {baseline_result.get('tokens_used', 0)} tokens, ${baseline_costs['total_cost']:.6f}, {baseline_latency:.0f}ms")
            print(f"  ✅ Optimized: {optimized_result.get('tokens_used', 0)} tokens, ${operating_cost:.6f}, {optimized_latency:.0f}ms")
            print(f"  💰 Savings: {token_savings} tokens ({token_savings_percent:.1f}%), ${cost_savings:.6f}")
            if quality_judge:
                print(f"  🏆 Quality: {quality_judge.get('winner', 'unknown')} (confidence: {quality_judge.get('confidence', 0):.2f})")
            if optimized_result.get("cache_hit"):
                print(f"  🎯 Cache: {optimized_result.get('cache_type', 'hit')}")
            
            # Save progress incrementally
            try:
                with open(json_file, "w") as f:
                    json.dump({
                        "summary": {"progress": f"{i}/{len(queries)}"},
                        "results": results,
                    }, f, indent=2, default=str)
            except Exception as save_error:
                print(f"  ⚠️  Warning: Could not save progress: {save_error}")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "query_num": i,
                "query": query,
                "error": str(e),
            })
    
    # Calculate summary
    total_token_savings = sum(r.get("comparison", {}).get("token_savings", 0) for r in results if "comparison" in r)
    total_cost_savings = total_baseline_cost - total_operating_cost
    total_cost_savings_percent = (total_cost_savings / total_baseline_cost * 100) if total_baseline_cost > 0 else 0
    
    cache_hits = sum(1 for r in results if r.get("tokenomics", {}).get("cache_hit", False))
    cache_hit_rate = (cache_hits / len(results)) * 100 if results else 0
    
    # Component savings breakdown
    total_memory_savings = sum(r.get("tokenomics", {}).get("component_savings", {}).get("memory_layer", 0) for r in results)
    total_orchestrator_savings = sum(r.get("tokenomics", {}).get("component_savings", {}).get("orchestrator", 0) for r in results)
    total_bandit_savings = sum(r.get("tokenomics", {}).get("component_savings", {}).get("bandit", 0) for r in results)
    
    summary = {
        "test_info": {
            "start_time": datetime.now().isoformat(),
            "total_queries": len(queries),
            "successful_queries": len([r for r in results if "error" not in r]),
        },
        "costs": {
            "total_baseline_cost": round(total_baseline_cost, 6),
            "total_operating_cost": round(total_operating_cost, 6),
            "total_evaluation_cost": round(total_evaluation_cost, 6),
            "total_cost_savings": round(total_cost_savings, 6),
            "total_cost_savings_percent": round(total_cost_savings_percent, 2),
        },
        "tokens": {
            "total_token_savings": total_token_savings,
            "total_token_savings_percent": round((total_token_savings / sum(r.get("baseline", {}).get("tokens_used", 0) for r in results if "baseline" in r)) * 100, 2) if sum(r.get("baseline", {}).get("tokens_used", 0) for r in results if "baseline" in r) > 0 else 0,
        },
        "cache": {
            "cache_hits": cache_hits,
            "cache_hit_rate": round(cache_hit_rate, 2),
        },
        "component_savings": {
            "memory_layer": total_memory_savings,
            "orchestrator": total_orchestrator_savings,
            "bandit": total_bandit_savings,
            "total": total_memory_savings + total_orchestrator_savings + total_bandit_savings,
        },
    }
    
    # Save results (output_dir and timestamp already created above)
    with open(json_file, "w") as f:
        json.dump({
            "summary": summary,
            "results": results,
        }, f, indent=2, default=str)
    
    # Generate markdown report
    md_file = output_dir / f"comprehensive_playground_test_{timestamp}.md"
    generate_markdown_report(summary, results, md_file)
    
    print("\n" + "=" * 80)
    print("TEST COMPLETED")
    print("=" * 80)
    print(f"Total Baseline Cost: ${total_baseline_cost:.6f}")
    print(f"Total Operating Cost: ${total_operating_cost:.6f}")
    print(f"Total Evaluation Cost: ${total_evaluation_cost:.6f}")
    print(f"Total Cost Savings: ${total_cost_savings:.6f} ({total_cost_savings_percent:.2f}%)")
    print(f"Total Token Savings: {total_token_savings} tokens")
    print(f"Cache Hit Rate: {cache_hit_rate:.1f}%")
    print(f"\nResults saved to:")
    print(f"  JSON: {json_file}")
    print(f"  Markdown: {md_file}")


def generate_markdown_report(summary: Dict, results: List[Dict], output_file: Path):
    """Generate detailed markdown report."""
    
    with open(output_file, "w") as f:
        f.write("# Comprehensive Playground Test Results\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Executive Summary
        f.write("## Executive Summary\n\n")
        f.write(f"- **Total Queries:** {summary['test_info']['total_queries']}\n")
        f.write(f"- **Successful Queries:** {summary['test_info']['successful_queries']}\n")
        f.write(f"- **Total Baseline Cost:** ${summary['costs']['total_baseline_cost']:.6f}\n")
        f.write(f"- **Total Operating Cost:** ${summary['costs']['total_operating_cost']:.6f}\n")
        f.write(f"- **Total Evaluation Cost:** ${summary['costs']['total_evaluation_cost']:.6f}\n")
        f.write(f"- **Total Cost Savings:** ${summary['costs']['total_cost_savings']:.6f} ({summary['costs']['total_cost_savings_percent']:.2f}%)\n")
        f.write(f"- **Total Token Savings:** {summary['tokens']['total_token_savings']} tokens ({summary['tokens']['total_token_savings_percent']:.2f}%)\n")
        f.write(f"- **Cache Hit Rate:** {summary['cache']['cache_hit_rate']:.2f}%\n\n")
        
        # Component Savings
        f.write("## Component Savings Breakdown\n\n")
        f.write(f"- **Memory Layer:** {summary['component_savings']['memory_layer']} tokens\n")
        f.write(f"- **Orchestrator:** {summary['component_savings']['orchestrator']} tokens\n")
        f.write(f"- **Bandit:** {summary['component_savings']['bandit']} tokens\n")
        f.write(f"- **Total:** {summary['component_savings']['total']} tokens\n\n")
        
        # Detailed Results
        f.write("## Detailed Results\n\n")
        
        for result in results:
            if "error" in result:
                f.write(f"### Query {result['query_num']}: {result['query'][:60]}...\n\n")
                f.write(f"**Error:** {result['error']}\n\n")
                continue
            
            baseline = result.get("baseline", {})
            tokenomics = result.get("tokenomics", {})
            comparison = result.get("comparison", {})
            costs = result.get("costs", {})
            judge = result.get("judge", {})
            
            f.write(f"### Query {result['query_num']}: {result['query']}\n\n")
            
            # Metrics
            f.write("#### Metrics\n\n")
            f.write(f"- **Token Savings:** {comparison.get('token_savings', 0)} tokens ({comparison.get('token_savings_percent', 0):.2f}%)\n")
            f.write(f"- **Cost Savings:** ${comparison.get('cost_savings', 0):.6f} ({comparison.get('cost_savings_percent', 0):.2f}%)\n")
            f.write(f"- **Latency Reduction:** {comparison.get('latency_reduction', 0):.2f}ms ({comparison.get('latency_reduction_percent', 0):.2f}%)\n\n")
            
            # Costs
            f.write("#### Cost Breakdown\n\n")
            f.write(f"- **Baseline Cost:** ${baseline.get('cost', 0):.6f}\n")
            f.write(f"- **Operating Cost:** ${costs.get('operating_cost', 0):.6f} (what platform spent to answer)\n")
            f.write(f"- **Evaluation Cost:** ${costs.get('evaluation_cost', 0):.6f} (what we spent to grade it)\n")
            f.write(f"- **Total Cost:** ${costs.get('total_cost', 0):.6f}\n\n")
            
            # Baseline
            f.write("#### Baseline Response\n\n")
            f.write(f"- **Model:** {baseline.get('model', 'unknown')}\n")
            f.write(f"- **Tokens:** {baseline.get('tokens_used', 0)} ({baseline.get('input_tokens', 0)} in, {baseline.get('output_tokens', 0)} out)\n")
            f.write(f"- **Latency:** {baseline.get('latency_ms', 0):.2f}ms\n")
            f.write(f"- **Response:** {baseline.get('response', '')[:200]}...\n\n")
            
            # Tokenomics
            f.write("#### Tokenomics Response\n\n")
            f.write(f"- **Model:** {tokenomics.get('model', 'unknown')}\n")
            f.write(f"- **Strategy:** {tokenomics.get('strategy', 'none')}\n")
            f.write(f"- **Complexity:** {tokenomics.get('complexity', 'unknown')}\n")
            f.write(f"- **Tokens:** {tokenomics.get('tokens_used', 0)} ({tokenomics.get('input_tokens', 0)} in, {tokenomics.get('output_tokens', 0)} out)\n")
            f.write(f"- **Latency:** {tokenomics.get('latency_ms', 0):.2f}ms\n")
            f.write(f"- **Cache:** {tokenomics.get('cache_type', 'none')} (hit: {tokenomics.get('cache_hit', False)})\n")
            if tokenomics.get('similarity'):
                f.write(f"- **Similarity:** {tokenomics.get('similarity'):.3f}\n")
            f.write(f"- **Response:** {tokenomics.get('response', '')[:200]}...\n\n")
            
            # Component Savings
            comp_savings = tokenomics.get("component_savings", {})
            if comp_savings:
                f.write("#### Component Savings\n\n")
                f.write(f"- **Memory Layer:** {comp_savings.get('memory_layer', 0)} tokens\n")
                f.write(f"- **Orchestrator:** {comp_savings.get('orchestrator', 0)} tokens\n")
                f.write(f"- **Bandit:** {comp_savings.get('bandit', 0)} tokens\n")
                f.write(f"- **Total:** {comp_savings.get('total_savings', 0)} tokens\n\n")
            
            # Compression
            compression = tokenomics.get("compression_metrics", {})
            if compression:
                f.write("#### Compression Metrics\n\n")
                f.write(f"- **Query Compressed:** {compression.get('query_compressed', False)}\n")
                f.write(f"- **Context Compressed:** {compression.get('context_compressed', False)}\n")
                if compression.get('query_compression_ratio'):
                    f.write(f"- **Query Compression Ratio:** {compression.get('query_compression_ratio'):.2f}\n")
                if compression.get('context_compression_ratio'):
                    f.write(f"- **Context Compression Ratio:** {compression.get('context_compression_ratio'):.2f}\n")
                f.write("\n")
            
            # Quality Judge
            if judge.get("enabled"):
                f.write("#### Quality Judge\n\n")
                judge_result = judge.get("result", {})
                f.write(f"- **Winner:** {judge_result.get('winner', 'unknown')}\n")
                f.write(f"- **Confidence:** {judge_result.get('confidence', 0):.2f}\n")
                f.write(f"- **Explanation:** {judge_result.get('explanation', 'N/A')}\n")
                f.write(f"- **Judge Cost:** ${judge.get('cost', 0):.6f}\n")
                f.write(f"- **Judge Tokens:** {judge.get('cost_breakdown', {}).get('tokens_used', 0)}\n")
                f.write(f"- **Judge Model:** {judge.get('cost_breakdown', {}).get('model', 'unknown')}\n\n")
            
            f.write("---\n\n")


if __name__ == "__main__":
    run_comprehensive_test()






