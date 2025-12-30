#!/usr/bin/env python3
"""Generate documentation from test results JSON file."""

import json
import sys
from pathlib import Path
from datetime import datetime

def generate_documentation(json_file: str):
    """Generate markdown documentation from JSON results."""
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    results = data.get('results', [])
    summary = data.get('summary', {})
    
    # Calculate totals
    total_baseline_cost = sum(r.get('baseline', {}).get('cost', 0) for r in results if 'baseline' in r)
    total_operating_cost = sum(r.get('costs', {}).get('operating_cost', 0) for r in results if 'costs' in r)
    total_evaluation_cost = sum(r.get('costs', {}).get('evaluation_cost', 0) for r in results if 'costs' in r)
    total_token_savings = sum(r.get('comparison', {}).get('token_savings', 0) for r in results if 'comparison' in r)
    total_baseline_tokens = sum(r.get('baseline', {}).get('tokens_used', 0) for r in results if 'baseline' in r)
    
    cache_hits = sum(1 for r in results if r.get('tokenomics', {}).get('cache_hit', False))
    
    # Component savings
    total_memory = sum(r.get('tokenomics', {}).get('component_savings', {}).get('memory_layer', 0) for r in results)
    total_orchestrator = sum(r.get('tokenomics', {}).get('component_savings', {}).get('orchestrator', 0) for r in results)
    total_bandit = sum(r.get('tokenomics', {}).get('component_savings', {}).get('bandit', 0) for r in results)
    
    # Generate markdown
    output_file = json_file.replace('.json', '.md')
    
    with open(output_file, 'w') as f:
        f.write("# Comprehensive Playground Test Results\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Test File:** {Path(json_file).name}\n\n")
        
        # Executive Summary
        f.write("## Executive Summary\n\n")
        f.write(f"- **Total Queries:** {len(results)}\n")
        f.write(f"- **Total Baseline Cost:** ${total_baseline_cost:.6f}\n")
        f.write(f"- **Total Operating Cost:** ${total_operating_cost:.6f} (what platform spent to answer queries)\n")
        f.write(f"- **Total Evaluation Cost:** ${total_evaluation_cost:.6f} (what we spent on judge to grade responses)\n")
        f.write(f"- **Total Cost Savings:** ${total_baseline_cost - total_operating_cost:.6f} ({((total_baseline_cost - total_operating_cost) / total_baseline_cost * 100) if total_baseline_cost > 0 else 0:.2f}%)\n")
        f.write(f"- **Total Token Savings:** {total_token_savings} tokens ({total_token_savings / total_baseline_tokens * 100 if total_baseline_tokens > 0 else 0:.2f}%)\n")
        f.write(f"- **Cache Hit Rate:** {cache_hits}/{len(results)} ({cache_hits/len(results)*100 if results else 0:.1f}%)\n\n")
        
        # Component Savings
        f.write("## Component Savings Breakdown\n\n")
        f.write(f"- **Memory Layer:** {total_memory} tokens\n")
        f.write(f"- **Orchestrator:** {total_orchestrator} tokens\n")
        f.write(f"- **Bandit:** {total_bandit} tokens\n")
        f.write(f"- **Total Component Savings:** {total_memory + total_orchestrator + total_bandit} tokens\n\n")
        
        # Detailed Results
        f.write("## Detailed Results\n\n")
        
        for result in results:
            if 'error' in result:
                f.write(f"### Query {result['query_num']}: {result['query'][:60]}...\n\n")
                f.write(f"**Error:** {result['error']}\n\n")
                continue
            
            baseline = result.get('baseline', {})
            tokenomics = result.get('tokenomics', {})
            comparison = result.get('comparison', {})
            costs = result.get('costs', {})
            judge = result.get('judge', {})
            
            f.write(f"### Query {result['query_num']}: {result['query']}\n\n")
            
            # Metrics
            f.write("#### Metrics\n\n")
            f.write(f"- **Token Savings:** {comparison.get('token_savings', 0)} tokens ({comparison.get('token_savings_percent', 0):.2f}%)\n")
            f.write(f"- **Cost Savings:** ${comparison.get('cost_savings', 0):.6f} ({comparison.get('cost_savings_percent', 0):.2f}%)\n")
            f.write(f"- **Latency Reduction:** {comparison.get('latency_reduction', 0):.2f}ms ({comparison.get('latency_reduction_percent', 0):.2f}%)\n\n")
            
            # Cost Breakdown
            f.write("#### Cost Breakdown (CRITICAL - Separated)\n\n")
            f.write(f"- **Baseline Cost:** ${baseline.get('cost', 0):.6f}\n")
            f.write(f"- **Operating Cost:** ${costs.get('operating_cost', 0):.6f} ⚡ (what platform spent to answer)\n")
            f.write(f"- **Evaluation Cost:** ${costs.get('evaluation_cost', 0):.6f} 🏆 (what we spent on judge to grade it)\n")
            f.write(f"- **Total Cost:** ${costs.get('total_cost', 0):.6f} (operating + evaluation)\n\n")
            
            # Baseline
            f.write("#### Baseline Response\n\n")
            f.write(f"- **Model:** {baseline.get('model', 'unknown')}\n")
            f.write(f"- **Tokens:** {baseline.get('tokens_used', 0)} ({baseline.get('input_tokens', 0)} in, {baseline.get('output_tokens', 0)} out)\n")
            f.write(f"- **Latency:** {baseline.get('latency_ms', 0):.2f}ms\n")
            f.write(f"- **Response:** {baseline.get('response', '')[:300]}...\n\n")
            
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
            f.write(f"- **Response:** {tokenomics.get('response', '')[:300]}...\n\n")
            
            # Component Savings
            comp_savings = tokenomics.get('component_savings', {})
            if comp_savings:
                f.write("#### Component Savings\n\n")
                f.write(f"- **Memory Layer:** {comp_savings.get('memory_layer', 0)} tokens\n")
                f.write(f"- **Orchestrator:** {comp_savings.get('orchestrator', 0)} tokens\n")
                f.write(f"- **Bandit:** {comp_savings.get('bandit', 0)} tokens\n")
                f.write(f"- **Total:** {comp_savings.get('total_savings', 0)} tokens\n\n")
            
            # Compression
            compression = tokenomics.get('compression_metrics', {})
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
            if judge.get('enabled'):
                f.write("#### Quality Judge\n\n")
                judge_result = judge.get('result', {})
                f.write(f"- **Winner:** {judge_result.get('winner', 'unknown')}\n")
                f.write(f"- **Confidence:** {judge_result.get('confidence', 0):.2f}\n")
                f.write(f"- **Explanation:** {judge_result.get('explanation', 'N/A')}\n")
                f.write(f"- **Judge Cost:** ${judge.get('cost', 0):.6f}\n")
                f.write(f"- **Judge Tokens:** {judge.get('cost_breakdown', {}).get('tokens_used', 0)}\n")
                f.write(f"- **Judge Model:** {judge.get('cost_breakdown', {}).get('model', 'unknown')}\n\n")
            
            f.write("---\n\n")
    
    print(f"Documentation generated: {output_file}")
    return output_file

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_documentation.py <json_file>")
        sys.exit(1)
    
    generate_documentation(sys.argv[1])






