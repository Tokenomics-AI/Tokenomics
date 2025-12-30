#!/usr/bin/env python3
"""
Platform Profiler - Forensic Audit Tool

This is NOT a test. It's a diagnostic tool that breaks down the cost and latency
of a single query into granular components to identify overhead and "bloat".
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, Optional
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

load_dotenv(project_root / '.env')

from tokenomics.core import TokenomicsPlatform
from tokenomics.config import TokenomicsConfig
from tokenomics.orchestrator.orchestrator import QueryPlan


class PlatformProfiler:
    """Forensic audit tool for platform overhead analysis."""
    
    def __init__(self):
        """Initialize profiler."""
        self.config = TokenomicsConfig.from_env()
        self.platform = None
        self.profile_results = {}
        
    def initialize_platform(self):
        """Initialize TokenomicsPlatform."""
        print("🔧 Initializing TokenomicsPlatform...")
        start = time.time()
        self.platform = TokenomicsPlatform(config=self.config)
        init_time = (time.time() - start) * 1000
        print(f"   ✅ Platform initialized in {init_time:.2f}ms\n")
        
    def profile_orchestrator(self, query: str) -> Dict:
        """Profile the Orchestrator (The Planner)."""
        print("=" * 60)
        print("STEP 1: PROFILING THE ORCHESTRATOR (The Planner)")
        print("=" * 60)
        
        orchestrator = self.platform.orchestrator
        
        # Measure plan_query time
        start = time.time()
        plan = orchestrator.plan_query(query=query, token_budget=None)
        planning_time = (time.time() - start) * 1000
        
        print(f"\n⏱️  Planning Time: {planning_time:.2f}ms")
        
        # Build the prompt to see what actually gets sent
        # Check what system prompt is actually used (default or custom)
        # The orchestrator doesn't have a default system prompt, so we'll use a standard one
        system_prompt = "You are a helpful AI assistant."
        built_prompt = orchestrator.build_prompt(plan, system_prompt=system_prompt)
        
        # Also check if there's a default system prompt in the platform
        # For now, we'll use the standard one that would be used in practice
        
        # Count tokens for each component
        user_query_tokens = orchestrator.count_tokens(query)
        system_prompt_tokens = orchestrator.count_tokens(system_prompt) if system_prompt else 0
        context_tokens = 0
        if plan.retrieved_context:
            context_tokens = sum(orchestrator.count_tokens(ctx) for ctx in plan.retrieved_context)
        total_input_tokens = orchestrator.count_tokens(built_prompt)
        
        # Extract allocations
        allocations_dict = {}
        system_allocated_tokens = 0
        for alloc in plan.allocations:
            allocations_dict[alloc.component] = {
                'tokens': alloc.tokens,
                'priority': alloc.priority,
                'utility': alloc.utility
            }
            if alloc.component == 'system_prompt':
                system_allocated_tokens = alloc.tokens
        
        print(f"\n📋 QUERY PLAN:")
        print(f"   Complexity: {plan.complexity.value}")
        print(f"   Token Budget: {plan.token_budget:,} tokens")
        print(f"   Model: {plan.model}")
        print(f"   Allocations:")
        for component, data in allocations_dict.items():
            print(f"      - {component}: {data['tokens']:,} tokens (priority: {data['priority']}, utility: {data['utility']:.2f})")
        
        print(f"\n📝 SYSTEM PROMPT GENERATED:")
        print(f"   '{system_prompt}'")
        print(f"   ({system_prompt_tokens} tokens)")
        
        print(f"\n📊 TOKEN BREAKDOWN:")
        print(f"   User Query:    {user_query_tokens:,} tokens (\"{query}\")")
        print(f"   System Bloat: {system_prompt_tokens:,} tokens (\"{system_prompt}\")")
        if context_tokens > 0:
            print(f"   Context:       {context_tokens:,} tokens")
        else:
            print(f"   Context:       0 tokens (no context retrieved)")
        print(f"   Total Input:   {total_input_tokens:,} tokens")
        
        bloat_ratio = (system_prompt_tokens / total_input_tokens * 100) if total_input_tokens > 0 else 0
        print(f"   Bloat Ratio:   {bloat_ratio:.1f}% (System / Total)")
        
        print(f"\n🔍 FULL PROMPT THAT WILL BE SENT TO LLM:")
        print("-" * 60)
        print(built_prompt)
        print("-" * 60)
        
        return {
            'planning_time_ms': planning_time,
            'user_query_tokens': user_query_tokens,
            'system_prompt_tokens': system_prompt_tokens,
            'system_allocated_tokens': system_allocated_tokens,
            'context_tokens': context_tokens,
            'total_input_tokens': total_input_tokens,
            'bloat_ratio': bloat_ratio,
            'system_prompt': system_prompt,
            'full_prompt': built_prompt,
            'plan': plan,
        }
    
    def profile_bandit(self, query: str, plan: QueryPlan) -> Dict:
        """Profile the Bandit (The Decider)."""
        print("\n" + "=" * 60)
        print("STEP 2: PROFILING THE BANDIT (The Decider)")
        print("=" * 60)
        
        bandit = self.platform.bandit
        
        # Check if state file exists and when it was last accessed
        state_file_info = None
        if bandit.state_file:
            state_path = Path(bandit.state_file)
            if state_path.exists():
                state_file_info = {
                    'exists': True,
                    'path': str(state_path),
                    'size': state_path.stat().st_size,
                    'modified': time.ctime(state_path.stat().st_mtime),
                }
            else:
                state_file_info = {
                    'exists': False,
                    'path': str(state_path),
                }
        
        # Measure strategy selection time
        complexity = plan.complexity.value
        context_quality_score = plan.context_quality_score
        
        # Check if state file will be read
        state_read_time = 0
        if bandit.state_file and Path(bandit.state_file).exists():
            start = time.time()
            try:
                # Check file read time (simulate)
                with open(bandit.state_file, 'r') as f:
                    _ = json.load(f)
                state_read_time = (time.time() - start) * 1000
            except:
                pass
        
        # Measure select_strategy_cost_aware time
        start = time.time()
        strategy = bandit.select_strategy_cost_aware(
            query_complexity=complexity,
            context_quality_score=context_quality_score
        )
        routing_time = (time.time() - start) * 1000
        
        print(f"\n⏱️  Routing Time: {routing_time:.2f}ms")
        
        if state_file_info:
            print(f"\n💾 STATE FILE CHECK:")
            if state_file_info['exists']:
                print(f"   File: {state_file_info['path']}")
                print(f"   Size: {state_file_info['size']:,} bytes")
                print(f"   Last Modified: {state_file_info['modified']}")
                print(f"   Estimated Read Time: {state_read_time:.2f}ms")
                print(f"   ⚠️  State file access may contribute to latency")
            else:
                print(f"   File: {state_file_info['path']} (does not exist)")
                print(f"   ✅ No state file to read")
        
        if strategy:
            print(f"\n🎯 SELECTED STRATEGY:")
            print(f"   Arm ID: {strategy.arm_id}")
            print(f"   Model: {strategy.model}")
            print(f"   Max Tokens: {strategy.max_tokens}")
        else:
            print(f"\n   ⚠️  No strategy selected")
        
        # Check bandit stats
        print(f"\n📊 BANDIT STATISTICS:")
        print(f"   Total Pulls: {bandit.total_pulls}")
        print(f"   Query Count: {bandit.query_count}")
        print(f"   Available Arms: {len(bandit.arms)}")
        for arm_id, arm in bandit.arms.items():
            print(f"      - {arm_id}: {arm.pulls} pulls, avg reward: {arm.average_reward:.4f}")
        
        return {
            'routing_time_ms': routing_time,
            'state_file_read_time_ms': state_read_time,
            'state_file_info': state_file_info,
            'strategy': strategy.arm_id if strategy else None,
            'strategy_model': strategy.model if strategy else None,
        }
    
    def profile_execution(self, query: str, plan: QueryPlan, strategy) -> Dict:
        """Profile the Execution (The Doer) - LLM call only."""
        print("\n" + "=" * 60)
        print("STEP 3: PROFILING THE EXECUTION (The Doer)")
        print("=" * 60)
        
        orchestrator = self.platform.orchestrator
        llm_provider = self.platform.llm_provider
        
        # Build the prompt
        system_prompt = "You are a helpful AI assistant."
        prompt = orchestrator.build_prompt(plan, system_prompt=system_prompt)
        
        # Override model if strategy specifies
        model = strategy.model if strategy else plan.model
        
        # Measure just the LLM call time
        print(f"\n📤 Sending to LLM:")
        print(f"   Model: {model}")
        print(f"   Prompt Length: {len(prompt)} characters")
        print(f"   Estimated Tokens: {orchestrator.count_tokens(prompt):,}")
        
        start = time.time()
        try:
            llm_response = llm_provider.generate(
                prompt,
                max_tokens=strategy.max_tokens if strategy else 512,
                temperature=0.3
            )
            execution_time = (time.time() - start) * 1000
            success = True
            response_tokens = llm_response.tokens_used if hasattr(llm_response, 'tokens_used') else 0
            response_text = llm_response.text if hasattr(llm_response, 'text') else ""
        except Exception as e:
            execution_time = (time.time() - start) * 1000
            success = False
            response_tokens = 0
            response_text = f"ERROR: {str(e)}"
            print(f"   ❌ LLM call failed: {e}")
        
        print(f"\n⏱️  Execution Time: {execution_time:.2f}ms")
        print(f"   Response Tokens: {response_tokens:,}")
        print(f"   Success: {'✅' if success else '❌'}")
        
        return {
            'execution_time_ms': execution_time,
            'response_tokens': response_tokens,
            'success': success,
            'response_preview': response_text[:100] + "..." if len(response_text) > 100 else response_text,
        }
    
    def generate_tax_receipt(self, orchestrator_profile: Dict, bandit_profile: Dict, execution_profile: Dict):
        """Generate the "Tax Receipt" output."""
        print("\n" + "=" * 60)
        print("=== PLATFORM TAX RECEIPT ===")
        print("=" * 60)
        
        # Latency Breakdown
        planning_time = orchestrator_profile['planning_time_ms']
        routing_time = bandit_profile['routing_time_ms']
        execution_time = execution_profile['execution_time_ms']
        total_overhead = planning_time + routing_time
        total_time = planning_time + routing_time + execution_time
        
        print("\n1. LATENCY BREAKDOWN:")
        print(f"   Planning (Orchestrator): {planning_time:>8.2f} ms")
        print(f"   Routing (Bandit):        {routing_time:>8.2f} ms")
        print(f"   Execution (OpenAI):      {execution_time:>8.2f} ms")
        print(f"   TOTAL OVERHEAD:          {total_overhead:>8.2f} ms (Planning + Routing)")
        print(f"   TOTAL TIME:              {total_time:>8.2f} ms")
        print(f"   Overhead %:              {(total_overhead / total_time * 100):>7.1f}%")
        
        # Token Breakdown
        user_tokens = orchestrator_profile['user_query_tokens']
        system_tokens = orchestrator_profile['system_prompt_tokens']
        context_tokens = orchestrator_profile['context_tokens']
        total_input = orchestrator_profile['total_input_tokens']
        bloat_ratio = orchestrator_profile['bloat_ratio']
        system_allocated = orchestrator_profile.get('system_allocated_tokens', system_tokens)
        
        print("\n2. TOKEN BREAKDOWN:")
        print(f"   User Query:    {user_tokens:>6,} tokens (\"{orchestrator_profile.get('query', 'N/A')}\")")
        print(f"   System Bloat:  {system_tokens:>6,} tokens (The hidden instructions)")
        if system_allocated > system_tokens:
            print(f"   System Waste:  {system_allocated - system_tokens:>6,} tokens (allocated but unused)")
        if context_tokens > 0:
            print(f"   Context:        {context_tokens:>6,} tokens (Retrieved from memory)")
        else:
            print(f"   Context:        {context_tokens:>6,} tokens (No context retrieved)")
        print(f"   Total Input:   {total_input:>6,} tokens")
        print(f"   Bloat Ratio:   {bloat_ratio:>6.1f}% (System / Total)")
        
        # Config Check
        print("\n3. CONFIG CHECK:")
        print(f"   Compression Enabled: {self.config.memory.enable_llmlingua}")
        print(f"   Cache Enabled:       {self.config.memory.use_exact_cache or self.config.memory.use_semantic_cache}")
        print(f"   Exact Cache:         {self.config.memory.use_exact_cache}")
        print(f"   Semantic Cache:      {self.config.memory.use_semantic_cache}")
        print(f"   Bandit Enabled:      {self.config.bandit.auto_save}")
        print(f"   State File:          {self.config.bandit.state_file or 'None'}")
        
        # Additional Diagnostics
        print("\n4. ADDITIONAL DIAGNOSTICS:")
        if bandit_profile.get('state_file_info') and bandit_profile['state_file_info'].get('exists'):
            print(f"   State File Size:     {bandit_profile['state_file_info']['size']:,} bytes")
            print(f"   State File Read:     {bandit_profile.get('state_file_read_time_ms', 0):.2f} ms")
        print(f"   Selected Strategy:   {bandit_profile.get('strategy', 'None')}")
        print(f"   Strategy Model:      {bandit_profile.get('strategy_model', 'None')}")
        
        print("\n" + "=" * 60)
        
        # Summary
        print("\n💡 KEY INSIGHTS:")
        if total_overhead > 100:
            print(f"   ⚠️  High overhead: {total_overhead:.0f}ms spent on planning/routing")
        else:
            print(f"   ✅ Low overhead: {total_overhead:.0f}ms spent on planning/routing")
        
        if bloat_ratio > 20:
            print(f"   ⚠️  High system prompt bloat: {bloat_ratio:.1f}% of input tokens")
        else:
            print(f"   ✅ Reasonable system prompt: {bloat_ratio:.1f}% of input tokens")
        
        if bandit_profile.get('state_file_read_time_ms', 0) > 10:
            print(f"   ⚠️  State file read may be slow: {bandit_profile['state_file_read_time_ms']:.2f}ms")
        else:
            print(f"   ✅ State file access is fast")
        
        print("=" * 60 + "\n")
    
    def run_profile(self, query: str = "What is 2+2?"):
        """Run the complete profile."""
        print("\n" + "=" * 60)
        print("PLATFORM PROFILER - FORENSIC AUDIT")
        print("=" * 60)
        print(f"\n🔍 Profiling query: \"{query}\"")
        print(f"   This tool breaks down cost and latency into granular components.\n")
        
        # Initialize
        self.initialize_platform()
        
        # Step 1: Profile Orchestrator
        orchestrator_profile = self.profile_orchestrator(query)
        orchestrator_profile['query'] = query  # Store query for receipt
        
        # Step 2: Profile Bandit
        plan = orchestrator_profile['plan']
        bandit_profile = self.profile_bandit(query, plan)
        
        # Step 3: Profile Execution
        strategy = None
        if bandit_profile.get('strategy'):
            # Get the actual strategy object
            for arm_id, arm in self.platform.bandit.arms.items():
                if arm_id == bandit_profile['strategy']:
                    strategy = arm.strategy
                    break
        
        execution_profile = self.profile_execution(query, plan, strategy)
        
        # Generate Tax Receipt
        self.generate_tax_receipt(orchestrator_profile, bandit_profile, execution_profile)
        
        return {
            'orchestrator': orchestrator_profile,
            'bandit': bandit_profile,
            'execution': execution_profile,
        }


if __name__ == "__main__":
    profiler = PlatformProfiler()
    
    # Default query
    query = "What is 2+2?"
    
    # Allow command line override
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    
    try:
        results = profiler.run_profile(query)
        print("\n✅ Profiling complete!")
    except Exception as e:
        print(f"\n❌ Profiling failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)







