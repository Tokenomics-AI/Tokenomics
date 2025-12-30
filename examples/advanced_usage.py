"""Advanced usage example with custom strategies and configuration."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenomics.core import TokenomicsPlatform
from tokenomics.config import TokenomicsConfig
from tokenomics.bandit import Strategy


def main():
    """Run advanced usage example."""
    # Create custom configuration
    config = TokenomicsConfig.from_env()
    
    # Customize memory settings
    config.memory.cache_size = 500
    config.memory.similarity_threshold = 0.8
    config.memory.use_semantic_cache = True
    
    # Customize orchestrator
    config.orchestrator.default_token_budget = 3000
    config.orchestrator.enable_multi_model_routing = True
    
    # Customize bandit
    config.bandit.algorithm = "ucb"
    config.bandit.reward_lambda = 0.0005  # Lower penalty for tokens
    
    # Initialize platform
    platform = TokenomicsPlatform(config=config)
    
    # Add custom strategies
    custom_strategies = [
        Strategy(
            arm_id="ultra_fast",
            model="gemini-flash",
            max_tokens=300,
            temperature=0.5,
            memory_mode="off",
            metadata={"use_case": "quick_answers"},
        ),
        Strategy(
            arm_id="high_quality",
            model="gemini-pro",
            max_tokens=2500,
            temperature=0.9,
            memory_mode="rich",
            rerank=True,
            n=3,
            metadata={"use_case": "detailed_analysis"},
        ),
    ]
    
    platform.bandit.add_strategies(custom_strategies)
    
    print("=" * 60)
    print("Tokenomics Platform - Advanced Usage")
    print("=" * 60)
    
    # Test different query types
    queries = [
        ("What is 2+2?", "quick"),
        ("Explain quantum computing in detail", "detailed"),
        ("What is 2+2?", "quick"),  # Should hit cache
    ]
    
    for query, query_type in queries:
        print(f"\nQuery: {query}")
        print(f"Type: {query_type}")
        print("-" * 60)
        
        result = platform.query(query)
        
        print(f"Response: {result['response'][:150]}...")
        print(f"Tokens: {result['tokens_used']}")
        print(f"Cache: {result['cache_hit']}")
        print(f"Strategy: {result['strategy']}")
        if result['reward']:
            print(f"Reward: {result['reward']:.4f}")
    
    # Show bandit learning
    print("\n" + "=" * 60)
    print("Bandit Statistics")
    print("=" * 60)
    stats = platform.bandit.stats()
    for arm_id, arm_stats in stats["arms"].items():
        print(f"{arm_id}:")
        print(f"  Pulls: {arm_stats['pulls']}")
        print(f"  Avg Reward: {arm_stats['average_reward']:.4f}")
    
    best = platform.bandit.get_best_strategy()
    if best:
        print(f"\nBest Strategy: {best.arm_id}")


if __name__ == "__main__":
    main()

