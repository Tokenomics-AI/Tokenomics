"""Basic usage example for Tokenomics platform."""

import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenomics.core import TokenomicsPlatform
from tokenomics.config import TokenomicsConfig
from usage_tracker import UsageTracker


def main():
    """Run basic usage example."""
    # Ensure .env is loaded
    import os
    from dotenv import load_dotenv
    from pathlib import Path
    load_dotenv(Path(__file__).parent.parent / '.env')
    
    # Set API key directly if not loaded from .env
    if not os.getenv("GEMINI_API_KEY"):
        os.environ["GEMINI_API_KEY"] = "AIzaSyCvSI80PtKuVejnkIiiNxjjN6PyRRngB1E"
    
    # Create configuration
    config = TokenomicsConfig.from_env()
    
    # Override for demo (use smaller cache, disable semantic cache to avoid TensorFlow issues)
    config.memory.cache_size = 100
    config.memory.use_semantic_cache = False  # Disable to avoid TensorFlow DLL issues on Windows
    config.orchestrator.default_token_budget = 2000
    
    # Try a different model if gemini-2.0-flash-exp has quota issues
    # config.llm.model = "gemini-pro"  # Uncomment to try a different model
    
    # Initialize platform
    platform = TokenomicsPlatform(config=config)
    
    # Initialize usage tracker
    tracker = UsageTracker(output_file="usage_report_basic.json")
    
    # Example queries
    queries = [
        "What is machine learning?",
        "Explain neural networks",
        "What is machine learning?",  # Duplicate - should hit cache
        "How do transformers work?",
    ]
    
    print("=" * 60)
    print("Tokenomics Platform - Basic Usage Example")
    print("=" * 60)
    print(f"Started at: {datetime.now().isoformat()}")
    print()
    
    for i, query in enumerate(queries, 1):
        print(f"\n[{i}] Query: {query}")
        print("-" * 60)
        
        start_time = time.time()
        result = platform.query(query)
        elapsed = (time.time() - start_time) * 1000
        
        # Record usage
        tracker.record_query(
            query=query,
            response=result['response'],
            tokens_used=result['tokens_used'],
            latency_ms=result['latency_ms'],
            cache_hit=result['cache_hit'],
            cache_type=result.get('cache_type', 'none'),
            strategy=result.get('strategy', 'none'),
            model=config.llm.model,
        )
        
        print(f"Response: {result['response'][:200]}...")
        print(f"Tokens used: {result['tokens_used']}")
        print(f"Cache hit: {result['cache_hit']} ({result.get('cache_type', 'none')})")
        print(f"Latency: {result['latency_ms']:.2f} ms")
        if result['strategy']:
            print(f"Strategy: {result['strategy']}")
        
        # Show cumulative stats
        if i > 1:
            summary = tracker.get_summary()
            print(f"\n  Cumulative: {summary['cache_hits']} cache hits, "
                  f"{summary['total_tokens_used']:,} tokens used, "
                  f"{summary['tokens_saved']:,} tokens saved ({summary['token_savings_rate_percent']:.1f}%)")
    
    # Save and print usage report
    tracker.save_report()
    tracker.print_summary()
    
    # Print platform statistics
    print("\n" + "=" * 60)
    print("Platform Statistics")
    print("=" * 60)
    stats = platform.get_stats()
    print(f"Memory cache size: {stats['memory']['size']}")
    print(f"Bandit pulls: {stats['bandit']['total_pulls']}")
    print(f"Best strategy: {platform.bandit.get_best_strategy().arm_id if platform.bandit.get_best_strategy() else 'N/A'}")
    print()
    print(f"Detailed usage report saved to: usage_report_basic.json")


if __name__ == "__main__":
    main()

