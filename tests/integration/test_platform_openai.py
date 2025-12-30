"""Test script to verify Tokenomics platform works with OpenAI API key."""

import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from tokenomics.core import TokenomicsPlatform
from tokenomics.config import TokenomicsConfig
from dotenv import load_dotenv

def main():
    """Test the platform with OpenAI."""
    print("=" * 60)
    print("Tokenomics Platform - OpenAI Test")
    print("=" * 60)
    print(f"Started at: {datetime.now().isoformat()}")
    print()
    
    # Load environment variables
    load_dotenv(Path(__file__).parent / '.env')
    
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        print("ERROR: OPENAI_API_KEY not found in environment variables")
        return False
    
    print(f"[OK] API Key found (starts with: {api_key[:10]}...)")
    print()
    
    # Create configuration
    try:
        config = TokenomicsConfig.from_env()
        # Override to use OpenAI
        config.llm.provider = "openai"
        config.llm.model = "gpt-4o-mini"
        config.llm.api_key = api_key
        
        # Enable semantic cache for testing
        config.memory.use_semantic_cache = True
        config.memory.cache_size = 100
        config.memory.similarity_threshold = 0.75
        config.memory.direct_return_threshold = 0.85
        
        print("[OK] Configuration created")
        print(f"  Provider: {config.llm.provider}")
        print(f"  Model: {config.llm.model}")
        print(f"  Semantic cache: {config.memory.use_semantic_cache}")
        print()
    except Exception as e:
        print(f"ERROR: Failed to create configuration: {e}")
        return False
    
    # Initialize platform
    try:
        print("Initializing platform...")
        platform = TokenomicsPlatform(config=config)
        print("[OK] Platform initialized successfully")
        print()
    except Exception as e:
        print(f"ERROR: Failed to initialize platform: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test queries
    test_queries = [
        "What is machine learning?",
        "Explain machine learning to me",  # Similar query - should hit semantic cache
        "What is machine learning?",  # Exact duplicate - should hit exact cache
    ]
    
    print("Running test queries...")
    print("-" * 60)
    
    total_tokens = 0
    cache_hits = 0
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n[{i}] Query: {query}")
        print("-" * 40)
        
        try:
            start_time = time.time()
            result = platform.query(query, use_cache=True, use_bandit=True)
            elapsed = (time.time() - start_time) * 1000
            
            tokens = result['tokens_used']
            cache_hit = result['cache_hit']
            cache_type = result.get('cache_type', 'none')
            similarity = result.get('similarity')
            
            total_tokens += tokens
            if cache_hit:
                cache_hits += 1
            
            print(f"[OK] Response received")
            print(f"  Tokens used: {tokens}")
            print(f"  Cache hit: {cache_hit} ({cache_type})")
            if similarity:
                print(f"  Similarity: {similarity:.3f}")
            print(f"  Latency: {elapsed:.2f} ms")
            print(f"  Response preview: {result['response'][:100]}...")
            
        except Exception as e:
            print(f"[ERROR] Query failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Total queries: {len(test_queries)}")
    print(f"Cache hits: {cache_hits} ({cache_hits/len(test_queries)*100:.1f}%)")
    print(f"Total tokens used: {total_tokens}")
    print(f"Tokens saved: {total_tokens if cache_hits > 0 else 0} (from cache hits)")
    
    # Get platform stats
    try:
        stats = platform.get_stats()
        print(f"\nPlatform Stats:")
        print(f"  Memory cache size: {stats['memory']['size']}")
        print(f"  Total tokens saved: {stats['memory'].get('total_tokens_saved', 0)}")
    except Exception as e:
        print(f"\nNote: Could not get platform stats: {e}")
    
    print("\n[OK] Platform test completed successfully!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

