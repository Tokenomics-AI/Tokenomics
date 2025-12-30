"""
Test semantic cache with similar (not identical) queries.

This test validates the tiered similarity matching:
- High similarity (>0.92): Direct return (0 tokens)
- Medium similarity (0.80-0.92): Use as context for shorter generation
- Low similarity (<0.80): Full LLM call
"""

# Disable TensorFlow before any imports (fixes Windows DLL issues)
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from tokenomics.core import TokenomicsPlatform
from tokenomics.config import TokenomicsConfig


def run_semantic_cache_test():
    """Run semantic cache test with similar queries."""
    print("=" * 80)
    print("SEMANTIC CACHE TEST - Similar Query Matching")
    print("=" * 80)
    print()
    print("This test demonstrates token savings from semantic similarity matching:")
    print("  - Same topic, different wording → should return cached response")
    print("  - Different topic → should make new LLM call")
    print()
    
    # Configure platform with semantic cache enabled
    # Make sure OPENAI_API_KEY is set in your .env file
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in your .env file.")
    
    config = TokenomicsConfig.from_env()
    config.llm.provider = "openai"
    config.llm.model = "gpt-4o-mini"
    config.llm.api_key = os.environ["OPENAI_API_KEY"]
    
    # Enable semantic cache with tiered matching
    config.memory.use_exact_cache = True
    config.memory.use_semantic_cache = True
    config.memory.similarity_threshold = 0.75  # Context matching threshold
    config.memory.direct_return_threshold = 0.85  # Direct return threshold
    config.memory.cache_size = 100
    
    print("[1/4] Initializing platform with semantic cache...")
    platform = TokenomicsPlatform(config=config)
    
    # Test queries grouped by topic (similar queries should match)
    query_groups = [
        # Group 1: Machine Learning
        {
            "topic": "Machine Learning",
            "queries": [
                "What is machine learning?",           # Original
                "Explain machine learning to me",      # Similar
                "Tell me about ML",                    # Similar (abbreviation)
                "How does machine learning work?",     # Similar
            ]
        },
        # Group 2: Neural Networks (different topic)
        {
            "topic": "Neural Networks",
            "queries": [
                "What are neural networks?",           # Original (new topic)
                "Explain neural nets",                 # Similar
                "How do neural networks work?",        # Similar
            ]
        },
        # Group 3: Python (different topic)  
        {
            "topic": "Python Programming",
            "queries": [
                "How to optimize Python code?",        # Original (new topic)
                "Python optimization techniques",      # Similar
            ]
        },
    ]
    
    results = []
    total_tokens = 0
    total_saved = 0
    api_calls = 0
    direct_returns = 0
    context_matches = 0
    
    print()
    print("[2/4] Running queries...")
    print("-" * 80)
    
    for group in query_groups:
        print(f"\n📁 Topic: {group['topic']}")
        print("-" * 40)
        
        for i, query in enumerate(group['queries']):
            print(f"\n  Query {i+1}: \"{query}\"")
            
            start_time = time.time()
            result = platform.query(query)
            latency = (time.time() - start_time) * 1000
            
            tokens = result.get("tokens_used", 0)
            cache_hit = result.get("cache_hit", False)
            cache_type = result.get("cache_type")
            similarity = result.get("similarity")
            
            total_tokens += tokens
            
            if cache_type == "exact":
                print(f"    ✓ EXACT CACHE HIT - 0 tokens")
                direct_returns += 1
                total_saved += 300  # Estimated savings
            elif cache_type == "semantic_direct":
                print(f"    ✓ SEMANTIC DIRECT RETURN (similarity: {similarity:.3f}) - 0 tokens")
                direct_returns += 1
                total_saved += 300  # Estimated savings
            elif cache_hit and not tokens:
                print(f"    ✓ CACHE HIT - 0 tokens")
                direct_returns += 1
                total_saved += 300
            elif cache_hit:
                print(f"    ~ Context-enhanced: {tokens} tokens, {latency:.0f}ms")
                context_matches += 1
                api_calls += 1
            else:
                print(f"    → Full API call: {tokens} tokens, {latency:.0f}ms")
                api_calls += 1
            
            results.append({
                "query": query,
                "topic": group['topic'],
                "tokens": tokens,
                "cache_type": cache_type,
                "similarity": similarity,
                "latency_ms": latency,
            })
    
    # Print summary
    print()
    print("=" * 80)
    print("[3/4] RESULTS SUMMARY")
    print("=" * 80)
    
    total_queries = len(results)
    
    print(f"""
📊 Query Statistics:
   Total queries:       {total_queries}
   API calls made:      {api_calls}
   Direct returns:      {direct_returns} (exact + semantic)
   Context matches:     {context_matches}

💰 Token Optimization:
   Tokens used:         {total_tokens}
   Estimated saved:     {total_saved} (from cache hits)
   
⚡ Cache Performance:
   Direct return rate:  {direct_returns}/{total_queries} ({100*direct_returns/total_queries:.1f}%)
""")
    
    # Breakdown by cache type
    exact_hits = sum(1 for r in results if r["cache_type"] == "exact")
    semantic_hits = sum(1 for r in results if r["cache_type"] == "semantic_direct")
    
    print(f"""🎯 Cache Hit Breakdown:
   Exact matches:       {exact_hits}
   Semantic direct:     {semantic_hits}
   Context-enhanced:    {context_matches}
   Full LLM calls:      {api_calls - context_matches}
""")
    
    # Memory stats
    stats = platform.get_stats()
    mem_stats = stats.get("memory", {})
    
    print("[4/4] Memory Layer Stats:")
    print(f"   Semantic cache enabled: {mem_stats.get('semantic_cache_enabled', False)}")
    if "preferences" in mem_stats:
        prefs = mem_stats["preferences"]
        print(f"   Learned preferences:")
        print(f"     - Tone: {prefs.get('tone')} (confidence: {prefs.get('confidence', 0):.2f})")
        print(f"     - Format: {prefs.get('format')}")
        print(f"     - Interactions: {prefs.get('interactions', 0)}")
    
    print()
    print("=" * 80)
    
    # Assessment
    if semantic_hits > 0:
        print("✅ SUCCESS: Semantic cache is working!")
        print("   Similar queries are returning cached responses without API calls.")
    elif direct_returns > 0:
        print("⚠️  PARTIAL: Only exact matches working, semantic matching may need tuning.")
        print("   Try lowering direct_return_threshold (currently 0.92)")
    else:
        print("❌ ISSUE: No cache hits detected. Check semantic cache configuration.")
    
    print("=" * 80)
    
    return results


def run_similarity_analysis():
    """Analyze similarity scores between queries."""
    print()
    print("=" * 80)
    print("SIMILARITY ANALYSIS")
    print("=" * 80)
    print()
    
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except ImportError:
        print("⚠️  sentence-transformers not installed, skipping similarity analysis")
        return
    
    print("Loading embedding model...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    # Test queries
    queries = [
        "What is machine learning?",
        "Explain machine learning to me",
        "Tell me about ML",
        "How does machine learning work?",
        "What are neural networks?",
        "Explain neural nets",
        "How to optimize Python code?",
    ]
    
    print("\nComputing embeddings...")
    embeddings = model.encode(queries)
    
    print("\nSimilarity Matrix (cosine similarity):")
    print("-" * 80)
    
    # Print header
    print(f"{'Query':<35} | ", end="")
    for i in range(len(queries)):
        print(f"Q{i+1:2d} ", end="")
    print()
    print("-" * 80)
    
    # Compute and print similarity matrix
    for i, q1 in enumerate(queries):
        print(f"{q1[:33]:<35} | ", end="")
        for j, q2 in enumerate(queries):
            # Cosine similarity
            sim = np.dot(embeddings[i], embeddings[j]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
            )
            
            # Color code: >0.92 green, 0.80-0.92 yellow, <0.80 red
            if sim >= 0.92:
                marker = "✓"
            elif sim >= 0.80:
                marker = "~"
            else:
                marker = " "
            
            print(f"{sim:.2f}{marker}", end="")
        print()
    
    print()
    print("Legend: ✓ = direct return (>0.92), ~ = context match (0.80-0.92)")
    print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test semantic cache")
    parser.add_argument("--analyze", action="store_true", help="Run similarity analysis only")
    args = parser.parse_args()
    
    if args.analyze:
        run_similarity_analysis()
    else:
        # Run main test
        results = run_semantic_cache_test()
        
        # Also run similarity analysis
        run_similarity_analysis()

