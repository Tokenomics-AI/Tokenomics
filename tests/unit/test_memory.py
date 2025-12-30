"""Tests for memory layer."""

import pytest
from tokenomics.memory import MemoryCache, SmartMemoryLayer


def test_exact_cache():
    """Test exact cache functionality."""
    cache = MemoryCache(max_size=10)
    
    # Store entry
    entry = cache.put("test query", "test response", tokens_used=100)
    assert entry.query == "test query"
    assert entry.response == "test response"
    
    # Retrieve exact match
    retrieved = cache.get_exact("test query")
    assert retrieved is not None
    assert retrieved.response == "test response"
    
    # Miss on different query
    assert cache.get_exact("different query") is None


def test_cache_eviction():
    """Test cache eviction."""
    cache = MemoryCache(max_size=3, eviction_policy="lru")
    
    # Fill cache
    cache.put("q1", "r1")
    cache.put("q2", "r2")
    cache.put("q3", "r3")
    
    # Access q1 to make it recently used
    cache.get_exact("q1")
    
    # Add q4 - should evict q2 (least recently used)
    cache.put("q4", "r4")
    
    assert cache.get_exact("q1") is not None
    assert cache.get_exact("q2") is None
    assert cache.get_exact("q3") is not None
    assert cache.get_exact("q4") is not None


def test_smart_memory_layer():
    """Test smart memory layer."""
    memory = SmartMemoryLayer(
        use_exact_cache=True,
        use_semantic_cache=False,  # Disable for faster tests
        cache_size=10,
    )
    
    # Store
    memory.store("test query", "test response", tokens_used=100)
    
    # Retrieve exact
    exact, semantic = memory.retrieve("test query")
    assert exact is not None
    assert exact.response == "test response"
    
    # Clear
    memory.clear()
    exact, semantic = memory.retrieve("test query")
    assert exact is None

