"""Caching layer for exact and semantic matching."""

import hashlib
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from collections import OrderedDict
import structlog

logger = structlog.get_logger()


@dataclass
class CacheEntry:
    """A single cache entry."""
    query: str
    response: str
    query_hash: str
    query_embedding: Optional[List[float]] = None
    metadata: Dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    tokens_used: int = 0

    def touch(self):
        """Update access metadata."""
        self.last_accessed = datetime.now()
        self.access_count += 1


class MemoryCache:
    """Two-layer cache: exact match + semantic similarity."""
    
    def __init__(
        self,
        max_size: int = 1000,
        eviction_policy: str = "lru",
        ttl_seconds: Optional[int] = None,
        similarity_threshold: float = 0.85,
        persistent_storage: Optional[str] = None,
    ):
        """
        Initialize memory cache.
        
        Args:
            max_size: Maximum number of entries
            eviction_policy: "lru" or "time-based"
            ttl_seconds: Time-to-live in seconds (None = no expiry)
            similarity_threshold: Minimum similarity for semantic matches
            persistent_storage: Path to SQLite database for persistence (None = in-memory only)
        """
        self.max_size = max_size
        self.eviction_policy = eviction_policy
        self.ttl_seconds = ttl_seconds
        self.similarity_threshold = similarity_threshold
        
        # Exact match cache: hash -> CacheEntry
        self._exact_cache: Dict[str, CacheEntry] = {}
        
        # LRU ordering for eviction
        self._lru_order: OrderedDict[str, None] = OrderedDict()
        
        # Persistent storage (optional)
        self.persistent_cache = None
        if persistent_storage:
            try:
                from .persistent_cache import PersistentCache
                self.persistent_cache = PersistentCache(persistent_storage)
                # Load existing entries from persistent storage
                self._load_from_persistent()
                logger.info("Loaded cache entries from persistent storage")
            except Exception as e:
                logger.warning("Failed to initialize persistent cache", error=str(e))
        
        logger.info(
            "MemoryCache initialized",
            max_size=max_size,
            eviction_policy=eviction_policy,
            ttl_seconds=ttl_seconds,
            persistent_storage=persistent_storage,
        )
    
    def _load_from_persistent(self):
        """Load cache entries from persistent storage."""
        if not self.persistent_cache:
            return
        
        try:
            entries = self.persistent_cache.get_all_entries()
            for entry in entries:
                # Check if expired
                if self._is_expired(entry):
                    continue
                
                # Add to in-memory cache
                self._exact_cache[entry.query_hash] = entry
                self._lru_order[entry.query_hash] = None
                
                # Evict if needed
                if len(self._exact_cache) >= self.max_size:
                    self._evict_if_needed()
            
            logger.info(
                "Loaded entries from persistent cache",
                num_entries=len(entries),
                loaded_to_memory=len(self._exact_cache),
            )
        except Exception as e:
            logger.error("Failed to load from persistent cache", error=str(e))
    
    def _hash_query(self, query: str) -> str:
        """Generate hash for exact matching."""
        return hashlib.sha256(query.encode()).hexdigest()
    
    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if entry has expired."""
        if self.ttl_seconds is None:
            return False
        age = (datetime.now() - entry.created_at).total_seconds()
        return age > self.ttl_seconds
    
    def _evict_if_needed(self):
        """Evict entries if cache is full."""
        while len(self._exact_cache) >= self.max_size:
            if self.eviction_policy == "lru":
                # Remove least recently used
                if self._lru_order:
                    oldest_key = next(iter(self._lru_order))
                    del self._exact_cache[oldest_key]
                    del self._lru_order[oldest_key]
                    logger.debug("Evicted LRU entry", key=oldest_key)
            elif self.eviction_policy == "time-based":
                # Remove oldest entry
                oldest_entry = min(
                    self._exact_cache.values(),
                    key=lambda e: e.created_at
                )
                oldest_key = oldest_entry.query_hash
                del self._exact_cache[oldest_key]
                if oldest_key in self._lru_order:
                    del self._lru_order[oldest_key]
                logger.debug("Evicted time-based entry", key=oldest_key)
    
    def get_exact(self, query: str) -> Optional[CacheEntry]:
        """Get exact match from cache."""
        query_hash = self._hash_query(query)
        
        if query_hash in self._exact_cache:
            entry = self._exact_cache[query_hash]
            
            # Check expiration
            if self._is_expired(entry):
                del self._exact_cache[query_hash]
                if query_hash in self._lru_order:
                    del self._lru_order[query_hash]
                return None
            
            # Update LRU order
            entry.touch()
            if query_hash in self._lru_order:
                self._lru_order.move_to_end(query_hash)
            else:
                self._lru_order[query_hash] = None
            
            logger.debug("Exact cache hit", query_hash=query_hash[:8])
            # Update persistent cache access
            if self.persistent_cache:
                try:
                    self.persistent_cache.update_access(query_hash)
                except Exception as e:
                    logger.warning("Failed to update persistent cache access", error=str(e))
            return entry
        
        # Try persistent cache if not in memory
        if self.persistent_cache:
            try:
                persistent_entry = self.persistent_cache.get(query_hash)
                if persistent_entry and not self._is_expired(persistent_entry):
                    # Add to in-memory cache
                    self._exact_cache[query_hash] = persistent_entry
                    self._lru_order[query_hash] = None
                    persistent_entry.touch()
                    self.persistent_cache.update_access(query_hash)
                    logger.debug("Exact cache hit from persistent storage", query_hash=query_hash[:8])
                    return persistent_entry
            except Exception as e:
                logger.warning("Failed to check persistent cache", error=str(e))
        
        logger.debug("Exact cache miss", query_hash=query_hash[:8])
        return None
    
    def put(
        self,
        query: str,
        response: str,
        query_embedding: Optional[List[float]] = None,
        tokens_used: int = 0,
        metadata: Optional[Dict] = None,
    ) -> CacheEntry:
        """Store entry in cache."""
        query_hash = self._hash_query(query)
        
        entry = CacheEntry(
            query=query,
            response=response,
            query_hash=query_hash,
            query_embedding=query_embedding,
            tokens_used=tokens_used,
            metadata=metadata or {},
        )
        
        # Evict if needed
        if query_hash not in self._exact_cache:
            self._evict_if_needed()
        
        self._exact_cache[query_hash] = entry
        self._lru_order[query_hash] = None
        self._lru_order.move_to_end(query_hash)
        
        logger.debug("Cached entry", query_hash=query_hash[:8], tokens_saved=tokens_used)
        
        # Store in persistent cache if available
        if self.persistent_cache:
            try:
                self.persistent_cache.put(entry)
            except Exception as e:
                logger.warning("Failed to store in persistent cache", error=str(e))
        
        return entry
    
    def get_all_entries(self) -> List[CacheEntry]:
        """Get all cache entries (for semantic search)."""
        # Filter expired entries
        valid_entries = [
            entry for entry in self._exact_cache.values()
            if not self._is_expired(entry)
        ]
        return valid_entries
    
    def clear(self):
        """Clear all cache entries."""
        self._exact_cache.clear()
        self._lru_order.clear()
        logger.info("Cache cleared")
    
    def stats(self) -> Dict:
        """Get cache statistics."""
        return {
            "size": len(self._exact_cache),
            "max_size": self.max_size,
            "eviction_policy": self.eviction_policy,
            "total_tokens_saved": sum(e.tokens_used for e in self._exact_cache.values()),
        }

