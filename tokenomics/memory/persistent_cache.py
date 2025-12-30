"""Persistent cache storage using SQLite."""

import sqlite3
import json
import hashlib
from typing import Optional, Dict, List
from datetime import datetime
from pathlib import Path
import structlog

from .cache import CacheEntry

logger = structlog.get_logger()


class PersistentCache:
    """SQLite-backed persistent cache storage."""
    
    def __init__(self, db_path: str = "tokenomics_cache.db"):
        """
        Initialize persistent cache.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        logger.info("PersistentCache initialized", db_path=str(self.db_path))
    
    def _init_db(self):
        """Initialize database schema."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Create cache entries table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cache_entries (
                query_hash TEXT PRIMARY KEY,
                query TEXT NOT NULL,
                response TEXT NOT NULL,
                tokens_used INTEGER DEFAULT 0,
                metadata TEXT,  -- JSON string
                created_at TEXT NOT NULL,
                last_accessed TEXT NOT NULL,
                access_count INTEGER DEFAULT 0
            )
        """)
        
        # Create index for faster lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_last_accessed 
            ON cache_entries(last_accessed)
        """)
        
        conn.commit()
        conn.close()
    
    def get(self, query_hash: str) -> Optional[CacheEntry]:
        """Retrieve cache entry by hash."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT query, response, tokens_used, metadata, 
                   created_at, last_accessed, access_count
            FROM cache_entries
            WHERE query_hash = ?
        """, (query_hash,))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        query, response, tokens_used, metadata_json, created_at, last_accessed, access_count = row
        
        # Parse metadata
        metadata = json.loads(metadata_json) if metadata_json else {}
        
        # Parse datetimes
        created_at_dt = datetime.fromisoformat(created_at)
        last_accessed_dt = datetime.fromisoformat(last_accessed)
        
        entry = CacheEntry(
            query=query,
            response=response,
            query_hash=query_hash,
            metadata=metadata,
            created_at=created_at_dt,
            last_accessed=last_accessed_dt,
            access_count=access_count,
            tokens_used=tokens_used,
        )
        
        return entry
    
    def put(self, entry: CacheEntry):
        """Store cache entry."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Serialize metadata
        metadata_json = json.dumps(entry.metadata) if entry.metadata else "{}"
        
        cursor.execute("""
            INSERT OR REPLACE INTO cache_entries
            (query_hash, query, response, tokens_used, metadata, 
             created_at, last_accessed, access_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            entry.query_hash,
            entry.query,
            entry.response,
            entry.tokens_used,
            metadata_json,
            entry.created_at.isoformat(),
            entry.last_accessed.isoformat(),
            entry.access_count,
        ))
        
        conn.commit()
        conn.close()
        logger.debug("Stored cache entry in persistent storage", query_hash=entry.query_hash[:8])
    
    def update_access(self, query_hash: str):
        """Update access metadata."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE cache_entries
            SET last_accessed = ?,
                access_count = access_count + 1
            WHERE query_hash = ?
        """, (datetime.now().isoformat(), query_hash))
        
        conn.commit()
        conn.close()
    
    def get_all_entries(self) -> List[CacheEntry]:
        """Get all cache entries."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT query_hash, query, response, tokens_used, metadata,
                   created_at, last_accessed, access_count
            FROM cache_entries
        """)
        
        entries = []
        for row in cursor.fetchall():
            query_hash, query, response, tokens_used, metadata_json, created_at, last_accessed, access_count = row
            
            metadata = json.loads(metadata_json) if metadata_json else {}
            created_at_dt = datetime.fromisoformat(created_at)
            last_accessed_dt = datetime.fromisoformat(last_accessed)
            
            entry = CacheEntry(
                query=query,
                response=response,
                query_hash=query_hash,
                metadata=metadata,
                created_at=created_at_dt,
                last_accessed=last_accessed_dt,
                access_count=access_count,
                tokens_used=tokens_used,
            )
            entries.append(entry)
        
        conn.close()
        return entries
    
    def clear(self):
        """Clear all cache entries."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("DELETE FROM cache_entries")
        conn.commit()
        conn.close()
        logger.info("Cleared persistent cache")
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM cache_entries")
        total_entries = cursor.fetchone()[0]
        
        cursor.execute("SELECT SUM(tokens_used) FROM cache_entries")
        total_tokens = cursor.fetchone()[0] or 0
        
        cursor.execute("SELECT SUM(access_count) FROM cache_entries")
        total_accesses = cursor.fetchone()[0] or 0
        
        conn.close()
        
        return {
            "total_entries": total_entries,
            "total_tokens": total_tokens,
            "total_accesses": total_accesses,
        }






