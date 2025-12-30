"""Data collection for escalation prediction ML model training."""

import sqlite3
import json
from typing import Optional, List
from pathlib import Path
from datetime import datetime
import structlog

logger = structlog.get_logger()


class EscalationDataCollector:
    """Collects training data for escalation prediction model."""
    
    def __init__(self, db_path: str = "escalation_prediction_data.db"):
        """
        Initialize escalation data collector.
        
        Args:
            db_path: Path to SQLite database for training data
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        logger.info("EscalationDataCollector initialized", db_path=str(self.db_path))
    
    def _init_db(self):
        """Initialize database schema."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS escalation_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                query_length INTEGER NOT NULL,
                complexity TEXT NOT NULL,
                context_quality_score REAL NOT NULL,
                query_tokens INTEGER NOT NULL,
                query_embedding TEXT,  -- JSON array of first 10 dimensions
                escalated INTEGER NOT NULL,  -- 0 or 1 (boolean)
                model_used TEXT,
                timestamp TEXT NOT NULL
            )
        """)
        
        # Create indexes for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_complexity 
            ON escalation_predictions(complexity)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_escalated 
            ON escalation_predictions(escalated)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON escalation_predictions(timestamp)
        """)
        
        conn.commit()
        conn.close()
    
    def record(
        self,
        query: str,
        query_length: int,
        complexity: str,
        context_quality_score: float,
        query_tokens: int,
        query_embedding: Optional[List[float]] = None,
        escalated: bool = False,
        model_used: Optional[str] = None,
    ):
        """
        Record an escalation outcome.
        
        Args:
            query: User query
            query_length: Query length in characters
            complexity: Query complexity (simple/medium/complex)
            context_quality_score: Context quality score (0.0-1.0)
            query_tokens: Number of tokens in query
            query_embedding: Optional query embedding vector
            escalated: Whether escalation actually occurred
            model_used: Model that was used
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Serialize embedding vector (first 10 dimensions)
        embedding_json = None
        if query_embedding:
            embedding_json = json.dumps(
                query_embedding[:10] if len(query_embedding) > 10 else query_embedding
            )
        
        cursor.execute("""
            INSERT INTO escalation_predictions
            (query, query_length, complexity, context_quality_score, query_tokens,
             query_embedding, escalated, model_used, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            query,
            query_length,
            complexity,
            context_quality_score,
            query_tokens,
            embedding_json,
            1 if escalated else 0,  # Store as integer (0 or 1)
            model_used,
            datetime.now().isoformat(),
        ))
        
        conn.commit()
        conn.close()
        
        logger.debug(
            "Recorded escalation prediction data",
            query_preview=query[:50],
            escalated=escalated,
        )
    
    def get_training_data(self, min_samples: int = 100) -> Optional[List[dict]]:
        """
        Get training data if enough samples available.
        
        Args:
            min_samples: Minimum number of samples required
        
        Returns:
            List of training samples or None if not enough data
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM escalation_predictions")
        count = cursor.fetchone()[0]
        
        if count < min_samples:
            conn.close()
            logger.info(
                "Not enough training data for escalation prediction",
                current_samples=count,
                required_samples=min_samples,
            )
            return None
        
        cursor.execute("""
            SELECT query, query_length, complexity, context_quality_score, 
                   query_tokens, query_embedding, escalated, model_used
            FROM escalation_predictions
            ORDER BY timestamp DESC
            LIMIT ?
        """, (min_samples,))
        
        samples = []
        for row in cursor.fetchall():
            query, query_length, complexity, context_quality, query_tokens, embedding_json, escalated_int, model_used = row
            
            embedding_vector = None
            if embedding_json:
                try:
                    embedding_vector = json.loads(embedding_json)
                except:
                    pass
            
            samples.append({
                "query": query,
                "query_length": query_length,
                "complexity": complexity,
                "context_quality_score": context_quality,
                "query_tokens": query_tokens,
                "embedding_vector": embedding_vector,
                "escalated": bool(escalated_int),  # Convert back to boolean
                "model_used": model_used,
            })
        
        conn.close()
        
        logger.info("Retrieved escalation training data", num_samples=len(samples))
        return samples
    
    def get_stats(self) -> dict:
        """Get data collection statistics."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM escalation_predictions")
        total_samples = cursor.fetchone()[0]
        
        if total_samples == 0:
            conn.close()
            return {
                "total_samples": 0,
                "escalation_rate": 0.0,
                "avg_context_quality": 0.0,
            }
        
        cursor.execute("""
            SELECT 
                AVG(escalated),
                AVG(context_quality_score),
                COUNT(*) FILTER (WHERE escalated = 1) as escalated_count
            FROM escalation_predictions
        """)
        avg_escalated, avg_context_quality, escalated_count = cursor.fetchone()
        
        conn.close()
        
        return {
            "total_samples": total_samples,
            "escalation_rate": round(avg_escalated * 100, 2) if avg_escalated else 0.0,
            "escalated_count": escalated_count or 0,
            "avg_context_quality": round(avg_context_quality, 3) if avg_context_quality else 0.0,
        }



