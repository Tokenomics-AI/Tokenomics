"""Data collection for ML model training."""

import sqlite3
import json
from typing import Optional, List
from pathlib import Path
from datetime import datetime
import structlog

logger = structlog.get_logger()


class DataCollector:
    """Collects training data for token prediction model."""
    
    def __init__(self, db_path: str = "token_prediction_data.db"):
        """
        Initialize data collector.
        
        Args:
            db_path: Path to SQLite database for training data
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        logger.info("DataCollector initialized", db_path=str(self.db_path))
    
    def _init_db(self):
        """Initialize database schema."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS token_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                query_length INTEGER NOT NULL,
                complexity TEXT NOT NULL,
                embedding_vector TEXT,  -- JSON array of first 10 dimensions
                predicted_tokens INTEGER,
                actual_output_tokens INTEGER NOT NULL,
                model_used TEXT,
                timestamp TEXT NOT NULL
            )
        """)
        
        # Create index for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_complexity 
            ON token_predictions(complexity)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON token_predictions(timestamp)
        """)
        
        conn.commit()
        conn.close()
    
    def record(
        self,
        query: str,
        query_length: int,
        complexity: str,
        embedding_vector: Optional[List[float]] = None,
        predicted_tokens: Optional[int] = None,
        actual_output_tokens: int = 0,
        model_used: Optional[str] = None,
    ):
        """Record a prediction and actual result."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Serialize embedding vector (first 10 dimensions)
        embedding_json = None
        if embedding_vector:
            embedding_json = json.dumps(embedding_vector[:10] if len(embedding_vector) > 10 else embedding_vector)
        
        cursor.execute("""
            INSERT INTO token_predictions
            (query, query_length, complexity, embedding_vector, predicted_tokens, 
             actual_output_tokens, model_used, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            query,
            query_length,
            complexity,
            embedding_json,
            predicted_tokens,
            actual_output_tokens,
            model_used,
            datetime.now().isoformat(),
        ))
        
        conn.commit()
        conn.close()
        
        logger.debug(
            "Recorded token prediction data",
            query_preview=query[:50],
            actual_tokens=actual_output_tokens,
        )
    
    def get_training_data(self, min_samples: int = 500) -> Optional[List[dict]]:
        """
        Get training data if enough samples available.
        
        Args:
            min_samples: Minimum number of samples required
        
        Returns:
            List of training samples or None if not enough data
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM token_predictions")
        count = cursor.fetchone()[0]
        
        if count < min_samples:
            conn.close()
            logger.info(
                "Not enough training data",
                current_samples=count,
                required_samples=min_samples,
            )
            return None
        
        cursor.execute("""
            SELECT query, query_length, complexity, embedding_vector, 
                   predicted_tokens, actual_output_tokens, model_used
            FROM token_predictions
            ORDER BY timestamp DESC
            LIMIT ?
        """, (min_samples,))
        
        samples = []
        for row in cursor.fetchall():
            query, query_length, complexity, embedding_json, predicted_tokens, actual_output_tokens, model_used = row
            
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
                "embedding_vector": embedding_vector,
                "predicted_tokens": predicted_tokens,
                "actual_output_tokens": actual_output_tokens,
                "model_used": model_used,
            })
        
        conn.close()
        
        logger.info("Retrieved training data", num_samples=len(samples))
        return samples
    
    def get_stats(self) -> dict:
        """Get data collection statistics."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM token_predictions")
        total_samples = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT AVG(actual_output_tokens), MIN(actual_output_tokens), MAX(actual_output_tokens)
            FROM token_predictions
        """)
        avg_tokens, min_tokens, max_tokens = cursor.fetchone()
        
        conn.close()
        
        return {
            "total_samples": total_samples,
            "avg_output_tokens": round(avg_tokens, 2) if avg_tokens else 0,
            "min_output_tokens": min_tokens or 0,
            "max_output_tokens": max_tokens or 0,
        }






