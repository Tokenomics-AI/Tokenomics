"""Unified data collection for all ML models (token prediction, escalation prediction, complexity classification)."""

import sqlite3
import json
from typing import Optional, List, Dict
from pathlib import Path
from datetime import datetime
import structlog

logger = structlog.get_logger()


class UnifiedDataCollector:
    """Unified data collector for all ML models."""
    
    def __init__(self, db_path: str = "ml_training_data.db"):
        """
        Initialize unified data collector.
        
        Args:
            db_path: Path to SQLite database for training data
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        logger.info("UnifiedDataCollector initialized", db_path=str(self.db_path))
    
    def _init_db(self):
        """Initialize database schema with all 3 tables."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Token predictions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS token_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                query_length INTEGER NOT NULL,
                complexity TEXT NOT NULL,
                embedding_vector TEXT,
                predicted_tokens INTEGER,
                actual_output_tokens INTEGER NOT NULL,
                model_used TEXT,
                timestamp TEXT NOT NULL
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_token_complexity 
            ON token_predictions(complexity)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_token_timestamp 
            ON token_predictions(timestamp)
        """)
        
        # Escalation predictions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS escalation_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                query_length INTEGER NOT NULL,
                complexity TEXT NOT NULL,
                context_quality_score REAL NOT NULL,
                query_tokens INTEGER NOT NULL,
                query_embedding TEXT,
                escalated INTEGER NOT NULL,
                model_used TEXT,
                timestamp TEXT NOT NULL
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_escalation_complexity 
            ON escalation_predictions(complexity)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_escalation_escalated 
            ON escalation_predictions(escalated)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_escalation_timestamp 
            ON escalation_predictions(timestamp)
        """)
        
        # Complexity predictions table (new)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS complexity_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                query_length INTEGER NOT NULL,
                query_tokens INTEGER NOT NULL,
                query_embedding TEXT,
                keyword_counts TEXT,
                predicted_complexity TEXT NOT NULL,
                actual_complexity TEXT,
                model_used TEXT,
                timestamp TEXT NOT NULL
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_complexity_predicted 
            ON complexity_predictions(predicted_complexity)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_complexity_actual 
            ON complexity_predictions(actual_complexity)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_complexity_timestamp 
            ON complexity_predictions(timestamp)
        """)
        
        conn.commit()
        conn.close()
    
    def record_token_prediction(
        self,
        query: str,
        query_length: int,
        complexity: str,
        embedding_vector: Optional[List[float]] = None,
        predicted_tokens: Optional[int] = None,
        actual_output_tokens: int = 0,
        model_used: Optional[str] = None,
    ):
        """Record a token prediction outcome."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        embedding_json = None
        if embedding_vector:
            embedding_json = json.dumps(
                embedding_vector[:10] if len(embedding_vector) > 10 else embedding_vector
            )
        
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
        
        logger.debug("Recorded token prediction data", query_preview=query[:50])
    
    def record_escalation_prediction(
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
        """Record an escalation prediction outcome."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
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
            1 if escalated else 0,
            model_used,
            datetime.now().isoformat(),
        ))
        
        conn.commit()
        conn.close()
        
        logger.debug("Recorded escalation prediction data", query_preview=query[:50])
    
    def record_complexity_prediction(
        self,
        query: str,
        query_length: int,
        query_tokens: int,
        query_embedding: Optional[List[float]] = None,
        keyword_counts: Optional[Dict[str, int]] = None,
        predicted_complexity: str = "simple",
        actual_complexity: Optional[str] = None,
        model_used: Optional[str] = None,
    ):
        """Record a complexity classification outcome."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        embedding_json = None
        if query_embedding:
            embedding_json = json.dumps(
                query_embedding[:10] if len(query_embedding) > 10 else query_embedding
            )
        
        keyword_counts_json = None
        if keyword_counts:
            keyword_counts_json = json.dumps(keyword_counts)
        
        cursor.execute("""
            INSERT INTO complexity_predictions
            (query, query_length, query_tokens, query_embedding, keyword_counts,
             predicted_complexity, actual_complexity, model_used, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            query,
            query_length,
            query_tokens,
            embedding_json,
            keyword_counts_json,
            predicted_complexity,
            actual_complexity,
            model_used,
            datetime.now().isoformat(),
        ))
        
        conn.commit()
        conn.close()
        
        logger.debug("Recorded complexity prediction data", query_preview=query[:50])
    
    def get_training_data(self, model_type: str = "token", min_samples: int = 100) -> Optional[List[dict]]:
        """
        Get training data for specific model type.
        
        Args:
            model_type: One of "token", "escalation", "complexity"
            min_samples: Minimum number of samples required
        
        Returns:
            List of training samples or None if not enough data
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        if model_type == "token":
            table = "token_predictions"
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            
            if count < min_samples:
                conn.close()
                logger.info("Not enough training data for token prediction", 
                          current_samples=count, required_samples=min_samples)
                return None
            
            cursor.execute(f"""
                SELECT query, query_length, complexity, embedding_vector, 
                       predicted_tokens, actual_output_tokens, model_used
                FROM {table}
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
        
        elif model_type == "escalation":
            table = "escalation_predictions"
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            
            if count < min_samples:
                conn.close()
                logger.info("Not enough training data for escalation prediction",
                          current_samples=count, required_samples=min_samples)
                return None
            
            cursor.execute(f"""
                SELECT query, query_length, complexity, context_quality_score, 
                       query_tokens, query_embedding, escalated, model_used
                FROM {table}
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
                    "escalated": bool(escalated_int),
                    "model_used": model_used,
                })
        
        elif model_type == "complexity":
            table = "complexity_predictions"
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            
            if count < min_samples:
                conn.close()
                logger.info("Not enough training data for complexity classification",
                          current_samples=count, required_samples=min_samples)
                return None
            
            cursor.execute(f"""
                SELECT query, query_length, query_tokens, query_embedding, keyword_counts,
                       predicted_complexity, actual_complexity, model_used
                FROM {table}
                ORDER BY timestamp DESC
                LIMIT ?
            """, (min_samples,))
            
            samples = []
            for row in cursor.fetchall():
                query, query_length, query_tokens, embedding_json, keyword_counts_json, predicted_complexity, actual_complexity, model_used = row
                
                embedding_vector = None
                if embedding_json:
                    try:
                        embedding_vector = json.loads(embedding_json)
                    except:
                        pass
                
                keyword_counts = None
                if keyword_counts_json:
                    try:
                        keyword_counts = json.loads(keyword_counts_json)
                    except:
                        pass
                
                samples.append({
                    "query": query,
                    "query_length": query_length,
                    "query_tokens": query_tokens,
                    "embedding_vector": embedding_vector,
                    "keyword_counts": keyword_counts,
                    "predicted_complexity": predicted_complexity,
                    "actual_complexity": actual_complexity,
                    "model_used": model_used,
                })
        
        else:
            conn.close()
            logger.error("Unknown model type", model_type=model_type)
            return None
        
        conn.close()
        logger.info(f"Retrieved {model_type} training data", num_samples=len(samples))
        return samples
    
    def get_stats(self, model_type: Optional[str] = None) -> Dict:
        """
        Get statistics for all model types or specific model type.
        
        Args:
            model_type: Optional model type ("token", "escalation", "complexity") 
                       or None for all stats
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        stats = {}
        
        # Token prediction stats
        cursor.execute("SELECT COUNT(*) FROM token_predictions")
        token_count = cursor.fetchone()[0]
        if token_count > 0:
            cursor.execute("""
                SELECT AVG(actual_output_tokens), MIN(actual_output_tokens), MAX(actual_output_tokens)
                FROM token_predictions
            """)
            avg_tokens, min_tokens, max_tokens = cursor.fetchone()
            stats["token_prediction"] = {
                "total_samples": token_count,
                "avg_output_tokens": round(avg_tokens, 2) if avg_tokens else 0,
                "min_output_tokens": min_tokens or 0,
                "max_output_tokens": max_tokens or 0,
            }
        else:
            stats["token_prediction"] = {"total_samples": 0}
        
        # Escalation prediction stats
        cursor.execute("SELECT COUNT(*) FROM escalation_predictions")
        escalation_count = cursor.fetchone()[0]
        if escalation_count > 0:
            cursor.execute("""
                SELECT AVG(escalated), AVG(context_quality_score), COUNT(*) FILTER (WHERE escalated = 1)
                FROM escalation_predictions
            """)
            avg_escalated, avg_context_quality, escalated_count = cursor.fetchone()
            stats["escalation_prediction"] = {
                "total_samples": escalation_count,
                "escalation_rate": round(avg_escalated * 100, 2) if avg_escalated else 0.0,
                "escalated_count": escalated_count or 0,
                "avg_context_quality": round(avg_context_quality, 3) if avg_context_quality else 0.0,
            }
        else:
            stats["escalation_prediction"] = {"total_samples": 0}
        
        # Complexity prediction stats
        cursor.execute("SELECT COUNT(*) FROM complexity_predictions")
        complexity_count = cursor.fetchone()[0]
        if complexity_count > 0:
            cursor.execute("""
                SELECT 
                    COUNT(*) FILTER (WHERE predicted_complexity = 'simple') as simple_count,
                    COUNT(*) FILTER (WHERE predicted_complexity = 'medium') as medium_count,
                    COUNT(*) FILTER (WHERE predicted_complexity = 'complex') as complex_count,
                    COUNT(*) FILTER (WHERE actual_complexity IS NOT NULL) as labeled_count
                FROM complexity_predictions
            """)
            simple_count, medium_count, complex_count, labeled_count = cursor.fetchone()
            stats["complexity_prediction"] = {
                "total_samples": complexity_count,
                "simple_count": simple_count or 0,
                "medium_count": medium_count or 0,
                "complex_count": complex_count or 0,
                "labeled_samples": labeled_count or 0,
            }
        else:
            stats["complexity_prediction"] = {"total_samples": 0}
        
        conn.close()
        
        # If specific model type requested, return only that model's stats
        if model_type:
            if model_type == "token":
                return stats.get("token_prediction", {"total_samples": 0})
            elif model_type == "escalation":
                return stats.get("escalation_prediction", {"total_samples": 0})
            elif model_type == "complexity":
                return stats.get("complexity_prediction", {"total_samples": 0})
        
        return stats
    
    # Adapter methods for backward compatibility with DataCollector and EscalationDataCollector interfaces
    def record(self, query: str, query_length: int, complexity: str,
               embedding_vector: Optional[List[float]] = None,
               predicted_tokens: Optional[int] = None,
               actual_output_tokens: int = 0,
               model_used: Optional[str] = None,
               # Escalation-specific parameters (optional, used to detect escalation calls)
               context_quality_score: Optional[float] = None,
               query_tokens: Optional[int] = None,
               query_embedding: Optional[List[float]] = None,
               escalated: Optional[bool] = None):
        """
        Adapter method that routes to appropriate record method based on parameters.
        Detects if this is a token prediction or escalation prediction call.
        """
        # If escalation-specific parameters are provided, use escalation record
        if context_quality_score is not None and query_tokens is not None and escalated is not None:
            # Use query_embedding if provided, otherwise fall back to embedding_vector
            escalation_embedding = query_embedding if query_embedding is not None else embedding_vector
            self.record_escalation_prediction(
                query=query,
                query_length=query_length,
                complexity=complexity,
                context_quality_score=context_quality_score,
                query_tokens=query_tokens,
                query_embedding=escalation_embedding,
                escalated=escalated,
                model_used=model_used,
            )
        else:
            # Otherwise, use token prediction record
            self.record_token_prediction(
                query=query,
                query_length=query_length,
                complexity=complexity,
                embedding_vector=embedding_vector,
                predicted_tokens=predicted_tokens,
                actual_output_tokens=actual_output_tokens,
                model_used=model_used,
            )
    
    def get_training_data_token(self, min_samples: int = 500) -> Optional[List[dict]]:
        """Get training data for token prediction (adapter for DataCollector interface)."""
        return self.get_training_data("token", min_samples=min_samples)
    




