"""Token prediction for dynamic max_tokens allocation."""

from typing import Optional, List, Dict
from pathlib import Path
import json
from datetime import datetime
import structlog

logger = structlog.get_logger()


class TokenPredictor:
    """Predicts optimal max_tokens based on query characteristics."""
    
    def __init__(self, data_collector=None, model_path: str = "models/token_predictor.pkl"):
        """
        Initialize token predictor.
        
        Args:
            data_collector: DataCollector instance for storing training data
            model_path: Path to saved model file (optional)
        """
        self.data_collector = data_collector
        self.ml_model = None  # Will be set when model is trained
        self.model_trained = False
        self.model_path = model_path
        self.metadata_path = model_path.replace(".pkl", "_metadata.json")
        
        # Try to load existing model
        self.load_model()
        
        logger.info("TokenPredictor initialized", model_trained=self.model_trained)
    
    def predict(
        self,
        query: str,
        complexity: str,
        query_tokens: int,
        query_embedding: Optional[List[float]] = None,
    ) -> int:
        """
        Predict optimal max_tokens for a query.
        
        Uses ML model if trained, otherwise uses heuristic.
        
        Args:
            query: User query
            complexity: Query complexity (simple/medium/complex)
            query_tokens: Number of tokens in query
            query_embedding: Optional query embedding vector
        
        Returns:
            Predicted max_tokens value
        """
        if self.model_trained and self.ml_model:
            # Use ML model prediction
            return self._predict_ml(query, complexity, query_tokens, query_embedding)
        else:
            # Use heuristic prediction
            return self._predict_heuristic(query, complexity, query_tokens)
    
    def _predict_heuristic(
        self,
        query: str,
        complexity: str,
        query_tokens: int,
    ) -> int:
        """
        Heuristic-based prediction (works immediately, no training needed).
        
        Args:
            query: User query
            complexity: Query complexity (simple/medium/complex)
            query_tokens: Number of tokens in query
        
        Returns:
            Predicted max_tokens
        """
        # Base tokens by complexity
        base_tokens = {
            "simple": 200,
            "medium": 500,
            "complex": 800,
        }.get(complexity, 500)
        
        # Length multiplier: longer queries typically need longer answers
        # Factor: 1.0 + (query_tokens / 100) * 0.1
        # Example: 50 token query → 1.05x, 100 token query → 1.1x
        length_factor = 1.0 + (query_tokens / 100) * 0.1
        
        # Query type detection (simple heuristics)
        query_lower = query.lower()
        
        # Explanation/analysis queries need more tokens
        if any(word in query_lower for word in ["explain", "describe", "analyze", "comprehensive", "detailed"]):
            type_multiplier = 1.3
        # Question queries are typically shorter
        elif query.strip().endswith("?") and len(query.split()) < 10:
            type_multiplier = 0.9
        else:
            type_multiplier = 1.0
        
        # Calculate prediction with 20% buffer for safety
        predicted = int(base_tokens * length_factor * type_multiplier * 1.2)
        
        # Cap at reasonable maximum (2000 tokens)
        predicted = min(predicted, 2000)
        
        # Ensure minimum (100 tokens)
        predicted = max(predicted, 100)
        
        logger.debug(
            "Token prediction (heuristic)",
            complexity=complexity,
            query_tokens=query_tokens,
            predicted_tokens=predicted,
            base_tokens=base_tokens,
            length_factor=round(length_factor, 2),
            type_multiplier=round(type_multiplier, 2),
        )
        
        return predicted
    
    def _predict_ml(
        self,
        query: str,
        complexity: str,
        query_tokens: int,
        query_embedding: Optional[List[float]] = None,
    ) -> int:
        """
        ML model-based prediction (when model is trained).
        
        Args:
            query: User query
            complexity: Query complexity (simple/medium/complex)
            query_tokens: Number of tokens in query
            query_embedding: Optional query embedding vector
        
        Returns:
            Predicted max_tokens
        """
        if not self.ml_model:
            logger.debug("ML model not available, using heuristic")
            return self._predict_heuristic(query, complexity, query_tokens)
        
        try:
            import numpy as np
            
            complexity_map = {"simple": 0, "medium": 1, "complex": 2}
            
            # Prepare features
            features = [query_tokens, complexity_map.get(complexity, 1)]
            
            # Add embedding features (first 10 dimensions)
            if query_embedding:
                embedding = query_embedding[:10] if len(query_embedding) >= 10 else query_embedding + [0.0] * (10 - len(query_embedding))
                features.extend(embedding)
            else:
                features.extend([0.0] * 10)
            
            # Predict
            X = np.array([features])
            predicted = int(self.ml_model.predict(X)[0])
            
            # Add 20% buffer and cap
            predicted = int(predicted * 1.2)
            predicted = min(predicted, 2000)
            predicted = max(predicted, 100)
            
            logger.debug(
                "Token prediction (ML model)",
                complexity=complexity,
                query_tokens=query_tokens,
                predicted_tokens=predicted,
            )
            
            return predicted
            
        except Exception as e:
            logger.warning("ML model prediction failed, using heuristic", error=str(e))
            return self._predict_heuristic(query, complexity, query_tokens)
    
    def record_prediction(
        self,
        query: str,
        complexity: str,
        query_tokens: int,
        query_embedding: Optional[List[float]] = None,
        predicted_tokens: int = 0,
        actual_output_tokens: int = 0,
        model_used: Optional[str] = None,
    ):
        """
        Record a prediction and actual result for training.
        
        Args:
            query: User query
            complexity: Query complexity
            query_tokens: Number of tokens in query
            query_embedding: Optional query embedding
            predicted_tokens: Predicted max_tokens
            actual_output_tokens: Actual output tokens generated
            model_used: Model that was used
        """
        if self.data_collector:
            try:
                self.data_collector.record(
                    query=query,
                    query_length=query_tokens,
                    complexity=complexity,
                    embedding_vector=query_embedding,
                    predicted_tokens=predicted_tokens,
                    actual_output_tokens=actual_output_tokens,
                    model_used=model_used,
                )
            except Exception as e:
                logger.warning("Failed to record prediction data", error=str(e))
    
    def train_model(self, min_samples: int = 500) -> bool:
        """
        Train ML model when enough data is available.
        
        Args:
            min_samples: Minimum number of samples required
        
        Returns:
            True if model was trained, False otherwise
        """
        if not self.data_collector:
            logger.warning("No data collector available for training")
            return False
        
        # Get token training data (explicitly specify model_type if UnifiedDataCollector)
        if hasattr(self.data_collector, 'get_training_data'):
            import inspect
            sig = inspect.signature(self.data_collector.get_training_data)
            if 'model_type' in sig.parameters:
                training_data = self.data_collector.get_training_data("token", min_samples=min_samples)
            else:
                # Old interface (DataCollector)
                training_data = self.data_collector.get_training_data(min_samples=min_samples)
        else:
            training_data = None
        
        if not training_data:
            logger.info("Not enough data for training", required_samples=min_samples)
            return False
        
        try:
            # Import XGBoost (optional dependency)
            try:
                import xgboost as xgb
            except ImportError:
                logger.warning("XGBoost not available, cannot train ML model")
                logger.info("Install with: pip install xgboost")
                return False
            
            # Prepare features and targets
            import numpy as np
            
            X = []
            y = []
            
            complexity_map = {"simple": 0, "medium": 1, "complex": 2}
            
            for sample in training_data:
                features = [
                    sample["query_length"],
                    complexity_map.get(sample["complexity"], 1),
                ]
                
                # Add embedding features (first 10 dimensions)
                if sample.get("embedding_vector"):
                    embedding = sample["embedding_vector"]
                    # Pad or truncate to 10 dimensions
                    if len(embedding) < 10:
                        embedding = embedding + [0.0] * (10 - len(embedding))
                    else:
                        embedding = embedding[:10]
                    features.extend(embedding)
                else:
                    # No embedding, use zeros
                    features.extend([0.0] * 10)
                
                X.append(features)
                y.append(sample["actual_output_tokens"])
            
            X = np.array(X)
            y = np.array(y)
            
            # Train XGBoost regressor
            self.ml_model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
            )
            
            self.ml_model.fit(X, y)
            self.model_trained = True
            
            logger.info(
                "ML model trained successfully",
                num_samples=len(training_data),
                model_type="XGBoost",
            )
            
            # Save model after training
            self.save_model()
            
            return True
            
        except Exception as e:
            logger.error("Failed to train ML model", error=str(e))
            return False
    
    def get_stats(self) -> Dict:
        """Get predictor statistics."""
        stats = {
            "model_trained": self.model_trained,
            "model_type": "XGBoost" if self.model_trained else "Heuristic",
        }
        
        if self.data_collector:
            # Get token-specific stats if UnifiedDataCollector
            if hasattr(self.data_collector, 'get_stats'):
                import inspect
                sig = inspect.signature(self.data_collector.get_stats)
                if 'model_type' in sig.parameters:
                    token_stats = self.data_collector.get_stats("token")
                    stats.update(token_stats)
                else:
                    # Old interface (DataCollector)
                    stats.update(self.data_collector.get_stats())
            else:
                stats.update(self.data_collector.get_stats())
        
        return stats
    
    def save_model(self, model_path: Optional[str] = None) -> bool:
        """
        Save trained model to disk.
        
        Args:
            model_path: Optional path to save model (defaults to self.model_path)
        
        Returns:
            True if saved successfully, False otherwise
        """
        if not self.model_trained or not self.ml_model:
            logger.warning("Cannot save model: model not trained")
            return False
        
        try:
            model_path = model_path or self.model_path
            model_file = Path(model_path)
            metadata_file = Path(model_path.replace(".pkl", "_metadata.json"))
            
            # Create models directory if it doesn't exist
            model_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Try joblib first (better for large numpy arrays), fallback to pickle
            try:
                import joblib
                joblib.dump(self.ml_model, str(model_file))
            except ImportError:
                import pickle
                with open(model_file, 'wb') as f:
                    pickle.dump(self.ml_model, f)
            
            # Save metadata
            metadata = {
                "model_type": "XGBoost Regressor",
                "trained_at": datetime.now().isoformat(),
                "model_trained": True,
                "feature_count": 12,  # query_length + complexity + 10 embedding dims
            }
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info("Token predictor model saved", model_path=str(model_file))
            return True
            
        except Exception as e:
            logger.error("Failed to save token predictor model", error=str(e))
            return False
    
    def load_model(self, model_path: Optional[str] = None) -> bool:
        """
        Load trained model from disk.
        
        Args:
            model_path: Optional path to load model from (defaults to self.model_path)
        
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            model_path = model_path or self.model_path
            model_file = Path(model_path)
            metadata_file = Path(model_path.replace(".pkl", "_metadata.json"))
            
            if not model_file.exists():
                logger.debug("Token predictor model file not found", model_path=str(model_file))
                return False
            
            # Try joblib first, fallback to pickle
            try:
                import joblib
                self.ml_model = joblib.load(str(model_file))
            except ImportError:
                import pickle
                with open(model_file, 'rb') as f:
                    self.ml_model = pickle.load(f)
            
            # Load metadata if available
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                        logger.debug("Loaded token predictor metadata", metadata=metadata)
                except Exception:
                    pass
            
            self.model_trained = True
            logger.info("Token predictor model loaded", model_path=str(model_file))
            return True
            
        except Exception as e:
            logger.warning("Failed to load token predictor model", error=str(e))
            self.model_trained = False
            self.ml_model = None
            return False






