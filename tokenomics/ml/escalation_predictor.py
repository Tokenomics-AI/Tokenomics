"""Escalation prediction for cascading inference using ML model."""

from typing import Optional, List, Dict
from pathlib import Path
import json
from datetime import datetime
import structlog

logger = structlog.get_logger()


class EscalationPredictor:
    """Predicts escalation likelihood using ML model."""
    
    def __init__(self, data_collector=None, model_path: str = "models/escalation_predictor.pkl"):
        """
        Initialize escalation predictor.
        
        Args:
            data_collector: EscalationDataCollector instance for storing training data
            model_path: Path to saved model file (optional)
        """
        self.data_collector = data_collector
        self.ml_model = None  # Will be set when model is trained
        self.model_trained = False
        self.model_path = model_path
        self.metadata_path = model_path.replace(".pkl", "_metadata.json")
        
        # Try to load existing model
        self.load_model()
        
        logger.info("EscalationPredictor initialized", model_trained=self.model_trained)
    
    def predict(
        self,
        query: str,
        complexity: str,
        context_quality_score: float,
        query_tokens: int,
        query_embedding: Optional[List[float]] = None,
    ) -> float:
        """
        Predict escalation probability (0.0-1.0).
        
        Returns probability that escalation will be needed.
        Uses ML model if trained, otherwise uses heuristic.
        
        Args:
            query: User query
            complexity: Query complexity (simple/medium/complex)
            context_quality_score: Context quality score (0.0-1.0)
            query_tokens: Number of tokens in query
            query_embedding: Optional query embedding vector
        
        Returns:
            Escalation probability (0.0 = unlikely, 1.0 = very likely)
        """
        if self.model_trained and self.ml_model:
            # Use ML model prediction
            return self._predict_ml(query, complexity, context_quality_score, query_tokens, query_embedding)
        else:
            # Use heuristic prediction
            return self._predict_heuristic(query, complexity, context_quality_score, query_tokens)
    
    def _predict_heuristic(
        self,
        query: str,
        complexity: str,
        context_quality_score: float,
        query_tokens: int,
    ) -> float:
        """
        Heuristic-based prediction (works immediately, no training needed).
        
        Args:
            query: User query
            complexity: Query complexity (simple/medium/complex)
            context_quality_score: Context quality score (0.0-1.0)
            query_tokens: Number of tokens in query
        
        Returns:
            Escalation probability (0.0-1.0)
        """
        probability = 0.0
        
        # Complexity factor
        complexity_weights = {"simple": 0.1, "medium": 0.3, "complex": 0.6}
        probability += complexity_weights.get(complexity, 0.3)
        
        # Context quality factor (low quality = higher escalation likelihood)
        if context_quality_score < 0.5:
            probability += 0.3
        elif context_quality_score < 0.7:
            probability += 0.15
        
        # Query length factor (very long queries might need premium)
        if query_tokens > 200:
            probability += 0.1
        elif query_tokens > 100:
            probability += 0.05
        
        # Normalize and clamp
        probability = max(0.0, min(1.0, probability))
        
        logger.debug(
            "Escalation prediction (heuristic)",
            complexity=complexity,
            context_quality=context_quality_score,
            query_tokens=query_tokens,
            predicted_probability=probability,
        )
        
        return probability
    
    def _predict_ml(
        self,
        query: str,
        complexity: str,
        context_quality_score: float,
        query_tokens: int,
        query_embedding: Optional[List[float]] = None,
    ) -> float:
        """
        ML model-based prediction (when model is trained).
        
        Args:
            query: User query
            complexity: Query complexity (simple/medium/complex)
            context_quality_score: Context quality score (0.0-1.0)
            query_tokens: Number of tokens in query
            query_embedding: Optional query embedding vector
        
        Returns:
            Escalation probability (0.0-1.0)
        """
        if not self.ml_model:
            logger.debug("ML model not available, using heuristic")
            return self._predict_heuristic(query, complexity, context_quality_score, query_tokens)
        
        try:
            import numpy as np
            
            complexity_map = {"simple": 0, "medium": 1, "complex": 2}
            
            # Prepare features: [query_tokens, complexity_encoded, context_quality, embedding[0:10]]
            features = [
                query_tokens,
                complexity_map.get(complexity, 1),
                context_quality_score,
            ]
            
            # Add embedding features (first 10 dimensions)
            if query_embedding:
                embedding = query_embedding[:10] if len(query_embedding) >= 10 else query_embedding + [0.0] * (10 - len(query_embedding))
                features.extend(embedding)
            else:
                features.extend([0.0] * 10)
            
            # Predict probability using XGBoost classifier
            X = np.array([features])
            # XGBClassifier.predict_proba() returns probabilities for each class
            # [probability_class_0, probability_class_1]
            # We want probability of escalation (class 1)
            probabilities = self.ml_model.predict_proba(X)[0]
            predicted_probability = probabilities[1]  # Probability of escalation (class 1)
            
            logger.debug(
                "Escalation prediction (ML model)",
                complexity=complexity,
                context_quality=context_quality_score,
                query_tokens=query_tokens,
                predicted_probability=predicted_probability,
            )
            
            return float(predicted_probability)
            
        except Exception as e:
            logger.warning("ML model prediction failed, using heuristic", error=str(e))
            return self._predict_heuristic(query, complexity, context_quality_score, query_tokens)
    
    def record_outcome(
        self,
        query: str,
        complexity: str,
        context_quality_score: float,
        query_tokens: int,
        query_embedding: Optional[List[float]] = None,
        escalated: bool = False,
        model_used: Optional[str] = None,
    ):
        """
        Record an escalation outcome for training.
        
        Args:
            query: User query
            complexity: Query complexity
            context_quality_score: Context quality score
            query_tokens: Number of tokens in query
            query_embedding: Optional query embedding
            escalated: Whether escalation actually occurred
            model_used: Model that was used
        """
        if self.data_collector:
            try:
                self.data_collector.record(
                    query=query,
                    query_length=len(query),
                    complexity=complexity,
                    context_quality_score=context_quality_score,
                    query_tokens=query_tokens,
                    query_embedding=query_embedding,
                    escalated=escalated,
                    model_used=model_used,
                )
            except Exception as e:
                logger.warning("Failed to record escalation outcome data", error=str(e))
    
    def train_model(self, min_samples: int = 100) -> bool:
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
        
        # Get escalation training data (explicitly specify model_type)
        if hasattr(self.data_collector, 'get_training_data'):
            # Check if it's UnifiedDataCollector (has model_type parameter)
            import inspect
            sig = inspect.signature(self.data_collector.get_training_data)
            if 'model_type' in sig.parameters:
                training_data = self.data_collector.get_training_data("escalation", min_samples=min_samples)
            else:
                # Old interface (EscalationDataCollector)
                training_data = self.data_collector.get_training_data(min_samples=min_samples)
        else:
            training_data = None
        
        if not training_data:
            logger.info("Not enough data for escalation prediction training", required_samples=min_samples)
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
                    sample["query_tokens"],
                    complexity_map.get(sample["complexity"], 1),
                    sample["context_quality_score"],
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
                # Target: 1 if escalated, 0 if not
                y.append(1 if sample["escalated"] else 0)
            
            X = np.array(X)
            y = np.array(y)
            
            # Train XGBoost binary classifier
            self.ml_model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss',
            )
            
            self.ml_model.fit(X, y)
            self.model_trained = True
            
            logger.info(
                "Escalation prediction ML model trained successfully",
                num_samples=len(training_data),
                model_type="XGBoost Classifier",
            )
            
            # Save model after training
            self.save_model()
            
            return True
            
        except Exception as e:
            logger.error("Failed to train escalation prediction ML model", error=str(e))
            return False
    
    def get_stats(self) -> Dict:
        """Get predictor statistics."""
        stats = {
            "model_trained": self.model_trained,
            "model_type": "XGBoost Classifier" if self.model_trained else "Heuristic",
        }
        
        if self.data_collector:
            # Get escalation-specific stats if UnifiedDataCollector
            if hasattr(self.data_collector, 'get_stats'):
                import inspect
                sig = inspect.signature(self.data_collector.get_stats)
                if 'model_type' in sig.parameters:
                    escalation_stats = self.data_collector.get_stats("escalation")
                    stats.update(escalation_stats)
                else:
                    # Old interface (EscalationDataCollector)
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
                "model_type": "XGBoost Classifier",
                "trained_at": datetime.now().isoformat(),
                "model_trained": True,
                "feature_count": 13,  # query_tokens + complexity + context_quality + 10 embedding dims
            }
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info("Escalation predictor model saved", model_path=str(model_file))
            return True
            
        except Exception as e:
            logger.error("Failed to save escalation predictor model", error=str(e))
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
                logger.debug("Escalation predictor model file not found", model_path=str(model_file))
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
                        logger.debug("Loaded escalation predictor metadata", metadata=metadata)
                except Exception:
                    pass
            
            self.model_trained = True
            logger.info("Escalation predictor model loaded", model_path=str(model_file))
            return True
            
        except Exception as e:
            logger.warning("Failed to load escalation predictor model", error=str(e))
            self.model_trained = False
            self.ml_model = None
            return False




