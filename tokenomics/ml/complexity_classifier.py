"""ML-based complexity classifier for query complexity analysis."""

from typing import Optional, List, Dict
from pathlib import Path
import json
from datetime import datetime
import structlog

logger = structlog.get_logger()


class ComplexityClassifier:
    """ML-based complexity classifier using XGBoost multi-class classifier."""
    
    def __init__(self, data_collector=None, tokenizer=None, model_path: str = "models/complexity_classifier.pkl"):
        """
        Initialize complexity classifier.
        
        Args:
            data_collector: UnifiedDataCollector instance for storing training data
            tokenizer: Optional tokenizer for counting tokens (tiktoken)
            model_path: Path to saved model file (optional)
        """
        self.data_collector = data_collector
        self.tokenizer = tokenizer
        self.ml_model = None  # Will be set when model is trained
        self.model_trained = False
        self.model_path = model_path
        self.metadata_path = model_path.replace(".pkl", "_metadata.json")
        self.class_to_complexity = None  # Will be set when model is trained/loaded
        
        # Try to load existing model
        self.load_model()
        
        # Keyword indicators (same as orchestrator)
        self.complex_indicators = [
            "design", "architecture", "compare", "analyze", "comprehensive",
            "detailed", "system", "pipeline", "implement", "optimize",
            "production", "enterprise", "scalable", "microservices",
            "explain the differences", "provide a comprehensive",
            "write a detailed", "create a complete"
        ]
        self.medium_indicators = [
            "how does", "explain", "difference", "work", "what are",
            "benefits", "use cases", "advantages", "disadvantages"
        ]
        self.comparison_patterns = ["vs", "versus", "difference between", "compare", "contrast"]
        
        logger.info("ComplexityClassifier initialized", model_trained=self.model_trained)
    
    def _extract_features(self, query: str, query_embedding: Optional[List[float]] = None) -> Dict:
        """
        Extract extended features from query.
        
        Returns:
            Dictionary with feature values
        """
        query_length = len(query)
        query_lower = query.lower()
        
        # Count tokens
        query_tokens = 0
        if self.tokenizer:
            try:
                query_tokens = len(self.tokenizer.encode(query))
            except:
                pass
        if query_tokens == 0:
            # Fallback: estimate tokens
            words = len(query.split())
            query_tokens = int(words * 0.75 * 1.2) if words > 0 else 0
        
        # Count keyword matches
        complex_score = sum(1 for ind in self.complex_indicators if ind in query_lower)
        medium_score = sum(1 for ind in self.medium_indicators if ind in query_lower)
        
        # Count question marks
        question_count = query.count('?')
        
        # Check for comparison patterns
        has_comparison = 1 if any(pattern in query_lower for pattern in self.comparison_patterns) else 0
        
        # Extract embedding (first 10 dimensions)
        embedding_features = [0.0] * 10
        if query_embedding:
            embedding_features = query_embedding[:10] if len(query_embedding) >= 10 else query_embedding + [0.0] * (10 - len(query_embedding))
        
        return {
            "query_length": query_length,
            "query_tokens": query_tokens,
            "embedding": embedding_features,
            "complex_score": complex_score,
            "medium_score": medium_score,
            "question_count": question_count,
            "has_comparison": has_comparison,
        }
    
    def _predict_heuristic(self, query: str) -> str:
        """
        Heuristic-based prediction (works immediately, no training needed).
        Uses same logic as orchestrator's analyze_complexity.
        """
        features = self._extract_features(query)
        
        query_tokens = features["query_tokens"]
        complex_score = features["complex_score"]
        question_count = features["question_count"]
        has_comparison = features["has_comparison"]
        
        # Enhanced classification logic (same as orchestrator)
        if complex_score >= 2 or (query_tokens >= 50 and complex_score >= 1) or \
           (question_count >= 2) or (has_comparison and query_tokens >= 30):
            return "complex"
        elif query_tokens >= 20 or complex_score >= 1 or features["medium_score"] >= 1:
            return "medium"
        else:
            return "simple"
    
    def predict(self, query: str, query_embedding: Optional[List[float]] = None) -> str:
        """
        Predict complexity (simple/medium/complex).
        Uses ML model if trained, otherwise uses heuristic fallback.
        
        Args:
            query: User query
            query_embedding: Optional query embedding vector
        
        Returns:
            Complexity level: "simple", "medium", or "complex"
        """
        if self.model_trained and self.ml_model:
            return self._predict_ml(query, query_embedding)
        else:
            return self._predict_heuristic(query)
    
    def _predict_ml(self, query: str, query_embedding: Optional[List[float]] = None) -> str:
        """
        ML model-based prediction (when model is trained).
        
        Args:
            query: User query
            query_embedding: Optional query embedding vector
        
        Returns:
            Complexity level: "simple", "medium", or "complex"
        """
        if not self.ml_model:
            logger.debug("ML model not available, using heuristic")
            return self._predict_heuristic(query)
        
        try:
            import numpy as np
            
            features = self._extract_features(query, query_embedding)
            
            # Prepare feature vector: [query_length, query_tokens, embedding[0:10], 
            #                          complex_score, medium_score, question_count, has_comparison]
            feature_vector = [
                features["query_length"],
                features["query_tokens"],
            ]
            feature_vector.extend(features["embedding"])
            feature_vector.extend([
                features["complex_score"],
                features["medium_score"],
                features["question_count"],
                features["has_comparison"],
            ])
            
            # Predict using XGBoost multi-class classifier
            X = np.array([feature_vector])
            # XGBClassifier.predict() returns class index
            predicted_class = self.ml_model.predict(X)[0]
            
            # Use stored class mapping if available, otherwise default mapping
            if hasattr(self, 'class_to_complexity'):
                predicted = self.class_to_complexity.get(predicted_class, "medium")
            else:
                complexity_map = {0: "simple", 1: "medium", 2: "complex"}
                predicted = complexity_map.get(predicted_class, "medium")
            
            logger.debug(
                "Complexity prediction (ML model)",
                query_preview=query[:50],
                predicted_complexity=predicted,
            )
            
            return predicted
            
        except Exception as e:
            logger.warning("ML model prediction failed, using heuristic", error=str(e))
            return self._predict_heuristic(query)
    
    def record_prediction(
        self,
        query: str,
        predicted_complexity: str,
        actual_complexity: Optional[str] = None,
        query_embedding: Optional[List[float]] = None,
        model_used: Optional[str] = None,
    ):
        """
        Record a complexity prediction for training.
        
        Args:
            query: User query
            predicted_complexity: Predicted complexity (simple/medium/complex)
            actual_complexity: Actual/ground truth complexity (optional)
            query_embedding: Optional query embedding
            model_used: Model that was used
        """
        if self.data_collector:
            try:
                features = self._extract_features(query, query_embedding)
                
                keyword_counts = {
                    "complex_score": features["complex_score"],
                    "medium_score": features["medium_score"],
                    "question_count": features["question_count"],
                    "has_comparison": features["has_comparison"],
                }
                
                self.data_collector.record_complexity_prediction(
                    query=query,
                    query_length=features["query_length"],
                    query_tokens=features["query_tokens"],
                    query_embedding=query_embedding,
                    keyword_counts=keyword_counts,
                    predicted_complexity=predicted_complexity,
                    actual_complexity=actual_complexity,
                    model_used=model_used,
                )
            except Exception as e:
                logger.warning("Failed to record complexity prediction data", error=str(e))
    
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
        
        training_data = self.data_collector.get_training_data("complexity", min_samples=min_samples)
        
        if not training_data:
            logger.info("Not enough data for complexity classification training", required_samples=min_samples)
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
                # Use actual_complexity if available, otherwise use predicted_complexity
                target_complexity = sample.get("actual_complexity") or sample.get("predicted_complexity", "simple")
                
                # Extract features
                query = sample["query"]
                query_embedding = sample.get("embedding_vector")
                
                features = self._extract_features(query, query_embedding)
                
                # Build feature vector
                feature_vector = [
                    features["query_length"],
                    features["query_tokens"],
                ]
                feature_vector.extend(features["embedding"])
                
                # Use keyword_counts from database if available, otherwise extract
                keyword_counts = sample.get("keyword_counts")
                if keyword_counts:
                    feature_vector.extend([
                        keyword_counts.get("complex_score", 0),
                        keyword_counts.get("medium_score", 0),
                        keyword_counts.get("question_count", 0),
                        keyword_counts.get("has_comparison", 0),
                    ])
                else:
                    feature_vector.extend([
                        features["complex_score"],
                        features["medium_score"],
                        features["question_count"],
                        features["has_comparison"],
                    ])
                
                X.append(feature_vector)
                y.append(complexity_map.get(target_complexity.lower(), 0))
            
            X = np.array(X)
            y = np.array(y)
            
            # Get unique classes and ensure they start from 0
            unique_classes = np.unique(y)
            if len(unique_classes) == 0:
                logger.error("No training samples available")
                return False
            
            # Map classes to 0-based indices if needed
            if unique_classes.min() > 0:
                # Classes don't start from 0, remap them
                class_mapping = {old: new for new, old in enumerate(sorted(unique_classes))}
                y = np.array([class_mapping[val] for val in y])
                unique_classes = np.unique(y)
            
            num_classes = len(unique_classes)
            
            # Train XGBoost multi-class classifier
            self.ml_model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric='mlogloss',  # Multi-class log loss
                objective='multi:softprob',  # Multi-class classification
                num_class=num_classes,  # Use actual number of classes
            )
            
            self.ml_model.fit(X, y)
            
            # Store class mapping for prediction
            self.class_to_complexity = {i: ["simple", "medium", "complex"][min(i, 2)] for i in range(num_classes)}
            self.model_trained = True
            
            logger.info(
                "Complexity classification ML model trained successfully",
                num_samples=len(training_data),
                model_type="XGBoost Multi-Class Classifier",
            )
            
            # Save model after training
            self.save_model()
            
            return True
            
        except Exception as e:
            logger.error("Failed to train complexity classification ML model", error=str(e))
            import traceback
            traceback.print_exc()
            return False
    
    def get_stats(self) -> Dict:
        """Get classifier statistics."""
        stats = {
            "model_trained": self.model_trained,
            "model_type": "XGBoost Multi-Class Classifier" if self.model_trained else "Heuristic",
        }
        
        if self.data_collector:
            all_stats = self.data_collector.get_stats()
            complexity_stats = all_stats.get("complexity_prediction", {})
            stats.update(complexity_stats)
        
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
            
            # Save model and class mapping together
            model_data = {
                "model": self.ml_model,
                "class_to_complexity": self.class_to_complexity,
            }
            
            # Try joblib first (better for large numpy arrays), fallback to pickle
            try:
                import joblib
                joblib.dump(model_data, str(model_file))
            except ImportError:
                import pickle
                with open(model_file, 'wb') as f:
                    pickle.dump(model_data, f)
            
            # Save metadata
            metadata = {
                "model_type": "XGBoost Multi-Class Classifier",
                "trained_at": datetime.now().isoformat(),
                "model_trained": True,
                "num_classes": len(self.class_to_complexity) if self.class_to_complexity else 0,
                "class_to_complexity": self.class_to_complexity,
                "feature_count": 14,  # query_length + query_tokens + 10 embedding dims + 4 keyword features
            }
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info("Complexity classifier model saved", model_path=str(model_file))
            return True
            
        except Exception as e:
            logger.error("Failed to save complexity classifier model", error=str(e))
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
                logger.debug("Complexity classifier model file not found", model_path=str(model_file))
                return False
            
            # Try joblib first, fallback to pickle
            try:
                import joblib
                model_data = joblib.load(str(model_file))
            except ImportError:
                import pickle
                with open(model_file, 'rb') as f:
                    model_data = pickle.load(f)
            
            # Handle both old format (just model) and new format (dict with model + mapping)
            if isinstance(model_data, dict):
                self.ml_model = model_data.get("model")
                self.class_to_complexity = model_data.get("class_to_complexity")
            else:
                # Old format: just the model
                self.ml_model = model_data
                self.class_to_complexity = None
            
            # Load metadata if available (may have class mapping)
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                        # Restore class mapping from metadata if not in model data
                        if not self.class_to_complexity and "class_to_complexity" in metadata:
                            self.class_to_complexity = metadata["class_to_complexity"]
                        logger.debug("Loaded complexity classifier metadata", metadata=metadata)
                except Exception:
                    pass
            
            self.model_trained = True
            logger.info("Complexity classifier model loaded", model_path=str(model_file))
            return True
            
        except Exception as e:
            logger.warning("Failed to load complexity classifier model", error=str(e))
            self.model_trained = False
            self.ml_model = None
            self.class_to_complexity = None
            return False



