"""
Machine Learning Rule Optimizer for COA/ROE System

This module provides advanced rule optimization capabilities including:
- Machine learning for rule optimization
- Real-time rule updates
- Advanced correlation algorithms
- Predictive rule analysis
- Adaptive rule learning
"""

import asyncio
import json
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import redis
from prometheus_client import Counter, Gauge, Histogram
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler

import logging
# Use standard logging instead of FLAGSHIP.common
get_logger = logging.getLogger

logger = get_logger(__name__)

# ============================================================================
# PROMETHEUS METRICS
# ============================================================================

RULE_OPTIMIZATION_OPERATIONS = Counter(
    "coa_roe_rule_optimization_operations_total",
    "Total rule optimization operations",
    ["operation"],
)
RULE_OPTIMIZATION_LATENCY = Histogram(
    "coa_roe_rule_optimization_latency_seconds",
    "Rule optimization latency",
    ["operation"],
)
RULE_OPTIMIZATION_ACCURACY = Gauge(
    "coa_roe_rule_optimization_accuracy", "Rule optimization accuracy"
)
RULE_OPTIMIZATION_PRECISION = Gauge(
    "coa_roe_rule_optimization_precision", "Rule optimization precision"
)
RULE_OPTIMIZATION_RECALL = Gauge(
    "coa_roe_rule_optimization_recall", "Rule optimization recall"
)

# ============================================================================
# RULE OPTIMIZATION MODELS
# ============================================================================


@dataclass
class RuleFeature:
    """Rule feature data structure."""

    rule_id: str
    feature_name: str
    feature_value: float
    feature_type: str  # 'numeric', 'categorical', 'temporal'
    importance: float = 0.0


@dataclass
class RulePrediction:
    """Rule prediction data structure."""

    rule_id: str
    prediction: float
    confidence: float
    features_used: List[str]
    timestamp: datetime


@dataclass
class OptimizationResult:
    """Optimization result data structure."""

    rule_id: str
    original_score: float
    optimized_score: float
    improvement: float
    changes: Dict[str, Any]
    timestamp: datetime


# ============================================================================
# RULE OPTIMIZER
# ============================================================================


class RuleOptimizer:
    """Machine learning-based rule optimizer."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the rule optimizer."""
        self.config = config or {}
        self.logger = logger

        # ML models
        self.classifier = None
        self.clusterer = None
        self.scaler = StandardScaler()

        # Training data
        self.training_data: List[Dict[str, Any]] = []
        self.training_lock = threading.RLock()

        # Rule features and predictions
        self.rule_features: Dict[str, List[RuleFeature]] = {}
        self.rule_predictions: Dict[str, RulePrediction] = {}
        self.optimization_results: List[OptimizationResult] = []

        # Model state
        self.model_trained = False
        self.last_training = None
        self.training_interval = self.config.get("training_interval", 3600)  # 1 hour

        # Redis client for model storage
        self.redis_client = None
        self._initialize_redis()

        # Start background tasks
        self._start_background_tasks()

        logger.info("Rule Optimizer initialized")

    def _initialize_redis(self):
        """Initialize Redis connection for model storage."""
        try:
            redis_url = self.config.get("redis_url", "redis://localhost:6379")
            self.redis_client = redis.Redis.from_url(redis_url, decode_responses=True)
            self.redis_client.ping()
            logger.info("Redis connection established for model storage")
        except Exception as e:
            logger.warning(f"Failed to initialize Redis for model storage: {e}")
            self.redis_client = None

    def _start_background_tasks(self):
        """Start background tasks for model training and optimization."""
        threading.Thread(target=self._model_training_worker, daemon=True).start()
        threading.Thread(target=self._rule_optimization_worker, daemon=True).start()

    def _model_training_worker(self):
        """Background worker for model training."""
        while True:
            try:
                if not self.model_trained or self._should_retrain():
                    self._train_models()

                time.sleep(self.training_interval)

            except Exception as e:
                logger.error(f"Model training worker error: {e}")
                time.sleep(300)  # 5 minutes

    def _rule_optimization_worker(self):
        """Background worker for rule optimization."""
        while True:
            try:
                # Perform rule optimization
                self._optimize_rules()

                time.sleep(1800)  # 30 minutes

            except Exception as e:
                logger.error(f"Rule optimization worker error: {e}")
                time.sleep(600)  # 10 minutes

    def _should_retrain(self) -> bool:
        """Check if models should be retrained."""
        if not self.last_training:
            return True

        time_since_training = datetime.now() - self.last_training
        return time_since_training.total_seconds() > self.training_interval

    def _train_models(self):
        """Train ML models."""
        try:
            start_time = time.time()

            if not self.training_data:
                logger.warning("No training data available for model training")
                return

            # Prepare training data
            X, y = self._prepare_training_data()

            if len(X) < 10:
                logger.warning("Insufficient training data for model training")
                return

            # Train classifier
            self.classifier = RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42
            )
            self.classifier.fit(X, y)

            # Train clusterer for rule grouping
            self.clusterer = KMeans(n_clusters=min(5, len(X)), random_state=42)
            self.clusterer.fit(X)

            # Update model state
            self.model_trained = True
            self.last_training = datetime.now()

            # Calculate model performance
            y_pred = self.classifier.predict(X)
            accuracy = accuracy_score(y, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y, y_pred, average="weighted"
            )

            # Update Prometheus metrics
            RULE_OPTIMIZATION_ACCURACY.set(accuracy)
            RULE_OPTIMIZATION_PRECISION.set(precision)
            RULE_OPTIMIZATION_RECALL.set(recall)

            # Store model in Redis
            self._store_models()

            training_time = time.time() - start_time
            RULE_OPTIMIZATION_LATENCY.labels(operation="training").observe(
                training_time
            )

            logger.info(
                f"Models trained successfully in {training_time:.2f}s - Accuracy: {accuracy:.3f}"
            )

        except Exception as e:
            logger.error(f"Error training models: {e}")

    def _prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for ML models."""
        try:
            # Convert training data to features
            features = []
            labels = []

            for data_point in self.training_data:
                # Extract features
                feature_vector = self._extract_features(data_point)
                if feature_vector is not None:
                    features.append(feature_vector)
                    labels.append(data_point.get("label", 0))

            if not features:
                return np.array([]), np.array([])

            # Convert to numpy arrays
            X = np.array(features)
            y = np.array(labels)

            # Scale features
            X = self.scaler.fit_transform(X)

            return X, y

        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return np.array([]), np.array([])

    def _extract_features(self, data_point: Dict[str, Any]) -> Optional[List[float]]:
        """Extract features from data point."""
        try:
            features = []

            # Extract rule-based features
            rule_id = data_point.get("rule_id", "")
            if rule_id in self.rule_features:
                rule_feats = self.rule_features[rule_id]
                for feat in rule_feats:
                    features.append(feat.feature_value)

            # Extract temporal features
            timestamp = data_point.get("timestamp")
            if timestamp:
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp)
                features.extend([timestamp.hour, timestamp.weekday(), timestamp.month])

            # Extract performance features
            features.extend(
                [
                    data_point.get("latency", 0.0),
                    data_point.get("throughput", 0),
                    data_point.get("error_rate", 0.0),
                ]
            )

            # Extract system features
            features.extend(
                [
                    data_point.get("cpu_usage", 0.0),
                    data_point.get("memory_usage", 0.0),
                    data_point.get("cache_hit_ratio", 0.0),
                ]
            )

            return features if features else None

        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return None

    def _store_models(self):
        """Store trained models in Redis."""
        try:
            if not self.redis_client or not self.model_trained:
                return

            # Serialize models
            if self.classifier:
                classifier_data = joblib.dumps(self.classifier)
                self.redis_client.set("coa_roe:models:classifier", classifier_data)

            if self.clusterer:
                clusterer_data = joblib.dumps(self.clusterer)
                self.redis_client.set("coa_roe:models:clusterer", clusterer_data)

            if hasattr(self.scaler, "scale_"):
                scaler_data = joblib.dumps(self.scaler)
                self.redis_client.set("coa_roe:models:scaler", scaler_data)

            # Store model metadata
            metadata = {
                "trained": self.model_trained,
                "last_training": (
                    self.last_training.isoformat() if self.last_training else None
                ),
                "training_data_size": len(self.training_data),
            }
            self.redis_client.set("coa_roe:models:metadata", json.dumps(metadata))

            logger.info("Models stored in Redis")

        except Exception as e:
            logger.error(f"Error storing models: {e}")

    def _load_models(self):
        """Load trained models from Redis."""
        try:
            if not self.redis_client:
                return

            # Load classifier
            classifier_data = self.redis_client.get("coa_roe:models:classifier")
            if classifier_data:
                self.classifier = joblib.loads(classifier_data)

            # Load clusterer
            clusterer_data = self.redis_client.get("coa_roe:models:clusterer")
            if clusterer_data:
                self.clusterer = joblib.loads(clusterer_data)

            # Load scaler
            scaler_data = self.redis_client.get("coa_roe:models:scaler")
            if scaler_data:
                self.scaler = joblib.loads(scaler_data)

            # Load metadata
            metadata_data = self.redis_client.get("coa_roe:models:metadata")
            if metadata_data:
                metadata = json.loads(metadata_data)
                self.model_trained = metadata.get("trained", False)
                last_training_str = metadata.get("last_training")
                if last_training_str:
                    self.last_training = datetime.fromisoformat(last_training_str)

            logger.info("Models loaded from Redis")

        except Exception as e:
            logger.error(f"Error loading models: {e}")

    async def add_training_data(self, data_point: Dict[str, Any]):
        """Add training data point."""
        try:
            with self.training_lock:
                self.training_data.append(data_point)

                # Keep only recent training data
                cutoff_time = datetime.now() - timedelta(days=7)
                self.training_data = [
                    dp
                    for dp in self.training_data
                    if self._get_timestamp(dp) > cutoff_time
                ]

            # Store in Redis
            if self.redis_client:
                key = f"coa_roe:training_data:{int(time.time())}"
                self.redis_client.setex(key, 604800, json.dumps(data_point))  # 7 days

            RULE_OPTIMIZATION_OPERATIONS.labels(operation="add_training_data").inc()

        except Exception as e:
            logger.error(f"Error adding training data: {e}")

    def _get_timestamp(self, data_point: Dict[str, Any]) -> datetime:
        """Get timestamp from data point."""
        timestamp = data_point.get("timestamp")
        if isinstance(timestamp, str):
            return datetime.fromisoformat(timestamp)
        elif isinstance(timestamp, datetime):
            return timestamp
        else:
            return datetime.now()

    async def predict_rule_performance(
        self, rule_id: str, features: Dict[str, Any]
    ) -> Optional[RulePrediction]:
        """Predict rule performance using ML models."""
        try:
            if not self.model_trained or not self.classifier:
                return None

            start_time = time.time()

            # Extract features
            feature_vector = self._extract_features({"rule_id": rule_id, **features})
            if feature_vector is None:
                return None

            # Scale features
            X = self.scaler.transform([feature_vector])

            # Make prediction
            prediction = self.classifier.predict_proba(X)[0][
                1
            ]  # Probability of success

            # Calculate confidence based on model certainty
            confidence = max(prediction, 1 - prediction)

            # Create prediction
            rule_prediction = RulePrediction(
                rule_id=rule_id,
                prediction=prediction,
                confidence=confidence,
                features_used=list(features.keys()),
                timestamp=datetime.now(),
            )

            # Store prediction
            self.rule_predictions[rule_id] = rule_prediction

            # Store in Redis
            if self.redis_client:
                key = f"coa_roe:predictions:{rule_id}"
                prediction_data = asdict(rule_prediction)
                prediction_data["timestamp"] = rule_prediction.timestamp.isoformat()
                self.redis_client.setex(key, 3600, json.dumps(prediction_data))

            prediction_time = time.time() - start_time
            RULE_OPTIMIZATION_LATENCY.labels(operation="prediction").observe(
                prediction_time
            )
            RULE_OPTIMIZATION_OPERATIONS.labels(operation="prediction").inc()

            return rule_prediction

        except Exception as e:
            logger.error(f"Error predicting rule performance: {e}")
            return None

    async def optimize_rule(
        self, rule_id: str, current_config: Dict[str, Any]
    ) -> Optional[OptimizationResult]:
        """Optimize rule configuration using ML models."""
        try:
            if not self.model_trained:
                return None

            start_time = time.time()

            # Get current performance
            current_score = await self._evaluate_rule_performance(
                rule_id, current_config
            )

            # Generate optimization suggestions
            optimization_suggestions = self._generate_optimization_suggestions(
                rule_id, current_config
            )

            # Apply optimizations
            optimized_config = self._apply_optimizations(
                current_config, optimization_suggestions
            )

            # Evaluate optimized performance
            optimized_score = await self._evaluate_rule_performance(
                rule_id, optimized_config
            )

            # Calculate improvement
            improvement = optimized_score - current_score

            # Create optimization result
            optimization_result = OptimizationResult(
                rule_id=rule_id,
                original_score=current_score,
                optimized_score=optimized_score,
                improvement=improvement,
                changes=optimization_suggestions,
                timestamp=datetime.now(),
            )

            # Store result
            self.optimization_results.append(optimization_result)

            # Store in Redis
            if self.redis_client:
                key = f"coa_roe:optimizations:{rule_id}:{int(time.time())}"
                result_data = asdict(optimization_result)
                result_data["timestamp"] = optimization_result.timestamp.isoformat()
                self.redis_client.setex(key, 86400, json.dumps(result_data))  # 24 hours

            optimization_time = time.time() - start_time
            RULE_OPTIMIZATION_LATENCY.labels(operation="optimization").observe(
                optimization_time
            )
            RULE_OPTIMIZATION_OPERATIONS.labels(operation="optimization").inc()

            return optimization_result

        except Exception as e:
            logger.error(f"Error optimizing rule: {e}")
            return None

    async def _evaluate_rule_performance(
        self, rule_id: str, config: Dict[str, Any]
    ) -> float:
        """Evaluate rule performance score."""
        try:
            # Use ML model to predict performance
            prediction = await self.predict_rule_performance(rule_id, config)
            if prediction:
                return prediction.prediction

            # Fallback to simple scoring
            return 0.5

        except Exception as e:
            logger.error(f"Error evaluating rule performance: {e}")
            return 0.0

    def _generate_optimization_suggestions(
        self, rule_id: str, config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate optimization suggestions for rule."""
        try:
            suggestions = {}

            # Analyze rule features
            if rule_id in self.rule_features:
                features = self.rule_features[rule_id]

                # Find low-importance features
                low_importance_features = [f for f in features if f.importance < 0.1]
                if low_importance_features:
                    suggestions["remove_low_importance_features"] = [
                        f.feature_name for f in low_importance_features
                    ]

                # Find high-importance features that could be optimized
                high_importance_features = [f for f in features if f.importance > 0.8]
                if high_importance_features:
                    suggestions["optimize_high_importance_features"] = [
                        f.feature_name for f in high_importance_features
                    ]

            # Generate temporal optimizations
            suggestions["temporal_optimizations"] = (
                self._generate_temporal_optimizations(config)
            )

            # Generate performance optimizations
            suggestions["performance_optimizations"] = (
                self._generate_performance_optimizations(config)
            )

            return suggestions

        except Exception as e:
            logger.error(f"Error generating optimization suggestions: {e}")
            return {}

    def _generate_temporal_optimizations(self, config: Dict[str, Any]) -> List[str]:
        """Generate temporal optimizations."""
        optimizations = []

        # Check for time-based optimizations
        if "time_window" in config:
            current_window = config["time_window"]
            if current_window > 3600:  # 1 hour
                optimizations.append("reduce_time_window")
            elif current_window < 60:  # 1 minute
                optimizations.append("increase_time_window")

        return optimizations

    def _generate_performance_optimizations(self, config: Dict[str, Any]) -> List[str]:
        """Generate performance optimizations."""
        optimizations = []

        # Check for performance-related optimizations
        if "batch_size" in config:
            current_batch = config["batch_size"]
            if current_batch > 1000:
                optimizations.append("reduce_batch_size")
            elif current_batch < 10:
                optimizations.append("increase_batch_size")

        if "cache_ttl" in config:
            current_ttl = config["cache_ttl"]
            if current_ttl > 3600:
                optimizations.append("reduce_cache_ttl")
            elif current_ttl < 60:
                optimizations.append("increase_cache_ttl")

        return optimizations

    def _apply_optimizations(
        self, config: Dict[str, Any], suggestions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply optimization suggestions to configuration."""
        try:
            optimized_config = config.copy()

            # Apply temporal optimizations
            if "temporal_optimizations" in suggestions:
                for opt in suggestions["temporal_optimizations"]:
                    if (
                        opt == "reduce_time_window"
                        and "time_window" in optimized_config
                    ):
                        optimized_config["time_window"] = max(
                            60, optimized_config["time_window"] // 2
                        )
                    elif (
                        opt == "increase_time_window"
                        and "time_window" in optimized_config
                    ):
                        optimized_config["time_window"] = min(
                            3600, optimized_config["time_window"] * 2
                        )

            # Apply performance optimizations
            if "performance_optimizations" in suggestions:
                for opt in suggestions["performance_optimizations"]:
                    if opt == "reduce_batch_size" and "batch_size" in optimized_config:
                        optimized_config["batch_size"] = max(
                            10, optimized_config["batch_size"] // 2
                        )
                    elif (
                        opt == "increase_batch_size"
                        and "batch_size" in optimized_config
                    ):
                        optimized_config["batch_size"] = min(
                            1000, optimized_config["batch_size"] * 2
                        )
                    elif opt == "reduce_cache_ttl" and "cache_ttl" in optimized_config:
                        optimized_config["cache_ttl"] = max(
                            60, optimized_config["cache_ttl"] // 2
                        )
                    elif (
                        opt == "increase_cache_ttl" and "cache_ttl" in optimized_config
                    ):
                        optimized_config["cache_ttl"] = min(
                            3600, optimized_config["cache_ttl"] * 2
                        )

            return optimized_config

        except Exception as e:
            logger.error(f"Error applying optimizations: {e}")
            return config

    def _optimize_rules(self):
        """Perform rule optimization for all rules."""
        try:
            # Get all rules that need optimization
            rules_to_optimize = self._get_rules_for_optimization()

            for rule_id in rules_to_optimize:
                try:
                    # Get current configuration
                    current_config = self._get_rule_config(rule_id)
                    if current_config:
                        # Optimize rule
                        asyncio.run(self.optimize_rule(rule_id, current_config))

                except Exception as e:
                    logger.error(f"Error optimizing rule {rule_id}: {e}")

        except Exception as e:
            logger.error(f"Error in rule optimization: {e}")

    def _get_rules_for_optimization(self) -> List[str]:
        """Get list of rules that need optimization."""
        try:
            # Get rules from Redis or memory
            if self.redis_client:
                rule_keys = self.redis_client.keys("coa_roe:rules:*")
                return [key.split(":")[-1] for key in rule_keys]
            else:
                return list(self.rule_features.keys())

        except Exception as e:
            logger.error(f"Error getting rules for optimization: {e}")
            return []

    def _get_rule_config(self, rule_id: str) -> Optional[Dict[str, Any]]:
        """Get rule configuration."""
        try:
            if self.redis_client:
                config_data = self.redis_client.get(f"coa_roe:rules:{rule_id}")
                if config_data:
                    return json.loads(config_data)

            return None

        except Exception as e:
            logger.error(f"Error getting rule config: {e}")
            return None

    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get optimization summary."""
        try:
            if not self.optimization_results:
                return {"total_optimizations": 0, "average_improvement": 0.0}

            improvements = [r.improvement for r in self.optimization_results]

            return {
                "total_optimizations": len(self.optimization_results),
                "average_improvement": np.mean(improvements),
                "max_improvement": np.max(improvements),
                "min_improvement": np.min(improvements),
                "recent_optimizations": [
                    asdict(r) for r in self.optimization_results[-10:]
                ],
            }

        except Exception as e:
            logger.error(f"Error getting optimization summary: {e}")
            return {"total_optimizations": 0, "average_improvement": 0.0}

    def get_rule_predictions(self) -> Dict[str, RulePrediction]:
        """Get all rule predictions."""
        return self.rule_predictions.copy()

    def get_model_status(self) -> Dict[str, Any]:
        """Get model status."""
        return {
            "trained": self.model_trained,
            "last_training": (
                self.last_training.isoformat() if self.last_training else None
            ),
            "training_data_size": len(self.training_data),
            "models_available": {
                "classifier": self.classifier is not None,
                "clusterer": self.clusterer is not None,
                "scaler": hasattr(self.scaler, "scale_"),
            },
        }
