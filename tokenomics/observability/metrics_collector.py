"""Metrics collector for observability."""

import time
from typing import Dict, Any, Optional, List
from collections import defaultdict
from threading import Lock
import structlog

logger = structlog.get_logger()


class MetricsCollector:
    """Collects metrics for observability."""
    
    def __init__(self):
        """Initialize metrics collector."""
        self.lock = Lock()
        self.metrics: Dict[str, Any] = {
            "queries": {
                "total": 0,
                "successful": 0,
                "failed": 0,
            },
            "latency": {
                "total_ms": 0.0,
                "count": 0,
                "p50": 0.0,
                "p95": 0.0,
                "p99": 0.0,
            },
            "tokens": {
                "total": 0,
                "input": 0,
                "output": 0,
            },
            "cache": {
                "hits": 0,
                "misses": 0,
            },
            "errors": defaultdict(int),
        }
        self.latency_history: List[float] = []
        logger.info("MetricsCollector initialized")
    
    def record_query(
        self,
        success: bool,
        latency_ms: float,
        tokens_used: int = 0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_hit: bool = False,
        error: Optional[str] = None,
    ):
        """Record a query execution."""
        with self.lock:
            self.metrics["queries"]["total"] += 1
            if success:
                self.metrics["queries"]["successful"] += 1
            else:
                self.metrics["queries"]["failed"] += 1
                if error:
                    self.metrics["errors"][error] += 1
            
            # Update latency
            self.metrics["latency"]["total_ms"] += latency_ms
            self.metrics["latency"]["count"] += 1
            self.latency_history.append(latency_ms)
            # Keep only last 1000 latencies
            if len(self.latency_history) > 1000:
                self.latency_history = self.latency_history[-1000:]
            
            # Calculate percentiles
            if self.latency_history:
                sorted_latencies = sorted(self.latency_history)
                self.metrics["latency"]["p50"] = sorted_latencies[len(sorted_latencies) // 2]
                self.metrics["latency"]["p95"] = sorted_latencies[int(len(sorted_latencies) * 0.95)]
                self.metrics["latency"]["p99"] = sorted_latencies[int(len(sorted_latencies) * 0.99)]
            
            # Update tokens
            self.metrics["tokens"]["total"] += tokens_used
            self.metrics["tokens"]["input"] += input_tokens
            self.metrics["tokens"]["output"] += output_tokens
            
            # Update cache
            if cache_hit:
                self.metrics["cache"]["hits"] += 1
            else:
                self.metrics["cache"]["misses"] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics snapshot."""
        with self.lock:
            metrics = self.metrics.copy()
            
            # Calculate averages
            if metrics["latency"]["count"] > 0:
                metrics["latency"]["avg_ms"] = metrics["latency"]["total_ms"] / metrics["latency"]["count"]
            else:
                metrics["latency"]["avg_ms"] = 0.0
            
            # Calculate cache hit rate
            total_cache_ops = metrics["cache"]["hits"] + metrics["cache"]["misses"]
            if total_cache_ops > 0:
                metrics["cache"]["hit_rate"] = (metrics["cache"]["hits"] / total_cache_ops) * 100
            else:
                metrics["cache"]["hit_rate"] = 0.0
            
            # Calculate success rate
            if metrics["queries"]["total"] > 0:
                metrics["queries"]["success_rate"] = (metrics["queries"]["successful"] / metrics["queries"]["total"]) * 100
            else:
                metrics["queries"]["success_rate"] = 0.0
            
            return metrics
    
    def reset(self):
        """Reset all metrics."""
        with self.lock:
            self.metrics = {
                "queries": {"total": 0, "successful": 0, "failed": 0},
                "latency": {"total_ms": 0.0, "count": 0, "p50": 0.0, "p95": 0.0, "p99": 0.0},
                "tokens": {"total": 0, "input": 0, "output": 0},
                "cache": {"hits": 0, "misses": 0},
                "errors": defaultdict(int),
            }
            self.latency_history = []
            logger.info("Metrics reset")






