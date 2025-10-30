"""
Performance Monitoring & Analytics System for COA/ROE System

This module provides comprehensive performance monitoring including:
- Real-time performance metrics collection
- Performance monitoring dashboards
- Alerting for performance degradation
- Performance trend analysis
- System health monitoring
"""

import json
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import psutil
import redis
from prometheus_client import Counter, Gauge, Histogram

import logging
# Use standard logging instead of FLAGSHIP.common
get_logger = logging.getLogger

logger = get_logger(__name__)

# ============================================================================
# PROMETHEUS METRICS
# ============================================================================

# Performance metrics
PERFORMANCE_OPERATIONS = Counter(
    "coa_roe_performance_operations_total",
    "Total performance operations",
    ["operation"],
)
PERFORMANCE_LATENCY = Histogram(
    "coa_roe_performance_latency_seconds", "Performance latency", ["operation"]
)
PERFORMANCE_THROUGHPUT = Counter(
    "coa_roe_performance_throughput_total", "Performance throughput", ["operation"]
)
PERFORMANCE_ERRORS = Counter(
    "coa_roe_performance_errors_total",
    "Performance errors",
    ["operation", "error_type"],
)

# System metrics
SYSTEM_CPU_USAGE = Gauge(
    "coa_roe_system_cpu_usage_percent", "System CPU usage percentage"
)
SYSTEM_MEMORY_USAGE = Gauge(
    "coa_roe_system_memory_usage_percent", "System memory usage percentage"
)
SYSTEM_DISK_USAGE = Gauge(
    "coa_roe_system_disk_usage_percent", "System disk usage percentage"
)
SYSTEM_NETWORK_IO = Counter(
    "coa_roe_system_network_io_bytes", "System network I/O", ["direction"]
)

# Cache metrics
CACHE_PERFORMANCE = Histogram(
    "coa_roe_cache_performance_seconds", "Cache performance", ["tier", "operation"]
)
CACHE_EFFICIENCY = Gauge("coa_roe_cache_efficiency_ratio", "Cache efficiency ratio")

# Rule correlation metrics
RULE_CORRELATION_LATENCY = Histogram(
    "coa_roe_rule_correlation_latency_seconds", "Rule correlation latency"
)
RULE_CORRELATION_SUCCESS = Counter(
    "coa_roe_rule_correlation_success_total", "Rule correlation success"
)
RULE_CORRELATION_FAILURES = Counter(
    "coa_roe_rule_correlation_failures_total", "Rule correlation failures"
)

# ============================================================================
# PERFORMANCE METRICS
# ============================================================================


@dataclass
class PerformanceMetric:
    """Performance metric data structure."""

    timestamp: datetime
    operation: str
    latency: float
    throughput: int
    error_count: int
    success_count: int
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, int]
    cache_hit_ratio: float
    rule_correlation_latency: float
    rule_correlation_success_rate: float


@dataclass
class AlertThreshold:
    """Alert threshold configuration."""

    metric_name: str
    warning_threshold: float
    critical_threshold: float
    duration: int  # seconds
    enabled: bool = True


# ============================================================================
# PERFORMANCE MONITOR
# ============================================================================


class PerformanceMonitor:
    """Comprehensive performance monitoring system."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the performance monitor."""
        self.config = config or {}
        self.logger = logger

        # Performance data storage
        self.metrics: List[PerformanceMetric] = []
        self.metrics_lock = threading.RLock()

        # Alert thresholds
        self.alert_thresholds: Dict[str, AlertThreshold] = {
            "latency": AlertThreshold("latency", 1.0, 5.0, 300),
            "cpu_usage": AlertThreshold("cpu_usage", 80.0, 95.0, 60),
            "memory_usage": AlertThreshold("memory_usage", 80.0, 95.0, 60),
            "disk_usage": AlertThreshold("disk_usage", 85.0, 95.0, 300),
            "cache_hit_ratio": AlertThreshold("cache_hit_ratio", 0.8, 0.6, 300),
            "rule_correlation_latency": AlertThreshold(
                "rule_correlation_latency", 0.5, 2.0, 300
            ),
            "error_rate": AlertThreshold("error_rate", 0.05, 0.1, 300),
        }

        # Alert state
        self.active_alerts: Dict[str, Dict[str, Any]] = {}
        self.alert_lock = threading.Lock()

        # Monitoring state
        self.monitoring_enabled = True
        self.monitoring_interval = self.config.get("monitoring_interval", 30)  # seconds
        self.metrics_retention = self.config.get("metrics_retention", 86400)  # 24 hours

        # Redis client for metrics storage
        self.redis_client = None
        self._initialize_redis()

        # Start monitoring
        self._start_monitoring()

        logger.info("Performance Monitor initialized")

    def _initialize_redis(self):
        """Initialize Redis connection for metrics storage."""
        try:
            redis_url = self.config.get("redis_url", "redis://localhost:6379")
            self.redis_client = redis.Redis.from_url(redis_url, decode_responses=True)
            self.redis_client.ping()
            logger.info("Redis connection established for metrics storage")
        except Exception as e:
            logger.warning(f"Failed to initialize Redis for metrics: {e}")
            self.redis_client = None

    def _start_monitoring(self):
        """Start the monitoring background task."""
        if self.monitoring_enabled:
            threading.Thread(target=self._monitoring_worker, daemon=True).start()
            logger.info("Performance monitoring started")

    def _monitoring_worker(self):
        """Background worker for performance monitoring."""
        while self.monitoring_enabled:
            try:
                # Collect system metrics
                metrics = self._collect_system_metrics()

                # Store metrics
                self._store_metrics(metrics)

                # Check alerts
                self._check_alerts(metrics)

                # Clean up old metrics
                self._cleanup_old_metrics()

                time.sleep(self.monitoring_interval)

            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                time.sleep(60)

    def _collect_system_metrics(self) -> PerformanceMetric:
        """Collect current system metrics."""
        try:
            # System metrics
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")
            network = psutil.net_io_counters()

            # Cache metrics (if available)
            cache_hit_ratio = self._get_cache_hit_ratio()

            # Rule correlation metrics (if available)
            rule_correlation_latency = self._get_rule_correlation_latency()
            rule_correlation_success_rate = self._get_rule_correlation_success_rate()

            # Create performance metric
            metric = PerformanceMetric(
                timestamp=datetime.now(),
                operation="system_monitoring",
                latency=0.0,  # Will be updated by specific operations
                throughput=0,  # Will be updated by specific operations
                error_count=0,  # Will be updated by specific operations
                success_count=0,  # Will be updated by specific operations
                cpu_usage=cpu_usage,
                memory_usage=memory.percent,
                disk_usage=disk.percent,
                network_io={
                    "bytes_sent": network.bytes_sent,
                    "bytes_recv": network.bytes_recv,
                },
                cache_hit_ratio=cache_hit_ratio,
                rule_correlation_latency=rule_correlation_latency,
                rule_correlation_success_rate=rule_correlation_success_rate,
            )

            # Update Prometheus metrics
            SYSTEM_CPU_USAGE.set(cpu_usage)
            SYSTEM_MEMORY_USAGE.set(memory.percent)
            SYSTEM_DISK_USAGE.set(disk.percent)
            SYSTEM_NETWORK_IO.labels(direction="sent").inc(network.bytes_sent)
            SYSTEM_NETWORK_IO.labels(direction="recv").inc(network.bytes_recv)

            return metric

        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return self._create_default_metric()

    def _get_cache_hit_ratio(self) -> float:
        """Get current cache hit ratio."""
        try:
            if self.redis_client:
                # Get cache statistics from Redis
                cache_stats = self.redis_client.hgetall("coa_roe:cache:stats")
                if cache_stats:
                    hits = int(cache_stats.get("hits", 0))
                    misses = int(cache_stats.get("misses", 0))
                    total = hits + misses
                    return hits / total if total > 0 else 0.0
        except Exception as e:
            logger.warning(f"Error getting cache hit ratio: {e}")

        return 0.0

    def _get_rule_correlation_latency(self) -> float:
        """Get current rule correlation latency."""
        try:
            if self.redis_client:
                # Get rule correlation metrics from Redis
                latency = self.redis_client.get("coa_roe:rule_correlation:latency")
                return float(latency) if latency else 0.0
        except Exception as e:
            logger.warning(f"Error getting rule correlation latency: {e}")

        return 0.0

    def _get_rule_correlation_success_rate(self) -> float:
        """Get current rule correlation success rate."""
        try:
            if self.redis_client:
                # Get rule correlation metrics from Redis
                success = int(
                    self.redis_client.get("coa_roe:rule_correlation:success") or 0
                )
                total = int(
                    self.redis_client.get("coa_roe:rule_correlation:total") or 0
                )
                return success / total if total > 0 else 1.0
        except Exception as e:
            logger.warning(f"Error getting rule correlation success rate: {e}")

        return 1.0

    def _create_default_metric(self) -> PerformanceMetric:
        """Create a default metric when collection fails."""
        return PerformanceMetric(
            timestamp=datetime.now(),
            operation="system_monitoring",
            latency=0.0,
            throughput=0,
            error_count=0,
            success_count=0,
            cpu_usage=0.0,
            memory_usage=0.0,
            disk_usage=0.0,
            network_io={"bytes_sent": 0, "bytes_recv": 0},
            cache_hit_ratio=0.0,
            rule_correlation_latency=0.0,
            rule_correlation_success_rate=1.0,
        )

    def _store_metrics(self, metric: PerformanceMetric):
        """Store metrics in memory and Redis."""
        try:
            # Store in memory
            with self.metrics_lock:
                self.metrics.append(metric)

                # Keep only recent metrics
                cutoff_time = datetime.now() - timedelta(seconds=self.metrics_retention)
                self.metrics = [m for m in self.metrics if m.timestamp > cutoff_time]

            # Store in Redis
            if self.redis_client:
                metric_data = asdict(metric)
                metric_data["timestamp"] = metric.timestamp.isoformat()

                # Store as JSON
                key = f"coa_roe:metrics:{int(metric.timestamp.timestamp())}"
                self.redis_client.setex(
                    key, self.metrics_retention, json.dumps(metric_data)
                )

                # Update latest metrics
                self.redis_client.set("coa_roe:metrics:latest", json.dumps(metric_data))

        except Exception as e:
            logger.error(f"Error storing metrics: {e}")

    def _check_alerts(self, metric: PerformanceMetric):
        """Check for alert conditions."""
        try:
            for alert_name, threshold in self.alert_thresholds.items():
                if not threshold.enabled:
                    continue

                # Get current value
                current_value = self._get_metric_value(metric, alert_name)

                # Check thresholds
                if current_value >= threshold.critical_threshold:
                    self._trigger_alert(
                        alert_name,
                        "CRITICAL",
                        current_value,
                        threshold.critical_threshold,
                    )
                elif current_value >= threshold.warning_threshold:
                    self._trigger_alert(
                        alert_name,
                        "WARNING",
                        current_value,
                        threshold.warning_threshold,
                    )
                else:
                    self._clear_alert(alert_name)

        except Exception as e:
            logger.error(f"Error checking alerts: {e}")

    def _get_metric_value(self, metric: PerformanceMetric, alert_name: str) -> float:
        """Get metric value for alert checking."""
        if alert_name == "latency":
            return metric.latency
        elif alert_name == "cpu_usage":
            return metric.cpu_usage
        elif alert_name == "memory_usage":
            return metric.memory_usage
        elif alert_name == "disk_usage":
            return metric.disk_usage
        elif alert_name == "cache_hit_ratio":
            return metric.cache_hit_ratio
        elif alert_name == "rule_correlation_latency":
            return metric.rule_correlation_latency
        elif alert_name == "error_rate":
            total = metric.error_count + metric.success_count
            return metric.error_count / total if total > 0 else 0.0
        else:
            return 0.0

    def _trigger_alert(
        self, alert_name: str, severity: str, current_value: float, threshold: float
    ):
        """Trigger an alert."""
        with self.alert_lock:
            alert_key = f"{alert_name}_{severity}"

            if alert_key not in self.active_alerts:
                alert = {
                    "alert_name": alert_name,
                    "severity": severity,
                    "current_value": current_value,
                    "threshold": threshold,
                    "timestamp": datetime.now(),
                    "message": f"{alert_name} is {severity}: {current_value:.2f} >= {threshold:.2f}",
                }

                self.active_alerts[alert_key] = alert

                # Log alert
                logger.warning(f"ALERT: {alert['message']}")

                # Store in Redis
                if self.redis_client:
                    alert_data = asdict(alert)
                    alert_data["timestamp"] = alert["timestamp"].isoformat()
                    self.redis_client.setex(
                        f"coa_roe:alerts:{alert_key}", 3600, json.dumps(alert_data)
                    )

    def _clear_alert(self, alert_name: str):
        """Clear an alert."""
        with self.alert_lock:
            alert_keys = [
                k for k in self.active_alerts.keys() if k.startswith(alert_name)
            ]
            for key in alert_keys:
                del self.active_alerts[key]

                # Remove from Redis
                if self.redis_client:
                    self.redis_client.delete(f"coa_roe:alerts:{key}")

    def _cleanup_old_metrics(self):
        """Clean up old metrics."""
        try:
            # Clean up memory metrics
            with self.metrics_lock:
                cutoff_time = datetime.now() - timedelta(seconds=self.metrics_retention)
                self.metrics = [m for m in self.metrics if m.timestamp > cutoff_time]

            # Clean up Redis metrics
            if self.redis_client:
                cutoff_timestamp = int(
                    (
                        datetime.now() - timedelta(seconds=self.metrics_retention)
                    ).timestamp()
                )
                keys = self.redis_client.keys("coa_roe:metrics:*")

                for key in keys:
                    try:
                        timestamp_str = key.split(":")[-1]
                        timestamp = int(timestamp_str)
                        if timestamp < cutoff_timestamp:
                            self.redis_client.delete(key)
                    except (ValueError, IndexError):
                        continue

        except Exception as e:
            logger.error(f"Error cleaning up old metrics: {e}")

    async def record_operation(
        self, operation: str, latency: float, success: bool = True
    ):
        """Record an operation performance."""
        try:
            # Update Prometheus metrics
            PERFORMANCE_OPERATIONS.labels(operation=operation).inc()
            PERFORMANCE_LATENCY.labels(operation=operation).observe(latency)
            PERFORMANCE_THROUGHPUT.labels(operation=operation).inc()

            if not success:
                PERFORMANCE_ERRORS.labels(
                    operation=operation, error_type="operation_failed"
                ).inc()

            # Store in Redis for real-time monitoring
            if self.redis_client:
                operation_data = {
                    "operation": operation,
                    "latency": latency,
                    "success": success,
                    "timestamp": datetime.now().isoformat(),
                }

                # Store operation metrics
                key = f"coa_roe:operations:{operation}:{int(time.time())}"
                self.redis_client.setex(key, 3600, json.dumps(operation_data))

                # Update operation statistics
                stats_key = f"coa_roe:operations:stats:{operation}"
                self.redis_client.hincrby(stats_key, "count", 1)
                self.redis_client.hincrby(
                    stats_key, "total_latency", int(latency * 1000)
                )

                if success:
                    self.redis_client.hincrby(stats_key, "success_count", 1)
                else:
                    self.redis_client.hincrby(stats_key, "error_count", 1)

        except Exception as e:
            logger.error(f"Error recording operation: {e}")

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        try:
            with self.metrics_lock:
                if not self.metrics:
                    return self._create_empty_summary()

                # Calculate statistics
                latencies = [m.latency for m in self.metrics]
                cpu_usage = [m.cpu_usage for m in self.metrics]
                memory_usage = [m.memory_usage for m in self.metrics]
                cache_hit_ratios = [m.cache_hit_ratio for m in self.metrics]

                summary = {
                    "timestamp": datetime.now().isoformat(),
                    "metrics_count": len(self.metrics),
                    "latency": {
                        "mean": np.mean(latencies) if latencies else 0.0,
                        "median": np.median(latencies) if latencies else 0.0,
                        "p95": np.percentile(latencies, 95) if latencies else 0.0,
                        "p99": np.percentile(latencies, 99) if latencies else 0.0,
                    },
                    "system": {
                        "cpu_usage_mean": np.mean(cpu_usage) if cpu_usage else 0.0,
                        "memory_usage_mean": (
                            np.mean(memory_usage) if memory_usage else 0.0
                        ),
                        "cache_hit_ratio_mean": (
                            np.mean(cache_hit_ratios) if cache_hit_ratios else 0.0
                        ),
                    },
                    "alerts": {
                        "active_count": len(self.active_alerts),
                        "active_alerts": list(self.active_alerts.values()),
                    },
                }

                return summary

        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return self._create_empty_summary()

    def _create_empty_summary(self) -> Dict[str, Any]:
        """Create empty performance summary."""
        return {
            "timestamp": datetime.now().isoformat(),
            "metrics_count": 0,
            "latency": {"mean": 0.0, "median": 0.0, "p95": 0.0, "p99": 0.0},
            "system": {
                "cpu_usage_mean": 0.0,
                "memory_usage_mean": 0.0,
                "cache_hit_ratio_mean": 0.0,
            },
            "alerts": {"active_count": 0, "active_alerts": []},
        }

    def get_metrics_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get metrics history for the specified hours."""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)

            with self.metrics_lock:
                recent_metrics = [m for m in self.metrics if m.timestamp > cutoff_time]

            return [asdict(m) for m in recent_metrics]

        except Exception as e:
            logger.error(f"Error getting metrics history: {e}")
            return []

    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get active alerts."""
        with self.alert_lock:
            return list(self.active_alerts.values())

    def update_alert_threshold(
        self,
        alert_name: str,
        warning_threshold: float,
        critical_threshold: float,
        duration: int = 300,
    ):
        """Update alert threshold."""
        if alert_name in self.alert_thresholds:
            self.alert_thresholds[alert_name] = AlertThreshold(
                alert_name, warning_threshold, critical_threshold, duration
            )
            logger.info(f"Updated alert threshold for {alert_name}")
        else:
            logger.warning(f"Unknown alert name: {alert_name}")

    def enable_monitoring(self, enabled: bool = True):
        """Enable or disable monitoring."""
        self.monitoring_enabled = enabled
        logger.info(f"Performance monitoring {'enabled' if enabled else 'disabled'}")

    def get_system_health(self) -> Dict[str, Any]:
        """Get system health status."""
        try:
            # Get current system metrics
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")

            # Determine health status
            health_status = "HEALTHY"
            if cpu_usage > 90 or memory.percent > 90 or disk.percent > 90:
                health_status = "CRITICAL"
            elif cpu_usage > 80 or memory.percent > 80 or disk.percent > 80:
                health_status = "WARNING"

            return {
                "status": health_status,
                "timestamp": datetime.now().isoformat(),
                "cpu_usage": cpu_usage,
                "memory_usage": memory.percent,
                "disk_usage": disk.percent,
                "active_alerts": len(self.active_alerts),
            }

        except Exception as e:
            logger.error(f"Error getting system health: {e}")
            return {
                "status": "UNKNOWN",
                "timestamp": datetime.now().isoformat(),
                "cpu_usage": 0.0,
                "memory_usage": 0.0,
                "disk_usage": 0.0,
                "active_alerts": 0,
            }
