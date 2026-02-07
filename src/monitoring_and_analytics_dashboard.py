"""
Monitoring and Analytics Dashboard

Build a comprehensive monitoring system to track usage, performance, and quality metrics.
Provides real-time visibility into RAG system behavior with:
- Metrics collection and storage
- Alerting for threshold violations
- User analytics and behavior tracking
- A/B testing framework for experiments
- Text-based visualization and reporting

All implementations use only the Python standard library for maximum compatibility.
"""

import hashlib
import json
import logging
import math
import random
import statistics
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics that can be collected."""
    COUNTER = "counter"  # Monotonically increasing value
    GAUGE = "gauge"  # Value that can go up or down
    HISTOGRAM = "histogram"  # Distribution of values
    TIMER = "timer"  # Duration measurements


class AlertSeverity(Enum):
    """Severity levels for alerts."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class ExperimentStatus(Enum):
    """Status of an A/B test experiment."""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"


@dataclass
class MetricPoint:
    """A single metric data point."""
    name: str
    value: float
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE


@dataclass
class Alert:
    """Represents a triggered alert."""
    alert_id: str
    name: str
    message: str
    severity: AlertSeverity
    metric_name: str
    threshold: float
    current_value: float
    triggered_at: float
    resolved_at: Optional[float] = None

    @property
    def is_active(self) -> bool:
        """Check if alert is still active."""
        return self.resolved_at is None

    @property
    def duration_seconds(self) -> float:
        """Get alert duration."""
        end_time = self.resolved_at or time.time()
        return end_time - self.triggered_at


@dataclass
class AlertRule:
    """Configuration for an alerting rule."""
    name: str
    metric_name: str
    condition: str  # "gt", "lt", "gte", "lte", "eq"
    threshold: float
    severity: AlertSeverity = AlertSeverity.WARNING
    cooldown_seconds: float = 300.0  # Minimum time between alerts
    last_triggered: Optional[float] = None

    def check(self, value: float) -> bool:
        """Check if the rule is triggered by a value."""
        if self.condition == "gt":
            return value > self.threshold
        elif self.condition == "lt":
            return value < self.threshold
        elif self.condition == "gte":
            return value >= self.threshold
        elif self.condition == "lte":
            return value <= self.threshold
        elif self.condition == "eq":
            return value == self.threshold
        return False


class DataStore:
    """
    Thread-safe time-series data store with JSON persistence.

    Stores metric data points with efficient retrieval by time range
    and metric name. Supports optional persistence to disk.

    Example:
        store = DataStore(persistence_path="/tmp/metrics.json")
        store.store(MetricPoint("response_time", 1.5, time.time()))
        points = store.query("response_time", start_time, end_time)
    """

    def __init__(self, max_points: int = 100000,
                 persistence_path: Optional[str] = None):
        """
        Initialize the data store.

        Args:
            max_points: Maximum data points to retain in memory
            persistence_path: Optional path for JSON persistence
        """
        self.max_points = max_points
        self.persistence_path = persistence_path
        self._data: Dict[str, List[MetricPoint]] = defaultdict(list)
        self._lock = threading.Lock()

        if persistence_path:
            self._load_from_disk()

    def store(self, point: MetricPoint) -> None:
        """
        Store a metric data point.

        Args:
            point: MetricPoint to store
        """
        with self._lock:
            self._data[point.name].append(point)

            # Trim if over capacity (remove oldest)
            if len(self._data[point.name]) > self.max_points:
                self._data[point.name] = self._data[point.name][-self.max_points:]

    def query(self, metric_name: str, start_time: Optional[float] = None,
              end_time: Optional[float] = None) -> List[MetricPoint]:
        """
        Query metric data points.

        Args:
            metric_name: Name of metric to query
            start_time: Start of time range (inclusive)
            end_time: End of time range (inclusive)

        Returns:
            List of matching MetricPoints
        """
        with self._lock:
            points = self._data.get(metric_name, [])

            if start_time is not None:
                points = [p for p in points if p.timestamp >= start_time]
            if end_time is not None:
                points = [p for p in points if p.timestamp <= end_time]

            return list(points)

    def get_latest(self, metric_name: str) -> Optional[MetricPoint]:
        """Get the most recent data point for a metric."""
        with self._lock:
            points = self._data.get(metric_name, [])
            return points[-1] if points else None

    def get_metrics(self) -> List[str]:
        """Get list of all metric names."""
        with self._lock:
            return list(self._data.keys())

    def persist(self) -> None:
        """Persist data to disk."""
        if not self.persistence_path:
            return

        with self._lock:
            data = {
                name: [
                    {
                        "name": p.name,
                        "value": p.value,
                        "timestamp": p.timestamp,
                        "tags": p.tags,
                        "metric_type": p.metric_type.value
                    }
                    for p in points
                ]
                for name, points in self._data.items()
            }

        try:
            with open(self.persistence_path, 'w') as f:
                json.dump(data, f)
        except IOError as e:
            logger.error(f"Failed to persist data: {e}")

    def _load_from_disk(self) -> None:
        """Load data from disk."""
        if not self.persistence_path:
            return

        try:
            path = Path(self.persistence_path)
            if path.exists():
                with open(path, 'r') as f:
                    data = json.load(f)

                for name, points in data.items():
                    self._data[name] = [
                        MetricPoint(
                            name=p["name"],
                            value=p["value"],
                            timestamp=p["timestamp"],
                            tags=p.get("tags", {}),
                            metric_type=MetricType(p.get("metric_type", "gauge"))
                        )
                        for p in points
                    ]
        except (IOError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to load persisted data: {e}")


class MetricsCollector:
    """
    Collects and aggregates metrics from RAG system operations.

    Provides a simple interface for recording various metric types
    including counters, gauges, histograms, and timers.

    Example:
        collector = MetricsCollector()
        collector.record_response_time(1.5, user_id="user123")
        collector.increment_counter("queries_total")
        summary = collector.get_summary("response_time")
    """

    def __init__(self, data_store: Optional[DataStore] = None):
        """
        Initialize the metrics collector.

        Args:
            data_store: DataStore for persistence (creates new if not provided)
        """
        self.data_store = data_store or DataStore()
        self._counters: Dict[str, float] = defaultdict(float)
        self._lock = threading.Lock()

    def record(self, name: str, value: float,
               metric_type: MetricType = MetricType.GAUGE,
               tags: Optional[Dict[str, str]] = None) -> None:
        """
        Record a metric value.

        Args:
            name: Metric name
            value: Metric value
            metric_type: Type of metric
            tags: Additional metadata tags
        """
        point = MetricPoint(
            name=name,
            value=value,
            timestamp=time.time(),
            tags=tags or {},
            metric_type=metric_type
        )
        self.data_store.store(point)

    def increment_counter(self, name: str, value: float = 1.0,
                         tags: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter metric."""
        with self._lock:
            self._counters[name] += value

        self.record(name, self._counters[name], MetricType.COUNTER, tags)

    def record_response_time(self, seconds: float, **tags) -> None:
        """Record a response time measurement."""
        self.record("response_time", seconds, MetricType.TIMER, tags)

    def record_accuracy(self, score: float, **tags) -> None:
        """Record an accuracy score (0-1)."""
        self.record("accuracy", score, MetricType.GAUGE, tags)

    def record_user_satisfaction(self, score: float, **tags) -> None:
        """Record a user satisfaction score (1-5)."""
        self.record("user_satisfaction", score, MetricType.GAUGE, tags)

    def record_cost(self, amount: float, **tags) -> None:
        """Record a cost amount."""
        self.record("cost", amount, MetricType.COUNTER, tags)
        self.increment_counter("cost_total", amount)

    def record_query_execution(self, data: Dict[str, Any]) -> None:
        """
        Record comprehensive query execution data.

        Args:
            data: Dictionary with query metrics (response_time, cost, accuracy, etc.)
        """
        tags = {}
        if "user_id" in data:
            tags["user_id"] = data["user_id"]

        if "response_time" in data:
            self.record_response_time(data["response_time"], **tags)

        if "accuracy_score" in data:
            self.record_accuracy(data["accuracy_score"], **tags)

        if "cost" in data:
            self.record_cost(data["cost"], **tags)

        if "retrieval_time" in data:
            self.record("retrieval_time", data["retrieval_time"], MetricType.TIMER, tags)

        if "generation_time" in data:
            self.record("generation_time", data["generation_time"], MetricType.TIMER, tags)

        self.increment_counter("queries_total", tags=tags)

    def get_summary(self, metric_name: str,
                    start_time: Optional[float] = None) -> Dict[str, Any]:
        """
        Get statistical summary for a metric.

        Args:
            metric_name: Name of metric to summarize
            start_time: Start time for analysis window

        Returns:
            Dictionary with count, mean, std, min, max, percentiles
        """
        points = self.data_store.query(metric_name, start_time)

        if not points:
            return {"count": 0}

        values = [p.value for p in points]
        values.sort()

        return {
            "count": len(values),
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0,
            "min": min(values),
            "max": max(values),
            "p50": values[int(len(values) * 0.5)],
            "p95": values[int(len(values) * 0.95)] if len(values) >= 20 else values[-1],
            "p99": values[int(len(values) * 0.99)] if len(values) >= 100 else values[-1]
        }


class AlertManager:
    """
    Manages alerting rules and notifications for system monitoring.

    Monitors metric thresholds and generates alerts when conditions
    are met. Supports multiple severity levels and cooldown periods.

    Example:
        manager = AlertManager(collector)
        manager.add_rule(AlertRule(
            name="High Latency",
            metric_name="response_time",
            condition="gt",
            threshold=5.0,
            severity=AlertSeverity.WARNING
        ))
        alerts = manager.check_rules()
    """

    def __init__(self, metrics_collector: MetricsCollector):
        """
        Initialize the alert manager.

        Args:
            metrics_collector: MetricsCollector to monitor
        """
        self.collector = metrics_collector
        self._rules: List[AlertRule] = []
        self._alerts: List[Alert] = []
        self._lock = threading.Lock()
        self._alert_handlers: List[Callable[[Alert], None]] = []

        # Add default rules
        self._add_default_rules()

    def _add_default_rules(self) -> None:
        """Add default alerting rules."""
        self._rules.extend([
            AlertRule(
                name="High Response Time",
                metric_name="response_time",
                condition="gt",
                threshold=5.0,
                severity=AlertSeverity.WARNING
            ),
            AlertRule(
                name="Critical Response Time",
                metric_name="response_time",
                condition="gt",
                threshold=10.0,
                severity=AlertSeverity.CRITICAL
            ),
            AlertRule(
                name="Low User Satisfaction",
                metric_name="user_satisfaction",
                condition="lt",
                threshold=3.0,
                severity=AlertSeverity.WARNING
            ),
            AlertRule(
                name="Low Accuracy",
                metric_name="accuracy",
                condition="lt",
                threshold=0.5,
                severity=AlertSeverity.WARNING
            )
        ])

    def add_rule(self, rule: AlertRule) -> None:
        """Add a new alerting rule."""
        with self._lock:
            self._rules.append(rule)

    def add_handler(self, handler: Callable[[Alert], None]) -> None:
        """Add an alert handler callback."""
        self._alert_handlers.append(handler)

    def check_rules(self) -> List[Alert]:
        """
        Check all rules against current metrics.

        Returns:
            List of newly triggered alerts
        """
        new_alerts = []
        current_time = time.time()

        with self._lock:
            for rule in self._rules:
                # Get latest metric value
                point = self.collector.data_store.get_latest(rule.metric_name)
                if point is None:
                    continue

                # Check cooldown
                if (rule.last_triggered and
                    current_time - rule.last_triggered < rule.cooldown_seconds):
                    continue

                # Check rule condition
                if rule.check(point.value):
                    alert = Alert(
                        alert_id=f"{rule.name}_{int(current_time)}",
                        name=rule.name,
                        message=f"{rule.name}: {point.value:.2f} {rule.condition} {rule.threshold}",
                        severity=rule.severity,
                        metric_name=rule.metric_name,
                        threshold=rule.threshold,
                        current_value=point.value,
                        triggered_at=current_time
                    )

                    self._alerts.append(alert)
                    new_alerts.append(alert)
                    rule.last_triggered = current_time

                    # Notify handlers
                    for handler in self._alert_handlers:
                        try:
                            handler(alert)
                        except Exception as e:
                            logger.error(f"Alert handler error: {e}")

                    logger.warning(f"Alert triggered: {alert.message}")

        return new_alerts

    def resolve_alert(self, alert_id: str) -> bool:
        """
        Resolve an active alert.

        Args:
            alert_id: ID of alert to resolve

        Returns:
            True if alert was found and resolved
        """
        with self._lock:
            for alert in self._alerts:
                if alert.alert_id == alert_id and alert.is_active:
                    alert.resolved_at = time.time()
                    return True
        return False

    def get_active_alerts(self) -> List[Alert]:
        """Get all currently active alerts."""
        with self._lock:
            return [a for a in self._alerts if a.is_active]

    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """Get alert history for the past N hours."""
        cutoff = time.time() - (hours * 3600)
        with self._lock:
            return [a for a in self._alerts if a.triggered_at > cutoff]


@dataclass
class Experiment:
    """Configuration for an A/B test experiment."""
    experiment_id: str
    name: str
    description: str
    variants: List[str]  # e.g., ["control", "treatment"]
    traffic_split: Dict[str, float]  # e.g., {"control": 0.5, "treatment": 0.5}
    status: ExperimentStatus = ExperimentStatus.DRAFT
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    ended_at: Optional[float] = None


@dataclass
class ExperimentResult:
    """Results from an experiment variant."""
    experiment_id: str
    variant: str
    user_id: str
    metric_name: str
    value: float
    timestamp: float = field(default_factory=time.time)


class ABTestingFramework:
    """
    A/B testing framework for RAG system experiments.

    Supports controlled experiments with:
    - Multiple variants with configurable traffic splits
    - User-consistent variant assignment
    - Statistical significance testing
    - Result aggregation and analysis

    Example:
        ab = ABTestingFramework()
        exp = ab.create_experiment("vector_db_test", ["chromadb", "pinecone"])
        variant = ab.get_variant(experiment_id, user_id)
        ab.record_result(experiment_id, variant, user_id, "accuracy", 0.85)
        analysis = ab.analyze_experiment(experiment_id)
    """

    def __init__(self):
        """Initialize the A/B testing framework."""
        self._experiments: Dict[str, Experiment] = {}
        self._results: List[ExperimentResult] = []
        self._assignments: Dict[Tuple[str, str], str] = {}  # (exp_id, user_id) -> variant
        self._lock = threading.Lock()

    def create_experiment(self, name: str, variants: List[str],
                         description: str = "",
                         traffic_split: Optional[Dict[str, float]] = None) -> Experiment:
        """
        Create a new experiment.

        Args:
            name: Experiment name
            variants: List of variant names
            description: Experiment description
            traffic_split: Traffic allocation per variant (defaults to equal split)

        Returns:
            Created Experiment object
        """
        exp_id = f"exp_{int(time.time())}_{hashlib.md5(name.encode()).hexdigest()[:8]}"

        if traffic_split is None:
            # Equal split
            split = 1.0 / len(variants)
            traffic_split = {v: split for v in variants}

        experiment = Experiment(
            experiment_id=exp_id,
            name=name,
            description=description,
            variants=variants,
            traffic_split=traffic_split
        )

        with self._lock:
            self._experiments[exp_id] = experiment

        logger.info(f"Created experiment: {name} ({exp_id})")
        return experiment

    def start_experiment(self, experiment_id: str) -> bool:
        """Start an experiment."""
        with self._lock:
            if experiment_id not in self._experiments:
                return False
            exp = self._experiments[experiment_id]
            exp.status = ExperimentStatus.RUNNING
            exp.started_at = time.time()
            return True

    def stop_experiment(self, experiment_id: str) -> bool:
        """Stop an experiment."""
        with self._lock:
            if experiment_id not in self._experiments:
                return False
            exp = self._experiments[experiment_id]
            exp.status = ExperimentStatus.COMPLETED
            exp.ended_at = time.time()
            return True

    def get_variant(self, experiment_id: str, user_id: str) -> Optional[str]:
        """
        Get the variant for a user in an experiment.

        Uses consistent hashing to ensure users always get the same variant.

        Args:
            experiment_id: ID of the experiment
            user_id: User identifier

        Returns:
            Variant name or None if experiment not found
        """
        with self._lock:
            if experiment_id not in self._experiments:
                return None

            exp = self._experiments[experiment_id]
            if exp.status != ExperimentStatus.RUNNING:
                return None

            # Check for existing assignment
            key = (experiment_id, user_id)
            if key in self._assignments:
                return self._assignments[key]

            # Deterministic assignment using hash
            hash_input = f"{experiment_id}:{user_id}"
            hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
            bucket = (hash_value % 10000) / 10000.0

            cumulative = 0.0
            for variant, split in exp.traffic_split.items():
                cumulative += split
                if bucket < cumulative:
                    self._assignments[key] = variant
                    return variant

            # Fallback to first variant
            variant = exp.variants[0]
            self._assignments[key] = variant
            return variant

    def record_result(self, experiment_id: str, variant: str,
                     user_id: str, metric_name: str, value: float) -> None:
        """
        Record a result for an experiment variant.

        Args:
            experiment_id: ID of the experiment
            variant: Variant that was used
            user_id: User identifier
            metric_name: Name of the metric
            value: Metric value
        """
        result = ExperimentResult(
            experiment_id=experiment_id,
            variant=variant,
            user_id=user_id,
            metric_name=metric_name,
            value=value
        )

        with self._lock:
            self._results.append(result)

    def analyze_experiment(self, experiment_id: str,
                          metric_name: str = "accuracy") -> Dict[str, Any]:
        """
        Analyze experiment results with statistical significance.

        Args:
            experiment_id: ID of the experiment to analyze
            metric_name: Metric to analyze

        Returns:
            Analysis results including means, confidence, and significance
        """
        with self._lock:
            if experiment_id not in self._experiments:
                return {"error": "Experiment not found"}

            exp = self._experiments[experiment_id]

            # Get results by variant
            variant_results: Dict[str, List[float]] = {v: [] for v in exp.variants}

            for result in self._results:
                if result.experiment_id == experiment_id and result.metric_name == metric_name:
                    if result.variant in variant_results:
                        variant_results[result.variant].append(result.value)

            # Calculate statistics for each variant
            variant_stats = {}
            for variant, values in variant_results.items():
                if values:
                    variant_stats[variant] = {
                        "count": len(values),
                        "mean": statistics.mean(values),
                        "std": statistics.stdev(values) if len(values) > 1 else 0,
                        "min": min(values),
                        "max": max(values)
                    }
                else:
                    variant_stats[variant] = {"count": 0}

            # Calculate statistical significance between variants
            significance = self._calculate_significance(variant_results, exp.variants)

            return {
                "experiment_id": experiment_id,
                "name": exp.name,
                "metric": metric_name,
                "status": exp.status.value,
                "variant_stats": variant_stats,
                "significance": significance,
                "total_samples": sum(len(v) for v in variant_results.values())
            }

    def _calculate_significance(self, variant_results: Dict[str, List[float]],
                                variants: List[str]) -> Dict[str, Any]:
        """Calculate statistical significance between variants using t-test."""
        if len(variants) < 2:
            return {"significant": False, "reason": "Need at least 2 variants"}

        # Get first two variants for comparison
        v1, v2 = variants[0], variants[1]
        values1, values2 = variant_results.get(v1, []), variant_results.get(v2, [])

        if len(values1) < 30 or len(values2) < 30:
            return {
                "significant": False,
                "reason": "Need at least 30 samples per variant",
                "samples": {"v1": len(values1), "v2": len(values2)}
            }

        # Welch's t-test approximation
        mean1, mean2 = statistics.mean(values1), statistics.mean(values2)
        var1, var2 = statistics.variance(values1), statistics.variance(values2)
        n1, n2 = len(values1), len(values2)

        se = math.sqrt(var1/n1 + var2/n2) if (var1/n1 + var2/n2) > 0 else 1
        t_stat = (mean1 - mean2) / se if se > 0 else 0

        # Approximate p-value (simplified)
        # For |t| > 2, generally significant at p < 0.05
        significant = abs(t_stat) > 1.96

        return {
            "significant": significant,
            "t_statistic": round(t_stat, 4),
            "effect_size": round(mean2 - mean1, 4),
            "comparison": f"{v1} vs {v2}",
            "winner": v2 if mean2 > mean1 and significant else (v1 if mean1 > mean2 and significant else None)
        }

    def get_experiments(self, status: Optional[ExperimentStatus] = None) -> List[Experiment]:
        """Get all experiments, optionally filtered by status."""
        with self._lock:
            experiments = list(self._experiments.values())
            if status:
                experiments = [e for e in experiments if e.status == status]
            return experiments


class AnalyticsEngine:
    """
    Analyzes usage patterns and generates insights from RAG system data.

    Provides analytics including:
    - Query pattern analysis
    - User behavior tracking
    - Performance trend identification
    - Usage forecasting

    Example:
        engine = AnalyticsEngine(collector)
        patterns = engine.analyze_query_patterns()
        trends = engine.get_performance_trends()
    """

    def __init__(self, metrics_collector: MetricsCollector):
        """
        Initialize the analytics engine.

        Args:
            metrics_collector: MetricsCollector to analyze
        """
        self.collector = metrics_collector
        self._user_sessions: Dict[str, List[Dict]] = defaultdict(list)
        self._lock = threading.Lock()

    def track_user_action(self, user_id: str, action: str,
                         metadata: Optional[Dict] = None) -> None:
        """
        Track a user action for analytics.

        Args:
            user_id: User identifier
            action: Action type (e.g., "query", "feedback")
            metadata: Additional action metadata
        """
        with self._lock:
            self._user_sessions[user_id].append({
                "action": action,
                "timestamp": time.time(),
                "metadata": metadata or {}
            })

    def get_user_analytics(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get user analytics for the specified time period.

        Args:
            hours: Number of hours to analyze

        Returns:
            User analytics including active users, actions, patterns
        """
        cutoff = time.time() - (hours * 3600)

        with self._lock:
            active_users = set()
            action_counts: Dict[str, int] = defaultdict(int)
            sessions_by_hour: Dict[int, int] = defaultdict(int)

            for user_id, actions in self._user_sessions.items():
                user_actions = [a for a in actions if a["timestamp"] > cutoff]
                if user_actions:
                    active_users.add(user_id)
                    for action in user_actions:
                        action_counts[action["action"]] += 1
                        hour = int((time.time() - action["timestamp"]) / 3600)
                        sessions_by_hour[hour] += 1

        return {
            "active_users": len(active_users),
            "total_actions": sum(action_counts.values()),
            "action_breakdown": dict(action_counts),
            "hourly_distribution": dict(sessions_by_hour),
            "period_hours": hours
        }

    def analyze_query_patterns(self, hours: int = 24) -> Dict[str, Any]:
        """
        Analyze query patterns and trends.

        Args:
            hours: Number of hours to analyze

        Returns:
            Query pattern analysis
        """
        cutoff = time.time() - (hours * 3600)

        # Get query metrics
        queries = self.collector.data_store.query("queries_total", cutoff)
        response_times = self.collector.data_store.query("response_time", cutoff)

        # Calculate hourly distribution
        hourly_queries: Dict[int, int] = defaultdict(int)
        for point in queries:
            hour = int((time.time() - point.timestamp) / 3600)
            hourly_queries[hour] += 1

        # Identify peak hours
        peak_hour = max(hourly_queries.items(), key=lambda x: x[1])[0] if hourly_queries else 0

        return {
            "total_queries": len(queries),
            "period_hours": hours,
            "queries_per_hour": len(queries) / hours if hours > 0 else 0,
            "peak_hour_offset": peak_hour,
            "hourly_distribution": dict(hourly_queries),
            "avg_response_time": statistics.mean([p.value for p in response_times]) if response_times else 0
        }

    def get_performance_trends(self, hours: int = 24,
                               interval_minutes: int = 60) -> Dict[str, Any]:
        """
        Get performance trends over time.

        Args:
            hours: Number of hours to analyze
            interval_minutes: Aggregation interval

        Returns:
            Performance trends with time series data
        """
        cutoff = time.time() - (hours * 3600)
        interval_seconds = interval_minutes * 60

        response_times = self.collector.data_store.query("response_time", cutoff)
        accuracies = self.collector.data_store.query("accuracy", cutoff)

        # Aggregate by interval
        def aggregate_by_interval(points: List[MetricPoint]) -> List[Dict]:
            if not points:
                return []

            result = []
            current_interval = int(points[0].timestamp / interval_seconds)
            interval_values = []

            for point in sorted(points, key=lambda p: p.timestamp):
                point_interval = int(point.timestamp / interval_seconds)

                if point_interval != current_interval:
                    if interval_values:
                        result.append({
                            "interval": current_interval,
                            "timestamp": current_interval * interval_seconds,
                            "mean": statistics.mean(interval_values),
                            "count": len(interval_values)
                        })
                    current_interval = point_interval
                    interval_values = []

                interval_values.append(point.value)

            # Add last interval
            if interval_values:
                result.append({
                    "interval": current_interval,
                    "timestamp": current_interval * interval_seconds,
                    "mean": statistics.mean(interval_values),
                    "count": len(interval_values)
                })

            return result

        return {
            "period_hours": hours,
            "interval_minutes": interval_minutes,
            "response_time_trend": aggregate_by_interval(response_times),
            "accuracy_trend": aggregate_by_interval(accuracies)
        }


class TextVisualizer:
    """
    Creates text-based visualizations for terminal display.

    Provides ASCII charts, tables, and dashboards for
    displaying metrics and analytics data.
    """

    def __init__(self, width: int = 60):
        """
        Initialize the visualizer.

        Args:
            width: Maximum width for visualizations
        """
        self.width = width

    def horizontal_bar(self, value: float, max_value: float,
                       width: int = 40, fill: str = "=") -> str:
        """Create a horizontal bar chart element."""
        if max_value <= 0:
            return ""
        ratio = min(value / max_value, 1.0)
        filled = int(width * ratio)
        return f"[{fill * filled}{' ' * (width - filled)}]"

    def sparkline(self, values: List[float], width: int = 20) -> str:
        """Create a sparkline from a list of values."""
        if not values:
            return " " * width

        chars = "▁▂▃▄▅▆▇█"
        min_val, max_val = min(values), max(values)
        range_val = max_val - min_val if max_val != min_val else 1

        # Sample values if too many
        if len(values) > width:
            step = len(values) / width
            values = [values[int(i * step)] for i in range(width)]

        result = ""
        for v in values:
            idx = int((v - min_val) / range_val * (len(chars) - 1))
            result += chars[idx]

        return result

    def table(self, headers: List[str], rows: List[List[str]]) -> str:
        """Create a formatted table."""
        if not rows:
            return ""

        # Calculate column widths
        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                if i < len(col_widths):
                    col_widths[i] = max(col_widths[i], len(str(cell)))

        # Build table
        separator = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
        header_row = "|" + "|".join(f" {h:<{col_widths[i]}} " for i, h in enumerate(headers)) + "|"

        lines = [separator, header_row, separator]

        for row in rows:
            row_str = "|" + "|".join(
                f" {str(row[i]) if i < len(row) else '':<{col_widths[i]}} "
                for i in range(len(headers))
            ) + "|"
            lines.append(row_str)

        lines.append(separator)
        return "\n".join(lines)

    def status_indicator(self, status: str) -> str:
        """Create a status indicator."""
        indicators = {
            "healthy": "[OK]",
            "warning": "[!!]",
            "critical": "[XX]",
            "unknown": "[??]"
        }
        return indicators.get(status.lower(), "[--]")

    def progress_bar(self, current: float, total: float,
                    width: int = 30, prefix: str = "") -> str:
        """Create a progress bar."""
        if total <= 0:
            ratio = 0
        else:
            ratio = min(current / total, 1.0)

        filled = int(width * ratio)
        bar = "=" * filled + ">" + " " * (width - filled - 1) if filled < width else "=" * width
        percent = f"{ratio * 100:.1f}%"

        return f"{prefix}[{bar}] {percent}"


class MonitoringAndAnalyticsDashboard:
    """
    Comprehensive monitoring dashboard for RAG systems.

    Integrates metrics collection, alerting, analytics, and A/B testing
    into a unified dashboard interface with visualization capabilities.

    Example:
        dashboard = MonitoringAndAnalyticsDashboard()

        # Record query execution
        dashboard.record_query_execution({
            'query': 'What is ML?',
            'response_time': 2.3,
            'cost': 0.02,
            'accuracy_score': 0.85
        })

        # Check for alerts
        dashboard.check_alerts()

        # Display dashboard
        dashboard.execute()
    """

    def __init__(self, persistence_path: Optional[str] = None):
        """
        Initialize the monitoring dashboard.

        Args:
            persistence_path: Optional path for data persistence
        """
        # Initialize components
        self.data_store = DataStore(persistence_path=persistence_path)
        self.metrics_collector = MetricsCollector(data_store=self.data_store)
        self.alert_manager = AlertManager(self.metrics_collector)
        self.ab_testing = ABTestingFramework()
        self.analytics_engine = AnalyticsEngine(self.metrics_collector)
        self.visualizer = TextVisualizer()

        self._start_time = time.time()
        logger.info("Monitoring and Analytics Dashboard initialized")

    def record_query_execution(self, data: Dict[str, Any]) -> None:
        """
        Record comprehensive query execution data.

        Args:
            data: Dictionary with query metrics
        """
        self.metrics_collector.record_query_execution(data)

        if "user_id" in data:
            self.analytics_engine.track_user_action(
                data["user_id"],
                "query",
                {"query": data.get("query", "")}
            )

    def check_alerts(self) -> List[Alert]:
        """Check all alerting rules and return any new alerts."""
        return self.alert_manager.check_rules()

    def get_system_status(self) -> Dict[str, Any]:
        """
        Get current system status overview.

        Returns:
            Dictionary with system health and key metrics
        """
        # Get recent metrics
        now = time.time()
        last_hour = now - 3600

        response_summary = self.metrics_collector.get_summary("response_time", last_hour)
        accuracy_summary = self.metrics_collector.get_summary("accuracy", last_hour)
        satisfaction_summary = self.metrics_collector.get_summary("user_satisfaction", last_hour)

        # Determine overall health
        active_alerts = self.alert_manager.get_active_alerts()
        critical_alerts = [a for a in active_alerts if a.severity == AlertSeverity.CRITICAL]

        if critical_alerts:
            status = "critical"
        elif active_alerts:
            status = "warning"
        else:
            status = "healthy"

        return {
            "status": status,
            "uptime_seconds": now - self._start_time,
            "active_alerts": len(active_alerts),
            "response_time": response_summary,
            "accuracy": accuracy_summary,
            "user_satisfaction": satisfaction_summary,
            "running_experiments": len(self.ab_testing.get_experiments(ExperimentStatus.RUNNING))
        }

    def get_dashboard_text(self) -> str:
        """
        Generate text-based dashboard display.

        Returns:
            Formatted dashboard string for terminal display
        """
        status = self.get_system_status()
        user_analytics = self.analytics_engine.get_user_analytics()
        trends = self.analytics_engine.get_performance_trends(hours=24)

        lines = []
        lines.append("=" * 60)
        lines.append("          RAG SYSTEM MONITORING DASHBOARD")
        lines.append("=" * 60)

        # System Status
        status_indicator = self.visualizer.status_indicator(status["status"])
        lines.append(f"\nSystem Status: {status_indicator} {status['status'].upper()}")
        lines.append(f"Uptime: {status['uptime_seconds']/3600:.1f} hours")
        lines.append(f"Active Alerts: {status['active_alerts']}")

        # Key Metrics
        lines.append("\n-- Key Metrics (Last Hour) --")

        if status["response_time"].get("count", 0) > 0:
            rt = status["response_time"]
            lines.append(f"Response Time: {rt['mean']:.2f}s (p95: {rt['p95']:.2f}s)")

            # Add visual bar
            bar = self.visualizer.horizontal_bar(rt['mean'], 5.0, width=30)
            lines.append(f"  {bar} {rt['mean']:.2f}/5.0s target")

        if status["accuracy"].get("count", 0) > 0:
            acc = status["accuracy"]
            lines.append(f"Accuracy: {acc['mean']:.1%} (min: {acc['min']:.1%})")

        if status["user_satisfaction"].get("count", 0) > 0:
            sat = status["user_satisfaction"]
            lines.append(f"User Satisfaction: {sat['mean']:.1f}/5.0")

        # User Analytics
        lines.append("\n-- User Analytics (24h) --")
        lines.append(f"Active Users: {user_analytics['active_users']}")
        lines.append(f"Total Actions: {user_analytics['total_actions']}")

        # Performance Trend
        if trends["response_time_trend"]:
            rt_values = [t["mean"] for t in trends["response_time_trend"]]
            sparkline = self.visualizer.sparkline(rt_values)
            lines.append(f"\nResponse Time Trend: {sparkline}")

        # Active Alerts
        active_alerts = self.alert_manager.get_active_alerts()
        if active_alerts:
            lines.append("\n-- Active Alerts --")
            for alert in active_alerts[:5]:
                severity = "[!]" if alert.severity == AlertSeverity.WARNING else "[X]"
                lines.append(f"  {severity} {alert.message}")

        # Running Experiments
        experiments = self.ab_testing.get_experiments(ExperimentStatus.RUNNING)
        if experiments:
            lines.append("\n-- Running Experiments --")
            for exp in experiments:
                lines.append(f"  - {exp.name}: {', '.join(exp.variants)}")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)

    def generate_report(self, hours: int = 24) -> Dict[str, Any]:
        """
        Generate a comprehensive analytics report.

        Args:
            hours: Number of hours to include in report

        Returns:
            Comprehensive report dictionary
        """
        return {
            "generated_at": datetime.now().isoformat(),
            "period_hours": hours,
            "system_status": self.get_system_status(),
            "user_analytics": self.analytics_engine.get_user_analytics(hours),
            "query_patterns": self.analytics_engine.analyze_query_patterns(hours),
            "performance_trends": self.analytics_engine.get_performance_trends(hours),
            "alert_history": [
                {
                    "name": a.name,
                    "message": a.message,
                    "severity": a.severity.value,
                    "triggered_at": a.triggered_at,
                    "active": a.is_active
                }
                for a in self.alert_manager.get_alert_history(hours)
            ]
        }

    def execute(self) -> None:
        """
        Display the monitoring dashboard.

        Prints the dashboard to stdout for terminal viewing.
        """
        print(self.get_dashboard_text())


def create_monitoring_and_analytics_dashboard(
    persistence_path: Optional[str] = None
) -> MonitoringAndAnalyticsDashboard:
    """
    Factory function for creating MonitoringAndAnalyticsDashboard instances.

    Args:
        persistence_path: Optional path for data persistence

    Returns:
        MonitoringAndAnalyticsDashboard: A new instance
    """
    return MonitoringAndAnalyticsDashboard(persistence_path=persistence_path)
