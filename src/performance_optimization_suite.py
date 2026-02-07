"""
Performance Optimization Suite

Implement caching, query optimization, and cost reduction strategies for production scale.
This module provides tools to optimize RAG system performance including:
- Intelligent caching with LRU eviction and TTL support
- Vector database query optimization
- Cost monitoring and budget management
- Performance profiling and bottleneck identification

All implementations use only the Python standard library for maximum compatibility.
"""

import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, TypeVar

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

T = TypeVar('T')


class CacheEntryStatus(Enum):
    """Status of a cache entry."""
    HIT = "hit"
    MISS = "miss"
    EXPIRED = "expired"
    EVICTED = "evicted"


@dataclass
class CacheEntry:
    """Represents a cached item with metadata."""
    value: Any
    created_at: float
    last_accessed: float
    ttl: float  # Time-to-live in seconds
    size_bytes: int = 0
    access_count: int = 0

    @property
    def is_expired(self) -> bool:
        """Check if the entry has expired."""
        return time.time() > self.created_at + self.ttl

    @property
    def age_seconds(self) -> float:
        """Get the age of the entry in seconds."""
        return time.time() - self.created_at


@dataclass
class CacheStats:
    """Statistics for cache performance."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0
    total_size_bytes: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class QueryCache:
    """
    LRU cache with TTL support for query results and embeddings.

    Implements an efficient caching strategy that:
    - Uses LRU (Least Recently Used) eviction when capacity is reached
    - Supports time-based expiration (TTL) for entries
    - Provides thread-safe operations for concurrent access
    - Tracks detailed statistics for performance monitoring

    Example:
        cache = QueryCache(max_size=1000, default_ttl=3600)
        cache.set("query_key", embedding_vector, ttl=7200)
        result = cache.get("query_key")  # Returns (value, True, age_seconds)
    """

    def __init__(self, max_size: int = 1000, default_ttl: float = 3600.0):
        """
        Initialize the query cache.

        Args:
            max_size: Maximum number of entries in cache
            default_ttl: Default time-to-live in seconds
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.Lock()
        self._stats = CacheStats()

    def _compute_key(self, key: Any) -> str:
        """Compute a string key from any hashable object."""
        if isinstance(key, str):
            return key
        return hashlib.md5(json.dumps(key, sort_keys=True).encode()).hexdigest()

    def _estimate_size(self, value: Any) -> int:
        """Estimate the size of a value in bytes."""
        try:
            return len(json.dumps(value).encode())
        except (TypeError, ValueError):
            return 0

    def get(self, key: Any) -> Tuple[Optional[Any], bool, float]:
        """
        Retrieve an item from cache.

        Args:
            key: Cache key (string or hashable object)

        Returns:
            Tuple of (value, cache_hit, age_seconds)
            - value: Cached value or None if not found
            - cache_hit: True if value was found and valid
            - age_seconds: Age of the cached entry (0 if miss)
        """
        str_key = self._compute_key(key)

        with self._lock:
            if str_key not in self._cache:
                self._stats.misses += 1
                return None, False, 0.0

            entry = self._cache[str_key]

            # Check expiration
            if entry.is_expired:
                self._stats.expirations += 1
                self._stats.total_size_bytes -= entry.size_bytes
                del self._cache[str_key]
                self._stats.misses += 1
                return None, False, 0.0

            # Update access tracking (LRU)
            entry.last_accessed = time.time()
            entry.access_count += 1
            self._cache.move_to_end(str_key)

            self._stats.hits += 1
            return entry.value, True, entry.age_seconds

    def set(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """
        Store an item in cache.

        Args:
            key: Cache key (string or hashable object)
            value: Value to cache
            ttl: Time-to-live in seconds (uses default if not specified)
        """
        str_key = self._compute_key(key)
        ttl = ttl if ttl is not None else self.default_ttl
        size = self._estimate_size(value)

        with self._lock:
            # Remove existing entry if present
            if str_key in self._cache:
                old_entry = self._cache[str_key]
                self._stats.total_size_bytes -= old_entry.size_bytes
                del self._cache[str_key]

            # Evict if at capacity
            while len(self._cache) >= self.max_size:
                oldest_key = next(iter(self._cache))
                oldest_entry = self._cache[oldest_key]
                self._stats.total_size_bytes -= oldest_entry.size_bytes
                self._stats.evictions += 1
                del self._cache[oldest_key]

            # Add new entry
            entry = CacheEntry(
                value=value,
                created_at=time.time(),
                last_accessed=time.time(),
                ttl=ttl,
                size_bytes=size
            )
            self._cache[str_key] = entry
            self._stats.total_size_bytes += size

    def delete(self, key: Any) -> bool:
        """
        Remove an item from cache.

        Args:
            key: Cache key to remove

        Returns:
            True if key was found and removed
        """
        str_key = self._compute_key(key)

        with self._lock:
            if str_key in self._cache:
                entry = self._cache[str_key]
                self._stats.total_size_bytes -= entry.size_bytes
                del self._cache[str_key]
                return True
        return False

    def clear(self) -> None:
        """Clear all entries from cache."""
        with self._lock:
            self._cache.clear()
            self._stats.total_size_bytes = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._stats.hits,
                "misses": self._stats.misses,
                "hit_rate": self._stats.hit_rate,
                "evictions": self._stats.evictions,
                "expirations": self._stats.expirations,
                "total_size_bytes": self._stats.total_size_bytes
            }

    def cleanup_expired(self) -> int:
        """
        Remove all expired entries from cache.

        Returns:
            Number of entries removed
        """
        removed = 0
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired
            ]
            for key in expired_keys:
                entry = self._cache[key]
                self._stats.total_size_bytes -= entry.size_bytes
                self._stats.expirations += 1
                del self._cache[key]
                removed += 1
        return removed


@dataclass
class VectorQueryConfig:
    """Configuration for vector database query optimization."""
    max_results: int = 10
    similarity_threshold: float = 0.5
    use_approximate_search: bool = False
    batch_size: int = 100
    timeout_seconds: float = 30.0


class VectorOptimizer:
    """
    Optimizes vector database queries for better performance.

    Provides strategies for:
    - Limiting result sets to reduce memory usage
    - Filtering by similarity thresholds
    - Batching queries for efficiency
    - Query result caching
    """

    def __init__(self, config: Optional[VectorQueryConfig] = None):
        """
        Initialize the vector optimizer.

        Args:
            config: Query configuration options
        """
        self.config = config or VectorQueryConfig()
        self._query_cache = QueryCache(max_size=500, default_ttl=1800)
        self._stats = {
            "queries_processed": 0,
            "results_filtered": 0,
            "cache_hits": 0,
            "total_latency_ms": 0.0
        }
        self._lock = threading.Lock()

    def optimize_query(self, query_embedding: List[float],
                       max_results: Optional[int] = None,
                       threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        Create an optimized query configuration.

        Args:
            query_embedding: The query embedding vector
            max_results: Override for max results
            threshold: Override for similarity threshold

        Returns:
            Optimized query parameters
        """
        return {
            "embedding": query_embedding,
            "max_results": max_results or self.config.max_results,
            "threshold": threshold or self.config.similarity_threshold,
            "use_approximate": self.config.use_approximate_search,
            "timeout": self.config.timeout_seconds
        }

    def filter_results(self, results: List[Dict[str, Any]],
                      threshold: Optional[float] = None,
                      max_results: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Filter and limit search results.

        Args:
            results: List of search results with 'score' field
            threshold: Minimum similarity score
            max_results: Maximum number of results to return

        Returns:
            Filtered and limited results list
        """
        threshold = threshold or self.config.similarity_threshold
        max_results = max_results or self.config.max_results

        # Filter by threshold
        filtered = [r for r in results if r.get('score', 0) >= threshold]

        with self._lock:
            self._stats["results_filtered"] += len(results) - len(filtered)

        # Sort by score descending and limit
        filtered.sort(key=lambda x: x.get('score', 0), reverse=True)
        return filtered[:max_results]

    def batch_queries(self, embeddings: List[List[float]]) -> List[List[List[float]]]:
        """
        Split embeddings into optimal batches for processing.

        Args:
            embeddings: List of embedding vectors

        Returns:
            List of batches, each containing embedding vectors
        """
        batch_size = self.config.batch_size
        return [
            embeddings[i:i + batch_size]
            for i in range(0, len(embeddings), batch_size)
        ]

    def get_cached_result(self, query_key: str) -> Tuple[Optional[List[Dict]], bool]:
        """
        Check for cached query results.

        Args:
            query_key: Unique key for the query

        Returns:
            Tuple of (cached_results, cache_hit)
        """
        result, hit, _ = self._query_cache.get(query_key)
        if hit:
            with self._lock:
                self._stats["cache_hits"] += 1
        return result, hit

    def cache_result(self, query_key: str, results: List[Dict[str, Any]],
                    ttl: Optional[float] = None) -> None:
        """
        Cache query results.

        Args:
            query_key: Unique key for the query
            results: Query results to cache
            ttl: Time-to-live in seconds
        """
        self._query_cache.set(query_key, results, ttl)

    def record_query(self, latency_ms: float) -> None:
        """Record query execution metrics."""
        with self._lock:
            self._stats["queries_processed"] += 1
            self._stats["total_latency_ms"] += latency_ms

    def get_stats(self) -> Dict[str, Any]:
        """Get optimizer statistics."""
        with self._lock:
            avg_latency = (
                self._stats["total_latency_ms"] / self._stats["queries_processed"]
                if self._stats["queries_processed"] > 0 else 0
            )
            return {
                **self._stats,
                "average_latency_ms": avg_latency,
                "cache_stats": self._query_cache.get_stats()
            }


@dataclass
class CostEvent:
    """Represents a billable API event."""
    event_type: str  # e.g., "embedding", "generation", "query"
    timestamp: float
    tokens: int
    cost: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BudgetAlert:
    """Represents a budget threshold alert."""
    threshold: float
    triggered_at: Optional[float] = None
    message: str = ""
    level: str = "warning"  # "warning" or "critical"


class CostMonitor:
    """
    Monitors API usage costs and enforces budgets.

    Tracks costs across different API operations, provides usage
    reports, and generates alerts when spending approaches limits.

    Example:
        monitor = CostMonitor(daily_budget=10.0)
        monitor.track_api_call("embedding", tokens=100, cost=0.001)
        report = monitor.get_cost_report()
    """

    # Default cost rates per 1K tokens
    DEFAULT_RATES = {
        "embedding": 0.0001,  # $0.0001 per 1K tokens
        "generation_input": 0.003,  # $0.003 per 1K tokens
        "generation_output": 0.015,  # $0.015 per 1K tokens
        "query": 0.001  # $0.001 per query
    }

    def __init__(self, daily_budget: float = 100.0,
                 monthly_budget: float = 2000.0,
                 cost_rates: Optional[Dict[str, float]] = None):
        """
        Initialize the cost monitor.

        Args:
            daily_budget: Maximum daily spending limit
            monthly_budget: Maximum monthly spending limit
            cost_rates: Custom cost rates (per 1K tokens or per operation)
        """
        self.daily_budget = daily_budget
        self.monthly_budget = monthly_budget
        self.cost_rates = cost_rates or self.DEFAULT_RATES.copy()

        self._events: List[CostEvent] = []
        self._lock = threading.Lock()

        # Set up budget alerts
        self._alerts: List[BudgetAlert] = [
            BudgetAlert(threshold=0.8, message="80% of daily budget consumed", level="warning"),
            BudgetAlert(threshold=0.95, message="95% of daily budget consumed", level="critical"),
        ]

    def track_api_call(self, event_type: str, tokens: int = 0,
                       cost: Optional[float] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> CostEvent:
        """
        Track an API call and its associated cost.

        Args:
            event_type: Type of API call (e.g., "embedding", "generation")
            tokens: Number of tokens used
            cost: Explicit cost (if None, calculated from rates)
            metadata: Additional event metadata

        Returns:
            The recorded CostEvent
        """
        if cost is None:
            rate = self.cost_rates.get(event_type, 0)
            cost = (tokens / 1000) * rate if tokens > 0 else rate

        event = CostEvent(
            event_type=event_type,
            timestamp=time.time(),
            tokens=tokens,
            cost=cost,
            metadata=metadata or {}
        )

        with self._lock:
            self._events.append(event)
            self._check_alerts()

        return event

    def _check_alerts(self) -> None:
        """Check if any budget alerts should be triggered."""
        daily_cost = self._get_period_cost(period="day")
        daily_ratio = daily_cost / self.daily_budget if self.daily_budget > 0 else 0

        for alert in self._alerts:
            if daily_ratio >= alert.threshold and alert.triggered_at is None:
                alert.triggered_at = time.time()
                logger.warning(f"Budget Alert: {alert.message} (${daily_cost:.2f}/${self.daily_budget:.2f})")

    def _get_period_cost(self, period: str = "day") -> float:
        """Get total cost for a time period."""
        now = time.time()
        if period == "day":
            cutoff = now - 86400  # 24 hours
        elif period == "hour":
            cutoff = now - 3600
        elif period == "month":
            cutoff = now - 2592000  # 30 days
        else:
            cutoff = 0

        return sum(e.cost for e in self._events if e.timestamp > cutoff)

    def get_cost_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive cost report.

        Returns:
            Dictionary with cost breakdown and analysis
        """
        with self._lock:
            now = time.time()
            day_ago = now - 86400
            hour_ago = now - 3600

            # Calculate costs by period
            hourly_cost = sum(e.cost for e in self._events if e.timestamp > hour_ago)
            daily_cost = sum(e.cost for e in self._events if e.timestamp > day_ago)
            total_cost = sum(e.cost for e in self._events)

            # Calculate by event type
            costs_by_type: Dict[str, float] = {}
            tokens_by_type: Dict[str, int] = {}
            counts_by_type: Dict[str, int] = {}

            for event in self._events:
                costs_by_type[event.event_type] = costs_by_type.get(event.event_type, 0) + event.cost
                tokens_by_type[event.event_type] = tokens_by_type.get(event.event_type, 0) + event.tokens
                counts_by_type[event.event_type] = counts_by_type.get(event.event_type, 0) + 1

            return {
                "total_cost": total_cost,
                "daily_cost": daily_cost,
                "hourly_cost": hourly_cost,
                "budget_remaining": self.daily_budget - daily_cost,
                "budget_used_percent": (daily_cost / self.daily_budget * 100) if self.daily_budget > 0 else 0,
                "costs_by_type": costs_by_type,
                "tokens_by_type": tokens_by_type,
                "calls_by_type": counts_by_type,
                "total_events": len(self._events),
                "alerts": [
                    {"message": a.message, "triggered": a.triggered_at is not None}
                    for a in self._alerts
                ]
            }

    def get_usage_trend(self, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get hourly usage trend.

        Args:
            hours: Number of hours to analyze

        Returns:
            List of hourly usage summaries
        """
        now = time.time()
        trend = []

        for i in range(hours):
            hour_start = now - (i + 1) * 3600
            hour_end = now - i * 3600

            hour_events = [
                e for e in self._events
                if hour_start <= e.timestamp < hour_end
            ]

            trend.append({
                "hour_offset": -i,
                "cost": sum(e.cost for e in hour_events),
                "tokens": sum(e.tokens for e in hour_events),
                "calls": len(hour_events)
            })

        return list(reversed(trend))

    def reset_alerts(self) -> None:
        """Reset all alert triggers."""
        for alert in self._alerts:
            alert.triggered_at = None


@dataclass
class TimingRecord:
    """Records timing for a profiled operation."""
    name: str
    start_time: float
    end_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        """Get duration in milliseconds."""
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time) * 1000


class PerformanceProfiler:
    """
    Profiles system performance and identifies bottlenecks.

    Provides timing measurement, latency analysis, and bottleneck
    identification for RAG system operations.

    Example:
        profiler = PerformanceProfiler()
        with profiler.profile_query("my_query"):
            # Perform RAG operations
            results = rag_system.query("What is ML?")

        report = profiler.get_performance_report()
    """

    def __init__(self, history_size: int = 1000):
        """
        Initialize the performance profiler.

        Args:
            history_size: Maximum number of timing records to retain
        """
        self.history_size = history_size
        self._records: List[TimingRecord] = []
        self._active_profiles: Dict[str, TimingRecord] = {}
        self._lock = threading.Lock()

    @contextmanager
    def profile_query(self, query_id: str, metadata: Optional[Dict] = None) -> Generator[None, None, None]:
        """
        Context manager for profiling a query.

        Args:
            query_id: Unique identifier for the query
            metadata: Additional metadata to record

        Yields:
            None (use as context manager)
        """
        record = TimingRecord(
            name=query_id,
            start_time=time.time(),
            metadata=metadata or {}
        )

        with self._lock:
            self._active_profiles[query_id] = record

        try:
            yield
        finally:
            record.end_time = time.time()
            with self._lock:
                del self._active_profiles[query_id]
                self._records.append(record)
                # Trim history
                if len(self._records) > self.history_size:
                    self._records = self._records[-self.history_size:]

    def record_operation(self, operation_name: str, duration_ms: float,
                        metadata: Optional[Dict] = None) -> None:
        """
        Record a completed operation timing.

        Args:
            operation_name: Name of the operation
            duration_ms: Duration in milliseconds
            metadata: Additional metadata
        """
        now = time.time()
        record = TimingRecord(
            name=operation_name,
            start_time=now - (duration_ms / 1000),
            end_time=now,
            metadata=metadata or {}
        )

        with self._lock:
            self._records.append(record)
            if len(self._records) > self.history_size:
                self._records = self._records[-self.history_size:]

    def get_performance_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive performance report.

        Returns:
            Performance analysis with latency statistics and bottlenecks
        """
        with self._lock:
            if not self._records:
                return {
                    "total_operations": 0,
                    "average_latency_ms": 0,
                    "p50_latency_ms": 0,
                    "p95_latency_ms": 0,
                    "p99_latency_ms": 0,
                    "bottlenecks": [],
                    "recommendations": []
                }

            durations = [r.duration_ms for r in self._records]
            durations.sort()

            total = len(durations)
            avg = sum(durations) / total

            # Calculate percentiles
            p50 = durations[int(total * 0.5)]
            p95 = durations[int(total * 0.95)] if total >= 20 else durations[-1]
            p99 = durations[int(total * 0.99)] if total >= 100 else durations[-1]

            # Group by operation name
            by_operation: Dict[str, List[float]] = {}
            for record in self._records:
                if record.name not in by_operation:
                    by_operation[record.name] = []
                by_operation[record.name].append(record.duration_ms)

            operation_stats = {
                name: {
                    "count": len(times),
                    "average_ms": sum(times) / len(times),
                    "max_ms": max(times),
                    "min_ms": min(times)
                }
                for name, times in by_operation.items()
            }

            # Identify bottlenecks (operations with high latency)
            bottlenecks = [
                {"operation": name, "average_ms": stats["average_ms"]}
                for name, stats in operation_stats.items()
                if stats["average_ms"] > p95
            ]
            bottlenecks.sort(key=lambda x: x["average_ms"], reverse=True)

            # Generate recommendations
            recommendations = self._generate_recommendations(
                avg, p95, bottlenecks, operation_stats
            )

            return {
                "total_operations": total,
                "average_latency_ms": round(avg, 2),
                "p50_latency_ms": round(p50, 2),
                "p95_latency_ms": round(p95, 2),
                "p99_latency_ms": round(p99, 2),
                "operations_by_type": operation_stats,
                "bottlenecks": bottlenecks[:5],  # Top 5 bottlenecks
                "recommendations": recommendations
            }

    def _generate_recommendations(self, avg: float, p95: float,
                                  bottlenecks: List[Dict],
                                  operation_stats: Dict) -> List[str]:
        """Generate performance improvement recommendations."""
        recommendations = []

        if avg > 1000:
            recommendations.append("Average latency exceeds 1s - consider implementing caching")

        if p95 > 5000:
            recommendations.append("P95 latency exceeds 5s - investigate slow queries")

        if bottlenecks:
            top_bottleneck = bottlenecks[0]["operation"]
            recommendations.append(f"Focus optimization on '{top_bottleneck}' - highest latency")

        if len(operation_stats) > 10:
            recommendations.append("Many operation types detected - consider consolidating")

        if not recommendations:
            recommendations.append("System performing within normal parameters")

        return recommendations

    def get_latency_histogram(self, buckets: int = 10) -> Dict[str, Any]:
        """
        Get latency distribution as a histogram.

        Args:
            buckets: Number of histogram buckets

        Returns:
            Histogram data with bucket counts
        """
        with self._lock:
            if not self._records:
                return {"buckets": [], "counts": []}

            durations = [r.duration_ms for r in self._records]
            min_val = min(durations)
            max_val = max(durations)
            bucket_size = (max_val - min_val) / buckets if max_val != min_val else 1

            bucket_bounds = [min_val + i * bucket_size for i in range(buckets + 1)]
            counts = [0] * buckets

            for d in durations:
                bucket_idx = min(int((d - min_val) / bucket_size), buckets - 1)
                counts[bucket_idx] += 1

            return {
                "buckets": [f"{bucket_bounds[i]:.0f}-{bucket_bounds[i+1]:.0f}ms"
                           for i in range(buckets)],
                "counts": counts
            }


class PerformanceOptimizationSuite:
    """
    Unified interface for RAG system performance optimization.

    Combines caching, vector optimization, cost monitoring, and
    performance profiling into a single, easy-to-use interface.

    Example:
        suite = PerformanceOptimizationSuite()

        # Cache an embedding
        suite.cache_embedding("What is RAG?", [0.1, 0.2, 0.3])

        # Track API cost
        suite.track_api_call("embedding", tokens=100)

        # Profile a query
        with suite.profile_query("query_123"):
            results = rag.query("What is RAG?")

        # Get reports
        report = suite.get_optimization_report()
    """

    def __init__(self, cache_size: int = 1000, cache_ttl: float = 3600.0,
                 daily_budget: float = 100.0):
        """
        Initialize the optimization suite.

        Args:
            cache_size: Maximum cache entries
            cache_ttl: Default cache TTL in seconds
            daily_budget: Daily cost budget
        """
        # Initialize components
        self._embedding_cache = QueryCache(max_size=cache_size, default_ttl=cache_ttl)
        self._response_cache = QueryCache(max_size=cache_size // 2, default_ttl=cache_ttl // 2)
        self._vector_optimizer = VectorOptimizer()
        self._cost_monitor = CostMonitor(daily_budget=daily_budget)
        self._profiler = PerformanceProfiler()

        logger.info(f"Performance Optimization Suite initialized (cache_size={cache_size}, budget=${daily_budget})")

    # Caching methods
    def cache_embedding(self, query: str, embedding: List[float],
                       ttl: Optional[float] = None) -> None:
        """Cache a query embedding."""
        self._embedding_cache.set(query, embedding, ttl)

    def get_cached_embedding(self, query: str) -> Tuple[Optional[List[float]], bool, float]:
        """Get cached embedding if available."""
        return self._embedding_cache.get(query)

    def cache_response(self, query: str, response: Dict[str, Any],
                      ttl: Optional[float] = None) -> None:
        """Cache a query response."""
        self._response_cache.set(query, response, ttl)

    def get_cached_response(self, query: str) -> Tuple[Optional[Dict], bool, float]:
        """Get cached response if available."""
        return self._response_cache.get(query)

    # Cost tracking methods
    def track_api_call(self, event_type: str, tokens: int = 0,
                       cost: Optional[float] = None) -> None:
        """Track an API call cost."""
        self._cost_monitor.track_api_call(event_type, tokens, cost)

    def get_cost_report(self) -> Dict[str, Any]:
        """Get cost monitoring report."""
        return self._cost_monitor.get_cost_report()

    # Performance profiling methods
    @contextmanager
    def profile_query(self, query_id: str) -> Generator[None, None, None]:
        """Profile a query execution."""
        with self._profiler.profile_query(query_id):
            yield

    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance profiling report."""
        return self._profiler.get_performance_report()

    # Vector optimization methods
    def optimize_vector_query(self, embedding: List[float],
                             max_results: Optional[int] = None) -> Dict[str, Any]:
        """Create optimized vector query parameters."""
        return self._vector_optimizer.optimize_query(embedding, max_results)

    def filter_search_results(self, results: List[Dict],
                             threshold: Optional[float] = None) -> List[Dict]:
        """Filter search results by similarity threshold."""
        return self._vector_optimizer.filter_results(results, threshold)

    # Unified reporting
    def get_optimization_report(self) -> Dict[str, Any]:
        """
        Get comprehensive optimization report.

        Returns:
            Combined report from all optimization components
        """
        return {
            "cache": {
                "embeddings": self._embedding_cache.get_stats(),
                "responses": self._response_cache.get_stats()
            },
            "cost": self._cost_monitor.get_cost_report(),
            "performance": self._profiler.get_performance_report(),
            "vector_optimization": self._vector_optimizer.get_stats(),
            "recommendations": self._generate_recommendations()
        }

    def _generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on current state."""
        recommendations = []

        # Check cache performance
        embed_stats = self._embedding_cache.get_stats()
        if embed_stats["hit_rate"] < 0.3 and embed_stats["hits"] + embed_stats["misses"] > 100:
            recommendations.append("Cache hit rate is low - consider increasing cache size")

        # Check cost
        cost_report = self._cost_monitor.get_cost_report()
        if cost_report["budget_used_percent"] > 80:
            recommendations.append("Approaching daily budget limit - review high-cost operations")

        # Check performance
        perf_report = self._profiler.get_performance_report()
        if perf_report.get("average_latency_ms", 0) > 2000:
            recommendations.append("Average latency is high - enable caching and optimize queries")

        if not recommendations:
            recommendations.append("System is performing optimally")

        return recommendations

    def execute(self) -> None:
        """
        Run the optimization suite and display status.

        Prints a summary of current optimization status and recommendations.
        """
        report = self.get_optimization_report()

        print("=" * 60)
        print("Performance Optimization Suite Status")
        print("=" * 60)

        # Cache status
        print("\n-- Cache Performance --")
        embed_cache = report["cache"]["embeddings"]
        print(f"Embedding Cache: {embed_cache['size']}/{embed_cache['max_size']} entries")
        print(f"Hit Rate: {embed_cache['hit_rate']:.1%}")

        # Cost status
        print("\n-- Cost Monitoring --")
        cost = report["cost"]
        print(f"Daily Cost: ${cost['daily_cost']:.2f} / ${cost['budget_remaining'] + cost['daily_cost']:.2f}")
        print(f"Budget Used: {cost['budget_used_percent']:.1f}%")

        # Performance status
        print("\n-- Performance Metrics --")
        perf = report["performance"]
        print(f"Operations: {perf['total_operations']}")
        print(f"Avg Latency: {perf['average_latency_ms']:.2f}ms")
        print(f"P95 Latency: {perf['p95_latency_ms']:.2f}ms")

        # Recommendations
        print("\n-- Recommendations --")
        for rec in report["recommendations"]:
            print(f"  - {rec}")

        print("=" * 60)


def create_performance_optimization_suite(cache_size: int = 1000,
                                          cache_ttl: float = 3600.0,
                                          daily_budget: float = 100.0) -> PerformanceOptimizationSuite:
    """
    Factory function for creating PerformanceOptimizationSuite instances.

    Args:
        cache_size: Maximum number of cache entries
        cache_ttl: Default cache TTL in seconds
        daily_budget: Daily cost budget in dollars

    Returns:
        PerformanceOptimizationSuite: A new instance
    """
    return PerformanceOptimizationSuite(
        cache_size=cache_size,
        cache_ttl=cache_ttl,
        daily_budget=daily_budget
    )
