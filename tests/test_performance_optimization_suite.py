"""
Tests for Performance Optimization Suite

Run: pytest
Watch: pytest-watch (install with: pip install pytest-watch)
Coverage: pytest --cov=src
"""

import time
import pytest
from src.performance_optimization_suite import (
    PerformanceOptimizationSuite,
    create_performance_optimization_suite,
    QueryCache,
    VectorOptimizer,
    CostMonitor,
    PerformanceProfiler,
    VectorQueryConfig
)


class TestPerformanceOptimizationSuite:
    """Test suite for PerformanceOptimizationSuite."""

    @pytest.fixture
    def instance(self):
        """Create a fresh instance for each test."""
        return create_performance_optimization_suite()

    def test_create_instance(self, instance):
        """Test that we can create an instance."""
        assert instance is not None
        assert isinstance(instance, PerformanceOptimizationSuite)

    def test_cache_embedding(self, instance):
        """Test caching an embedding."""
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        instance.cache_embedding("test query", embedding)

        result, hit, age = instance.get_cached_embedding("test query")
        assert hit is True
        assert result == embedding
        assert age >= 0

    def test_cache_response(self, instance):
        """Test caching a response."""
        response = {"answer": "This is a test response", "sources": []}
        instance.cache_response("test query", response)

        result, hit, age = instance.get_cached_response("test query")
        assert hit is True
        assert result == response

    def test_track_api_call(self, instance):
        """Test tracking an API call."""
        instance.track_api_call("embedding", tokens=100)

        report = instance.get_cost_report()
        assert report["total_events"] >= 1
        assert "embedding" in report["calls_by_type"]

    def test_profile_query(self, instance):
        """Test query profiling."""
        with instance.profile_query("test_query_1"):
            time.sleep(0.01)  # Simulate work

        report = instance.get_performance_report()
        assert report["total_operations"] >= 1

    def test_get_optimization_report(self, instance):
        """Test getting a comprehensive optimization report."""
        # Add some data
        instance.cache_embedding("q1", [0.1, 0.2])
        instance.track_api_call("generation", tokens=50)

        report = instance.get_optimization_report()

        assert "cache" in report
        assert "cost" in report
        assert "performance" in report
        assert "recommendations" in report


class TestQueryCache:
    """Test suite for QueryCache."""

    @pytest.fixture
    def cache(self):
        """Create a fresh QueryCache for each test."""
        return QueryCache(max_size=10, default_ttl=60.0)

    def test_set_and_get(self, cache):
        """Test basic set and get operations."""
        cache.set("key1", "value1")
        value, hit, age = cache.get("key1")

        assert hit is True
        assert value == "value1"
        assert age >= 0

    def test_cache_miss(self, cache):
        """Test cache miss for non-existent key."""
        value, hit, age = cache.get("nonexistent")

        assert hit is False
        assert value is None
        assert age == 0

    def test_lru_eviction(self, cache):
        """Test LRU eviction when cache is full."""
        # Fill cache
        for i in range(10):
            cache.set(f"key{i}", f"value{i}")

        # Add one more (should evict oldest)
        cache.set("key10", "value10")

        # First key should be evicted
        value, hit, age = cache.get("key0")
        assert hit is False

        # Latest key should exist
        value, hit, age = cache.get("key10")
        assert hit is True

    def test_ttl_expiration(self, cache):
        """Test TTL-based expiration."""
        cache.set("key1", "value1", ttl=0.1)  # 100ms TTL

        # Should exist initially
        value, hit, age = cache.get("key1")
        assert hit is True

        # Wait for expiration
        time.sleep(0.15)

        # Should be expired now
        value, hit, age = cache.get("key1")
        assert hit is False

    def test_delete(self, cache):
        """Test delete operation."""
        cache.set("key1", "value1")

        result = cache.delete("key1")
        assert result is True

        value, hit, age = cache.get("key1")
        assert hit is False

    def test_clear(self, cache):
        """Test clearing the cache."""
        cache.set("key1", "value1")
        cache.set("key2", "value2")

        cache.clear()

        _, hit1, _ = cache.get("key1")
        _, hit2, _ = cache.get("key2")

        assert hit1 is False
        assert hit2 is False

    def test_get_stats(self, cache):
        """Test getting cache statistics."""
        cache.set("key1", "value1")
        cache.get("key1")  # Hit
        cache.get("key2")  # Miss

        stats = cache.get_stats()

        assert stats["size"] == 1
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5


class TestVectorOptimizer:
    """Test suite for VectorOptimizer."""

    @pytest.fixture
    def optimizer(self):
        """Create a fresh VectorOptimizer for each test."""
        return VectorOptimizer()

    def test_optimize_query(self, optimizer):
        """Test query optimization."""
        embedding = [0.1, 0.2, 0.3]
        query = optimizer.optimize_query(embedding)

        assert "embedding" in query
        assert "max_results" in query
        assert "threshold" in query

    def test_filter_results(self, optimizer):
        """Test filtering search results."""
        results = [
            {"doc_id": "1", "score": 0.9},
            {"doc_id": "2", "score": 0.7},
            {"doc_id": "3", "score": 0.3},
            {"doc_id": "4", "score": 0.1},
        ]

        filtered = optimizer.filter_results(results, threshold=0.5)

        assert len(filtered) == 2
        assert filtered[0]["doc_id"] == "1"
        assert filtered[1]["doc_id"] == "2"

    def test_filter_results_with_limit(self, optimizer):
        """Test filtering with max results limit."""
        results = [
            {"doc_id": "1", "score": 0.9},
            {"doc_id": "2", "score": 0.8},
            {"doc_id": "3", "score": 0.7},
        ]

        filtered = optimizer.filter_results(results, threshold=0.5, max_results=2)

        assert len(filtered) == 2

    def test_batch_queries(self, optimizer):
        """Test batching queries."""
        embeddings = [[0.1] * 100 for _ in range(250)]

        batches = optimizer.batch_queries(embeddings)

        assert len(batches) == 3
        assert len(batches[0]) == 100
        assert len(batches[1]) == 100
        assert len(batches[2]) == 50

    def test_cache_result(self, optimizer):
        """Test caching query results."""
        results = [{"doc_id": "1", "score": 0.9}]

        optimizer.cache_result("query_key", results)
        cached, hit = optimizer.get_cached_result("query_key")

        assert hit is True
        assert cached == results


class TestCostMonitor:
    """Test suite for CostMonitor."""

    @pytest.fixture
    def monitor(self):
        """Create a fresh CostMonitor for each test."""
        return CostMonitor(daily_budget=10.0)

    def test_track_api_call(self, monitor):
        """Test tracking an API call."""
        event = monitor.track_api_call("embedding", tokens=1000)

        assert event.event_type == "embedding"
        assert event.tokens == 1000
        assert event.cost > 0

    def test_explicit_cost(self, monitor):
        """Test tracking with explicit cost."""
        event = monitor.track_api_call("custom", cost=0.50)

        assert event.cost == 0.50

    def test_get_cost_report(self, monitor):
        """Test generating a cost report."""
        monitor.track_api_call("embedding", tokens=500)
        monitor.track_api_call("generation_input", tokens=200)

        report = monitor.get_cost_report()

        assert "total_cost" in report
        assert "daily_cost" in report
        assert "budget_remaining" in report
        assert "costs_by_type" in report

    def test_budget_alerts(self, monitor):
        """Test budget alert triggering."""
        # Spend 85% of budget
        monitor.track_api_call("custom", cost=8.5)

        report = monitor.get_cost_report()

        # Should have triggered the 80% alert
        alerts = [a for a in report["alerts"] if a["triggered"]]
        assert len(alerts) >= 1

    def test_usage_trend(self, monitor):
        """Test getting usage trend."""
        monitor.track_api_call("embedding", tokens=100)

        trend = monitor.get_usage_trend(hours=1)

        assert isinstance(trend, list)


class TestPerformanceProfiler:
    """Test suite for PerformanceProfiler."""

    @pytest.fixture
    def profiler(self):
        """Create a fresh PerformanceProfiler for each test."""
        return PerformanceProfiler()

    def test_profile_query(self, profiler):
        """Test profiling a query."""
        with profiler.profile_query("query_1"):
            time.sleep(0.01)

        report = profiler.get_performance_report()

        assert report["total_operations"] == 1
        assert report["average_latency_ms"] >= 10

    def test_record_operation(self, profiler):
        """Test recording an operation directly."""
        profiler.record_operation("test_op", 50.0)

        report = profiler.get_performance_report()

        assert report["total_operations"] == 1

    def test_multiple_operations(self, profiler):
        """Test profiling multiple operations."""
        profiler.record_operation("op1", 10.0)
        profiler.record_operation("op2", 20.0)
        profiler.record_operation("op3", 30.0)

        report = profiler.get_performance_report()

        assert report["total_operations"] == 3
        assert report["average_latency_ms"] == 20.0

    def test_percentiles(self, profiler):
        """Test percentile calculations."""
        # Add 100 operations with varying latencies
        for i in range(100):
            profiler.record_operation(f"op_{i}", float(i))

        report = profiler.get_performance_report()

        assert report["p50_latency_ms"] == 50.0
        assert report["p95_latency_ms"] >= 90.0

    def test_latency_histogram(self, profiler):
        """Test latency histogram generation."""
        for i in range(50):
            profiler.record_operation(f"op_{i}", float(i * 10))

        histogram = profiler.get_latency_histogram(buckets=5)

        assert "buckets" in histogram
        assert "counts" in histogram
        assert len(histogram["buckets"]) == 5

    def test_recommendations(self, profiler):
        """Test that recommendations are generated."""
        # Add some slow operations
        for i in range(10):
            profiler.record_operation(f"slow_op_{i}", 2000.0)

        report = profiler.get_performance_report()

        assert len(report["recommendations"]) > 0


class TestAcceptanceCriteria:
    """Tests for acceptance criteria."""

    @pytest.fixture
    def instance(self):
        """Create a fresh instance for each test."""
        return create_performance_optimization_suite()

    def test_implements_intelligent_caching_for_repeated_queries(self, instance):
        """Test: Implements intelligent caching for repeated queries and embeddings"""
        # Cache embedding
        embedding = [0.1, 0.2, 0.3]
        instance.cache_embedding("test query", embedding)

        # Retrieve from cache
        result, hit, age = instance.get_cached_embedding("test query")
        assert hit is True
        assert result == embedding

        # Cache response
        response = {"answer": "test"}
        instance.cache_response("test query", response)

        result, hit, age = instance.get_cached_response("test query")
        assert hit is True
        assert result == response

    def test_optimizes_vector_database_queries(self, instance):
        """Test: Optimizes vector database queries and indexing strategies"""
        embedding = [0.1, 0.2, 0.3]

        # Test query optimization
        optimized = instance.optimize_vector_query(embedding, max_results=5)
        assert optimized["max_results"] == 5

        # Test result filtering
        results = [
            {"doc_id": "1", "score": 0.9},
            {"doc_id": "2", "score": 0.3},
        ]
        filtered = instance.filter_search_results(results, threshold=0.5)
        assert len(filtered) == 1

    def test_includes_cost_monitoring_and_budget_alerts(self, instance):
        """Test: Includes cost monitoring and budget alerts for API usage"""
        # Track some API calls
        instance.track_api_call("embedding", tokens=1000)
        instance.track_api_call("generation", tokens=500)

        # Get cost report
        report = instance.get_cost_report()

        assert "total_cost" in report
        assert "daily_cost" in report
        assert "budget_remaining" in report
        assert "alerts" in report

    def test_provides_performance_profiling_and_bottleneck_identification(self, instance):
        """Test: Provides performance profiling and bottleneck identification"""
        # Profile some queries
        with instance.profile_query("query_1"):
            time.sleep(0.01)

        with instance.profile_query("query_2"):
            time.sleep(0.02)

        # Get performance report
        report = instance.get_performance_report()

        assert "total_operations" in report
        assert "average_latency_ms" in report
        assert "bottlenecks" in report
        assert "recommendations" in report
