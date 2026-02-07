"""
Tests for Monitoring and Analytics Dashboard

Run: pytest
Watch: pytest-watch (install with: pip install pytest-watch)
Coverage: pytest --cov=src
"""

import time
import pytest
from src.monitoring_and_analytics_dashboard import (
    MonitoringAndAnalyticsDashboard,
    create_monitoring_and_analytics_dashboard,
    MetricsCollector,
    DataStore,
    AlertManager,
    ABTestingFramework,
    AnalyticsEngine,
    TextVisualizer,
    MetricType,
    MetricPoint,
    AlertRule,
    AlertSeverity,
    ExperimentStatus
)


class TestMonitoringAndAnalyticsDashboard:
    """Test suite for MonitoringAndAnalyticsDashboard."""

    @pytest.fixture
    def instance(self):
        """Create a fresh instance for each test."""
        return create_monitoring_and_analytics_dashboard()

    def test_create_instance(self, instance):
        """Test that we can create an instance."""
        assert instance is not None
        assert isinstance(instance, MonitoringAndAnalyticsDashboard)

    def test_instance_has_components(self, instance):
        """Test that instance has all required components."""
        assert instance.data_store is not None
        assert instance.metrics_collector is not None
        assert instance.alert_manager is not None
        assert instance.ab_testing is not None
        assert instance.analytics_engine is not None

    def test_record_query_execution(self, instance):
        """Test recording query execution data."""
        instance.record_query_execution({
            'query': 'What is machine learning?',
            'response_time': 2.3,
            'cost': 0.02,
            'accuracy_score': 0.85,
            'user_id': 'user123'
        })

        # Verify metrics were recorded
        status = instance.get_system_status()
        assert status["response_time"]["count"] >= 1

    def test_get_system_status(self, instance):
        """Test getting system status."""
        status = instance.get_system_status()

        assert "status" in status
        assert "uptime_seconds" in status
        assert "active_alerts" in status
        assert status["status"] in ["healthy", "warning", "critical"]

    def test_check_alerts(self, instance):
        """Test checking for alerts."""
        alerts = instance.check_alerts()
        assert isinstance(alerts, list)

    def test_get_dashboard_text(self, instance):
        """Test generating dashboard text."""
        text = instance.get_dashboard_text()

        assert isinstance(text, str)
        assert "MONITORING DASHBOARD" in text
        assert "System Status" in text

    def test_generate_report(self, instance):
        """Test generating a comprehensive report."""
        report = instance.generate_report(hours=24)

        assert "generated_at" in report
        assert "system_status" in report
        assert "user_analytics" in report
        assert "query_patterns" in report


class TestDataStore:
    """Test suite for DataStore."""

    @pytest.fixture
    def store(self):
        """Create a fresh DataStore for each test."""
        return DataStore(max_points=100)

    def test_store_and_query(self, store):
        """Test storing and querying data points."""
        point = MetricPoint(
            name="test_metric",
            value=42.0,
            timestamp=time.time()
        )
        store.store(point)

        results = store.query("test_metric")
        assert len(results) == 1
        assert results[0].value == 42.0

    def test_query_with_time_range(self, store):
        """Test querying with time range."""
        now = time.time()

        # Store points at different times
        for i in range(5):
            point = MetricPoint(
                name="test_metric",
                value=float(i),
                timestamp=now - (100 - i * 10)
            )
            store.store(point)

        # Query recent points only
        results = store.query("test_metric", start_time=now - 50)
        assert len(results) < 5

    def test_get_latest(self, store):
        """Test getting the latest point."""
        for i in range(5):
            point = MetricPoint(
                name="test_metric",
                value=float(i),
                timestamp=time.time()
            )
            store.store(point)

        latest = store.get_latest("test_metric")
        assert latest is not None
        assert latest.value == 4.0

    def test_get_metrics(self, store):
        """Test getting list of metric names."""
        store.store(MetricPoint("metric1", 1.0, time.time()))
        store.store(MetricPoint("metric2", 2.0, time.time()))

        metrics = store.get_metrics()
        assert "metric1" in metrics
        assert "metric2" in metrics


class TestMetricsCollector:
    """Test suite for MetricsCollector."""

    @pytest.fixture
    def collector(self):
        """Create a fresh MetricsCollector for each test."""
        return MetricsCollector()

    def test_record(self, collector):
        """Test recording a metric."""
        collector.record("test_metric", 42.0)

        points = collector.data_store.query("test_metric")
        assert len(points) >= 1

    def test_increment_counter(self, collector):
        """Test incrementing a counter."""
        collector.increment_counter("test_counter")
        collector.increment_counter("test_counter")

        points = collector.data_store.query("test_counter")
        assert points[-1].value == 2.0

    def test_record_response_time(self, collector):
        """Test recording response time."""
        collector.record_response_time(1.5, user_id="user1")

        points = collector.data_store.query("response_time")
        assert len(points) >= 1
        assert points[0].value == 1.5

    def test_record_accuracy(self, collector):
        """Test recording accuracy score."""
        collector.record_accuracy(0.85)

        points = collector.data_store.query("accuracy")
        assert len(points) >= 1
        assert points[0].value == 0.85

    def test_record_query_execution(self, collector):
        """Test recording comprehensive query execution data."""
        collector.record_query_execution({
            'response_time': 2.0,
            'cost': 0.05,
            'accuracy_score': 0.9,
            'user_id': 'test_user'
        })

        rt_points = collector.data_store.query("response_time")
        assert len(rt_points) >= 1

    def test_get_summary(self, collector):
        """Test getting statistical summary."""
        for i in range(10):
            collector.record("test_metric", float(i))

        summary = collector.get_summary("test_metric")

        assert summary["count"] == 10
        assert summary["mean"] == 4.5
        assert "min" in summary
        assert "max" in summary


class TestAlertManager:
    """Test suite for AlertManager."""

    @pytest.fixture
    def collector(self):
        """Create a MetricsCollector for alert testing."""
        return MetricsCollector()

    @pytest.fixture
    def alert_manager(self, collector):
        """Create a fresh AlertManager for each test."""
        return AlertManager(collector)

    def test_add_rule(self, alert_manager):
        """Test adding an alert rule."""
        rule = AlertRule(
            name="Test Alert",
            metric_name="test_metric",
            condition="gt",
            threshold=10.0
        )
        alert_manager.add_rule(rule)

        assert len(alert_manager._rules) >= 1

    def test_check_rules_triggers_alert(self, alert_manager, collector):
        """Test that rules trigger alerts correctly."""
        # Add a rule
        rule = AlertRule(
            name="High Value Alert",
            metric_name="test_metric",
            condition="gt",
            threshold=10.0,
            cooldown_seconds=0  # No cooldown for testing
        )
        alert_manager.add_rule(rule)

        # Record a high value
        collector.record("test_metric", 15.0)

        # Check rules
        alerts = alert_manager.check_rules()

        assert len(alerts) >= 1
        assert any(a.name == "High Value Alert" for a in alerts)

    def test_alert_cooldown(self, alert_manager, collector):
        """Test alert cooldown period."""
        rule = AlertRule(
            name="Cooldown Test",
            metric_name="test_metric",
            condition="gt",
            threshold=10.0,
            cooldown_seconds=300
        )
        alert_manager.add_rule(rule)

        collector.record("test_metric", 15.0)
        alerts1 = alert_manager.check_rules()

        collector.record("test_metric", 20.0)
        alerts2 = alert_manager.check_rules()

        # Second check should not trigger due to cooldown
        cooldown_alerts = [a for a in alerts2 if a.name == "Cooldown Test"]
        assert len(cooldown_alerts) == 0

    def test_get_active_alerts(self, alert_manager, collector):
        """Test getting active alerts."""
        rule = AlertRule(
            name="Active Test",
            metric_name="test_metric",
            condition="gt",
            threshold=10.0,
            cooldown_seconds=0
        )
        alert_manager.add_rule(rule)

        collector.record("test_metric", 15.0)
        alert_manager.check_rules()

        active = alert_manager.get_active_alerts()
        assert len(active) >= 1

    def test_resolve_alert(self, alert_manager, collector):
        """Test resolving an alert."""
        rule = AlertRule(
            name="Resolve Test",
            metric_name="test_metric",
            condition="gt",
            threshold=10.0,
            cooldown_seconds=0
        )
        alert_manager.add_rule(rule)

        collector.record("test_metric", 15.0)
        alerts = alert_manager.check_rules()

        if alerts:
            result = alert_manager.resolve_alert(alerts[0].alert_id)
            assert result is True

            active = alert_manager.get_active_alerts()
            assert not any(a.alert_id == alerts[0].alert_id for a in active)


class TestABTestingFramework:
    """Test suite for ABTestingFramework."""

    @pytest.fixture
    def ab_testing(self):
        """Create a fresh ABTestingFramework for each test."""
        return ABTestingFramework()

    def test_create_experiment(self, ab_testing):
        """Test creating an experiment."""
        exp = ab_testing.create_experiment(
            "test_experiment",
            ["control", "treatment"]
        )

        assert exp.name == "test_experiment"
        assert len(exp.variants) == 2
        assert exp.status == ExperimentStatus.DRAFT

    def test_start_experiment(self, ab_testing):
        """Test starting an experiment."""
        exp = ab_testing.create_experiment("test", ["A", "B"])

        result = ab_testing.start_experiment(exp.experiment_id)
        assert result is True

        exp = ab_testing._experiments[exp.experiment_id]
        assert exp.status == ExperimentStatus.RUNNING

    def test_get_variant(self, ab_testing):
        """Test getting a variant for a user."""
        exp = ab_testing.create_experiment("test", ["control", "treatment"])
        ab_testing.start_experiment(exp.experiment_id)

        variant = ab_testing.get_variant(exp.experiment_id, "user123")

        assert variant in ["control", "treatment"]

    def test_consistent_variant_assignment(self, ab_testing):
        """Test that variant assignment is consistent."""
        exp = ab_testing.create_experiment("test", ["A", "B"])
        ab_testing.start_experiment(exp.experiment_id)

        # Same user should always get same variant
        variant1 = ab_testing.get_variant(exp.experiment_id, "user123")
        variant2 = ab_testing.get_variant(exp.experiment_id, "user123")

        assert variant1 == variant2

    def test_record_result(self, ab_testing):
        """Test recording experiment results."""
        exp = ab_testing.create_experiment("test", ["A", "B"])
        ab_testing.start_experiment(exp.experiment_id)

        ab_testing.record_result(
            exp.experiment_id,
            "A",
            "user123",
            "accuracy",
            0.85
        )

        assert len(ab_testing._results) == 1

    def test_analyze_experiment(self, ab_testing):
        """Test analyzing experiment results."""
        exp = ab_testing.create_experiment("test", ["control", "treatment"])
        ab_testing.start_experiment(exp.experiment_id)

        # Record some results
        for i in range(50):
            ab_testing.record_result(exp.experiment_id, "control", f"user_c{i}", "accuracy", 0.80)
            ab_testing.record_result(exp.experiment_id, "treatment", f"user_t{i}", "accuracy", 0.85)

        analysis = ab_testing.analyze_experiment(exp.experiment_id, "accuracy")

        assert "variant_stats" in analysis
        assert "significance" in analysis
        assert analysis["total_samples"] == 100


class TestAnalyticsEngine:
    """Test suite for AnalyticsEngine."""

    @pytest.fixture
    def collector(self):
        """Create a MetricsCollector for analytics."""
        return MetricsCollector()

    @pytest.fixture
    def engine(self, collector):
        """Create a fresh AnalyticsEngine for each test."""
        return AnalyticsEngine(collector)

    def test_track_user_action(self, engine):
        """Test tracking user actions."""
        engine.track_user_action("user123", "query", {"query_text": "test"})

        analytics = engine.get_user_analytics()
        assert analytics["active_users"] >= 1

    def test_get_user_analytics(self, engine):
        """Test getting user analytics."""
        engine.track_user_action("user1", "query")
        engine.track_user_action("user2", "query")
        engine.track_user_action("user1", "feedback")

        analytics = engine.get_user_analytics()

        assert analytics["active_users"] == 2
        assert analytics["total_actions"] == 3

    def test_analyze_query_patterns(self, engine, collector):
        """Test analyzing query patterns."""
        for i in range(10):
            collector.increment_counter("queries_total")
            collector.record_response_time(1.0)

        patterns = engine.analyze_query_patterns()

        assert patterns["total_queries"] >= 10
        assert "queries_per_hour" in patterns

    def test_get_performance_trends(self, engine, collector):
        """Test getting performance trends."""
        for i in range(10):
            collector.record_response_time(float(i) / 10)
            collector.record_accuracy(0.8 + float(i) / 100)

        trends = engine.get_performance_trends()

        assert "response_time_trend" in trends
        assert "accuracy_trend" in trends


class TestTextVisualizer:
    """Test suite for TextVisualizer."""

    @pytest.fixture
    def viz(self):
        """Create a fresh TextVisualizer for each test."""
        return TextVisualizer()

    def test_horizontal_bar(self, viz):
        """Test horizontal bar creation."""
        bar = viz.horizontal_bar(50, 100)
        assert "[" in bar
        assert "]" in bar

    def test_sparkline(self, viz):
        """Test sparkline creation."""
        values = [1, 2, 3, 4, 5, 4, 3, 2, 1]
        line = viz.sparkline(values)

        assert len(line) > 0
        # Sparkline uses unicode block characters
        assert any(c in line for c in "▁▂▃▄▅▆▇█")

    def test_table(self, viz):
        """Test table creation."""
        headers = ["Name", "Value"]
        rows = [["Metric1", "100"], ["Metric2", "200"]]

        table = viz.table(headers, rows)

        assert "Name" in table
        assert "Value" in table
        assert "Metric1" in table

    def test_status_indicator(self, viz):
        """Test status indicator."""
        assert viz.status_indicator("healthy") == "[OK]"
        assert viz.status_indicator("warning") == "[!!]"
        assert viz.status_indicator("critical") == "[XX]"

    def test_progress_bar(self, viz):
        """Test progress bar."""
        bar = viz.progress_bar(50, 100)

        assert "[" in bar
        assert "]" in bar
        assert "50.0%" in bar


class TestAcceptanceCriteria:
    """Tests for acceptance criteria."""

    @pytest.fixture
    def instance(self):
        """Create a fresh instance for each test."""
        return create_monitoring_and_analytics_dashboard()

    def test_tracks_key_metrics(self, instance):
        """Test: Tracks key metrics: response time, accuracy, user satisfaction, cost per query"""
        # Record query execution with all metrics
        instance.record_query_execution({
            'response_time': 2.3,
            'accuracy_score': 0.85,
            'cost': 0.02,
            'user_id': 'user123'
        })

        # Record user satisfaction separately
        instance.metrics_collector.record_user_satisfaction(4.5, user_id="user123")

        # Verify metrics are tracked
        status = instance.get_system_status()
        assert status["response_time"]["count"] >= 1
        assert status["accuracy"]["count"] >= 1
        assert status["user_satisfaction"]["count"] >= 1

    def test_implements_alerting(self, instance):
        """Test: Implements alerting for system failures and performance degradation"""
        # Add a custom alert rule
        rule = AlertRule(
            name="Test High Latency",
            metric_name="response_time",
            condition="gt",
            threshold=1.0,
            severity=AlertSeverity.WARNING,
            cooldown_seconds=0
        )
        instance.alert_manager.add_rule(rule)

        # Record a high latency value
        instance.metrics_collector.record_response_time(5.0)

        # Check alerts
        alerts = instance.check_alerts()

        # Should have triggered alert
        assert any(a.name == "Test High Latency" for a in alerts)

    def test_provides_user_analytics(self, instance):
        """Test: Provides user analytics and usage patterns visualization"""
        # Track some user actions
        instance.analytics_engine.track_user_action("user1", "query")
        instance.analytics_engine.track_user_action("user2", "query")
        instance.analytics_engine.track_user_action("user1", "feedback")

        analytics = instance.analytics_engine.get_user_analytics()

        assert analytics["active_users"] >= 2
        assert analytics["total_actions"] >= 3
        assert "action_breakdown" in analytics

        # Test visualization
        dashboard = instance.get_dashboard_text()
        assert "User Analytics" in dashboard

    def test_includes_ab_testing_framework(self, instance):
        """Test: Includes A/B testing framework for system improvements"""
        # Create an experiment
        exp = instance.ab_testing.create_experiment(
            "retrieval_test",
            ["algorithm_a", "algorithm_b"],
            description="Testing retrieval algorithms"
        )

        # Start the experiment
        instance.ab_testing.start_experiment(exp.experiment_id)

        # Get variant for users
        variant1 = instance.ab_testing.get_variant(exp.experiment_id, "user1")
        variant2 = instance.ab_testing.get_variant(exp.experiment_id, "user2")

        assert variant1 in ["algorithm_a", "algorithm_b"]
        assert variant2 in ["algorithm_a", "algorithm_b"]

        # Record results
        for i in range(40):
            instance.ab_testing.record_result(
                exp.experiment_id, "algorithm_a", f"user_a{i}", "accuracy", 0.80
            )
            instance.ab_testing.record_result(
                exp.experiment_id, "algorithm_b", f"user_b{i}", "accuracy", 0.85
            )

        # Analyze results
        analysis = instance.ab_testing.analyze_experiment(exp.experiment_id, "accuracy")

        assert "variant_stats" in analysis
        assert "significance" in analysis
