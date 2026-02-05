"""
Tests for Retrieval Evaluation Framework

Run: pytest
Watch: pytest-watch (install with: pip install pytest-watch)
Coverage: pytest --cov=src
"""

import pytest
from src.retrieval_evaluation_framework import (
    RetrievalEvaluationFramework,
    create_retrieval_evaluation_framework,
    TestDatasetManager,
    MetricsCalculator,
    MockRetrievalStrategy,
    StrategyComparator,
    ReportGenerator,
)


class TestRetrievalEvaluationFramework:
    """Test suite for RetrievalEvaluationFramework."""

    @pytest.fixture
    def instance(self):
        """Create a fresh instance for each test."""
        return create_retrieval_evaluation_framework()

    def test_create_instance(self, instance):
        """Test that we can create an instance."""
        assert instance is not None
        assert isinstance(instance, RetrievalEvaluationFramework)

    def test_execute_returns_results(self, instance):
        """Test that execute returns demonstration results."""
        result = instance.execute()
        assert result is not None
        assert "dataset" in result
        assert "strategies_evaluated" in result
        assert "evaluation_results" in result
        assert "report" in result


class TestAcceptanceCriteria:
    """Tests for acceptance criteria."""

    @pytest.fixture
    def instance(self):
        """Create a fresh instance for each test."""
        return create_retrieval_evaluation_framework()

    def test_creates_test_dataset_with_questions_and_expected_r(self, instance):
        """Test: Creates test dataset with questions and expected relevant chunks"""
        # Generate synthetic dataset
        dataset = instance.generate_synthetic_dataset(size=5)

        assert "questions" in dataset
        assert "ground_truth" in dataset
        assert "corpus" in dataset

        # Verify dataset structure
        assert len(dataset["questions"]) == 5
        assert len(dataset["ground_truth"]) == 5
        assert len(dataset["corpus"]) > 0

        # Each question should have ground truth
        for question in dataset["questions"]:
            assert question in dataset["ground_truth"]
            relevant_chunks = dataset["ground_truth"][question]
            assert len(relevant_chunks) > 0

        # Corpus should have id and text
        for chunk in dataset["corpus"]:
            assert "id" in chunk
            assert "text" in chunk

    def test_implements_metrics_like_precision_k_recall_k_and(self, instance):
        """Test: Implements metrics like precision@k, recall@k, and MRR"""
        metrics = instance.get_metrics_calculator()

        # Test precision@k
        retrieved = ["a", "b", "c", "d", "e"]
        relevant = ["a", "c", "f"]

        # precision@5: 2 relevant in top 5 / 5 = 0.4
        assert metrics.precision_at_k(retrieved, relevant, k=5) == pytest.approx(0.4)
        # precision@3: 2 relevant in top 3 / 3 = 0.666...
        assert metrics.precision_at_k(retrieved, relevant, k=3) == pytest.approx(2/3)
        # precision@1: 1 relevant in top 1 / 1 = 1.0
        assert metrics.precision_at_k(retrieved, relevant, k=1) == pytest.approx(1.0)

        # Test recall@k
        # recall@5: 2 relevant in top 5 / 3 total relevant = 0.666...
        assert metrics.recall_at_k(retrieved, relevant, k=5) == pytest.approx(2/3)
        # recall@3: 2 relevant in top 3 / 3 total relevant = 0.666...
        assert metrics.recall_at_k(retrieved, relevant, k=3) == pytest.approx(2/3)
        # recall@1: 1 relevant in top 1 / 3 total relevant = 0.333...
        assert metrics.recall_at_k(retrieved, relevant, k=1) == pytest.approx(1/3)

        # Test reciprocal rank
        # First relevant item "a" is at position 1, so RR = 1/1 = 1.0
        assert metrics.reciprocal_rank(retrieved, relevant) == pytest.approx(1.0)

        # Test with first relevant at position 2
        retrieved2 = ["x", "a", "c", "d", "e"]
        assert metrics.reciprocal_rank(retrieved2, relevant) == pytest.approx(0.5)

        # Test MRR
        all_retrieved = [["a", "b"], ["x", "c"], ["y", "z"]]
        all_relevant = [["a"], ["c"], ["w"]]
        # RR: 1/1, 1/2, 0 -> MRR = (1 + 0.5 + 0) / 3 = 0.5
        assert metrics.mean_reciprocal_rank(all_retrieved, all_relevant) == pytest.approx(0.5)

    def test_compares_different_embedding_models_and_chunking_s(self, instance):
        """Test: Compares different embedding models and chunking strategies"""
        # Generate dataset
        instance.generate_synthetic_dataset(size=5)

        # Create strategies with different characteristics
        strategies = instance.create_default_strategies()
        assert len(strategies) >= 2

        # Evaluate strategies
        results = instance.evaluate_strategies()

        # Should have results for each strategy
        assert len(results) == len(strategies)

        # Each result should have metrics
        for strategy_name, strategy_results in results.items():
            assert "aggregated_metrics" in strategy_results
            metrics = strategy_results["aggregated_metrics"]

            # Check key metrics exist
            assert "precision@5" in metrics
            assert "recall@5" in metrics
            assert "mrr" in metrics

            # Metrics should be in valid range [0, 1]
            assert 0 <= metrics["precision@5"] <= 1
            assert 0 <= metrics["recall@5"] <= 1
            assert 0 <= metrics["mrr"] <= 1

    def test_generates_evaluation_reports_with_actionable_insig(self, instance):
        """Test: Generates evaluation reports with actionable insights"""
        # Setup and evaluate
        instance.generate_synthetic_dataset(size=5)
        instance.create_default_strategies()
        instance.evaluate_strategies()

        # Generate report
        report = instance.generate_report()

        # Report should have key sections
        assert "summary" in report
        assert "strategy_summaries" in report
        assert "rankings" in report
        assert "insights" in report
        assert "recommendations" in report

        # Summary should have overview info
        summary = report["summary"]
        assert "num_strategies" in summary
        assert summary["num_strategies"] > 0

        # Should have insights
        assert len(report["insights"]) > 0

        # Should have recommendations
        assert len(report["recommendations"]) > 0

        # Rankings should include different orderings
        rankings = report["rankings"]
        assert "by_precision" in rankings
        assert "by_recall" in rankings
        assert "by_mrr" in rankings

        # Get formatted report
        formatted = instance.get_formatted_report()
        assert isinstance(formatted, str)
        assert "RETRIEVAL EVALUATION REPORT" in formatted
        assert "INSIGHTS" in formatted
        assert "RECOMMENDATIONS" in formatted


class TestMetricsCalculator:
    """Unit tests for MetricsCalculator."""

    @pytest.fixture
    def calculator(self):
        """Create a fresh MetricsCalculator."""
        return MetricsCalculator()

    def test_precision_empty_results(self, calculator):
        """Test precision with empty results."""
        assert calculator.precision_at_k([], ["a", "b"], k=5) == 0.0

    def test_recall_empty_relevant(self, calculator):
        """Test recall with no relevant items."""
        assert calculator.recall_at_k(["a", "b"], [], k=5) == 0.0

    def test_reciprocal_rank_no_relevant_found(self, calculator):
        """Test RR when no relevant items are retrieved."""
        assert calculator.reciprocal_rank(["a", "b", "c"], ["x", "y", "z"]) == 0.0

    def test_calculate_all_metrics(self, calculator):
        """Test calculating all metrics at once."""
        retrieved = ["a", "b", "c", "d", "e"]
        relevant = ["a", "c"]

        metrics = calculator.calculate_all_metrics(retrieved, relevant, k_values=[3, 5])

        assert "precision@3" in metrics
        assert "precision@5" in metrics
        assert "recall@3" in metrics
        assert "recall@5" in metrics
        assert "reciprocal_rank" in metrics


class TestTestDatasetManager:
    """Unit tests for TestDatasetManager."""

    @pytest.fixture
    def manager(self):
        """Create a fresh TestDatasetManager."""
        return TestDatasetManager()

    def test_create_dataset(self, manager):
        """Test creating a custom dataset."""
        questions = ["Q1?", "Q2?"]
        ground_truth = {"Q1?": ["chunk1"], "Q2?": ["chunk2", "chunk3"]}
        corpus = [
            {"id": "chunk1", "text": "Text 1"},
            {"id": "chunk2", "text": "Text 2"},
            {"id": "chunk3", "text": "Text 3"},
        ]

        success = manager.create_test_dataset(questions, ground_truth, corpus)
        assert success is True

        assert manager.get_questions() == questions
        assert manager.get_ground_truth("Q1?") == ["chunk1"]
        assert manager.get_corpus() == corpus

    def test_generate_synthetic_dataset(self, manager):
        """Test generating synthetic dataset."""
        dataset = manager.generate_synthetic_dataset(size=5)

        assert len(dataset["questions"]) == 5
        assert len(dataset["corpus"]) > 0

    def test_get_corpus_ids(self, manager):
        """Test getting all corpus IDs."""
        manager.generate_synthetic_dataset(size=3)
        ids = manager.get_corpus_ids()

        assert isinstance(ids, set)
        assert len(ids) > 0


class TestMockRetrievalStrategy:
    """Unit tests for MockRetrievalStrategy."""

    def test_strategy_creation(self):
        """Test creating a mock strategy."""
        strategy = MockRetrievalStrategy("test", accuracy=0.8, noise_level=0.2)
        assert strategy.name == "test"
        assert strategy.accuracy == 0.8
        assert strategy.noise_level == 0.2

    def test_strategy_retrieval(self):
        """Test strategy retrieval."""
        strategy = MockRetrievalStrategy("test", accuracy=1.0, noise_level=0.0)

        corpus = [
            {"id": "chunk1", "text": "Text 1"},
            {"id": "chunk2", "text": "Text 2"},
            {"id": "chunk3", "text": "Text 3"},
        ]
        relevant = ["chunk1", "chunk2"]

        results = strategy.retrieve("test query", corpus, relevant, top_k=3)

        assert len(results) <= 3
        assert all(isinstance(r, str) for r in results)


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def instance(self):
        """Create a fresh instance for each test."""
        return create_retrieval_evaluation_framework()

    def test_empty_dataset_handling(self, instance):
        """Test handling of empty dataset."""
        # Try to create empty dataset
        success = instance.create_test_dataset([], {}, [])
        assert success is False

    def test_custom_k_values(self, instance):
        """Test setting custom k values."""
        instance.set_k_values([1, 3, 10])

        instance.generate_synthetic_dataset(size=3)
        instance.create_default_strategies()
        results = instance.evaluate_strategies()

        # Check metrics are calculated for custom k values
        for strategy_results in results.values():
            metrics = strategy_results["aggregated_metrics"]
            assert "precision@1" in metrics
            assert "precision@3" in metrics
            assert "precision@10" in metrics
