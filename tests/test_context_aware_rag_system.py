"""
Tests for Context-Aware RAG System

Run: pytest
Watch: pytest-watch (install with: pip install pytest-watch)
Coverage: pytest --cov=src
"""

import pytest
from src.context_aware_rag_system import (
    ContextAwareRagSystem,
    create_context_aware_rag_system,
    QueryType,
    RetrievalQuality,
    ConfidenceLevel,
    PromptStrategy,
    RetrievedContext,
    RetrievalAnalyzer,
    QueryClassifier,
    ContextManager,
    PromptTemplateManager,
    ResponseGenerator,
)


class TestContextAwareRagSystem:
    """Test suite for ContextAwareRagSystem."""

    @pytest.fixture
    def instance(self):
        """Create a fresh instance for each test."""
        return create_context_aware_rag_system()

    def test_create_instance(self, instance):
        """Test that we can create an instance."""
        assert instance is not None
        assert isinstance(instance, ContextAwareRagSystem)

    def test_execute_returns_dict(self, instance):
        """Test that execute returns a dictionary with expected keys."""
        result = instance.execute(
            query="What is the capital of France?",
            retrieved_contexts=[
                {
                    "content": "Paris is the capital and most populous city of France.",
                    "confidence_score": 0.95,
                    "source": "encyclopedia.com/france"
                }
            ]
        )
        assert isinstance(result, dict)
        assert "response" in result
        assert "confidence_level" in result
        assert "retrieval_quality" in result
        assert "query_type" in result
        assert "sources_used" in result
        assert "prompt_strategy" in result


class TestAcceptanceCriteria:
    """Tests for acceptance criteria."""

    @pytest.fixture
    def instance(self):
        """Create a fresh instance for each test."""
        return create_context_aware_rag_system()

    @pytest.fixture
    def high_confidence_contexts(self):
        """High confidence context for testing."""
        return [
            {
                "content": "Paris is the capital of France.",
                "confidence_score": 0.95,
                "source": "encyclopedia.com"
            },
            {
                "content": "France is a country in Western Europe.",
                "confidence_score": 0.9,
                "source": "geography.org"
            }
        ]

    @pytest.fixture
    def low_confidence_contexts(self):
        """Low confidence context for testing."""
        return [
            {
                "content": "Some information that may be related.",
                "confidence_score": 0.3,
                "source": "uncertain-source.com"
            }
        ]

    def test_analyzes_retrieval_confidence_scores_to_adjust_pro(self, instance):
        """Test: Analyzes retrieval confidence scores to adjust prompt strategy"""
        # Test with high confidence - should use direct answer strategy
        high_result = instance.execute(
            query="What is the capital of France?",
            retrieved_contexts=[
                {"content": "Paris is the capital.", "confidence_score": 0.95, "source": "a.com"}
            ]
        )
        assert high_result["prompt_strategy"] == "direct_answer"
        assert high_result["retrieval_quality"] == "excellent"

        # Test with low confidence - should use cautious reasoning
        low_result = instance.execute(
            query="What is something obscure?",
            retrieved_contexts=[
                {"content": "Maybe this.", "confidence_score": 0.2, "source": "b.com"}
            ]
        )
        assert low_result["prompt_strategy"] == "cautious_reasoning"
        assert low_result["retrieval_quality"] == "poor"

        # Test with medium confidence
        medium_result = instance.execute(
            query="What about this?",
            retrieved_contexts=[
                {"content": "Some info.", "confidence_score": 0.6, "source": "c.com"}
            ]
        )
        assert medium_result["retrieval_quality"] == "good"

    def test_handles_cases_with_too_much_or_too_little_retrieve(self, instance):
        """Test: Handles cases with too much or too little retrieved context"""
        # Test with no contexts
        empty_result = instance.execute(
            query="What is unknown?",
            retrieved_contexts=[]
        )
        assert empty_result["retrieval_quality"] == "poor"
        assert len(empty_result["sources_used"]) == 0

        # Test with many contexts - should handle truncation
        many_contexts = [
            {"content": f"Content {i} " * 100, "confidence_score": 0.9 - (i * 0.05), "source": f"source{i}.com"}
            for i in range(10)
        ]
        result = instance.execute(
            query="What is this about?",
            retrieved_contexts=many_contexts,
            max_context_length=2000
        )
        # Should still produce valid output
        assert result["response"] is not None
        assert result["retrieval_quality"] in ["excellent", "good", "poor"]

        # Test that high confidence contexts are prioritized
        mixed_contexts = [
            {"content": "Low confidence content", "confidence_score": 0.2, "source": "low.com"},
            {"content": "High confidence content", "confidence_score": 0.95, "source": "high.com"},
        ]
        mixed_result = instance.execute(
            query="Test query",
            retrieved_contexts=mixed_contexts
        )
        # High confidence source should be used
        assert "high.com" in mixed_result["sources_used"]

    def test_implements_query_classification_to_select_appropri(self, instance):
        """Test: Implements query classification to select appropriate prompt templates"""
        # Test factual query classification
        factual_result = instance.execute(
            query="What is the capital of Japan?",
            retrieved_contexts=[
                {"content": "Tokyo is the capital.", "confidence_score": 0.9, "source": "a.com"}
            ]
        )
        assert factual_result["query_type"] == "factual"

        # Test analytical query classification
        analytical_result = instance.execute(
            query="Why did the economy crash and what are the implications?",
            retrieved_contexts=[
                {"content": "Economic analysis here.", "confidence_score": 0.9, "source": "b.com"}
            ]
        )
        assert analytical_result["query_type"] == "analytical"
        assert analytical_result["prompt_strategy"] == "analytical_deep_dive"

        # Test creative query classification
        creative_result = instance.execute(
            query="Imagine what the future of AI might look like",
            retrieved_contexts=[
                {"content": "AI trends discussion.", "confidence_score": 0.9, "source": "c.com"}
            ]
        )
        assert creative_result["query_type"] == "creative"
        assert creative_result["prompt_strategy"] == "creative_synthesis"

    def test_includes_confidence_indicators_in_responses(self, instance):
        """Test: Includes confidence indicators in responses"""
        # High confidence should indicate high confidence level
        high_result = instance.execute(
            query="Simple factual question",
            retrieved_contexts=[
                {"content": "Clear answer here.", "confidence_score": 0.95, "source": "a.com"}
            ]
        )
        assert high_result["confidence_level"] == "high"

        # Medium confidence
        medium_result = instance.execute(
            query="Question",
            retrieved_contexts=[
                {"content": "Partial answer.", "confidence_score": 0.6, "source": "b.com"}
            ]
        )
        assert medium_result["confidence_level"] == "medium"

        # Low confidence should indicate low confidence level
        low_result = instance.execute(
            query="Obscure question",
            retrieved_contexts=[
                {"content": "Uncertain answer.", "confidence_score": 0.2, "source": "c.com"}
            ]
        )
        assert low_result["confidence_level"] == "low"


class TestRetrievalAnalyzer:
    """Tests for RetrievalAnalyzer class."""

    def test_analyze_confidence_excellent(self):
        """Test excellent quality assessment."""
        contexts = [
            RetrievedContext(content="A", confidence_score=0.9, source="a.com"),
            RetrievedContext(content="B", confidence_score=0.85, source="b.com"),
        ]
        result = RetrievalAnalyzer.analyze_confidence(contexts)
        assert result["quality"] == RetrievalQuality.EXCELLENT

    def test_analyze_confidence_good(self):
        """Test good quality assessment."""
        contexts = [
            RetrievedContext(content="A", confidence_score=0.6, source="a.com"),
            RetrievedContext(content="B", confidence_score=0.55, source="b.com"),
        ]
        result = RetrievalAnalyzer.analyze_confidence(contexts)
        assert result["quality"] == RetrievalQuality.GOOD

    def test_analyze_confidence_poor(self):
        """Test poor quality assessment."""
        contexts = [
            RetrievedContext(content="A", confidence_score=0.3, source="a.com"),
            RetrievedContext(content="B", confidence_score=0.2, source="b.com"),
        ]
        result = RetrievalAnalyzer.analyze_confidence(contexts)
        assert result["quality"] == RetrievalQuality.POOR

    def test_analyze_confidence_empty(self):
        """Test with no contexts."""
        result = RetrievalAnalyzer.analyze_confidence([])
        assert result["quality"] == RetrievalQuality.POOR
        assert result["avg_confidence"] == 0.0


class TestQueryClassifier:
    """Tests for QueryClassifier class."""

    def test_classify_factual(self):
        """Test factual query classification."""
        assert QueryClassifier.classify("What is Python?") == QueryType.FACTUAL
        assert QueryClassifier.classify("Who is the president?") == QueryType.FACTUAL
        assert QueryClassifier.classify("When was it founded?") == QueryType.FACTUAL

    def test_classify_analytical(self):
        """Test analytical query classification."""
        assert QueryClassifier.classify("Why did the stock market crash?") == QueryType.ANALYTICAL
        assert QueryClassifier.classify("Explain how photosynthesis works") == QueryType.ANALYTICAL
        assert QueryClassifier.classify("Compare Python and JavaScript") == QueryType.ANALYTICAL

    def test_classify_creative(self):
        """Test creative query classification."""
        assert QueryClassifier.classify("Imagine a world without electricity") == QueryType.CREATIVE
        assert QueryClassifier.classify("Suggest some ideas for my project") == QueryType.CREATIVE
        assert QueryClassifier.classify("What if dinosaurs still existed?") == QueryType.CREATIVE


class TestContextManager:
    """Tests for ContextManager class."""

    def test_process_contexts_normal(self):
        """Test normal context processing."""
        contexts = [
            RetrievedContext(content="Short content", confidence_score=0.9, source="a.com"),
        ]
        result = ContextManager.process_contexts(contexts)
        assert result["contexts_used"] == 1
        assert result["was_truncated"] is False

    def test_process_contexts_truncation(self):
        """Test context truncation when too long."""
        contexts = [
            RetrievedContext(content="A" * 3000, confidence_score=0.9, source="a.com"),
            RetrievedContext(content="B" * 3000, confidence_score=0.8, source="b.com"),
        ]
        result = ContextManager.process_contexts(contexts, max_length=4000)
        assert result["was_truncated"] is True

    def test_process_contexts_empty(self):
        """Test with empty context list."""
        result = ContextManager.process_contexts([])
        assert result["contexts_used"] == 0
        assert result["was_truncated"] is False

    def test_process_contexts_prioritizes_high_confidence(self):
        """Test that high confidence contexts are prioritized."""
        contexts = [
            RetrievedContext(content="Low", confidence_score=0.3, source="low.com"),
            RetrievedContext(content="High", confidence_score=0.95, source="high.com"),
        ]
        result = ContextManager.process_contexts(contexts)
        # First context should be the high confidence one
        assert result["contexts"][0].source == "high.com"


class TestPromptTemplateManager:
    """Tests for PromptTemplateManager class."""

    def test_select_strategy_excellent_factual(self):
        """Test strategy selection for excellent quality factual queries."""
        strategy = PromptTemplateManager.select_strategy(
            QueryType.FACTUAL, RetrievalQuality.EXCELLENT
        )
        assert strategy == PromptStrategy.DIRECT_ANSWER

    def test_select_strategy_poor_quality(self):
        """Test strategy selection for poor quality retrievals."""
        strategy = PromptTemplateManager.select_strategy(
            QueryType.FACTUAL, RetrievalQuality.POOR
        )
        assert strategy == PromptStrategy.CAUTIOUS_REASONING

    def test_select_strategy_analytical(self):
        """Test strategy selection for analytical queries."""
        strategy = PromptTemplateManager.select_strategy(
            QueryType.ANALYTICAL, RetrievalQuality.GOOD
        )
        assert strategy == PromptStrategy.ANALYTICAL_DEEP_DIVE

    def test_select_strategy_creative(self):
        """Test strategy selection for creative queries."""
        strategy = PromptTemplateManager.select_strategy(
            QueryType.CREATIVE, RetrievalQuality.EXCELLENT
        )
        assert strategy == PromptStrategy.CREATIVE_SYNTHESIS

    def test_get_template(self):
        """Test template retrieval."""
        template = PromptTemplateManager.get_template(PromptStrategy.DIRECT_ANSWER)
        assert template is not None
        assert "{context}" in template
        assert "{question}" in template


class TestResponseGenerator:
    """Tests for ResponseGenerator class."""

    def test_format_context_for_prompt(self):
        """Test context formatting."""
        contexts = [
            RetrievedContext(content="Test content", confidence_score=0.9, source="test.com"),
        ]
        formatted = ResponseGenerator.format_context_for_prompt(contexts)
        assert "Test content" in formatted
        assert "test.com" in formatted
        assert "0.9" in formatted

    def test_format_context_empty(self):
        """Test formatting with empty contexts."""
        formatted = ResponseGenerator.format_context_for_prompt([])
        assert "No relevant context" in formatted
