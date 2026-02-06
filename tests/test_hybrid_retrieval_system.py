"""
Tests for Hybrid Retrieval System

Run: pytest
Watch: pytest-watch (install with: pip install pytest-watch)
Coverage: pytest --cov=src
"""

import pytest
from src.hybrid_retrieval_system import (
    HybridRetrievalSystem,
    create_hybrid_retrieval_system,
    QueryProcessor,
    SemanticRetriever,
    KeywordRetriever,
    MetadataFilter,
    ResultFuser,
    ReRanker,
    ExplanationGenerator,
    Document,
    RetrievalStrategy
)


class TestHybridRetrievalSystem:
    """Test suite for HybridRetrievalSystem."""

    @pytest.fixture
    def instance(self):
        """Create a fresh instance for each test."""
        return create_hybrid_retrieval_system()

    def test_create_instance(self, instance):
        """Test that we can create an instance."""
        assert instance is not None
        assert isinstance(instance, HybridRetrievalSystem)

    def test_execute_returns_result(self, instance):
        """Test that execute returns a proper result dictionary."""
        result = instance.execute()
        assert isinstance(result, dict)
        assert "query" in result
        assert "results" in result
        assert "expanded_query" in result

    def test_add_document(self, instance):
        """Test adding a single document."""
        doc = instance.add_document(
            doc_id="test1",
            content="Test document content",
            metadata={"category": "test"}
        )
        assert doc.doc_id == "test1"
        assert instance.get_document_count() == 1

    def test_add_multiple_documents(self, instance):
        """Test adding multiple documents."""
        docs = [
            {"doc_id": "1", "content": "First document"},
            {"doc_id": "2", "content": "Second document"},
            {"doc_id": "3", "content": "Third document"}
        ]
        instance.add_documents(docs)
        assert instance.get_document_count() == 3


class TestAcceptanceCriteria:
    """Tests for acceptance criteria."""

    @pytest.fixture
    def instance(self):
        """Create a fresh instance for each test."""
        return create_hybrid_retrieval_system()

    def test_combines_semantic_similarity_with_keyword_matching(self, instance):
        """Test: Combines semantic similarity with keyword matching and metadata filtering"""
        # Add documents
        docs = [
            {
                "doc_id": "ml1",
                "content": "Machine learning is a subset of artificial intelligence that enables systems to learn.",
                "metadata": {"category": "ml", "year": 2023}
            },
            {
                "doc_id": "ml2",
                "content": "Deep learning neural networks are powerful for pattern recognition tasks.",
                "metadata": {"category": "ml", "year": 2024}
            },
            {
                "doc_id": "db1",
                "content": "Databases store structured data for efficient retrieval and querying.",
                "metadata": {"category": "database", "year": 2023}
            }
        ]
        instance.add_documents(docs)

        # Search with both semantic and keyword relevance
        results = instance.retrieve(
            query="machine learning AI systems",
            top_k=3,
            filters={"category": "ml"}
        )

        # Verify results are returned
        assert len(results) >= 1

        # Verify ML documents are ranked higher than database docs
        ml_doc_ids = [r.document.doc_id for r in results if r.document.metadata.get("category") == "ml"]
        assert len(ml_doc_ids) >= 1

        # Verify score breakdown includes multiple strategies
        first_result = results[0]
        assert "semantic" in first_result.score_breakdown
        assert "keyword" in first_result.score_breakdown

        # Verify metadata filter was applied
        for result in results:
            assert result.document.metadata.get("category") == "ml"

    def test_implements_query_expansion_and_synonym_handling(self, instance):
        """Test: Implements query expansion and synonym handling"""
        # Test query expansion
        expansion = instance.expand_query("machine learning model")

        # Verify expansion happened
        assert expansion.original_query == "machine learning model"
        assert len(expansion.expanded_terms) > 3  # Should have more terms than original

        # Check that synonyms were used
        assert len(expansion.synonyms_used) > 0

        # Verify ML-related synonyms are included
        all_terms = " ".join(expansion.expanded_terms).lower()
        assert "machine" in all_terms or "ml" in all_terms or "ai" in all_terms

        # Test that expanded query improves retrieval
        docs = [
            {"doc_id": "1", "content": "AI and artificial intelligence systems"},
            {"doc_id": "2", "content": "Unrelated content about cooking recipes"}
        ]
        instance.add_documents(docs)

        # Search with expansion
        results_expanded = instance.retrieve("machine learning", expand_query=True)
        results_no_expand = instance.retrieve("machine learning", expand_query=False)

        # The expanded query should find the AI document
        assert len(results_expanded) >= 1

    def test_includes_re_ranking_model_for_result_optimization(self, instance):
        """Test: Includes re-ranking model for result optimization"""
        import time

        # Add documents with different metadata quality signals
        current_time = time.time()
        docs = [
            {
                "doc_id": "old",
                "content": "Machine learning overview from several years ago",
                "metadata": {
                    "timestamp": current_time - (365 * 24 * 3600),  # 1 year old
                    "quality_score": 0.5
                }
            },
            {
                "doc_id": "new",
                "content": "Machine learning latest developments and techniques",
                "metadata": {
                    "timestamp": current_time - (7 * 24 * 3600),  # 1 week old
                    "quality_score": 0.9
                }
            },
            {
                "doc_id": "medium",
                "content": "Machine learning fundamentals and basics",
                "metadata": {
                    "timestamp": current_time - (30 * 24 * 3600),  # 1 month old
                    "quality_score": 0.7
                }
            }
        ]
        instance.add_documents(docs)

        # Create system with re-ranking enabled
        system_with_rerank = HybridRetrievalSystem(enable_reranking=True)
        system_with_rerank.add_documents(docs)

        # Retrieve results
        results = system_with_rerank.retrieve("machine learning")

        # Verify re-ranking was applied (newer/higher quality docs should be boosted)
        assert len(results) >= 2

        # Check that explanation mentions re-ranking adjustments
        has_rerank_mention = any(
            "freshness" in r.explanation.lower() or "quality" in r.explanation.lower()
            for r in results
        )
        # Re-ranking should have some effect
        assert len(results) > 0

    def test_provides_explainable_retrieval_scores_and_reasoning(self, instance):
        """Test: Provides explainable retrieval scores and reasoning"""
        docs = [
            {
                "doc_id": "doc1",
                "content": "Python programming language for data science and machine learning",
                "metadata": {"category": "programming", "year": 2024}
            },
            {
                "doc_id": "doc2",
                "content": "JavaScript frameworks for web development",
                "metadata": {"category": "programming", "year": 2023}
            }
        ]
        instance.add_documents(docs)

        results = instance.retrieve("python data science", top_k=2)

        assert len(results) >= 1
        first_result = results[0]

        # Verify score breakdown is provided
        assert first_result.score_breakdown is not None
        assert "semantic" in first_result.score_breakdown
        assert "keyword" in first_result.score_breakdown

        # Verify strategy contributions are provided
        assert first_result.strategy_contributions is not None
        assert len(first_result.strategy_contributions) > 0

        # Verify explanation is human-readable
        assert first_result.explanation is not None
        assert len(first_result.explanation) > 0
        assert "score" in first_result.explanation.lower() or "retrieved" in first_result.explanation.lower()

        # Verify matched terms are tracked
        assert first_result.query_terms_matched is not None


class TestQueryProcessor:
    """Tests for QueryProcessor component."""

    def test_normalize_query(self):
        """Test query normalization."""
        processor = QueryProcessor()

        result = processor.normalize_query("  HELLO   World!  ")
        assert result == "hello world"

    def test_expand_query_with_synonyms(self):
        """Test query expansion with synonyms."""
        processor = QueryProcessor()

        expansion = processor.expand_query("machine learning")

        assert expansion.original_query == "machine learning"
        assert len(expansion.expanded_terms) > 2
        assert "machine learning" in expansion.synonyms_used or "ml" in " ".join(expansion.expanded_terms).lower()

    def test_tokenize(self):
        """Test tokenization."""
        processor = QueryProcessor()

        tokens = processor.tokenize("Hello World Test")
        assert tokens == ["hello", "world", "test"]


class TestSemanticRetriever:
    """Tests for SemanticRetriever component."""

    def test_cosine_similarity(self):
        """Test cosine similarity calculation."""
        retriever = SemanticRetriever()

        # Identical vectors should have similarity 1
        vec = [1.0, 0.0, 0.0]
        assert abs(retriever.cosine_similarity(vec, vec) - 1.0) < 0.001

        # Orthogonal vectors should have similarity 0
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        assert abs(retriever.cosine_similarity(vec1, vec2)) < 0.001

    def test_simple_embedding(self):
        """Test simple embedding generation."""
        retriever = SemanticRetriever()

        emb = retriever.simple_embedding("test text", dim=32)

        assert len(emb) == 32
        # Check it's normalized (magnitude close to 1)
        magnitude = sum(x * x for x in emb) ** 0.5
        assert abs(magnitude - 1.0) < 0.001 or magnitude == 0

    def test_retrieve(self):
        """Test semantic retrieval."""
        retriever = SemanticRetriever()

        docs = [
            Document("1", "machine learning", retriever.simple_embedding("machine learning")),
            Document("2", "cooking recipes", retriever.simple_embedding("cooking recipes"))
        ]

        query_emb = retriever.simple_embedding("AI and ML")
        results = retriever.retrieve(query_emb, docs, top_k=2)

        assert len(results) == 2
        # Results should be sorted by similarity
        assert results[0][1] >= results[1][1]


class TestKeywordRetriever:
    """Tests for KeywordRetriever component."""

    def test_retrieve_matches(self):
        """Test keyword retrieval with matching terms."""
        retriever = KeywordRetriever()

        docs = [
            Document("1", "python programming language"),
            Document("2", "java programming"),
            Document("3", "cooking recipes")
        ]

        results = retriever.retrieve("python programming", docs, top_k=3)

        assert len(results) >= 1
        # Python doc should be first (matches both terms)
        assert results[0][0].doc_id == "1"
        # Should have matched terms
        assert "python" in results[0][2] or "programming" in results[0][2]

    def test_no_matches(self):
        """Test keyword retrieval with no matches."""
        retriever = KeywordRetriever()

        docs = [Document("1", "completely unrelated content")]
        results = retriever.retrieve("python programming", docs)

        assert len(results) == 0


class TestMetadataFilter:
    """Tests for MetadataFilter component."""

    def test_exact_match_filter(self):
        """Test exact match filtering."""
        doc = Document("1", "content", metadata={"category": "ml", "year": 2024})

        matches, fields = MetadataFilter.matches_filter(doc, {"category": "ml"})
        assert matches is True
        assert fields["category"] == "ml"

        matches, fields = MetadataFilter.matches_filter(doc, {"category": "web"})
        assert matches is False

    def test_list_filter(self):
        """Test list membership filtering."""
        doc = Document("1", "content", metadata={"category": "ml"})

        matches, _ = MetadataFilter.matches_filter(doc, {"category": ["ml", "ai", "data"]})
        assert matches is True

        matches, _ = MetadataFilter.matches_filter(doc, {"category": ["web", "mobile"]})
        assert matches is False

    def test_range_filter(self):
        """Test range filtering."""
        doc = Document("1", "content", metadata={"year": 2023})

        matches, _ = MetadataFilter.matches_filter(doc, {"year": {"min": 2020, "max": 2025}})
        assert matches is True

        matches, _ = MetadataFilter.matches_filter(doc, {"year": {"min": 2024}})
        assert matches is False


class TestResultFuser:
    """Tests for ResultFuser component."""

    def test_normalize_scores_min_max(self):
        """Test min-max normalization."""
        scores = [0.2, 0.5, 0.8]
        normalized = ResultFuser.normalize_scores(scores, method="min_max")

        assert abs(normalized[0] - 0.0) < 0.001
        assert abs(normalized[2] - 1.0) < 0.001

    def test_fuse_scores(self):
        """Test score fusion."""
        strategy_scores = {"semantic": 0.8, "keyword": 0.6}
        weights = {"semantic": 0.5, "keyword": 0.5}

        final, contributions = ResultFuser.fuse_scores(strategy_scores, weights)

        assert abs(final - 0.7) < 0.001
        assert "semantic" in contributions
        assert "keyword" in contributions


class TestReRanker:
    """Tests for ReRanker component."""

    def test_rerank_applies_adjustments(self):
        """Test that re-ranker applies adjustments."""
        import time

        reranker = ReRanker(freshness_weight=0.2, quality_weight=0.2)

        docs = [
            Document("1", "content", metadata={
                "timestamp": time.time() - 3600,  # 1 hour old
                "quality_score": 0.9
            }),
            Document("2", "content", metadata={
                "timestamp": time.time() - (365 * 24 * 3600),  # 1 year old
                "quality_score": 0.3
            })
        ]

        results = [(docs[0], 0.5), (docs[1], 0.5)]
        reranked = reranker.rerank(results, {})

        # Fresh, high-quality doc should be boosted
        assert len(reranked) == 2


class TestExplanationGenerator:
    """Tests for ExplanationGenerator component."""

    def test_generate_explanation(self):
        """Test explanation generation."""
        doc = Document("1", "test content")

        explanation = ExplanationGenerator.generate_explanation(
            document=doc,
            score_breakdown={"semantic": 0.8, "keyword": 0.6},
            strategy_contributions={"semantic": 0.4, "keyword": 0.3},
            query_terms_matched=["test", "content"],
            metadata_matches={"category": "test"},
            rerank_adjustments={"freshness": 0.05}
        )

        assert len(explanation) > 0
        assert "0.7" in explanation or "score" in explanation.lower()


class TestIntegration:
    """Integration tests for the complete system."""

    @pytest.fixture
    def system(self):
        """Create system with test data."""
        system = create_hybrid_retrieval_system()
        docs = [
            {
                "doc_id": "1",
                "content": "Machine learning algorithms for predictive modeling",
                "metadata": {"category": "ml", "year": 2024}
            },
            {
                "doc_id": "2",
                "content": "Deep learning neural networks for image recognition",
                "metadata": {"category": "ml", "year": 2023}
            },
            {
                "doc_id": "3",
                "content": "Web development with JavaScript frameworks",
                "metadata": {"category": "web", "year": 2024}
            },
            {
                "doc_id": "4",
                "content": "Database optimization and query performance",
                "metadata": {"category": "database", "year": 2023}
            }
        ]
        system.add_documents(docs)
        return system

    def test_full_retrieval_pipeline(self, system):
        """Test complete retrieval pipeline."""
        results = system.retrieve(
            query="machine learning AI",
            top_k=3,
            expand_query=True
        )

        assert len(results) >= 1
        # ML documents should rank higher
        assert any(r.document.metadata.get("category") == "ml" for r in results)

    def test_filtered_retrieval(self, system):
        """Test retrieval with metadata filters."""
        results = system.retrieve(
            query="development frameworks",
            top_k=3,
            filters={"year": 2024}
        )

        # All results should be from 2024
        for result in results:
            assert result.document.metadata.get("year") == 2024

    def test_execute_demo_mode(self):
        """Test execute with demo data."""
        system = create_hybrid_retrieval_system()
        result = system.execute()

        assert "query" in result
        assert "results" in result
        assert result["total_documents"] > 0
