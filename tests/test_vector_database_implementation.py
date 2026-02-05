"""
Tests for Vector Database Implementation

Run: pytest
Watch: pytest-watch (install with: pip install pytest-watch)
Coverage: pytest --cov=src
"""

import pytest
from src.vector_database_implementation import VectorDatabaseImplementation, create_vector_database_implementation


class TestVectorDatabaseImplementation:
    """Test suite for VectorDatabaseImplementation."""

    @pytest.fixture
    def instance(self):
        """Create a fresh instance for each test."""
        return create_vector_database_implementation()

    def test_create_instance(self, instance):
        """Test that we can create an instance."""
        assert instance is not None
        assert isinstance(instance, VectorDatabaseImplementation)

    def test_execute_returns_results(self, instance):
        """Test that execute returns demonstration results."""
        result = instance.execute()
        assert result is not None
        assert "stored_count" in result
        assert "search_results" in result
        assert "performance_metrics" in result


class TestAcceptanceCriteria:
    """Tests for acceptance criteria."""

    @pytest.fixture
    def instance(self):
        """Create a fresh instance for each test."""
        return create_vector_database_implementation()

    def test_successfully_stores_document_chunks_as_embeddings(self, instance):
        """Test: Successfully stores document chunks as embeddings in vector database"""
        # Store an embedding
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        metadata = {"doc_id": "test1", "source": "test.pdf"}
        text = "This is a test document chunk."

        doc_id = instance.store_embedding(embedding, metadata, text)

        # Verify storage
        assert doc_id is not None
        assert instance.count() == 1

        # Retrieve and verify
        retrieved = instance.get_embedding(doc_id)
        assert retrieved is not None
        assert retrieved["embedding"] == embedding
        assert retrieved["metadata"] == metadata
        assert retrieved["text"] == text

    def test_implements_efficient_similarity_search_with_config(self, instance):
        """Test: Implements efficient similarity search with configurable top-k results"""
        # Store multiple embeddings
        embeddings = [
            ([0.1, 0.2, 0.3, 0.4, 0.5], {"doc_id": "doc1"}, "Document 1"),
            ([0.15, 0.25, 0.35, 0.45, 0.55], {"doc_id": "doc2"}, "Document 2"),
            ([0.9, 0.1, 0.1, 0.1, 0.1], {"doc_id": "doc3"}, "Document 3"),
            ([0.12, 0.22, 0.32, 0.42, 0.52], {"doc_id": "doc4"}, "Document 4"),
        ]

        for emb, meta, text in embeddings:
            instance.store_embedding(emb, meta, text)

        # Search with top_k=2
        query = [0.11, 0.21, 0.31, 0.41, 0.51]
        results = instance.similarity_search(query, top_k=2)

        assert len(results) == 2
        # Results should be sorted by similarity
        assert results[0]["score"] >= results[1]["score"]

        # Search with different top_k
        results_3 = instance.similarity_search(query, top_k=3)
        assert len(results_3) == 3

    def test_includes_metadata_filtering_capabilities(self, instance):
        """Test: Includes metadata filtering capabilities"""
        # Store embeddings with different metadata
        instance.store_embedding(
            [0.1, 0.2, 0.3],
            {"source": "manual.pdf", "type": "intro"},
            "Introduction text"
        )
        instance.store_embedding(
            [0.15, 0.25, 0.35],
            {"source": "manual.pdf", "type": "body"},
            "Body text"
        )
        instance.store_embedding(
            [0.12, 0.22, 0.32],
            {"source": "guide.pdf", "type": "intro"},
            "Guide intro"
        )

        # Search with filter
        query = [0.11, 0.21, 0.31]
        filtered_results = instance.similarity_search(
            query,
            top_k=5,
            filters={"source": "manual.pdf"}
        )

        # Should only return documents from manual.pdf
        assert len(filtered_results) == 2
        for result in filtered_results:
            assert result["metadata"]["source"] == "manual.pdf"

        # Filter by type
        type_results = instance.similarity_search(
            query,
            top_k=5,
            filters={"type": "intro"}
        )
        assert len(type_results) == 2
        for result in type_results:
            assert result["metadata"]["type"] == "intro"

    def test_provides_performance_benchmarks_for_query_response(self, instance):
        """Test: Provides performance benchmarks for query response times"""
        # Store some embeddings
        for i in range(10):
            instance.store_embedding(
                [0.1 * i, 0.2 * i, 0.3 * i, 0.4 * i, 0.5 * i],
                {"doc_id": f"doc{i}"},
                f"Document {i}"
            )

        # Perform some queries
        query = [0.5, 0.5, 0.5, 0.5, 0.5]
        for _ in range(5):
            instance.similarity_search(query, top_k=3)

        # Get performance metrics
        metrics = instance.get_performance_metrics()

        # Verify metrics exist
        assert "avg_query_time" in metrics
        assert "avg_store_time" in metrics
        assert "total_embeddings" in metrics
        assert "total_queries" in metrics

        # Verify values are reasonable
        assert metrics["total_embeddings"] == 10
        assert metrics["total_queries"] == 5
        assert metrics["avg_query_time"] >= 0
        assert metrics["avg_store_time"] >= 0


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def instance(self):
        """Create a fresh instance for each test."""
        return create_vector_database_implementation()

    def test_empty_embedding_raises_error(self, instance):
        """Test that empty embedding raises ValueError."""
        with pytest.raises(ValueError):
            instance.store_embedding([], {}, "text")

    def test_dimension_mismatch_raises_error(self, instance):
        """Test that dimension mismatch raises ValueError."""
        instance.store_embedding([0.1, 0.2, 0.3], {}, "text1")
        with pytest.raises(ValueError):
            instance.store_embedding([0.1, 0.2], {}, "text2")  # Different dimension

    def test_search_empty_database_returns_empty(self, instance):
        """Test that searching empty database returns empty list."""
        results = instance.similarity_search([0.1, 0.2, 0.3], top_k=5)
        assert results == []

    def test_delete_embedding(self, instance):
        """Test embedding deletion."""
        doc_id = instance.store_embedding([0.1, 0.2, 0.3], {}, "text")
        assert instance.count() == 1

        result = instance.delete_embedding(doc_id)
        assert result is True
        assert instance.count() == 0

    def test_clear_database(self, instance):
        """Test clearing the database."""
        for i in range(5):
            instance.store_embedding([0.1 * i, 0.2, 0.3], {}, f"text{i}")

        assert instance.count() == 5
        instance.clear()
        assert instance.count() == 0

    def test_batch_store_embeddings(self, instance):
        """Test batch storage of embeddings."""
        items = [
            {"embedding": [0.1, 0.2, 0.3], "metadata": {"id": "1"}, "text": "Text 1"},
            {"embedding": [0.2, 0.3, 0.4], "metadata": {"id": "2"}, "text": "Text 2"},
            {"embedding": [0.3, 0.4, 0.5], "metadata": {"id": "3"}, "text": "Text 3"},
        ]

        ids = instance.store_embeddings_batch(items)

        assert len(ids) == 3
        assert instance.count() == 3
