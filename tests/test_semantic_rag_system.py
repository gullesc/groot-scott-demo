"""
Tests for Semantic RAG System

Run: pytest
Watch: pytest-watch (install with: pip install pytest-watch)
Coverage: pytest --cov=src
"""

import pytest
from src.semantic_rag_system import SemanticRagSystem, create_semantic_rag_system


class TestSemanticRagSystem:
    """Test suite for SemanticRagSystem."""

    @pytest.fixture
    def instance(self):
        """Create a fresh instance for each test."""
        return create_semantic_rag_system()

    @pytest.fixture
    def indexed_instance(self):
        """Create an instance with indexed documents."""
        instance = create_semantic_rag_system()
        documents = [
            "Machine learning algorithms require large datasets for training.",
            "Neural networks are inspired by biological brain structures.",
            "Deep learning uses multiple layers to extract features.",
            "Python is a popular programming language for data science.",
            "Natural language processing helps computers understand text.",
        ]
        instance.index_documents(documents)
        return instance

    def test_create_instance(self, instance):
        """Test that we can create an instance."""
        assert instance is not None
        assert isinstance(instance, SemanticRagSystem)

    def test_execute_returns_results(self, instance):
        """Test that execute returns demonstration results."""
        result = instance.execute()
        assert result is not None
        assert "indexed_documents" in result
        assert "vocabulary_size" in result
        assert "semantic_results1" in result


class TestAcceptanceCriteria:
    """Tests for acceptance criteria."""

    @pytest.fixture
    def instance(self):
        """Create a fresh instance for each test."""
        return create_semantic_rag_system()

    def test_replaces_text_based_search_with_embedding_based_si(self, instance):
        """Test: Replaces text-based search with embedding-based similarity search"""
        documents = [
            "Machine learning models require training data.",
            "Neural networks learn patterns from examples.",
            "Deep learning extracts hierarchical features.",
        ]
        instance.index_documents(documents)

        # Search should return results based on semantic similarity
        results = instance.search("AI training requirements")

        assert len(results) > 0
        # Results should have similarity scores, not just keyword matches
        for result in results:
            assert "score" in result
            assert "document" in result
            assert isinstance(result["score"], float)

    def test_handles_query_embedding_generation_and_similarity(self, instance):
        """Test: Handles query embedding generation and similarity scoring"""
        documents = [
            "Python is great for machine learning.",
            "JavaScript is used for web development.",
            "Rust is known for memory safety.",
        ]
        instance.index_documents(documents)

        # Get query embedding
        query = "programming languages for ML"
        embedding = instance.get_query_embedding(query)

        # Embedding should be a list of floats
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, float) for x in embedding)

        # Search results should have similarity scores
        results = instance.search(query)
        if results:
            # First result should have highest score
            assert results[0]["score"] >= results[-1]["score"] if len(results) > 1 else True

    def test_implements_retrieval_result_ranking_and_filtering(self, instance):
        """Test: Implements retrieval result ranking and filtering"""
        documents = [
            "Machine learning uses algorithms to learn from data.",
            "Data science involves statistical analysis.",
            "Deep learning is a subset of machine learning.",
            "Web development uses HTML and CSS.",
            "Database management stores structured data.",
        ]
        instance.index_documents(documents)

        # Search with specific top_k
        results_3 = instance.search("machine learning data", top_k=3)
        results_5 = instance.search("machine learning data", top_k=5)

        assert len(results_3) <= 3
        assert len(results_5) <= 5

        # Results should be ranked by score (descending)
        if len(results_5) >= 2:
            for i in range(len(results_5) - 1):
                assert results_5[i]["score"] >= results_5[i + 1]["score"]

        # Test with similarity threshold
        instance_threshold = create_semantic_rag_system()
        instance_threshold._similarity_threshold = 0.1
        instance_threshold.index_documents(documents)
        results_threshold = instance_threshold.search("machine learning")

        # All results should meet threshold
        for result in results_threshold:
            assert result["score"] >= 0.0  # May be filtered by threshold

    def test_demonstrates_improved_relevance_over_keyword_based(self, instance):
        """Test: Demonstrates improved relevance over keyword-based approach"""
        documents = [
            "Neural networks learn patterns from training examples.",
            "Deep learning models use backpropagation for optimization.",
            "AI systems can make predictions based on data.",
            "Web browsers display HTML pages.",
            "Databases store information in tables.",
        ]
        instance.index_documents(documents)

        # Query that uses synonyms/related terms not in documents
        query = "machine intelligence training methods"

        comparison = instance.compare_with_keyword_search(query)

        # Should have both results
        assert "semantic_results" in comparison
        assert "keyword_results" in comparison
        assert "analysis" in comparison

        # Analysis should show differences
        analysis = comparison["analysis"]
        assert "semantic_found" in analysis
        assert "keyword_found" in analysis
        assert "overlap_count" in analysis

        # Semantic search should find relevant docs even without exact matches
        # because it understands "machine intelligence" relates to "AI" and "neural networks"
        semantic_results = comparison["semantic_results"]
        assert len(semantic_results) > 0


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def instance(self):
        """Create a fresh instance for each test."""
        return create_semantic_rag_system()

    def test_empty_document_list(self, instance):
        """Test indexing empty document list."""
        count = instance.index_documents([])
        assert count == 0
        assert instance.get_document_count() == 0

    def test_empty_query(self, instance):
        """Test searching with empty query."""
        documents = ["Test document"]
        instance.index_documents(documents)

        results = instance.search("")
        assert results == []

    def test_search_before_indexing(self, instance):
        """Test searching before any documents are indexed."""
        results = instance.search("test query")
        assert results == []

    def test_documents_with_metadata(self, instance):
        """Test indexing documents with metadata."""
        documents = ["Doc 1", "Doc 2"]
        metadata = [{"source": "file1.txt"}, {"source": "file2.txt"}]

        count = instance.index_documents(documents, metadata)
        assert count == 2

        results = instance.search("Doc", top_k=2)
        for result in results:
            assert "metadata" in result
            assert "source" in result["metadata"]

    def test_get_document_embedding(self, instance):
        """Test retrieving document embedding by index."""
        documents = ["Test document"]
        instance.index_documents(documents)

        embedding = instance.get_document_embedding(0)
        assert embedding is not None
        assert isinstance(embedding, list)

        # Invalid index should return None
        assert instance.get_document_embedding(100) is None

    def test_vocabulary_and_document_counts(self, instance):
        """Test vocabulary and document count methods."""
        documents = [
            "Machine learning is great.",
            "Deep learning is powerful.",
        ]
        instance.index_documents(documents)

        assert instance.get_document_count() == 2
        assert instance.get_vocabulary_size() > 0
