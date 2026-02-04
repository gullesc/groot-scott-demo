"""
Tests for Basic RAG Question-Answering System

Run: pytest
Watch: pytest-watch (install with: pip install pytest-watch)
Coverage: pytest --cov=src
"""

import pytest
from src.basic_rag_question_answering_system import (
    BasicRagQuestionAnsweringSystem,
    create_basic_rag_question_answering_system,
    RAGError,
    TextProcessor,
    TFIDFCalculator,
    SimilarityCalculator,
    ContextFormatter,
    SourceTracker,
)


class TestBasicRagQuestionAnsweringSystem:
    """Test suite for BasicRagQuestionAnsweringSystem."""

    @pytest.fixture
    def instance(self):
        """Create a fresh instance for each test."""
        return create_basic_rag_question_answering_system()

    def test_create_instance(self, instance):
        """Test that we can create an instance."""
        assert instance is not None
        assert isinstance(instance, BasicRagQuestionAnsweringSystem)

    def test_add_documents(self, instance):
        """Test adding documents to the system."""
        documents = [
            {"text": "Python is a programming language.", "source": "doc1.txt", "chunk_id": "1"},
            {"text": "Machine learning uses algorithms.", "source": "doc2.txt", "chunk_id": "2"},
        ]
        instance.add_documents(documents)

        assert len(instance.documents) == 2
        assert len(instance.document_vectors) == 2

    def test_retrieve_returns_relevant_chunks(self, instance):
        """Test that retrieval returns relevant chunks."""
        documents = [
            {"text": "Solar panels convert sunlight into electricity.", "source": "energy.txt", "chunk_id": "1"},
            {"text": "Python programming is popular for data science.", "source": "code.txt", "chunk_id": "2"},
            {"text": "Renewable energy includes solar and wind power.", "source": "green.txt", "chunk_id": "3"},
        ]
        instance.add_documents(documents)

        chunks, scores = instance.retrieve("What is solar energy?")

        # Should retrieve energy-related documents
        assert len(chunks) > 0
        assert len(scores) > 0
        assert all(s >= 0 for s in scores)


class TestTextProcessor:
    """Tests for TextProcessor."""

    def test_tokenize(self):
        """Test basic tokenization."""
        tokens = TextProcessor.tokenize("Hello World! This is a Test.")
        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens

    def test_tokenize_removes_punctuation(self):
        """Test that punctuation is handled."""
        tokens = TextProcessor.tokenize("Hello, world! How are you?")
        assert "hello" in tokens
        assert "," not in tokens
        assert "!" not in tokens

    def test_preprocess_with_stopwords(self):
        """Test preprocessing with stopword removal."""
        tokens = TextProcessor.preprocess("This is a test of the system")
        assert "this" not in tokens
        assert "is" not in tokens
        assert "a" not in tokens
        assert "test" in tokens
        assert "system" in tokens

    def test_preprocess_without_stopwords(self):
        """Test preprocessing without stopword removal."""
        tokens = TextProcessor.preprocess("This is a test", remove_stopwords=False)
        assert "this" in tokens
        assert "is" in tokens
        assert "a" in tokens


class TestTFIDFCalculator:
    """Tests for TFIDFCalculator."""

    @pytest.fixture
    def calculator(self):
        return TFIDFCalculator()

    def test_calculate_tf(self, calculator):
        """Test term frequency calculation."""
        tokens = ["hello", "world", "hello"]
        tf = calculator.calculate_tf(tokens)

        assert tf["hello"] == pytest.approx(2 / 3)
        assert tf["world"] == pytest.approx(1 / 3)

    def test_fit_and_idf(self, calculator):
        """Test IDF calculation after fitting."""
        documents = [
            ["python", "programming"],
            ["python", "data", "science"],
            ["machine", "learning", "data"],
        ]
        calculator.fit(documents)

        # Python appears in 2 of 3 docs
        idf_python = calculator.calculate_idf("python")
        # Machine appears in 1 of 3 docs
        idf_machine = calculator.calculate_idf("machine")

        # Machine should have higher IDF (rarer term)
        assert idf_machine > idf_python

    def test_calculate_tfidf(self, calculator):
        """Test TF-IDF calculation."""
        documents = [
            ["the", "quick", "brown", "fox"],
            ["the", "lazy", "dog"],
        ]
        calculator.fit(documents)

        tfidf = calculator.calculate_tfidf(["quick", "fox"])
        assert "quick" in tfidf
        assert "fox" in tfidf
        assert all(v > 0 for v in tfidf.values())


class TestSimilarityCalculator:
    """Tests for SimilarityCalculator."""

    def test_identical_vectors(self):
        """Test similarity of identical vectors."""
        vec = {"a": 1.0, "b": 2.0, "c": 3.0}
        similarity = SimilarityCalculator.cosine_similarity(vec, vec)
        assert similarity == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        """Test similarity of vectors with no common terms."""
        vec1 = {"a": 1.0, "b": 2.0}
        vec2 = {"c": 1.0, "d": 2.0}
        similarity = SimilarityCalculator.cosine_similarity(vec1, vec2)
        assert similarity == 0.0

    def test_partial_overlap(self):
        """Test similarity of vectors with partial overlap."""
        vec1 = {"a": 1.0, "b": 2.0, "c": 3.0}
        vec2 = {"b": 2.0, "c": 3.0, "d": 4.0}
        similarity = SimilarityCalculator.cosine_similarity(vec1, vec2)
        assert 0 < similarity < 1

    def test_empty_vectors(self):
        """Test similarity with empty vectors."""
        assert SimilarityCalculator.cosine_similarity({}, {"a": 1.0}) == 0.0
        assert SimilarityCalculator.cosine_similarity({"a": 1.0}, {}) == 0.0
        assert SimilarityCalculator.cosine_similarity({}, {}) == 0.0


class TestContextFormatter:
    """Tests for ContextFormatter."""

    def test_format_context(self):
        """Test context formatting for Claude."""
        chunks = [
            {"text": "First chunk content.", "source": "doc1.txt", "chunk_id": "1"},
            {"text": "Second chunk content.", "source": "doc2.txt", "chunk_id": "2"},
        ]
        scores = [0.9, 0.7]

        context = ContextFormatter.format_context(chunks, scores)

        assert "First chunk content" in context
        assert "Second chunk content" in context
        assert "doc1.txt" in context
        assert "doc2.txt" in context
        assert "0.90" in context
        assert "0.70" in context

    def test_format_empty_context(self):
        """Test formatting when no chunks are provided."""
        context = ContextFormatter.format_context([], [])
        assert "No relevant context" in context

    def test_format_prompt(self):
        """Test full prompt formatting."""
        question = "What is Python?"
        context = "Python is a programming language."

        prompt = ContextFormatter.format_prompt(question, context)

        assert question in prompt
        assert context in prompt
        assert "answer" in prompt.lower() or "question" in prompt.lower()


class TestSourceTracker:
    """Tests for SourceTracker."""

    def test_format_citations(self):
        """Test citation formatting."""
        chunks = [
            {"source": "doc1.txt", "chunk_id": "1", "metadata": {"page_number": 5}},
            {"source": "doc2.txt", "chunk_id": "2", "metadata": {}},
        ]

        citations = SourceTracker.format_citations(chunks)

        assert len(citations) == 2
        assert "doc1.txt" in citations[0]
        assert "page 5" in citations[0]
        assert "doc2.txt" in citations[1]


class TestAcceptanceCriteria:
    """Tests for acceptance criteria."""

    @pytest.fixture
    def instance(self):
        """Create a fresh instance for each test."""
        return create_basic_rag_question_answering_system()

    def test_accepts_user_questions_and_searches_through_documents(self, instance):
        """Test: Accepts user questions and searches through document chunks"""
        documents = [
            {"text": "The capital of France is Paris. Paris is known for the Eiffel Tower.", "source": "geo.txt", "chunk_id": "1"},
            {"text": "Python is a high-level programming language.", "source": "tech.txt", "chunk_id": "2"},
            {"text": "Machine learning is a subset of artificial intelligence.", "source": "ai.txt", "chunk_id": "3"},
        ]
        instance.add_documents(documents)

        # Test retrieval for geography question
        chunks, scores = instance.retrieve("What is the capital of France?")

        assert len(chunks) > 0
        # The geography document should be retrieved
        sources = [c.get("source") for c in chunks]
        assert "geo.txt" in sources or any("paris" in c.get("text", "").lower() for c in chunks)

    def test_uses_basic_text_similarity_for_retrieval(self, instance):
        """Test: Uses basic text similarity for retrieval (TF-IDF or keyword matching)"""
        documents = [
            {"text": "Solar energy is renewable and sustainable.", "source": "solar.txt", "chunk_id": "1"},
            {"text": "Wind turbines generate electricity from wind.", "source": "wind.txt", "chunk_id": "2"},
            {"text": "Cooking recipes for Italian pasta dishes.", "source": "food.txt", "chunk_id": "3"},
        ]
        instance.add_documents(documents)

        # Query about energy should retrieve energy documents
        chunks, scores = instance.retrieve("What are renewable energy sources?")

        # Should have similarity scores
        assert len(scores) > 0
        assert all(isinstance(s, float) for s in scores)

        # Energy documents should score higher than food document
        chunk_sources = [c.get("source") for c in chunks]
        if "food.txt" in chunk_sources:
            food_idx = chunk_sources.index("food.txt")
            # Food should be ranked lower (higher index = lower score since sorted desc)
            energy_sources = {"solar.txt", "wind.txt"}
            for i, src in enumerate(chunk_sources):
                if src in energy_sources:
                    assert scores[i] >= scores[food_idx]

    def test_formats_retrieved_context_for_claude_api_calls(self, instance):
        """Test: Formats retrieved context for Claude API calls"""
        chunks = [
            {"text": "Important context information.", "source": "source1.txt", "chunk_id": "1"},
        ]
        scores = [0.85]

        context = ContextFormatter.format_context(chunks, scores)
        prompt = ContextFormatter.format_prompt("What is the answer?", context)

        # Verify the prompt structure
        assert "Important context information" in prompt
        assert "source1.txt" in prompt
        assert "What is the answer?" in prompt
        # Should have instructions for Claude
        assert "context" in prompt.lower()

    def test_returns_answers_with_source_citations(self, instance):
        """Test: Returns answers with source citations"""
        documents = [
            {"text": "The speed of light is approximately 299,792 km/s.", "source": "physics.txt", "chunk_id": "1"},
            {"text": "Water boils at 100 degrees Celsius at sea level.", "source": "chemistry.txt", "chunk_id": "2"},
        ]
        instance.add_documents(documents)

        # Retrieve chunks for a question
        chunks, scores = instance.retrieve("What is the speed of light?")

        # Get citations
        citations = SourceTracker.format_citations(chunks)

        # Should have citations
        assert len(citations) > 0
        # Citations should reference the source files
        all_citations = " ".join(citations)
        assert "physics.txt" in all_citations or "chemistry.txt" in all_citations

    def test_execute_requires_question(self, instance):
        """Test that execute raises error without question."""
        with pytest.raises(RAGError):
            instance.execute(question=None)

    def test_execute_with_documents_and_question(self, instance):
        """Test execute with both documents and question (without API call)."""
        documents = [
            {"text": "The RAG system retrieves documents based on similarity matching.", "source": "test.txt", "chunk_id": "1"},
            {"text": "Machine learning algorithms process data efficiently.", "source": "ml.txt", "chunk_id": "2"},
        ]

        # This will fail on API call but we can test the retrieval part
        instance.add_documents(documents)
        chunks, scores = instance.retrieve("How does the RAG system work with documents?")

        assert len(chunks) > 0
        # Should retrieve the RAG-related document
        sources = [c.get("source") for c in chunks]
        assert "test.txt" in sources
