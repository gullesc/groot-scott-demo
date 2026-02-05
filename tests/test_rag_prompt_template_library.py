"""
Tests for RAG Prompt Template Library

Run: pytest
Watch: pytest-watch (install with: pip install pytest-watch)
Coverage: pytest --cov=src
"""

import pytest
from src.rag_prompt_template_library import (
    RagPromptTemplateLibrary,
    create_rag_prompt_template_library,
    QueryType,
    DocumentDomain,
    RetrievedDocument,
    ContextFormatter,
    TemplateManager,
    FallbackHandler,
    FACTUAL_QA_TEMPLATE,
    SUMMARIZATION_TEMPLATE,
    COMPARISON_TEMPLATE,
    ANALYSIS_TEMPLATE,
    DOMAIN_MODIFIERS,
    LOW_CONFIDENCE_FALLBACK,
    NO_CONTEXT_FALLBACK,
)


class TestRagPromptTemplateLibrary:
    """Test suite for RagPromptTemplateLibrary."""

    @pytest.fixture
    def instance(self):
        """Create a fresh instance for each test."""
        return create_rag_prompt_template_library()

    def test_create_instance(self, instance):
        """Test that we can create an instance."""
        assert instance is not None
        assert isinstance(instance, RagPromptTemplateLibrary)

    def test_execute_returns_dict(self, instance):
        """Test that execute returns a dictionary with expected keys."""
        result = instance.execute(
            query_type="factual_qa",
            question="What is Python?",
            documents=[{"content": "Python is a programming language.", "source": "docs.python.org"}]
        )
        assert isinstance(result, dict)
        assert "prompt" in result
        assert "confidence_level" in result
        assert "used_fallback" in result
        assert "document_count" in result


class TestAcceptanceCriteria:
    """Tests for acceptance criteria."""

    @pytest.fixture
    def instance(self):
        """Create a fresh instance for each test."""
        return create_rag_prompt_template_library()

    @pytest.fixture
    def sample_documents(self):
        """Sample documents for testing."""
        return [
            {"content": "Document 1 content about topic A.", "source": "source1.com", "confidence_score": 0.9},
            {"content": "Document 2 content about topic B.", "source": "source2.com", "confidence_score": 0.8},
        ]

    def test_includes_templates_for_factual_q_a_summarization(self, instance):
        """Test: Includes templates for factual Q&A, summarization, comparison, and analysis"""
        # Verify all four query type templates exist
        assert QueryType.FACTUAL_QA is not None
        assert QueryType.SUMMARIZATION is not None
        assert QueryType.COMPARISON is not None
        assert QueryType.ANALYSIS is not None

        # Verify templates are defined
        assert FACTUAL_QA_TEMPLATE is not None
        assert SUMMARIZATION_TEMPLATE is not None
        assert COMPARISON_TEMPLATE is not None
        assert ANALYSIS_TEMPLATE is not None

        # Verify templates contain key elements
        assert "factual" in FACTUAL_QA_TEMPLATE.lower() or "answer" in FACTUAL_QA_TEMPLATE.lower()
        assert "summary" in SUMMARIZATION_TEMPLATE.lower() or "summarize" in SUMMARIZATION_TEMPLATE.lower()
        assert "compar" in COMPARISON_TEMPLATE.lower()
        assert "analysis" in ANALYSIS_TEMPLATE.lower() or "analyze" in ANALYSIS_TEMPLATE.lower()

        # Verify we can get templates for each type
        for query_type in [QueryType.FACTUAL_QA, QueryType.SUMMARIZATION, QueryType.COMPARISON, QueryType.ANALYSIS]:
            template = instance.get_template_for_type(query_type)
            assert template is not None
            assert len(template) > 100  # Templates should be substantial

    def test_implements_dynamic_context_insertion_with_proper_f(self, instance, sample_documents):
        """Test: Implements dynamic context insertion with proper formatting"""
        result = instance.execute(
            query_type="factual_qa",
            question="What is the topic?",
            documents=sample_documents
        )

        prompt = result["prompt"]

        # Verify documents are inserted into the prompt
        assert "Document 1 content" in prompt
        assert "Document 2 content" in prompt

        # Verify source attribution is included
        assert "source1.com" in prompt
        assert "source2.com" in prompt

        # Verify proper formatting with source markers
        assert "[Source 1]" in prompt or "Source 1" in prompt
        assert "[Source 2]" in prompt or "Source 2" in prompt

        # Verify confidence scores are included
        assert "0.9" in prompt or "Confidence" in prompt

    def test_provides_templates_for_different_document_types_t(self, instance, sample_documents):
        """Test: Provides templates for different document types (technical, legal, academic)"""
        # Verify domain modifiers exist
        assert DocumentDomain.TECHNICAL in DOMAIN_MODIFIERS
        assert DocumentDomain.LEGAL in DOMAIN_MODIFIERS
        assert DocumentDomain.ACADEMIC in DOMAIN_MODIFIERS

        # Test each domain type
        for domain in ["technical", "legal", "academic"]:
            result = instance.execute(
                query_type="factual_qa",
                question="What is this about?",
                documents=sample_documents,
                domain=domain
            )

            prompt = result["prompt"]

            # Verify domain-specific content is included
            if domain == "technical":
                assert "technical" in prompt.lower()
            elif domain == "legal":
                assert "legal" in prompt.lower()
            elif domain == "academic":
                assert "academic" in prompt.lower() or "scholarly" in prompt.lower()

        # Verify domain modifiers contain appropriate guidance
        assert "technical terminology" in DOMAIN_MODIFIERS[DocumentDomain.TECHNICAL].lower()
        assert "legal terminology" in DOMAIN_MODIFIERS[DocumentDomain.LEGAL].lower()
        assert "scholarly" in DOMAIN_MODIFIERS[DocumentDomain.ACADEMIC].lower()

    def test_includes_fallback_prompts_for_low_confidence_retri(self, instance):
        """Test: Includes fallback prompts for low-confidence retrievals"""
        # Test with low confidence documents
        low_confidence_docs = [
            {"content": "Possibly related content.", "source": "source.com", "confidence_score": 0.2},
        ]

        result = instance.execute(
            query_type="factual_qa",
            question="What is the answer?",
            documents=low_confidence_docs
        )

        # Verify fallback was used
        assert result["used_fallback"] is True
        assert result["confidence_level"] == "low"

        # Verify fallback template content
        assert "limited" in result["prompt"].lower() or "uncertainty" in result["prompt"].lower()

        # Test with no documents
        result_no_docs = instance.execute(
            query_type="factual_qa",
            question="What is the answer?",
            documents=[]
        )

        assert result_no_docs["used_fallback"] is True
        assert "no relevant" in result_no_docs["prompt"].lower() or "no documents" in result_no_docs["prompt"].lower()

        # Verify fallback templates exist
        assert LOW_CONFIDENCE_FALLBACK is not None
        assert NO_CONTEXT_FALLBACK is not None
        assert "limited" in LOW_CONFIDENCE_FALLBACK.lower()
        assert "no relevant" in NO_CONTEXT_FALLBACK.lower()


class TestContextFormatter:
    """Tests for ContextFormatter class."""

    def test_format_documents_with_metadata(self):
        """Test document formatting includes all metadata."""
        docs = [
            RetrievedDocument(
                content="Test content",
                source="test.com",
                confidence_score=0.85,
                title="Test Title",
                author="Test Author",
                page=42
            )
        ]

        formatted = ContextFormatter.format_documents(docs)

        assert "Test content" in formatted
        assert "test.com" in formatted
        assert "Test Title" in formatted
        assert "Test Author" in formatted
        assert "Page: 42" in formatted
        assert "0.85" in formatted

    def test_format_documents_empty_list(self):
        """Test formatting with empty document list."""
        formatted = ContextFormatter.format_documents([])
        assert "No documents" in formatted

    def test_sort_by_confidence(self):
        """Test document sorting by confidence score."""
        docs = [
            RetrievedDocument(content="Low", source="a.com", confidence_score=0.3),
            RetrievedDocument(content="High", source="b.com", confidence_score=0.9),
            RetrievedDocument(content="Medium", source="c.com", confidence_score=0.6),
        ]

        sorted_docs = ContextFormatter.sort_by_confidence(docs)

        assert sorted_docs[0].content == "High"
        assert sorted_docs[1].content == "Medium"
        assert sorted_docs[2].content == "Low"

    def test_filter_by_confidence(self):
        """Test document filtering by minimum confidence."""
        docs = [
            RetrievedDocument(content="Low", source="a.com", confidence_score=0.3),
            RetrievedDocument(content="High", source="b.com", confidence_score=0.9),
        ]

        filtered = ContextFormatter.filter_by_confidence(docs, min_confidence=0.5)

        assert len(filtered) == 1
        assert filtered[0].content == "High"


class TestFallbackHandler:
    """Tests for FallbackHandler class."""

    def test_assess_confidence_high(self):
        """Test high confidence assessment."""
        docs = [
            RetrievedDocument(content="A", source="a.com", confidence_score=0.9),
            RetrievedDocument(content="B", source="b.com", confidence_score=0.8),
        ]

        confidence = FallbackHandler.assess_confidence(docs)
        assert confidence == "high"

    def test_assess_confidence_medium(self):
        """Test medium confidence assessment."""
        docs = [
            RetrievedDocument(content="A", source="a.com", confidence_score=0.5),
            RetrievedDocument(content="B", source="b.com", confidence_score=0.6),
        ]

        confidence = FallbackHandler.assess_confidence(docs)
        assert confidence == "medium"

    def test_assess_confidence_low(self):
        """Test low confidence assessment."""
        docs = [
            RetrievedDocument(content="A", source="a.com", confidence_score=0.2),
            RetrievedDocument(content="B", source="b.com", confidence_score=0.3),
        ]

        confidence = FallbackHandler.assess_confidence(docs)
        assert confidence == "low"

    def test_assess_confidence_empty(self):
        """Test confidence assessment with no documents."""
        confidence = FallbackHandler.assess_confidence([])
        assert confidence == "low"

    def test_get_fallback_template_needed(self):
        """Test fallback template selection when confidence is low."""
        docs = [
            RetrievedDocument(content="A", source="a.com", confidence_score=0.2),
        ]

        template = FallbackHandler.get_fallback_template(docs)
        assert template is not None
        assert "limited" in template.lower()

    def test_get_fallback_template_not_needed(self):
        """Test no fallback when confidence is sufficient."""
        docs = [
            RetrievedDocument(content="A", source="a.com", confidence_score=0.9),
        ]

        template = FallbackHandler.get_fallback_template(docs)
        assert template is None

    def test_get_fallback_template_no_documents(self):
        """Test fallback template for empty document list."""
        template = FallbackHandler.get_fallback_template([])
        assert template is not None
        assert "no relevant" in template.lower()
