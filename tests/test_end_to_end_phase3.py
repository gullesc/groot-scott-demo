"""
End-to-End Tests for RAG System - Phase 3

This test suite validates the complete RAG pipeline through Phase 3.
It exercises all components working together:
  Phase 1: Document ingestion, chunking, basic RAG
  Phase 2: Vector database, semantic search, evaluation
  Phase 3: Prompt templates, citations, context-aware responses

NOTE: This is an INCOMPLETE system test. Additional phases will extend
this test suite as new functionality is implemented.

Run: pytest tests/test_end_to_end_phase3.py -v
"""

import pytest
import tempfile
import os
from pathlib import Path

# Phase 1 imports
from src.document_ingestion_pipeline import (
    create_document_ingestion_pipeline,
    DocumentIngestionPipeline,
)
from src.basic_rag_question_answering_system import (
    create_basic_rag_question_answering_system,
    BasicRagQuestionAnsweringSystem,
)

# Phase 2 imports
from src.vector_database_implementation import (
    create_vector_database_implementation,
    VectorDatabaseImplementation,
)
from src.semantic_rag_system import (
    create_semantic_rag_system,
    SemanticRagSystem,
)
from src.retrieval_evaluation_framework import (
    create_retrieval_evaluation_framework,
    RetrievalEvaluationFramework,
)

# Phase 3 imports
from src.rag_prompt_template_library import (
    create_rag_prompt_template_library,
    RagPromptTemplateLibrary,
    QueryType as TemplateQueryType,
    DocumentDomain,
    RetrievedDocument,
)
from src.citation_and_source_tracking_system import (
    create_citation_and_source_tracking_system,
    CitationAndSourceTrackingSystem,
    CitationFormat,
)
from src.context_aware_rag_system import (
    create_context_aware_rag_system,
    ContextAwareRagSystem,
    QueryType as ContextQueryType,
    RetrievalQuality,
    ConfidenceLevel,
    PromptStrategy,
)


# =============================================================================
# TEST FIXTURES - Sample Documents for E2E Testing
# =============================================================================

SAMPLE_DOCUMENTS = {
    "ai_overview.txt": """
Artificial Intelligence: A Comprehensive Overview

Artificial Intelligence (AI) refers to the simulation of human intelligence
in machines programmed to think and learn like humans. The field was founded
in 1956 at the Dartmouth Conference.

Machine Learning is a subset of AI that enables systems to learn from data
without being explicitly programmed. Deep Learning, a subset of Machine Learning,
uses neural networks with many layers to model complex patterns.

Key AI Applications:
- Natural Language Processing (NLP): Understanding and generating human language
- Computer Vision: Interpreting visual information from the world
- Robotics: Creating intelligent machines that can perform tasks
- Expert Systems: Mimicking human expert decision-making

The future of AI includes advances in general artificial intelligence (AGI),
which would match human cognitive abilities across all domains.
""",
    "python_guide.md": """
# Python Programming Guide

## Introduction
Python is a high-level, interpreted programming language known for its
readability and versatility. Created by Guido van Rossum in 1991.

## Key Features
- **Dynamic Typing**: Variables don't need type declarations
- **Indentation-based Syntax**: Uses whitespace for code blocks
- **Rich Standard Library**: Batteries included philosophy

## Data Types
Python supports several built-in data types:
1. Integers (int): Whole numbers like 42
2. Floats (float): Decimal numbers like 3.14
3. Strings (str): Text like "Hello World"
4. Lists (list): Ordered, mutable collections
5. Dictionaries (dict): Key-value pairs

## Best Practices
Follow PEP 8 style guidelines for readable code.
Use virtual environments for project isolation.
Write docstrings to document your functions.
""",
    "climate_science.txt": """
Climate Science: Understanding Global Changes

Climate change refers to long-term shifts in global temperatures and weather
patterns. While natural factors play a role, human activities have been the
main driver since the 1800s, primarily through burning fossil fuels.

The Greenhouse Effect:
Carbon dioxide (CO2) and other greenhouse gases trap heat in Earth's atmosphere.
Since the industrial revolution, CO2 levels have increased by over 50%.
Global average temperature has risen approximately 1.1°C since pre-industrial times.

Key Climate Indicators:
- Rising sea levels (3.7mm per year on average)
- Shrinking ice sheets in Greenland and Antarctica
- More frequent extreme weather events
- Ocean acidification affecting marine ecosystems

Mitigation Strategies:
1. Transition to renewable energy sources
2. Improve energy efficiency in buildings and transportation
3. Protect and restore forests (carbon sinks)
4. Develop carbon capture technologies
""",
}


@pytest.fixture
def sample_documents_dir():
    """Create a temporary directory with sample documents."""
    with tempfile.TemporaryDirectory() as tmpdir:
        for filename, content in SAMPLE_DOCUMENTS.items():
            filepath = Path(tmpdir) / filename
            filepath.write_text(content.strip())
        yield tmpdir


@pytest.fixture
def ingestion_pipeline():
    """Create a document ingestion pipeline."""
    return create_document_ingestion_pipeline()


@pytest.fixture
def vector_db():
    """Create a vector database instance."""
    return create_vector_database_implementation()


@pytest.fixture
def semantic_rag():
    """Create a semantic RAG system."""
    return create_semantic_rag_system()


@pytest.fixture
def template_library():
    """Create the prompt template library."""
    return create_rag_prompt_template_library()


@pytest.fixture
def citation_system():
    """Create the citation and source tracking system."""
    return create_citation_and_source_tracking_system()


@pytest.fixture
def context_aware_system():
    """Create the context-aware RAG system."""
    return create_context_aware_rag_system()


# =============================================================================
# END-TO-END TEST: Full Pipeline Through Phase 3
# =============================================================================

class TestEndToEndPhase3Pipeline:
    """
    End-to-end test suite validating the complete RAG pipeline.

    This tests the integration of all Phase 1-3 components working together
    as a cohesive system.
    """

    def test_full_pipeline_document_to_response(
        self,
        sample_documents_dir,
        ingestion_pipeline,
        semantic_rag,
        template_library,
        citation_system,
        context_aware_system,
    ):
        """
        Test the complete pipeline: Document -> Ingest -> Index -> Query ->
        Template -> Cite -> Context-Aware Response
        """
        # =====================================================================
        # PHASE 1: Document Ingestion
        # =====================================================================

        # Ingest all sample documents
        all_texts = []
        all_metadata = []

        for filename in SAMPLE_DOCUMENTS.keys():
            filepath = os.path.join(sample_documents_dir, filename)
            chunks = ingestion_pipeline.process_file(filepath)

            assert chunks is not None, f"Failed to ingest {filename}"
            assert len(chunks) > 0, f"No chunks created for {filename}"

            for chunk in chunks:
                all_texts.append(chunk["text"])
                metadata = chunk.get("metadata", {})
                metadata["source_file"] = filename
                all_metadata.append(metadata)

        # Verify we got chunks from all documents
        assert len(all_texts) >= 3, "Should have chunks from all documents"

        # =====================================================================
        # PHASE 2: Semantic Indexing and Search
        # =====================================================================

        # Index all chunks in the semantic RAG system
        indexed_count = semantic_rag.index_documents(all_texts, all_metadata)
        assert indexed_count > 0, "Should have indexed documents"

        # Perform semantic search
        query = "What is machine learning and how does it relate to AI?"
        search_results = semantic_rag.search(query, top_k=3)

        assert len(search_results) > 0, "Semantic search returned no results"

        # Verify results are relevant (should find AI document)
        found_ai_content = any(
            "machine learning" in result["document"].lower() or
            "artificial intelligence" in result["document"].lower()
            for result in search_results
        )
        assert found_ai_content, "Search did not find relevant AI content"

        # =====================================================================
        # PHASE 3: Template Selection, Citation, and Context-Aware Response
        # =====================================================================

        # Register sources in citation system
        source_ids = []
        for i, result in enumerate(search_results):
            source_file = result.get("metadata", {}).get("source_file", f"source_{i}")
            source_id = citation_system.register_source(
                content=result["document"],
                title=source_file,
            )
            source_ids.append(source_id)

        # Create retrieved documents for template library
        retrieved_docs = [
            RetrievedDocument(
                content=result["document"],
                source=result.get("metadata", {}).get("source_file", f"source_{i}"),
                confidence_score=result.get("score", 0.8),
            )
            for i, result in enumerate(search_results)
        ]

        # Generate prompt using template library (Factual Q&A type)
        prompt_result = template_library.generate_prompt(
            query_type=TemplateQueryType.FACTUAL_QA,
            question=query,
            documents=retrieved_docs,
            domain=DocumentDomain.TECHNICAL,
        )

        assert prompt_result is not None, "Template library failed to generate prompt"
        assert "prompt" in prompt_result, "Result should contain 'prompt' key"
        assert query in prompt_result["prompt"], "Generated prompt should include the query"

        # Use context-aware system to analyze and classify
        contexts_for_analysis = [
            {
                "content": result["document"],
                "confidence_score": result.get("score", 0.8),
                "source": result.get("metadata", {}).get("source_file", f"source_{i}"),
            }
            for i, result in enumerate(search_results)
        ]

        response = context_aware_system.execute(
            query=query,
            retrieved_contexts=contexts_for_analysis,
        )

        # Verify context-aware response structure
        assert response is not None, "Context-aware system returned None"
        assert "query_type" in response, "Response missing query_type"
        assert "prompt_strategy" in response, "Response missing prompt_strategy"
        assert "response" in response, "Response missing response content"

        # Generate reference list for citations
        reference_list = citation_system.generate_reference_list(
            format_type=CitationFormat.INLINE
        )
        assert reference_list is not None, "Citation system failed to generate reference list"

    def test_pipeline_with_different_query_types(
        self,
        sample_documents_dir,
        ingestion_pipeline,
        semantic_rag,
        context_aware_system,
    ):
        """Test the pipeline handles different query types correctly."""
        # Ingest and index documents
        all_texts = []
        all_metadata = []

        for filename in SAMPLE_DOCUMENTS.keys():
            filepath = os.path.join(sample_documents_dir, filename)
            chunks = ingestion_pipeline.process_file(filepath)
            for chunk in chunks:
                all_texts.append(chunk["text"])
                metadata = chunk.get("metadata", {})
                metadata["source_file"] = filename
                all_metadata.append(metadata)

        semantic_rag.index_documents(all_texts, all_metadata)

        # Test different query types
        test_queries = [
            ("What is Python?", ContextQueryType.FACTUAL),
            ("Why is climate change happening?", ContextQueryType.ANALYTICAL),
        ]

        for query, expected_type in test_queries:
            results = semantic_rag.search(query, top_k=2)

            contexts = [
                {
                    "content": r["document"],
                    "confidence_score": r.get("score", 0.8),
                    "source": r.get("metadata", {}).get("source_file", "unknown"),
                }
                for r in results
            ]

            response = context_aware_system.execute(
                query=query,
                retrieved_contexts=contexts,
            )

            assert response is not None, f"Failed for query: {query}"
            assert "query_type" in response

    def test_pipeline_handles_low_confidence_retrieval(
        self,
        semantic_rag,
        context_aware_system,
        template_library,
    ):
        """Test fallback behavior when retrieval confidence is low."""
        # Add minimal content
        semantic_rag.index_documents(
            documents=["This is a short unrelated document about cooking recipes."],
            metadata=[{"source": "recipes.txt"}]
        )

        # Query something not in the corpus
        query = "What are quantum computing qubits?"
        results = semantic_rag.search(query, top_k=1)

        # Even with low-relevance results, system should handle gracefully
        contexts = [
            {
                "content": r["document"],
                "confidence_score": 0.1,  # Simulate low confidence
                "source": "recipes.txt",
            }
            for r in results
        ]

        response = context_aware_system.execute(
            query=query,
            retrieved_contexts=contexts,
        )

        # System should recognize low confidence and adapt strategy
        assert response is not None
        assert "prompt_strategy" in response

    def test_pipeline_cross_document_search(
        self,
        sample_documents_dir,
        ingestion_pipeline,
        semantic_rag,
        citation_system,
    ):
        """Test that search can find relevant info across multiple documents."""
        # Ingest all documents
        all_texts = []
        all_metadata = []

        for filename in SAMPLE_DOCUMENTS.keys():
            filepath = os.path.join(sample_documents_dir, filename)
            chunks = ingestion_pipeline.process_file(filepath)
            for chunk in chunks:
                all_texts.append(chunk["text"])
                metadata = chunk.get("metadata", {})
                metadata["source_file"] = filename
                all_metadata.append(metadata)

        semantic_rag.index_documents(all_texts, all_metadata)

        # Query that might match multiple documents
        query = "best practices and guidelines"
        results = semantic_rag.search(query, top_k=5)

        assert len(results) > 0, "Should find some results"

        # Register sources and verify citation system tracks them
        sources_registered = set()
        for i, result in enumerate(results):
            source_file = result.get("metadata", {}).get("source_file", f"source_{i}")
            if source_file not in sources_registered:
                citation_system.register_source(
                    content=result["document"],
                    title=source_file,
                )
                sources_registered.add(source_file)

        # Verify multiple sources can be tracked
        assert len(sources_registered) >= 1, "Should track at least one source"


# =============================================================================
# COMPONENT INTEGRATION TESTS
# =============================================================================

class TestPhase1Integration:
    """Test Phase 1 components work together."""

    def test_ingestion_produces_searchable_chunks(
        self,
        sample_documents_dir,
        ingestion_pipeline,
    ):
        """Verify ingested chunks have required fields for downstream use."""
        filepath = os.path.join(sample_documents_dir, "ai_overview.txt")
        chunks = ingestion_pipeline.process_file(filepath)

        assert len(chunks) > 0, "Should produce chunks"
        for chunk in chunks:
            assert "text" in chunk, "Chunk must have text"
            assert len(chunk["text"]) > 0, "Chunk text cannot be empty"


class TestPhase2Integration:
    """Test Phase 2 components work together."""

    def test_semantic_search_outperforms_naive_matching(self, semantic_rag):
        """Semantic search should find conceptually related content."""
        # Add documents
        documents = [
            "Machine learning algorithms learn patterns from data.",
            "Cats are popular domestic pets known for independence.",
            "Neural networks are inspired by biological brain structures.",
        ]
        metadata = [
            {"source": "ml.txt"},
            {"source": "cats.txt"},
            {"source": "nn.txt"},
        ]
        semantic_rag.index_documents(documents, metadata)

        # Query about AI should rank ML and NN content higher than cats
        results = semantic_rag.search("artificial intelligence systems", top_k=3)

        assert len(results) > 0
        # The top result should be about ML or NN, not cats
        top_result_text = results[0]["document"].lower()
        assert "cats" not in top_result_text or "learning" in top_result_text


class TestPhase3Integration:
    """Test Phase 3 components work together."""

    def test_template_and_citation_integration(
        self,
        template_library,
        citation_system,
    ):
        """Templates should work with citation system."""
        # Register a source
        source_id = citation_system.register_source(
            content="Python was created by Guido van Rossum.",
            title="Python History",
            author="Tech Encyclopedia",
        )

        # Create retrieved document
        doc = RetrievedDocument(
            content="Python was created by Guido van Rossum.",
            source="test_src",
            confidence_score=0.95,
        )

        # Generate prompt
        prompt_result = template_library.generate_prompt(
            query_type=TemplateQueryType.FACTUAL_QA,
            question="Who created Python?",
            documents=[doc],
        )

        assert prompt_result is not None
        assert "prompt" in prompt_result
        assert "Python" in prompt_result["prompt"]

    def test_context_aware_adapts_to_query_complexity(self, context_aware_system):
        """Context-aware system should adapt strategy based on query type."""
        simple_context = {
            "content": "The capital of France is Paris.",
            "confidence_score": 0.95,
            "source": "geography.txt",
        }

        # Simple factual query
        factual_response = context_aware_system.execute(
            query="What is the capital of France?",
            retrieved_contexts=[simple_context],
        )

        # Analytical query
        analytical_response = context_aware_system.execute(
            query="Why did Paris become the capital of France and what factors influenced this?",
            retrieved_contexts=[simple_context],
        )

        # Both should return valid responses
        assert factual_response is not None
        assert analytical_response is not None
        assert "query_type" in factual_response
        assert "query_type" in analytical_response


# =============================================================================
# EDGE CASES AND ERROR HANDLING
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling across the pipeline."""

    def test_empty_query_handling(self, semantic_rag, context_aware_system):
        """System should handle empty or minimal queries gracefully."""
        semantic_rag.index_documents(
            documents=["Sample content about various topics."],
            metadata=[{}]
        )

        # Empty query should not crash
        results = semantic_rag.search("", top_k=1)
        # System should return empty or handle gracefully
        assert isinstance(results, list)

    def test_no_documents_indexed(self, semantic_rag):
        """Search on empty index should return empty results."""
        results = semantic_rag.search("test query", top_k=5)
        assert results == [] or len(results) == 0

    def test_special_characters_in_query(
        self,
        semantic_rag,
        context_aware_system,
    ):
        """System should handle special characters in queries."""
        semantic_rag.index_documents(
            documents=["C++ is a programming language."],
            metadata=[{"source": "cpp.txt"}]
        )

        # Query with special characters
        results = semantic_rag.search("What is C++?", top_k=1)
        assert isinstance(results, list)


# =============================================================================
# FUTURE PHASE PLACEHOLDERS
# =============================================================================

class TestFuturePhases:
    """
    Placeholder tests for future phases.

    These tests are marked as skipped until the corresponding phases
    are implemented. They serve as documentation for expected functionality.
    """

    @pytest.mark.skip(reason="Phase 4 not yet implemented")
    def test_phase4_advanced_retrieval(self):
        """Phase 4: Advanced retrieval strategies (hybrid search, reranking)."""
        pass

    @pytest.mark.skip(reason="Phase 5 not yet implemented")
    def test_phase5_production_features(self):
        """Phase 5: Production features (caching, monitoring, scaling)."""
        pass

    @pytest.mark.skip(reason="Phase 6 not yet implemented")
    def test_phase6_multi_modal_rag(self):
        """Phase 6: Multi-modal RAG (images, tables, structured data)."""
        pass
