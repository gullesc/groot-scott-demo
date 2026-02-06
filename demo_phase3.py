#!/usr/bin/env python3
"""
RAG System Demo - Phase 3

This demo shows the complete RAG pipeline in action.
Run with: python3 demo_phase3.py
"""

import tempfile
import os
from pathlib import Path

# Import all components
from src.document_ingestion_pipeline import create_document_ingestion_pipeline
from src.semantic_rag_system import create_semantic_rag_system
from src.rag_prompt_template_library import (
    create_rag_prompt_template_library,
    QueryType,
    DocumentDomain,
    RetrievedDocument,
)
from src.citation_and_source_tracking_system import (
    create_citation_and_source_tracking_system,
    CitationFormat,
)
from src.context_aware_rag_system import create_context_aware_rag_system


def print_header(text: str):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_step(step_num: int, text: str):
    """Print a step indicator."""
    print(f"\n[Step {step_num}] {text}")
    print("-" * 50)


def main():
    print_header("RAG System Demo - Phases 1-3")
    print("\nThis demo walks through the complete RAG pipeline:")
    print("  Phase 1: Document ingestion and chunking")
    print("  Phase 2: Semantic search with embeddings")
    print("  Phase 3: Prompt templates, citations, context-aware responses")

    # =========================================================================
    # Setup: Create sample documents
    # =========================================================================
    print_step(0, "Creating sample documents...")

    sample_docs = {
        "ai_basics.txt": """
Artificial Intelligence (AI) is the simulation of human intelligence in machines.
Machine Learning is a subset of AI where systems learn from data without explicit programming.
Deep Learning uses neural networks with multiple layers to model complex patterns.
Key applications include natural language processing, computer vision, and robotics.
The field was founded at the Dartmouth Conference in 1956.
        """,
        "python_intro.md": """
# Python Programming

Python is a high-level programming language created by Guido van Rossum in 1991.
It emphasizes code readability with significant whitespace.
Python supports multiple paradigms: procedural, object-oriented, and functional.
Popular for web development, data science, AI/ML, and automation.
        """,
    }

    # Create temp directory with documents
    tmpdir = tempfile.mkdtemp()
    for filename, content in sample_docs.items():
        Path(tmpdir, filename).write_text(content.strip())
    print(f"Created {len(sample_docs)} sample documents in temp directory")

    # =========================================================================
    # PHASE 1: Document Ingestion
    # =========================================================================
    print_header("PHASE 1: Document Ingestion")

    print_step(1, "Initializing ingestion pipeline...")
    pipeline = create_document_ingestion_pipeline()

    print_step(2, "Processing documents into chunks...")
    all_texts = []
    all_metadata = []

    for filename in sample_docs.keys():
        filepath = os.path.join(tmpdir, filename)
        chunks = pipeline.process_file(filepath)

        print(f"\n  {filename}:")
        print(f"    - Produced {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            all_texts.append(chunk["text"])
            metadata = chunk.get("metadata", {})
            metadata["source"] = filename
            all_metadata.append(metadata)

            # Show first chunk preview
            if i == 0:
                preview = chunk["text"][:80].replace("\n", " ")
                print(f"    - First chunk: \"{preview}...\"")

    print(f"\n  Total chunks ready for indexing: {len(all_texts)}")

    # =========================================================================
    # PHASE 2: Semantic Search
    # =========================================================================
    print_header("PHASE 2: Semantic Search")

    print_step(3, "Indexing documents with TF-IDF embeddings...")
    semantic_rag = create_semantic_rag_system()
    indexed = semantic_rag.index_documents(all_texts, all_metadata)
    print(f"  Indexed {indexed} document chunks")

    print_step(4, "Performing semantic search...")
    query = "What is machine learning and how does it relate to AI?"
    print(f"\n  Query: \"{query}\"")

    results = semantic_rag.search(query, top_k=3)
    print(f"\n  Found {len(results)} relevant results:")

    for i, result in enumerate(results):
        score = result["score"]
        source = result["metadata"].get("source", "unknown")
        preview = result["document"][:60].replace("\n", " ")
        print(f"\n  [{i+1}] Score: {score:.3f} | Source: {source}")
        print(f"      \"{preview}...\"")

    # =========================================================================
    # PHASE 3: Intelligent Response Generation
    # =========================================================================
    print_header("PHASE 3: Intelligent Response Generation")

    # Step 5: Citation tracking
    print_step(5, "Registering sources for citation...")
    citation_system = create_citation_and_source_tracking_system()

    for i, result in enumerate(results):
        source_id = citation_system.register_source(
            content=result["document"],
            title=result["metadata"].get("source", f"Source {i+1}"),
        )
        print(f"  Registered: {result['metadata'].get('source', f'Source {i+1}')}")

    # Step 6: Prompt template generation
    print_step(6, "Generating prompt with template library...")
    template_library = create_rag_prompt_template_library()

    retrieved_docs = [
        RetrievedDocument(
            content=r["document"],
            source=r["metadata"].get("source", f"source_{i}"),
            confidence_score=r["score"],
        )
        for i, r in enumerate(results)
    ]

    prompt_result = template_library.generate_prompt(
        query_type=QueryType.FACTUAL_QA,
        question=query,
        documents=retrieved_docs,
        domain=DocumentDomain.TECHNICAL,
    )

    print(f"\n  Query Type: FACTUAL_QA")
    print(f"  Domain: TECHNICAL")
    print(f"  Confidence Level: {prompt_result.get('confidence_level', 'N/A')}")
    print(f"  Used Fallback: {prompt_result.get('used_fallback', False)}")
    print(f"\n  Generated Prompt Preview:")
    prompt_preview = prompt_result["prompt"][:200].replace("\n", "\n    ")
    print(f"    {prompt_preview}...")

    # Step 7: Context-aware response
    print_step(7, "Generating context-aware response...")
    context_aware = create_context_aware_rag_system()

    contexts = [
        {
            "content": r["document"],
            "confidence_score": r["score"],
            "source": r["metadata"].get("source", "unknown"),
        }
        for r in results
    ]

    response = context_aware.execute(
        query=query,
        retrieved_contexts=contexts,
    )

    print(f"\n  Query Classification: {response['query_type']}")
    print(f"  Prompt Strategy: {response['prompt_strategy']}")
    print(f"  Confidence Level: {response['confidence_level']}")
    print(f"\n  Response Preview:")
    response_preview = response["response"][:300].replace("\n", "\n    ")
    print(f"    {response_preview}...")

    # Step 8: Generate citations
    print_step(8, "Generating reference list...")
    references = citation_system.generate_reference_list(
        format_type=CitationFormat.INLINE
    )
    print(f"\n  References:\n{references}")

    # =========================================================================
    # Summary
    # =========================================================================
    print_header("Demo Complete!")
    print("""
What we demonstrated:

  Phase 1 - Document Ingestion:
    - Loaded multiple document formats (TXT, MD)
    - Split into searchable chunks with metadata

  Phase 2 - Semantic Search:
    - Created TF-IDF embeddings for all chunks
    - Found relevant content using semantic similarity
    - Ranked results by relevance score

  Phase 3 - Intelligent Response:
    - Classified query type (factual vs analytical)
    - Selected appropriate prompt template
    - Tracked and formatted source citations
    - Adapted response strategy based on confidence

NOTE: This system is incomplete. Future phases will add:
  - Phase 4: Hybrid search, reranking
  - Phase 5: Caching, monitoring, production features
  - Phase 6: Multi-modal RAG (images, tables)
    """)

    # Cleanup
    import shutil
    shutil.rmtree(tmpdir)


if __name__ == "__main__":
    main()
