"""
Phase deliverables package.

This module exports all deliverable implementations.
"""

try:
    from .document_ingestion_pipeline import DocumentIngestionPipeline
except (ImportError, SyntaxError):
    DocumentIngestionPipeline = None  # type: ignore

try:
    from .basic_rag_question_answering_system import BasicRagQuestionAnsweringSystem
except (ImportError, SyntaxError):
    BasicRagQuestionAnsweringSystem = None  # type: ignore

__all__ = ["DocumentIngestionPipeline", "BasicRagQuestionAnsweringSystem"]
