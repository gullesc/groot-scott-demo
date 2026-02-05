"""
RAG Prompt Template Library

A comprehensive library of prompt templates for different RAG use cases and query types.
This module provides structured, reusable prompt templates that optimize retrieval-augmented
generation workflows with Anthropic Claude.
"""

from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass


class QueryType(Enum):
    """Supported query types for RAG templates."""
    FACTUAL_QA = "factual_qa"
    SUMMARIZATION = "summarization"
    COMPARISON = "comparison"
    ANALYSIS = "analysis"


class DocumentDomain(Enum):
    """Supported document domain types."""
    TECHNICAL = "technical"
    LEGAL = "legal"
    ACADEMIC = "academic"
    GENERAL = "general"


@dataclass
class RetrievedDocument:
    """Represents a retrieved document chunk with metadata."""
    content: str
    source: str
    confidence_score: float = 1.0
    title: Optional[str] = None
    author: Optional[str] = None
    page: Optional[int] = None


# ============================================================================
# BASE PROMPT TEMPLATES
# ============================================================================

# Template for factual Q&A - direct, precise answers with citations
FACTUAL_QA_TEMPLATE = """You are a helpful assistant that answers questions based on the provided context.
Your task is to provide accurate, factual answers using ONLY the information from the given sources.

## Instructions
- Answer the question directly and concisely
- ALWAYS cite your sources using [Source N] format
- If the context doesn't contain enough information, say so clearly
- Do NOT make up information or use knowledge outside the provided context
- Quote directly from sources when appropriate

## Retrieved Context
{context}

## Question
{question}

## Your Answer
Provide a factual answer based solely on the context above. Include citations."""

# Template for summarization - comprehensive synthesis of documents
SUMMARIZATION_TEMPLATE = """You are a skilled summarizer tasked with creating a comprehensive summary of the provided documents.

## Instructions
- Synthesize the key information from all provided sources
- Organize the summary in a logical, coherent manner
- Maintain the original meaning while condensing the content
- Cite sources for major points using [Source N] format
- Highlight any conflicting information between sources

## Retrieved Context
{context}

## Summarization Request
{question}

## Your Summary
Provide a comprehensive summary that captures the essential information from all sources."""

# Template for comparison - analyzing similarities and differences
COMPARISON_TEMPLATE = """You are an analyst tasked with comparing and contrasting information from multiple sources.

## Instructions
- Identify key similarities between the sources
- Highlight important differences or contradictions
- Organize your comparison in a structured format
- Use specific evidence from each source with [Source N] citations
- Provide a balanced analysis without favoring any single source

## Retrieved Context
{context}

## Comparison Request
{question}

## Your Comparison
Provide a structured comparison addressing similarities, differences, and key insights."""

# Template for analysis - deep examination and interpretation
ANALYSIS_TEMPLATE = """You are an expert analyst tasked with providing deep analysis of the given information.

## Instructions
- Examine the provided context thoroughly
- Identify patterns, trends, and key insights
- Draw logical conclusions supported by evidence
- Consider multiple perspectives where applicable
- Cite all claims with [Source N] format
- Acknowledge limitations in the available information

## Retrieved Context
{context}

## Analysis Request
{question}

## Your Analysis
Provide a thorough analysis with evidence-based conclusions and citations."""

# ============================================================================
# DOMAIN-SPECIFIC TEMPLATE MODIFIERS
# ============================================================================

DOMAIN_MODIFIERS = {
    DocumentDomain.TECHNICAL: """
## Domain Context: Technical Documentation
- Use precise technical terminology
- Include code references or specifications when present
- Explain technical concepts clearly
- Note version numbers or compatibility information when available""",

    DocumentDomain.LEGAL: """
## Domain Context: Legal Documentation
- Use precise legal terminology
- Note jurisdiction-specific information
- Highlight important disclaimers or limitations
- Be especially careful about accuracy - legal advice requires precision
- Include relevant dates, case numbers, or statute references""",

    DocumentDomain.ACADEMIC: """
## Domain Context: Academic Sources
- Maintain scholarly tone and rigor
- Note methodological approaches when discussed
- Highlight statistical findings or research conclusions
- Acknowledge limitations mentioned in the research
- Use formal academic citation style""",

    DocumentDomain.GENERAL: """
## Domain Context: General Information
- Use clear, accessible language
- Provide context for specialized terms
- Focus on practical, actionable information"""
}

# ============================================================================
# FALLBACK TEMPLATES FOR LOW-CONFIDENCE SCENARIOS
# ============================================================================

LOW_CONFIDENCE_FALLBACK = """You are a helpful assistant responding to a question with limited relevant context.

## Important Notice
The retrieval system found limited relevant information for this query.
Please acknowledge this uncertainty in your response.

## Instructions
- Clearly state that the available context is limited
- Provide what information IS available with appropriate caveats
- Avoid speculation or assumptions beyond the provided context
- Suggest what additional information might be helpful
- Still cite any sources used with [Source N] format

## Retrieved Context (Limited Relevance)
{context}

## Question
{question}

## Your Response
Acknowledge the limited context and provide what assistance you can based on available information."""

NO_CONTEXT_FALLBACK = """You are a helpful assistant, but the retrieval system found no relevant documents for this query.

## Important Notice
No relevant context was retrieved for this question.

## Instructions
- Acknowledge that no relevant sources were found
- Do NOT attempt to answer from general knowledge
- Suggest how the user might refine their query
- Recommend alternative approaches to finding the information

## Question
{question}

## Your Response
Explain that no relevant context was found and provide helpful suggestions for the user."""


class ContextFormatter:
    """Handles formatting of retrieved documents into context strings."""

    @staticmethod
    def format_documents(documents: List[RetrievedDocument]) -> str:
        """
        Format a list of retrieved documents into a numbered context string.

        Args:
            documents: List of RetrievedDocument objects

        Returns:
            Formatted context string with source attribution
        """
        if not documents:
            return "[No documents retrieved]"

        formatted_parts = []
        for i, doc in enumerate(documents, 1):
            # Build source attribution
            source_info = f"Source: {doc.source}"
            if doc.title:
                source_info = f"{doc.title} | {source_info}"
            if doc.author:
                source_info += f" | Author: {doc.author}"
            if doc.page is not None:
                source_info += f" | Page: {doc.page}"

            # Format the document chunk
            formatted_parts.append(
                f"[Source {i}]\n"
                f"{source_info}\n"
                f"Confidence: {doc.confidence_score:.2f}\n"
                f"---\n"
                f"{doc.content}\n"
            )

        return "\n".join(formatted_parts)

    @staticmethod
    def filter_by_confidence(
        documents: List[RetrievedDocument],
        min_confidence: float = 0.0
    ) -> List[RetrievedDocument]:
        """Filter documents by minimum confidence score."""
        return [doc for doc in documents if doc.confidence_score >= min_confidence]

    @staticmethod
    def sort_by_confidence(
        documents: List[RetrievedDocument],
        descending: bool = True
    ) -> List[RetrievedDocument]:
        """Sort documents by confidence score."""
        return sorted(documents, key=lambda d: d.confidence_score, reverse=descending)


class TemplateManager:
    """Manages template selection and customization."""

    TEMPLATES = {
        QueryType.FACTUAL_QA: FACTUAL_QA_TEMPLATE,
        QueryType.SUMMARIZATION: SUMMARIZATION_TEMPLATE,
        QueryType.COMPARISON: COMPARISON_TEMPLATE,
        QueryType.ANALYSIS: ANALYSIS_TEMPLATE,
    }

    @classmethod
    def get_template(cls, query_type: QueryType) -> str:
        """Get the base template for a query type."""
        return cls.TEMPLATES.get(query_type, FACTUAL_QA_TEMPLATE)

    @classmethod
    def apply_domain_modifier(cls, template: str, domain: DocumentDomain) -> str:
        """Apply domain-specific modifications to a template."""
        modifier = DOMAIN_MODIFIERS.get(domain, DOMAIN_MODIFIERS[DocumentDomain.GENERAL])
        # Insert domain modifier after the first section
        parts = template.split("## Instructions", 1)
        if len(parts) == 2:
            return parts[0] + modifier + "\n\n## Instructions" + parts[1]
        return template + modifier


class FallbackHandler:
    """Handles low-confidence and edge case scenarios."""

    # Confidence thresholds
    HIGH_CONFIDENCE_THRESHOLD = 0.7
    LOW_CONFIDENCE_THRESHOLD = 0.4

    @classmethod
    def assess_confidence(
        cls,
        documents: List[RetrievedDocument]
    ) -> str:
        """
        Assess overall retrieval confidence.

        Returns: "high", "medium", or "low"
        """
        if not documents:
            return "low"

        avg_confidence = sum(d.confidence_score for d in documents) / len(documents)
        max_confidence = max(d.confidence_score for d in documents)

        # Consider both average and max confidence
        if avg_confidence >= cls.HIGH_CONFIDENCE_THRESHOLD or max_confidence >= 0.9:
            return "high"
        elif avg_confidence >= cls.LOW_CONFIDENCE_THRESHOLD:
            return "medium"
        else:
            return "low"

    @classmethod
    def get_fallback_template(
        cls,
        documents: List[RetrievedDocument]
    ) -> Optional[str]:
        """
        Get appropriate fallback template if confidence is low.

        Returns: Fallback template string or None if confidence is sufficient
        """
        if not documents:
            return NO_CONTEXT_FALLBACK

        confidence = cls.assess_confidence(documents)
        if confidence == "low":
            return LOW_CONFIDENCE_FALLBACK

        return None


class CitationManager:
    """Manages citation formatting across templates."""

    @staticmethod
    def format_citation_instructions(domain: DocumentDomain) -> str:
        """Generate citation instructions based on domain."""
        if domain == DocumentDomain.ACADEMIC:
            return """

## Citation Format
Use academic-style citations:
- In-text: [Source N, Author Year] where available
- Include page numbers when referencing specific passages
- List all sources at the end of your response"""
        elif domain == DocumentDomain.LEGAL:
            return """

## Citation Format
Use legal citation style:
- Reference specific sections, clauses, or paragraphs
- Include document dates and version numbers
- Use [Source N, Section/Clause] format"""
        else:
            return """

## Citation Format
- Use [Source N] for inline citations
- Reference the specific source number from the context
- Include multiple citations when information comes from multiple sources"""


class RagPromptTemplateLibrary:
    """
    RAG Prompt Template Library

    A comprehensive library for generating sophisticated prompts that optimize
    retrieval-augmented generation workflows with Anthropic Claude.
    """

    def __init__(self):
        """Initialize the RagPromptTemplateLibrary instance."""
        self.context_formatter = ContextFormatter()
        self.template_manager = TemplateManager()
        self.fallback_handler = FallbackHandler()
        self.citation_manager = CitationManager()

    def generate_prompt(
        self,
        query_type: QueryType,
        question: str,
        documents: List[RetrievedDocument],
        domain: DocumentDomain = DocumentDomain.GENERAL,
        use_fallback: bool = True
    ) -> Dict[str, Any]:
        """
        Generate a complete prompt for the given query and documents.

        Args:
            query_type: The type of query (factual_qa, summarization, etc.)
            question: The user's question or request
            documents: List of retrieved document chunks
            domain: The domain type of the documents
            use_fallback: Whether to use fallback templates for low confidence

        Returns:
            Dictionary containing:
                - prompt: The complete prompt string
                - confidence_level: Assessment of retrieval confidence
                - used_fallback: Whether a fallback template was used
                - document_count: Number of documents included
        """
        # Assess confidence and potentially use fallback
        confidence_level = self.fallback_handler.assess_confidence(documents)
        fallback_template = None

        if use_fallback:
            fallback_template = self.fallback_handler.get_fallback_template(documents)

        # Sort documents by confidence
        sorted_docs = self.context_formatter.sort_by_confidence(documents)

        # Format context
        context = self.context_formatter.format_documents(sorted_docs)

        # Select and customize template
        if fallback_template:
            template = fallback_template
            used_fallback = True
        else:
            template = self.template_manager.get_template(query_type)
            template = self.template_manager.apply_domain_modifier(template, domain)
            used_fallback = False

        # Build the final prompt
        prompt = template.format(context=context, question=question)

        # Add citation instructions for non-fallback prompts
        if not used_fallback:
            citation_instructions = self.citation_manager.format_citation_instructions(domain)
            prompt = prompt + citation_instructions

        return {
            "prompt": prompt,
            "confidence_level": confidence_level,
            "used_fallback": used_fallback,
            "document_count": len(documents)
        }

    def get_template_for_type(self, query_type: QueryType) -> str:
        """Get the raw template for a specific query type."""
        return self.template_manager.get_template(query_type)

    def get_domain_modifier(self, domain: DocumentDomain) -> str:
        """Get the domain modifier for a specific document domain."""
        return DOMAIN_MODIFIERS.get(domain, DOMAIN_MODIFIERS[DocumentDomain.GENERAL])

    def get_fallback_template(self, has_documents: bool = True) -> str:
        """Get the appropriate fallback template."""
        if has_documents:
            return LOW_CONFIDENCE_FALLBACK
        return NO_CONTEXT_FALLBACK

    def execute(
        self,
        query_type: str = "factual_qa",
        question: str = "",
        documents: Optional[List[Dict[str, Any]]] = None,
        domain: str = "general",
        confidence_scores: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Main entry point for generating RAG prompts.

        Args:
            query_type: One of "factual_qa", "summarization", "comparison", "analysis"
            question: The user's question or request
            documents: List of document dicts with 'content', 'source', and optional metadata
            domain: One of "technical", "legal", "academic", "general"
            confidence_scores: Optional list of confidence scores for each document

        Returns:
            Dictionary with the generated prompt and metadata
        """
        # Convert string query type to enum
        try:
            qt = QueryType(query_type)
        except ValueError:
            qt = QueryType.FACTUAL_QA

        # Convert string domain to enum
        try:
            dm = DocumentDomain(domain)
        except ValueError:
            dm = DocumentDomain.GENERAL

        # Convert document dicts to RetrievedDocument objects
        doc_list = []
        if documents:
            for i, doc in enumerate(documents):
                confidence = 1.0
                if confidence_scores and i < len(confidence_scores):
                    confidence = confidence_scores[i]
                elif "confidence_score" in doc:
                    confidence = doc["confidence_score"]

                doc_list.append(RetrievedDocument(
                    content=doc.get("content", ""),
                    source=doc.get("source", f"source_{i+1}"),
                    confidence_score=confidence,
                    title=doc.get("title"),
                    author=doc.get("author"),
                    page=doc.get("page")
                ))

        return self.generate_prompt(qt, question, doc_list, dm)


def create_rag_prompt_template_library() -> RagPromptTemplateLibrary:
    """
    Factory function for creating RagPromptTemplateLibrary instances.

    Returns:
        RagPromptTemplateLibrary: A new instance of RagPromptTemplateLibrary
    """
    return RagPromptTemplateLibrary()
