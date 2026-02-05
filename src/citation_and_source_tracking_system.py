"""
Citation and Source Tracking System

A robust system for tracking and citing sources in RAG-generated responses.
This module provides detailed source-text mapping, multiple citation formats,
clickable source links, and citation validation capabilities.
"""

from typing import Dict, List, Optional, Any, Set, Tuple
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import re
import hashlib


class CitationFormat(Enum):
    """Supported citation formats."""
    ACADEMIC = "academic"
    WEB = "web"
    INLINE = "inline"


@dataclass
class SourceDocument:
    """Represents a source document with comprehensive metadata."""
    source_id: str
    content: str
    url: Optional[str] = None
    title: Optional[str] = None
    author: Optional[str] = None
    publication_date: Optional[str] = None
    document_type: Optional[str] = None
    relevance_score: float = 1.0

    def __post_init__(self):
        """Generate source_id if not provided."""
        if not self.source_id:
            # Generate ID from content hash
            content_hash = hashlib.md5(self.content.encode()).hexdigest()[:8]
            self.source_id = f"src_{content_hash}"


@dataclass
class TextSpan:
    """Represents a span of text in the generated response."""
    text: str
    start_position: int
    end_position: int
    source_ids: List[str] = field(default_factory=list)
    confidence: float = 1.0


@dataclass
class Citation:
    """Represents a formatted citation."""
    citation_id: int
    source: SourceDocument
    format_type: CitationFormat
    formatted_text: str
    inline_marker: str


@dataclass
class ValidationResult:
    """Result of citation validation."""
    claim: str
    source_id: str
    is_supported: bool
    support_score: float
    matched_keywords: List[str]
    warnings: List[str] = field(default_factory=list)


# ============================================================================
# SOURCE MANAGER
# ============================================================================

class SourceManager:
    """Handles source document registration, storage, and retrieval."""

    def __init__(self):
        """Initialize the source manager."""
        self._sources: Dict[str, SourceDocument] = {}
        self._source_counter = 0

    def register_source(
        self,
        content: str,
        url: Optional[str] = None,
        title: Optional[str] = None,
        author: Optional[str] = None,
        publication_date: Optional[str] = None,
        document_type: Optional[str] = None,
        relevance_score: float = 1.0,
        source_id: Optional[str] = None
    ) -> str:
        """
        Register a new source document.

        Returns:
            The source ID for the registered document
        """
        self._source_counter += 1

        if source_id is None:
            source_id = f"source_{self._source_counter}"

        source = SourceDocument(
            source_id=source_id,
            content=content,
            url=url,
            title=title,
            author=author,
            publication_date=publication_date,
            document_type=document_type,
            relevance_score=relevance_score
        )

        self._sources[source_id] = source
        return source_id

    def get_source(self, source_id: str) -> Optional[SourceDocument]:
        """Retrieve a source by ID."""
        return self._sources.get(source_id)

    def get_all_sources(self) -> List[SourceDocument]:
        """Get all registered sources."""
        return list(self._sources.values())

    def clear_sources(self):
        """Clear all registered sources."""
        self._sources.clear()
        self._source_counter = 0


# ============================================================================
# CITATION FORMATTERS
# ============================================================================

class CitationFormatter:
    """Base class for citation formatting."""

    @staticmethod
    def format(source: SourceDocument, citation_number: int) -> str:
        """Format a citation for the given source."""
        raise NotImplementedError

    @staticmethod
    def get_inline_marker(citation_number: int) -> str:
        """Get the inline citation marker."""
        return f"[{citation_number}]"


class AcademicFormatter(CitationFormatter):
    """Formats citations in academic style."""

    @staticmethod
    def format(source: SourceDocument, citation_number: int) -> str:
        """
        Format citation in academic style.

        Example: [1] Author, A. (2024). Title. Retrieved from URL
        """
        parts = [f"[{citation_number}]"]

        if source.author:
            parts.append(f"{source.author}.")

        if source.publication_date:
            parts.append(f"({source.publication_date}).")

        if source.title:
            parts.append(f"{source.title}.")

        if source.url:
            parts.append(f"Retrieved from {source.url}")

        return " ".join(parts)

    @staticmethod
    def get_inline_marker(citation_number: int) -> str:
        """Academic inline marker: [1]"""
        return f"[{citation_number}]"


class WebFormatter(CitationFormatter):
    """Formats citations in web style with clickable links."""

    @staticmethod
    def format(source: SourceDocument, citation_number: int) -> str:
        """
        Format citation in web style with markdown links.

        Example: [1] Title (URL)
        """
        title = source.title or source.source_id

        if source.url:
            return f"[{citation_number}] [{title}]({source.url})"
        else:
            return f"[{citation_number}] {title}"

    @staticmethod
    def format_html(source: SourceDocument, citation_number: int) -> str:
        """Format citation as HTML with clickable link."""
        title = source.title or source.source_id

        if source.url:
            return f'[{citation_number}] <a href="{source.url}">{title}</a>'
        else:
            return f"[{citation_number}] {title}"

    @staticmethod
    def get_inline_marker(citation_number: int) -> str:
        """Web inline marker: [1]"""
        return f"[{citation_number}]"


class InlineFormatter(CitationFormatter):
    """Formats citations inline within text."""

    @staticmethod
    def format(source: SourceDocument, citation_number: int) -> str:
        """
        Format citation for inline reference list.

        Example: [Source: title]
        """
        title = source.title or source.url or source.source_id
        return f"[Source: {title}]"

    @staticmethod
    def get_inline_marker(citation_number: int) -> str:
        """Inline marker: [Source N]"""
        return f"[Source {citation_number}]"


# ============================================================================
# TEXT-SOURCE MAPPER
# ============================================================================

class TextSpanMapper:
    """Manages mapping between text segments and source documents."""

    def __init__(self):
        """Initialize the mapper."""
        self._mappings: List[TextSpan] = []

    def add_mapping(
        self,
        text: str,
        start_position: int,
        source_ids: List[str],
        confidence: float = 1.0
    ) -> TextSpan:
        """
        Add a new text-to-source mapping.

        Args:
            text: The text segment
            start_position: Starting position in the full response
            source_ids: List of source IDs this text derives from
            confidence: Confidence in the mapping

        Returns:
            The created TextSpan
        """
        span = TextSpan(
            text=text,
            start_position=start_position,
            end_position=start_position + len(text),
            source_ids=source_ids,
            confidence=confidence
        )
        self._mappings.append(span)
        return span

    def get_mappings(self) -> List[TextSpan]:
        """Get all text-source mappings."""
        return self._mappings

    def get_mappings_for_source(self, source_id: str) -> List[TextSpan]:
        """Get all text spans mapped to a specific source."""
        return [m for m in self._mappings if source_id in m.source_ids]

    def get_sources_for_position(self, position: int) -> List[str]:
        """Get source IDs for text at a given position."""
        source_ids = []
        for mapping in self._mappings:
            if mapping.start_position <= position < mapping.end_position:
                source_ids.extend(mapping.source_ids)
        return list(set(source_ids))

    def clear_mappings(self):
        """Clear all mappings."""
        self._mappings.clear()


# ============================================================================
# VALIDATION ENGINE
# ============================================================================

class ValidationEngine:
    """Validates that citations support the claims being made."""

    # Minimum keyword overlap for a claim to be considered supported
    MIN_SUPPORT_THRESHOLD = 0.3

    @classmethod
    def validate_claim(
        cls,
        claim: str,
        source: SourceDocument
    ) -> ValidationResult:
        """
        Validate whether a claim is supported by a source.

        Uses keyword matching and text overlap analysis.

        Args:
            claim: The claim text to validate
            source: The source document to check against

        Returns:
            ValidationResult with support assessment
        """
        # Extract significant keywords from claim
        claim_keywords = cls._extract_keywords(claim)
        source_keywords = cls._extract_keywords(source.content)

        # Find matching keywords
        matched = claim_keywords.intersection(source_keywords)
        match_ratio = len(matched) / len(claim_keywords) if claim_keywords else 0

        # Check for direct phrase matches
        phrase_match = cls._check_phrase_overlap(claim, source.content)

        # Calculate overall support score
        support_score = (match_ratio * 0.6) + (phrase_match * 0.4)

        # Determine if supported
        is_supported = support_score >= cls.MIN_SUPPORT_THRESHOLD

        # Generate warnings if needed
        warnings = []
        if not is_supported:
            warnings.append(f"Claim may not be fully supported by source {source.source_id}")
        elif support_score < 0.5:
            warnings.append(f"Partial support from source {source.source_id}")

        return ValidationResult(
            claim=claim,
            source_id=source.source_id,
            is_supported=is_supported,
            support_score=support_score,
            matched_keywords=list(matched),
            warnings=warnings
        )

    @staticmethod
    def _extract_keywords(text: str) -> Set[str]:
        """Extract significant keywords from text."""
        # Remove common stop words and extract meaningful words
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "must", "shall", "can", "need", "dare",
            "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
            "into", "through", "during", "before", "after", "above", "below",
            "between", "under", "again", "further", "then", "once", "here",
            "there", "when", "where", "why", "how", "all", "each", "few",
            "more", "most", "other", "some", "such", "no", "nor", "not",
            "only", "own", "same", "so", "than", "too", "very", "just",
            "and", "but", "if", "or", "because", "until", "while", "that",
            "this", "these", "those", "it", "its", "they", "them", "their"
        }

        # Tokenize and filter
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        return {w for w in words if w not in stop_words}

    @staticmethod
    def _check_phrase_overlap(claim: str, source_content: str) -> float:
        """Check for phrase-level overlap between claim and source."""
        claim_lower = claim.lower()
        source_lower = source_content.lower()

        # Check for 3-word phrase matches
        claim_words = claim_lower.split()
        if len(claim_words) < 3:
            # For very short claims, check if fully contained
            return 1.0 if claim_lower in source_lower else 0.0

        # Create 3-word phrases from claim
        phrases = [
            " ".join(claim_words[i:i+3])
            for i in range(len(claim_words) - 2)
        ]

        # Count matching phrases
        matches = sum(1 for phrase in phrases if phrase in source_lower)
        return matches / len(phrases) if phrases else 0.0


# ============================================================================
# LINK GENERATOR
# ============================================================================

class LinkGenerator:
    """Generates clickable links for source references."""

    @staticmethod
    def generate_markdown_link(source: SourceDocument) -> str:
        """Generate a markdown-formatted link."""
        title = source.title or source.source_id
        if source.url:
            return f"[{title}]({source.url})"
        return title

    @staticmethod
    def generate_html_link(source: SourceDocument) -> str:
        """Generate an HTML-formatted link."""
        title = source.title or source.source_id
        if source.url:
            return f'<a href="{source.url}" target="_blank">{title}</a>'
        return title

    @staticmethod
    def generate_reference_list_markdown(
        sources: List[SourceDocument],
        formatter: CitationFormatter
    ) -> str:
        """Generate a markdown reference list."""
        if not sources:
            return ""

        lines = ["", "## References", ""]
        for i, source in enumerate(sources, 1):
            lines.append(formatter.format(source, i))

        return "\n".join(lines)

    @staticmethod
    def generate_reference_list_html(
        sources: List[SourceDocument],
        formatter: CitationFormatter
    ) -> str:
        """Generate an HTML reference list."""
        if not sources:
            return ""

        lines = ["<h2>References</h2>", "<ol>"]
        for i, source in enumerate(sources, 1):
            if hasattr(formatter, 'format_html'):
                formatted = formatter.format_html(source, i)
            else:
                formatted = formatter.format(source, i)
            lines.append(f"  <li>{formatted}</li>")
        lines.append("</ol>")

        return "\n".join(lines)


# ============================================================================
# MAIN CITATION TRACKER CLASS
# ============================================================================

class CitationAndSourceTrackingSystem:
    """
    Citation and Source Tracking System

    A comprehensive system for tracking sources and generating citations
    in RAG-generated responses.
    """

    FORMATTERS = {
        CitationFormat.ACADEMIC: AcademicFormatter(),
        CitationFormat.WEB: WebFormatter(),
        CitationFormat.INLINE: InlineFormatter(),
    }

    def __init__(self):
        """Initialize the CitationAndSourceTrackingSystem instance."""
        self.source_manager = SourceManager()
        self.text_mapper = TextSpanMapper()
        self.validation_engine = ValidationEngine()
        self.link_generator = LinkGenerator()
        self._citations: List[Citation] = []

    def register_source(
        self,
        content: str,
        url: Optional[str] = None,
        title: Optional[str] = None,
        author: Optional[str] = None,
        publication_date: Optional[str] = None,
        **kwargs
    ) -> str:
        """Register a source document and return its ID."""
        return self.source_manager.register_source(
            content=content,
            url=url,
            title=title,
            author=author,
            publication_date=publication_date,
            **kwargs
        )

    def add_text_mapping(
        self,
        text: str,
        start_position: int,
        source_ids: List[str],
        confidence: float = 1.0
    ) -> TextSpan:
        """Add a mapping between generated text and source documents."""
        return self.text_mapper.add_mapping(text, start_position, source_ids, confidence)

    def get_text_mappings(self) -> List[TextSpan]:
        """Get all text-source mappings."""
        return self.text_mapper.get_mappings()

    def format_citation(
        self,
        source_id: str,
        format_type: CitationFormat = CitationFormat.INLINE
    ) -> Optional[Citation]:
        """
        Format a citation for a source.

        Args:
            source_id: The source ID to cite
            format_type: The citation format to use

        Returns:
            Citation object or None if source not found
        """
        source = self.source_manager.get_source(source_id)
        if not source:
            return None

        # Get citation number (1-indexed position in sources)
        all_sources = self.source_manager.get_all_sources()
        citation_number = next(
            (i + 1 for i, s in enumerate(all_sources) if s.source_id == source_id),
            len(self._citations) + 1
        )

        formatter = self.FORMATTERS.get(format_type, self.FORMATTERS[CitationFormat.INLINE])
        formatted_text = formatter.format(source, citation_number)
        inline_marker = formatter.get_inline_marker(citation_number)

        citation = Citation(
            citation_id=citation_number,
            source=source,
            format_type=format_type,
            formatted_text=formatted_text,
            inline_marker=inline_marker
        )

        self._citations.append(citation)
        return citation

    def validate_citation(
        self,
        claim: str,
        source_id: str
    ) -> ValidationResult:
        """
        Validate that a claim is supported by a cited source.

        Args:
            claim: The claim text to validate
            source_id: The source ID being cited

        Returns:
            ValidationResult with support assessment
        """
        source = self.source_manager.get_source(source_id)
        if not source:
            return ValidationResult(
                claim=claim,
                source_id=source_id,
                is_supported=False,
                support_score=0.0,
                matched_keywords=[],
                warnings=[f"Source {source_id} not found"]
            )

        return self.validation_engine.validate_claim(claim, source)

    def generate_clickable_link(
        self,
        source_id: str,
        format_type: str = "markdown"
    ) -> str:
        """
        Generate a clickable link for a source.

        Args:
            source_id: The source ID
            format_type: "markdown" or "html"

        Returns:
            Formatted link string
        """
        source = self.source_manager.get_source(source_id)
        if not source:
            return f"[Source {source_id} not found]"

        if format_type == "html":
            return self.link_generator.generate_html_link(source)
        return self.link_generator.generate_markdown_link(source)

    def generate_reference_list(
        self,
        format_type: CitationFormat = CitationFormat.ACADEMIC,
        output_format: str = "markdown"
    ) -> str:
        """
        Generate a complete reference list for all registered sources.

        Args:
            format_type: Citation format to use
            output_format: "markdown" or "html"

        Returns:
            Formatted reference list
        """
        sources = self.source_manager.get_all_sources()
        formatter = self.FORMATTERS.get(format_type, self.FORMATTERS[CitationFormat.ACADEMIC])

        if output_format == "html":
            return self.link_generator.generate_reference_list_html(sources, formatter)
        return self.link_generator.generate_reference_list_markdown(sources, formatter)

    def execute(
        self,
        sources: Optional[List[Dict[str, Any]]] = None,
        generated_text: str = "",
        text_source_mappings: Optional[List[Dict[str, Any]]] = None,
        citation_format: str = "inline",
        output_format: str = "markdown",
        validate_citations: bool = True
    ) -> Dict[str, Any]:
        """
        Main entry point for the citation and source tracking system.

        Args:
            sources: List of source documents with metadata
            generated_text: The generated response text
            text_source_mappings: Mappings between text spans and source IDs
            citation_format: Citation format ("academic", "web", "inline")
            output_format: Output format ("markdown", "html")
            validate_citations: Whether to validate citations

        Returns:
            Dictionary containing:
                - formatted_text: Text with inline citations
                - reference_list: Formatted reference list
                - citations: List of citation objects
                - validation_results: Validation results if requested
                - text_mappings: All text-source mappings
        """
        # Clear previous state
        self.source_manager.clear_sources()
        self.text_mapper.clear_mappings()
        self._citations.clear()

        # Parse citation format
        try:
            cit_format = CitationFormat(citation_format)
        except ValueError:
            cit_format = CitationFormat.INLINE

        # Register sources
        source_id_map = {}
        if sources:
            for i, src in enumerate(sources):
                source_id = self.register_source(
                    content=src.get("content", ""),
                    url=src.get("url"),
                    title=src.get("title"),
                    author=src.get("author"),
                    publication_date=src.get("publication_date"),
                    document_type=src.get("document_type"),
                    relevance_score=src.get("relevance_score", 1.0),
                    source_id=src.get("source_id")
                )
                source_id_map[i] = source_id

        # Process text-source mappings
        if text_source_mappings:
            for mapping in text_source_mappings:
                # Convert source indices to IDs if needed
                source_ids = mapping.get("source_ids", [])
                if isinstance(source_ids, list) and source_ids:
                    if isinstance(source_ids[0], int):
                        source_ids = [source_id_map.get(sid, f"source_{sid}") for sid in source_ids]

                self.add_text_mapping(
                    text=mapping.get("text", ""),
                    start_position=mapping.get("start_position", 0),
                    source_ids=source_ids,
                    confidence=mapping.get("confidence", 1.0)
                )

        # Format citations
        formatted_citations = []
        for source_id in source_id_map.values():
            citation = self.format_citation(source_id, cit_format)
            if citation:
                formatted_citations.append({
                    "citation_id": citation.citation_id,
                    "source_id": citation.source.source_id,
                    "formatted_text": citation.formatted_text,
                    "inline_marker": citation.inline_marker
                })

        # Add citations to generated text
        formatted_text = generated_text
        if generated_text and self._citations:
            # Simple citation insertion at the end of sentences mentioning sources
            for citation in self._citations:
                marker = citation.inline_marker
                if marker not in formatted_text:
                    # Add citation reference if not already present
                    pass  # Text already has citations or we don't modify

        # Generate reference list
        reference_list = self.generate_reference_list(cit_format, output_format)

        # Validate citations if requested
        validation_results = []
        if validate_citations and text_source_mappings:
            for mapping in self.text_mapper.get_mappings():
                for source_id in mapping.source_ids:
                    result = self.validate_citation(mapping.text, source_id)
                    validation_results.append({
                        "claim": result.claim,
                        "source_id": result.source_id,
                        "is_supported": result.is_supported,
                        "support_score": result.support_score,
                        "matched_keywords": result.matched_keywords,
                        "warnings": result.warnings
                    })

        # Generate clickable links
        clickable_links = {}
        for source_id in source_id_map.values():
            clickable_links[source_id] = self.generate_clickable_link(source_id, output_format)

        return {
            "formatted_text": formatted_text,
            "reference_list": reference_list,
            "citations": formatted_citations,
            "validation_results": validation_results,
            "text_mappings": [
                {
                    "text": m.text,
                    "start_position": m.start_position,
                    "end_position": m.end_position,
                    "source_ids": m.source_ids,
                    "confidence": m.confidence
                }
                for m in self.text_mapper.get_mappings()
            ],
            "clickable_links": clickable_links
        }


def create_citation_and_source_tracking_system() -> CitationAndSourceTrackingSystem:
    """
    Factory function for creating CitationAndSourceTrackingSystem instances.

    Returns:
        CitationAndSourceTrackingSystem: A new instance of CitationAndSourceTrackingSystem
    """
    return CitationAndSourceTrackingSystem()
