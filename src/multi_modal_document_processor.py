"""
Multi-Modal Document Processor

Extend your pipeline to extract and process images, tables, and structured content from documents.
This module provides simulated OCR, vision capabilities, and multi-modal content processing
using only the Python standard library for educational purposes.
"""

import json
import csv
import re
import uuid
import io
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum


class ContentType(Enum):
    """Types of content that can be processed."""
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    JSON = "json"
    CSV = "csv"
    UNKNOWN = "unknown"


class RelationshipType(Enum):
    """Types of relationships between content elements."""
    CONTAINS = "contains"          # Parent contains child
    REFERENCES = "references"      # One element references another
    FOLLOWS = "follows"            # Sequential relationship
    DESCRIBES = "describes"        # One element describes another (e.g., caption)
    RELATED_TO = "related_to"      # General relationship


@dataclass
class ProcessedContent:
    """Represents a processed content element with metadata."""
    content_id: str
    content_type: ContentType
    original_data: Any
    extracted_text: str
    searchable_text: str
    metadata: Dict[str, Any]
    position: Optional[int] = None
    confidence_score: float = 1.0
    processing_notes: List[str] = field(default_factory=list)


@dataclass
class ContentRelationship:
    """Represents a relationship between two content elements."""
    source_id: str
    target_id: str
    relationship_type: RelationshipType
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessingResult:
    """Complete result of multi-modal document processing."""
    document_id: str
    processed_contents: List[ProcessedContent]
    relationships: List[ContentRelationship]
    errors: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    processing_time: float = 0.0


# ============================================================================
# IMAGE PROCESSING (Simulated OCR and Vision)
# ============================================================================

class ImageProcessor:
    """
    Simulates image processing with OCR and vision capabilities.

    In a production system, this would integrate with actual OCR services
    (like Tesseract) or vision APIs (like Claude's vision capabilities).
    For educational purposes, we simulate these capabilities.
    """

    # Simulated patterns for different image types
    IMAGE_TYPE_PATTERNS = {
        "chart": ["chart", "graph", "plot", "visualization", "data"],
        "diagram": ["diagram", "flow", "architecture", "schema", "uml"],
        "screenshot": ["screenshot", "screen", "ui", "interface", "app"],
        "photo": ["photo", "image", "picture", "photograph"],
        "document": ["document", "scan", "page", "text", "pdf"]
    }

    @classmethod
    def process_image(
        cls,
        image_data: str,
        context: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ProcessedContent:
        """
        Process an image and extract text/descriptions.

        Args:
            image_data: Base64-encoded image data or data URL
            context: Surrounding text context (e.g., caption)
            metadata: Additional metadata about the image

        Returns:
            ProcessedContent with extracted information
        """
        content_id = str(uuid.uuid4())
        processing_notes = []

        # Parse image data
        image_info = cls._parse_image_data(image_data)
        processing_notes.append(f"Detected format: {image_info.get('format', 'unknown')}")

        # Simulate OCR text extraction
        ocr_text = cls._simulate_ocr(image_data, context)
        if ocr_text:
            processing_notes.append("OCR text extracted")

        # Simulate vision analysis
        vision_description = cls._simulate_vision_analysis(image_data, context)
        processing_notes.append("Vision analysis completed")

        # Determine image type from context
        image_type = cls._detect_image_type(context or "")

        # Build extracted text
        extracted_parts = []
        if vision_description:
            extracted_parts.append(f"[Image Description: {vision_description}]")
        if ocr_text:
            extracted_parts.append(f"[OCR Text: {ocr_text}]")
        if context:
            extracted_parts.append(f"[Context: {context}]")

        extracted_text = " ".join(extracted_parts) if extracted_parts else "[No text extracted]"

        # Build searchable text (normalized for search)
        searchable_text = cls._build_searchable_text(ocr_text, vision_description, context)

        # Build metadata
        content_metadata = {
            "image_format": image_info.get("format", "unknown"),
            "image_type": image_type,
            "has_ocr_text": bool(ocr_text),
            "has_context": bool(context),
            "data_size_bytes": len(image_data),
            **(metadata or {})
        }

        return ProcessedContent(
            content_id=content_id,
            content_type=ContentType.IMAGE,
            original_data=image_data[:100] + "..." if len(image_data) > 100 else image_data,
            extracted_text=extracted_text,
            searchable_text=searchable_text,
            metadata=content_metadata,
            confidence_score=0.7 if ocr_text else 0.5,
            processing_notes=processing_notes
        )

    @classmethod
    def _parse_image_data(cls, image_data: str) -> Dict[str, Any]:
        """Parse image data URL or base64 to extract format information."""
        if image_data.startswith("data:"):
            # Parse data URL format: data:image/jpeg;base64,/9j/4AAQ...
            match = re.match(r"data:image/(\w+);base64,", image_data)
            if match:
                return {"format": match.group(1), "encoding": "base64"}
        return {"format": "unknown", "encoding": "unknown"}

    @classmethod
    def _simulate_ocr(cls, image_data: str, context: Optional[str]) -> str:
        """
        Simulate OCR text extraction from an image.

        In production, this would use Tesseract or cloud OCR services.
        """
        # Simulate OCR by analyzing context hints
        if context:
            # Extract any quoted text from context as "OCR results"
            quotes = re.findall(r'"([^"]+)"', context)
            if quotes:
                return " ".join(quotes)

            # Check for numeric data hints
            numbers = re.findall(r'\b\d+(?:\.\d+)?%?\b', context)
            if numbers and len(numbers) >= 2:
                return f"Data values: {', '.join(numbers)}"

        return ""

    @classmethod
    def _simulate_vision_analysis(cls, image_data: str, context: Optional[str]) -> str:
        """
        Simulate vision model analysis of an image.

        In production, this would use Claude's vision capabilities.
        """
        image_type = cls._detect_image_type(context or "")

        descriptions = {
            "chart": "A data visualization showing quantitative information with labeled axes and data points",
            "diagram": "A diagram illustrating relationships and connections between concepts or components",
            "screenshot": "A screenshot capturing a user interface or application view",
            "photo": "A photograph depicting a scene or subject",
            "document": "A scanned document containing text and possibly structured content"
        }

        base_description = descriptions.get(image_type, "An image containing visual content")

        # Enhance with context if available
        if context:
            return f"{base_description}. Context suggests: {context[:100]}"

        return base_description

    @classmethod
    def _detect_image_type(cls, context: str) -> str:
        """Detect image type based on context keywords."""
        context_lower = context.lower()

        for img_type, keywords in cls.IMAGE_TYPE_PATTERNS.items():
            if any(keyword in context_lower for keyword in keywords):
                return img_type

        return "unknown"

    @classmethod
    def _build_searchable_text(
        cls,
        ocr_text: str,
        vision_description: str,
        context: Optional[str]
    ) -> str:
        """Build normalized searchable text from image processing results."""
        parts = []

        if ocr_text:
            parts.append(ocr_text.lower())
        if vision_description:
            parts.append(vision_description.lower())
        if context:
            parts.append(context.lower())

        text = " ".join(parts)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text


# ============================================================================
# TABLE PROCESSING
# ============================================================================

class TableProcessor:
    """
    Processes HTML tables and converts them to searchable formats.

    Handles various table structures including irregular tables,
    merged cells, and tables with headers.
    """

    @classmethod
    def process_table(
        cls,
        table_data: str,
        context: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ProcessedContent:
        """
        Process a table (HTML or plain text) and convert to searchable format.

        Args:
            table_data: HTML table string or plain text table
            context: Surrounding context about the table
            metadata: Additional metadata

        Returns:
            ProcessedContent with table data in searchable format
        """
        content_id = str(uuid.uuid4())
        processing_notes = []

        # Detect table format and parse
        if "<table" in table_data.lower() or "<tr" in table_data.lower():
            parsed_table = cls._parse_html_table(table_data)
            processing_notes.append("Parsed HTML table format")
        else:
            parsed_table = cls._parse_text_table(table_data)
            processing_notes.append("Parsed plain text table format")

        # Extract headers and rows
        headers = parsed_table.get("headers", [])
        rows = parsed_table.get("rows", [])

        processing_notes.append(f"Found {len(headers)} columns and {len(rows)} data rows")

        # Build extracted text representation
        extracted_text = cls._build_table_text(headers, rows)

        # Build searchable text (flattened for search)
        searchable_text = cls._build_searchable_table_text(headers, rows, context)

        # Build metadata
        content_metadata = {
            "column_count": len(headers),
            "row_count": len(rows),
            "headers": headers,
            "has_context": bool(context),
            "table_format": "html" if "<table" in table_data.lower() else "text",
            **(metadata or {})
        }

        return ProcessedContent(
            content_id=content_id,
            content_type=ContentType.TABLE,
            original_data=parsed_table,
            extracted_text=extracted_text,
            searchable_text=searchable_text,
            metadata=content_metadata,
            confidence_score=0.9 if headers else 0.7,
            processing_notes=processing_notes
        )

    @classmethod
    def _parse_html_table(cls, html: str) -> Dict[str, Any]:
        """Parse an HTML table into structured data."""
        headers = []
        rows = []

        # Extract headers from <th> tags
        th_pattern = re.compile(r'<th[^>]*>(.*?)</th>', re.IGNORECASE | re.DOTALL)
        header_matches = th_pattern.findall(html)
        headers = [cls._clean_html_text(h) for h in header_matches]

        # Extract rows from <tr> tags
        tr_pattern = re.compile(r'<tr[^>]*>(.*?)</tr>', re.IGNORECASE | re.DOTALL)
        td_pattern = re.compile(r'<td[^>]*>(.*?)</td>', re.IGNORECASE | re.DOTALL)

        for tr_match in tr_pattern.findall(html):
            cells = td_pattern.findall(tr_match)
            if cells:
                row = [cls._clean_html_text(cell) for cell in cells]
                rows.append(row)

        # If no headers found, use first row as headers
        if not headers and rows:
            headers = rows[0]
            rows = rows[1:]

        return {"headers": headers, "rows": rows}

    @classmethod
    def _parse_text_table(cls, text: str) -> Dict[str, Any]:
        """Parse a plain text table (pipe-delimited or space-delimited)."""
        lines = [line.strip() for line in text.strip().split('\n') if line.strip()]

        if not lines:
            return {"headers": [], "rows": []}

        # Detect delimiter
        if '|' in lines[0]:
            delimiter = '|'
        elif '\t' in lines[0]:
            delimiter = '\t'
        else:
            delimiter = None  # Space-delimited

        headers = []
        rows = []

        for i, line in enumerate(lines):
            # Skip separator lines (like |---|---|)
            if re.match(r'^[\s\-|+]+$', line):
                continue

            if delimiter:
                cells = [cell.strip() for cell in line.split(delimiter) if cell.strip()]
            else:
                cells = line.split()

            if i == 0 and not headers:
                headers = cells
            else:
                rows.append(cells)

        return {"headers": headers, "rows": rows}

    @classmethod
    def _clean_html_text(cls, html: str) -> str:
        """Remove HTML tags and clean text."""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', html)
        # Decode common HTML entities
        text = text.replace('&nbsp;', ' ')
        text = text.replace('&amp;', '&')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        text = text.replace('&quot;', '"')
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    @classmethod
    def _build_table_text(cls, headers: List[str], rows: List[List[str]]) -> str:
        """Build a readable text representation of a table."""
        lines = []

        if headers:
            lines.append("| " + " | ".join(headers) + " |")
            lines.append("|" + "|".join(["---"] * len(headers)) + "|")

        for row in rows:
            lines.append("| " + " | ".join(row) + " |")

        return "\n".join(lines)

    @classmethod
    def _build_searchable_table_text(
        cls,
        headers: List[str],
        rows: List[List[str]],
        context: Optional[str]
    ) -> str:
        """Build flattened searchable text from table data."""
        parts = []

        # Add context
        if context:
            parts.append(context.lower())

        # Add headers
        parts.extend([h.lower() for h in headers])

        # Add all cell values
        for row in rows:
            parts.extend([cell.lower() for cell in row])

        return " ".join(parts)


# ============================================================================
# STRUCTURED DATA PROCESSING (JSON, CSV)
# ============================================================================

class StructuredDataProcessor:
    """
    Processes structured data formats like JSON and CSV.

    Extracts searchable text while preserving structure and metadata.
    """

    @classmethod
    def process_json(
        cls,
        json_data: str,
        context: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ProcessedContent:
        """
        Process JSON data and extract searchable content.

        Args:
            json_data: JSON string to process
            context: Surrounding context
            metadata: Additional metadata

        Returns:
            ProcessedContent with indexed JSON data
        """
        content_id = str(uuid.uuid4())
        processing_notes = []

        # Parse JSON
        try:
            parsed_data = json.loads(json_data)
            processing_notes.append("Successfully parsed JSON")
        except json.JSONDecodeError as e:
            processing_notes.append(f"JSON parse error: {str(e)}")
            parsed_data = {"_error": str(e), "_raw": json_data[:500]}

        # Extract key-value pairs for searching
        key_values = cls._extract_key_values(parsed_data)
        processing_notes.append(f"Extracted {len(key_values)} key-value pairs")

        # Build extracted text
        extracted_text = cls._build_json_text(parsed_data)

        # Build searchable text
        searchable_text = cls._build_searchable_json_text(key_values, context)

        # Detect JSON structure type
        structure_type = cls._detect_json_structure(parsed_data)

        # Build metadata
        content_metadata = {
            "structure_type": structure_type,
            "key_count": len(key_values),
            "has_nested": cls._has_nested_objects(parsed_data),
            "has_arrays": cls._has_arrays(parsed_data),
            "has_context": bool(context),
            **(metadata or {})
        }

        return ProcessedContent(
            content_id=content_id,
            content_type=ContentType.JSON,
            original_data=parsed_data,
            extracted_text=extracted_text,
            searchable_text=searchable_text,
            metadata=content_metadata,
            confidence_score=0.95 if "_error" not in parsed_data else 0.3,
            processing_notes=processing_notes
        )

    @classmethod
    def process_csv(
        cls,
        csv_data: str,
        context: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ProcessedContent:
        """
        Process CSV data and extract searchable content.

        Args:
            csv_data: CSV string to process
            context: Surrounding context
            metadata: Additional metadata

        Returns:
            ProcessedContent with indexed CSV data
        """
        content_id = str(uuid.uuid4())
        processing_notes = []

        # Parse CSV
        try:
            reader = csv.reader(io.StringIO(csv_data))
            rows = list(reader)
            processing_notes.append("Successfully parsed CSV")
        except csv.Error as e:
            processing_notes.append(f"CSV parse error: {str(e)}")
            rows = []

        # Extract headers and data
        headers = rows[0] if rows else []
        data_rows = rows[1:] if len(rows) > 1 else []

        processing_notes.append(f"Found {len(headers)} columns and {len(data_rows)} data rows")

        # Build extracted text
        extracted_text = cls._build_csv_text(headers, data_rows)

        # Build searchable text
        searchable_text = cls._build_searchable_csv_text(headers, data_rows, context)

        # Build metadata
        content_metadata = {
            "column_count": len(headers),
            "row_count": len(data_rows),
            "headers": headers,
            "has_context": bool(context),
            **(metadata or {})
        }

        return ProcessedContent(
            content_id=content_id,
            content_type=ContentType.CSV,
            original_data={"headers": headers, "rows": data_rows},
            extracted_text=extracted_text,
            searchable_text=searchable_text,
            metadata=content_metadata,
            confidence_score=0.9 if headers else 0.6,
            processing_notes=processing_notes
        )

    @classmethod
    def _extract_key_values(
        cls,
        data: Any,
        prefix: str = ""
    ) -> List[Tuple[str, str]]:
        """Recursively extract key-value pairs from nested data."""
        pairs = []

        if isinstance(data, dict):
            for key, value in data.items():
                full_key = f"{prefix}.{key}" if prefix else key
                if isinstance(value, (dict, list)):
                    pairs.extend(cls._extract_key_values(value, full_key))
                else:
                    pairs.append((full_key, str(value)))
        elif isinstance(data, list):
            for i, item in enumerate(data):
                full_key = f"{prefix}[{i}]"
                if isinstance(item, (dict, list)):
                    pairs.extend(cls._extract_key_values(item, full_key))
                else:
                    pairs.append((full_key, str(item)))
        else:
            pairs.append((prefix, str(data)))

        return pairs

    @classmethod
    def _build_json_text(cls, data: Any) -> str:
        """Build a formatted text representation of JSON data."""
        return json.dumps(data, indent=2, default=str)

    @classmethod
    def _build_searchable_json_text(
        cls,
        key_values: List[Tuple[str, str]],
        context: Optional[str]
    ) -> str:
        """Build searchable text from extracted key-value pairs."""
        parts = []

        if context:
            parts.append(context.lower())

        for key, value in key_values:
            parts.append(key.lower())
            parts.append(str(value).lower())

        return " ".join(parts)

    @classmethod
    def _detect_json_structure(cls, data: Any) -> str:
        """Detect the type of JSON structure."""
        if isinstance(data, dict):
            if all(isinstance(v, (str, int, float, bool, type(None))) for v in data.values()):
                return "flat_object"
            return "nested_object"
        elif isinstance(data, list):
            if all(isinstance(item, dict) for item in data):
                return "array_of_objects"
            return "array"
        return "primitive"

    @classmethod
    def _has_nested_objects(cls, data: Any) -> bool:
        """Check if data contains nested objects."""
        if isinstance(data, dict):
            return any(isinstance(v, (dict, list)) for v in data.values())
        elif isinstance(data, list):
            return any(isinstance(item, (dict, list)) for item in data)
        return False

    @classmethod
    def _has_arrays(cls, data: Any) -> bool:
        """Check if data contains arrays."""
        if isinstance(data, list):
            return True
        if isinstance(data, dict):
            return any(isinstance(v, list) for v in data.values())
        return False

    @classmethod
    def _build_csv_text(cls, headers: List[str], rows: List[List[str]]) -> str:
        """Build a formatted text representation of CSV data."""
        lines = []
        if headers:
            lines.append(",".join(headers))
        for row in rows[:10]:  # Limit to first 10 rows for text representation
            lines.append(",".join(row))
        if len(rows) > 10:
            lines.append(f"... and {len(rows) - 10} more rows")
        return "\n".join(lines)

    @classmethod
    def _build_searchable_csv_text(
        cls,
        headers: List[str],
        rows: List[List[str]],
        context: Optional[str]
    ) -> str:
        """Build searchable text from CSV data."""
        parts = []

        if context:
            parts.append(context.lower())

        parts.extend([h.lower() for h in headers])

        for row in rows:
            parts.extend([cell.lower() for cell in row])

        return " ".join(parts)


# ============================================================================
# RELATIONSHIP MAPPING
# ============================================================================

class RelationshipMapper:
    """
    Maintains and tracks relationships between content elements.

    Creates a graph-like structure to represent how different content
    types within a document connect and influence each other.
    """

    def __init__(self):
        """Initialize the relationship mapper."""
        self._relationships: List[ContentRelationship] = []
        self._content_registry: Dict[str, ProcessedContent] = {}

    def register_content(self, content: ProcessedContent) -> None:
        """Register a processed content element."""
        self._content_registry[content.content_id] = content

    def add_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: RelationshipType,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ContentRelationship:
        """
        Add a relationship between two content elements.

        Args:
            source_id: ID of the source content
            target_id: ID of the target content
            relationship_type: Type of relationship
            metadata: Additional relationship metadata

        Returns:
            The created ContentRelationship
        """
        relationship = ContentRelationship(
            source_id=source_id,
            target_id=target_id,
            relationship_type=relationship_type,
            metadata=metadata or {}
        )
        self._relationships.append(relationship)
        return relationship

    def get_relationships_for(
        self,
        content_id: str,
        direction: str = "both"
    ) -> List[ContentRelationship]:
        """
        Get all relationships for a content element.

        Args:
            content_id: ID of the content to query
            direction: "outgoing", "incoming", or "both"

        Returns:
            List of related ContentRelationships
        """
        results = []

        for rel in self._relationships:
            if direction in ("both", "outgoing") and rel.source_id == content_id:
                results.append(rel)
            elif direction in ("both", "incoming") and rel.target_id == content_id:
                results.append(rel)

        return results

    def get_related_content(
        self,
        content_id: str,
        relationship_type: Optional[RelationshipType] = None
    ) -> List[ProcessedContent]:
        """
        Get content elements related to a given content.

        Args:
            content_id: ID of the content to query
            relationship_type: Optional filter by relationship type

        Returns:
            List of related ProcessedContent elements
        """
        related_ids = set()

        for rel in self._relationships:
            if relationship_type and rel.relationship_type != relationship_type:
                continue

            if rel.source_id == content_id:
                related_ids.add(rel.target_id)
            elif rel.target_id == content_id:
                related_ids.add(rel.source_id)

        return [
            self._content_registry[cid]
            for cid in related_ids
            if cid in self._content_registry
        ]

    def infer_relationships(
        self,
        contents: List[ProcessedContent]
    ) -> List[ContentRelationship]:
        """
        Automatically infer relationships between content elements.

        Uses position, context, and content analysis to detect relationships.

        Args:
            contents: List of processed content to analyze

        Returns:
            List of inferred relationships
        """
        inferred = []

        # Sort by position if available
        sorted_contents = sorted(
            contents,
            key=lambda c: c.position if c.position is not None else float('inf')
        )

        # Infer sequential relationships
        for i in range(len(sorted_contents) - 1):
            current = sorted_contents[i]
            next_content = sorted_contents[i + 1]

            # Add FOLLOWS relationship for sequential content
            if current.position is not None and next_content.position is not None:
                rel = self.add_relationship(
                    current.content_id,
                    next_content.content_id,
                    RelationshipType.FOLLOWS,
                    {"position_gap": next_content.position - current.position}
                )
                inferred.append(rel)

        # Infer DESCRIBES relationships (text describing images/tables)
        for i, content in enumerate(contents):
            if content.content_type == ContentType.TEXT:
                # Check if text describes nearby non-text content
                for j, other in enumerate(contents):
                    if other.content_type in (ContentType.IMAGE, ContentType.TABLE):
                        # Check for proximity and descriptive keywords
                        if abs(i - j) <= 1:  # Adjacent content
                            if self._is_descriptive_text(content.extracted_text, other):
                                rel = self.add_relationship(
                                    content.content_id,
                                    other.content_id,
                                    RelationshipType.DESCRIBES,
                                    {"reason": "proximity_and_context"}
                                )
                                inferred.append(rel)

        return inferred

    def _is_descriptive_text(
        self,
        text: str,
        target: ProcessedContent
    ) -> bool:
        """Check if text appears to describe another content element."""
        text_lower = text.lower()
        descriptive_patterns = [
            r"figure\s*\d*",
            r"table\s*\d*",
            r"chart\s*\d*",
            r"image\s*\d*",
            r"shows?\s",
            r"displays?\s",
            r"illustrates?\s",
            r"depicts?\s"
        ]

        return any(re.search(pattern, text_lower) for pattern in descriptive_patterns)

    def get_all_relationships(self) -> List[ContentRelationship]:
        """Get all registered relationships."""
        return self._relationships.copy()

    def clear(self) -> None:
        """Clear all relationships and registered content."""
        self._relationships.clear()
        self._content_registry.clear()


# ============================================================================
# CONTENT UNIFIER
# ============================================================================

class ContentUnifier:
    """
    Merges processed content into a consistent, unified output format.

    Handles combining different content types while preserving
    relationships and metadata.
    """

    @classmethod
    def unify_contents(
        cls,
        contents: List[ProcessedContent],
        relationships: List[ContentRelationship],
        document_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Unify multiple processed contents into a single output.

        Args:
            contents: List of processed content elements
            relationships: List of content relationships
            document_metadata: Overall document metadata

        Returns:
            Unified document representation
        """
        # Build combined searchable text
        all_searchable_text = " ".join(c.searchable_text for c in contents)

        # Organize content by type
        content_by_type = {}
        for content in contents:
            type_name = content.content_type.value
            if type_name not in content_by_type:
                content_by_type[type_name] = []
            content_by_type[type_name].append(cls._content_to_dict(content))

        # Format relationships
        formatted_relationships = [
            {
                "source": rel.source_id,
                "target": rel.target_id,
                "type": rel.relationship_type.value,
                "metadata": rel.metadata
            }
            for rel in relationships
        ]

        # Build unified output
        return {
            "combined_searchable_text": all_searchable_text,
            "content_count": len(contents),
            "content_by_type": content_by_type,
            "relationships": formatted_relationships,
            "relationship_count": len(relationships),
            "document_metadata": document_metadata or {},
            "content_types_present": list(content_by_type.keys())
        }

    @classmethod
    def _content_to_dict(cls, content: ProcessedContent) -> Dict[str, Any]:
        """Convert ProcessedContent to dictionary."""
        return {
            "id": content.content_id,
            "type": content.content_type.value,
            "extracted_text": content.extracted_text,
            "searchable_text": content.searchable_text,
            "metadata": content.metadata,
            "position": content.position,
            "confidence_score": content.confidence_score,
            "processing_notes": content.processing_notes
        }


# ============================================================================
# MAIN PROCESSOR CLASS
# ============================================================================

class MultiModalDocumentProcessor:
    """
    Multi-Modal Document Processor

    Extends the RAG pipeline to extract and process images, tables,
    and structured content from documents. Maintains relationships
    between different content types and produces unified, searchable output.
    """

    def __init__(self):
        """Initialize the MultiModalDocumentProcessor instance."""
        self.image_processor = ImageProcessor()
        self.table_processor = TableProcessor()
        self.structured_processor = StructuredDataProcessor()
        self.relationship_mapper = RelationshipMapper()
        self.content_unifier = ContentUnifier()

        # Processing statistics
        self._processing_stats = {
            "documents_processed": 0,
            "images_processed": 0,
            "tables_processed": 0,
            "json_processed": 0,
            "csv_processed": 0,
            "relationships_created": 0,
            "errors_encountered": 0
        }

    def detect_content_type(self, content: Any) -> ContentType:
        """
        Detect the type of content from its structure or format.

        Args:
            content: Content to analyze (string or dict)

        Returns:
            Detected ContentType
        """
        if isinstance(content, dict):
            content_type = content.get("type", "").lower()

            if content_type == "image" or "data" in content:
                data = content.get("data", "")
                if isinstance(data, str) and (data.startswith("data:image") or "base64" in data):
                    return ContentType.IMAGE

            if content_type == "table" or "rows" in content or "headers" in content:
                return ContentType.TABLE

            if content_type == "json":
                return ContentType.JSON

            if content_type == "csv":
                return ContentType.CSV

            # Check for raw content with type hints
            raw = content.get("content", content.get("data", ""))
            if isinstance(raw, str):
                return self._detect_string_content_type(raw)

            return ContentType.JSON  # Default for dicts

        elif isinstance(content, str):
            return self._detect_string_content_type(content)

        return ContentType.UNKNOWN

    def _detect_string_content_type(self, content: str) -> ContentType:
        """Detect content type from a string."""
        content_stripped = content.strip()

        # Check for HTML table
        if "<table" in content_stripped.lower() or "<tr" in content_stripped.lower():
            return ContentType.TABLE

        # Check for base64 image
        if content_stripped.startswith("data:image") or "/9j/" in content_stripped:
            return ContentType.IMAGE

        # Check for JSON
        if (content_stripped.startswith('{') and content_stripped.endswith('}')) or \
           (content_stripped.startswith('[') and content_stripped.endswith(']')):
            try:
                json.loads(content_stripped)
                return ContentType.JSON
            except json.JSONDecodeError:
                pass

        # Check for CSV (multiple lines with consistent delimiters)
        lines = content_stripped.split('\n')
        if len(lines) > 1:
            first_line_commas = lines[0].count(',')
            if first_line_commas > 0 and all(
                line.count(',') == first_line_commas
                for line in lines[:5] if line.strip()
            ):
                return ContentType.CSV

        return ContentType.TEXT

    def process_content(
        self,
        content: Any,
        content_type: Optional[ContentType] = None,
        context: Optional[str] = None,
        position: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ProcessedContent:
        """
        Process a single content element.

        Args:
            content: Content to process (dict or string)
            content_type: Type of content (auto-detected if not provided)
            context: Surrounding context
            position: Position in document
            metadata: Additional metadata

        Returns:
            ProcessedContent with extracted information
        """
        # Auto-detect content type if not provided
        if content_type is None:
            content_type = self.detect_content_type(content)

        # Extract raw data from dict if needed
        if isinstance(content, dict):
            raw_data = content.get("data", content.get("content", json.dumps(content)))
            context = context or content.get("context")
            metadata = {**(metadata or {}), **{k: v for k, v in content.items() if k not in ("data", "content", "context", "type")}}
        else:
            raw_data = content

        # Process based on type
        if content_type == ContentType.IMAGE:
            result = self.image_processor.process_image(raw_data, context, metadata)
            self._processing_stats["images_processed"] += 1
        elif content_type == ContentType.TABLE:
            result = self.table_processor.process_table(raw_data, context, metadata)
            self._processing_stats["tables_processed"] += 1
        elif content_type == ContentType.JSON:
            result = self.structured_processor.process_json(raw_data, context, metadata)
            self._processing_stats["json_processed"] += 1
        elif content_type == ContentType.CSV:
            result = self.structured_processor.process_csv(raw_data, context, metadata)
            self._processing_stats["csv_processed"] += 1
        else:
            # Handle as plain text
            result = ProcessedContent(
                content_id=str(uuid.uuid4()),
                content_type=ContentType.TEXT,
                original_data=raw_data,
                extracted_text=raw_data if isinstance(raw_data, str) else str(raw_data),
                searchable_text=raw_data.lower() if isinstance(raw_data, str) else str(raw_data).lower(),
                metadata=metadata or {},
                confidence_score=1.0,
                processing_notes=["Processed as plain text"]
            )

        # Set position
        result.position = position

        # Register with relationship mapper
        self.relationship_mapper.register_content(result)

        return result

    def process_document(
        self,
        contents: List[Any],
        document_metadata: Optional[Dict[str, Any]] = None,
        infer_relationships: bool = True
    ) -> ProcessingResult:
        """
        Process a complete document with multiple content elements.

        Args:
            contents: List of content elements to process
            document_metadata: Metadata about the document
            infer_relationships: Whether to automatically infer relationships

        Returns:
            ProcessingResult with all processed content and relationships
        """
        import time
        start_time = time.time()

        document_id = str(uuid.uuid4())
        processed_contents = []
        errors = []

        # Clear relationship mapper for new document
        self.relationship_mapper.clear()

        # Process each content element
        for i, content in enumerate(contents):
            try:
                processed = self.process_content(
                    content=content,
                    position=i
                )
                processed_contents.append(processed)
            except Exception as e:
                errors.append({
                    "position": i,
                    "error": str(e),
                    "content_preview": str(content)[:100]
                })
                self._processing_stats["errors_encountered"] += 1

        # Infer relationships if requested
        relationships = []
        if infer_relationships and len(processed_contents) > 1:
            relationships = self.relationship_mapper.infer_relationships(processed_contents)
            self._processing_stats["relationships_created"] += len(relationships)

        # Get all relationships (including any manually added)
        all_relationships = self.relationship_mapper.get_all_relationships()

        # Update stats
        self._processing_stats["documents_processed"] += 1

        processing_time = time.time() - start_time

        return ProcessingResult(
            document_id=document_id,
            processed_contents=processed_contents,
            relationships=all_relationships,
            errors=errors,
            metadata=document_metadata or {},
            processing_time=processing_time
        )

    def get_unified_output(self, result: ProcessingResult) -> Dict[str, Any]:
        """
        Get unified output from a processing result.

        Args:
            result: ProcessingResult to unify

        Returns:
            Unified document representation
        """
        return self.content_unifier.unify_contents(
            contents=result.processed_contents,
            relationships=result.relationships,
            document_metadata={
                **result.metadata,
                "document_id": result.document_id,
                "processing_time": result.processing_time,
                "error_count": len(result.errors)
            }
        )

    def add_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: RelationshipType,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ContentRelationship:
        """
        Manually add a relationship between content elements.

        Args:
            source_id: Source content ID
            target_id: Target content ID
            relationship_type: Type of relationship
            metadata: Relationship metadata

        Returns:
            Created ContentRelationship
        """
        rel = self.relationship_mapper.add_relationship(
            source_id, target_id, relationship_type, metadata
        )
        self._processing_stats["relationships_created"] += 1
        return rel

    def get_statistics(self) -> Dict[str, int]:
        """Get processing statistics."""
        return self._processing_stats.copy()

    def execute(
        self,
        contents: Optional[List[Any]] = None,
        document_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Main entry point for document processing.

        Args:
            contents: List of content elements to process
            document_metadata: Optional document metadata

        Returns:
            Unified processing result
        """
        if contents is None:
            # Demo with sample content
            contents = [
                {
                    "type": "image",
                    "data": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
                    "context": "Figure 1: Sales data visualization showing Q1 performance"
                },
                {
                    "type": "table",
                    "data": "<table><tr><th>Product</th><th>Sales</th></tr><tr><td>Widget A</td><td>1000</td></tr><tr><td>Widget B</td><td>1500</td></tr></table>",
                    "context": "Table 1: Product sales summary"
                },
                {
                    "type": "json",
                    "data": '{"config": {"api_key": "xxx", "model": "claude-3", "max_tokens": 1000}}',
                    "context": "API configuration settings"
                },
                {
                    "type": "csv",
                    "data": "name,department,salary\nAlice,Engineering,85000\nBob,Sales,75000",
                    "context": "Employee data export"
                }
            ]

        # Process the document
        result = self.process_document(contents, document_metadata)

        # Get unified output
        unified = self.get_unified_output(result)

        # Add processing stats
        unified["processing_statistics"] = self.get_statistics()
        unified["errors"] = result.errors

        return unified


def create_multi_modal_document_processor() -> MultiModalDocumentProcessor:
    """
    Factory function for creating MultiModalDocumentProcessor instances.

    Returns:
        MultiModalDocumentProcessor: A new instance of MultiModalDocumentProcessor
    """
    return MultiModalDocumentProcessor()
