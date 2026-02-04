"""
Document Ingestion Pipeline

Build a Python script that can load, clean, and chunk various document formats (PDF, TXT, MD)

This module implements a foundational component of a RAG system that processes various
document formats and prepares them for retrieval operations. It transforms raw documents
into structured, searchable chunks with metadata.
"""

import os
import re
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any


class DocumentProcessingError(Exception):
    """Custom exception for document processing errors."""
    pass


class BaseDocumentProcessor(ABC):
    """
    Abstract base class defining the interface for document processors.

    Each document type (TXT, MD, PDF) has its own processor that implements
    this interface, enabling consistent handling across formats.
    """

    @abstractmethod
    def extract_text(self, file_path: str) -> str:
        """
        Extract raw text content from the document.

        Args:
            file_path: Path to the document file

        Returns:
            Extracted text content as a string
        """
        pass

    @abstractmethod
    def get_page_info(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Get page-level information for the document.

        Args:
            file_path: Path to the document file

        Returns:
            List of dictionaries with page information (page_number, start_pos, end_pos)
        """
        pass


class TextDocumentProcessor(BaseDocumentProcessor):
    """
    Processor for plain text (.txt) files.

    Handles encoding detection and text extraction for plain text documents.
    Text files are treated as single-page documents.
    """

    # Common encodings to try when reading text files
    ENCODINGS = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'ascii']

    def extract_text(self, file_path: str) -> str:
        """Extract text from a .txt file with encoding detection."""
        for encoding in self.ENCODINGS:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                return content
            except UnicodeDecodeError:
                continue
            except Exception as e:
                raise DocumentProcessingError(f"Error reading file {file_path}: {str(e)}")

        raise DocumentProcessingError(
            f"Unable to decode file {file_path} with any supported encoding"
        )

    def get_page_info(self, file_path: str) -> List[Dict[str, Any]]:
        """Text files are treated as single-page documents."""
        text = self.extract_text(file_path)
        return [{"page_number": 1, "start_pos": 0, "end_pos": len(text)}]


class MarkdownDocumentProcessor(BaseDocumentProcessor):
    """
    Processor for Markdown (.md) files.

    Handles markdown documents while preserving important structural elements.
    Markdown headers (##) can be used to infer logical sections.
    """

    ENCODINGS = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']

    def extract_text(self, file_path: str) -> str:
        """Extract text from a .md file, preserving structure."""
        for encoding in self.ENCODINGS:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                return content
            except UnicodeDecodeError:
                continue
            except Exception as e:
                raise DocumentProcessingError(f"Error reading file {file_path}: {str(e)}")

        raise DocumentProcessingError(
            f"Unable to decode file {file_path} with any supported encoding"
        )

    def get_page_info(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Markdown files use headers as logical page boundaries.
        Each top-level section (# Header) is treated as a 'page'.
        """
        text = self.extract_text(file_path)

        # Find all top-level headers to create logical pages
        header_pattern = re.compile(r'^#\s+.+$', re.MULTILINE)
        matches = list(header_pattern.finditer(text))

        if not matches:
            # No headers found, treat as single page
            return [{"page_number": 1, "start_pos": 0, "end_pos": len(text)}]

        pages = []
        for i, match in enumerate(matches):
            start_pos = match.start()
            end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            pages.append({
                "page_number": i + 1,
                "start_pos": start_pos,
                "end_pos": end_pos
            })

        # Include any content before the first header
        if matches[0].start() > 0:
            pages.insert(0, {
                "page_number": 0,
                "start_pos": 0,
                "end_pos": matches[0].start()
            })

        return pages


class PDFDocumentProcessor(BaseDocumentProcessor):
    """
    Processor for PDF files using standard library only.

    LIMITATION: This implementation can only extract text from simple, text-based PDFs.
    It parses the PDF structure to find text streams. For production use, consider
    libraries like PyPDF2 or pdfplumber for more robust PDF handling.
    """

    def extract_text(self, file_path: str) -> str:
        """
        Extract text from a PDF file using basic parsing.

        This is a simplified implementation that works with basic text-based PDFs.
        It looks for text streams in the PDF structure.
        """
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
        except Exception as e:
            raise DocumentProcessingError(f"Error reading PDF file {file_path}: {str(e)}")

        # Check PDF header
        if not content.startswith(b'%PDF'):
            raise DocumentProcessingError(f"File {file_path} is not a valid PDF")

        extracted_text = []

        # Look for text between BT (Begin Text) and ET (End Text) markers
        # This is a simplified approach that works for basic PDFs
        try:
            # Find stream objects which may contain text
            stream_pattern = re.compile(b'stream\r?\n(.*?)\r?\nendstream', re.DOTALL)
            streams = stream_pattern.findall(content)

            for stream in streams:
                # Try to decode text from the stream
                text = self._extract_text_from_stream(stream)
                if text:
                    extracted_text.append(text)

            # Also look for text objects directly in the content
            text_pattern = re.compile(b'\\(([^)]+)\\)')
            text_matches = text_pattern.findall(content)
            for match in text_matches:
                try:
                    decoded = match.decode('utf-8', errors='ignore')
                    if len(decoded) > 2 and decoded.isprintable():
                        extracted_text.append(decoded)
                except:
                    pass

        except Exception as e:
            raise DocumentProcessingError(f"Error parsing PDF {file_path}: {str(e)}")

        result = ' '.join(extracted_text)

        # Clean up the extracted text
        result = re.sub(r'\s+', ' ', result).strip()

        if not result:
            # If we couldn't extract text, the PDF might be image-based
            raise DocumentProcessingError(
                f"Could not extract text from PDF {file_path}. "
                "The PDF may be image-based or use unsupported encoding."
            )

        return result

    def _extract_text_from_stream(self, stream: bytes) -> str:
        """
        Attempt to extract readable text from a PDF stream.

        PDF streams can be compressed or encoded in various ways.
        This basic implementation handles uncompressed text.
        """
        text_parts = []

        # Look for text showing operators: Tj, TJ, ', "
        # Tj shows a text string
        # TJ shows an array of text strings
        tj_pattern = re.compile(b'\\(([^)]+)\\)\\s*Tj', re.DOTALL)
        matches = tj_pattern.findall(stream)

        for match in matches:
            try:
                # Decode the text, handling PDF escape sequences
                decoded = self._decode_pdf_string(match)
                if decoded:
                    text_parts.append(decoded)
            except:
                pass

        return ' '.join(text_parts)

    def _decode_pdf_string(self, data: bytes) -> str:
        """Decode a PDF string, handling common escape sequences."""
        try:
            # Handle PDF escape sequences
            text = data.replace(b'\\n', b'\n')
            text = text.replace(b'\\r', b'\r')
            text = text.replace(b'\\t', b'\t')
            text = text.replace(b'\\(', b'(')
            text = text.replace(b'\\)', b')')
            text = text.replace(b'\\\\', b'\\')

            return text.decode('utf-8', errors='ignore')
        except:
            return ''

    def get_page_info(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Get page information from PDF.

        Note: This simplified implementation treats the PDF as a single page.
        For accurate page detection, a full PDF library would be needed.
        """
        text = self.extract_text(file_path)
        return [{"page_number": 1, "start_pos": 0, "end_pos": len(text)}]


class ChunkingEngine:
    """
    Implements intelligent text chunking with overlap handling.

    Uses a sliding window approach with boundary detection to create
    chunks that maintain semantic coherence by avoiding mid-word or
    mid-sentence breaks where possible.
    """

    def __init__(self, chunk_size: int = 1000, overlap_size: int = 200):
        """
        Initialize the chunking engine.

        Args:
            chunk_size: Target size for each chunk in characters (default: 1000)
            overlap_size: Number of characters to overlap between chunks (default: 200)
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap_size < 0:
            raise ValueError("overlap_size cannot be negative")
        if overlap_size >= chunk_size:
            raise ValueError("overlap_size must be less than chunk_size")

        self.chunk_size = chunk_size
        self.overlap_size = overlap_size

    def chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks with intelligent boundary detection.

        Args:
            text: The text to chunk

        Returns:
            List of chunk dictionaries with text, start_pos, and end_pos
        """
        if not text or not text.strip():
            return []

        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            # Calculate the end position for this chunk
            end = min(start + self.chunk_size, text_length)

            # If we're not at the end of the text, find a good breaking point
            if end < text_length:
                end = self._find_break_point(text, start, end)

            # Extract the chunk text
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "start_pos": start,
                    "end_pos": end
                })

            # Move start position, accounting for overlap
            if end >= text_length:
                break

            # Calculate next start position with overlap
            next_start = end - self.overlap_size

            # Make sure we're making progress
            if next_start <= start:
                next_start = start + 1

            start = next_start

        return chunks

    def _find_break_point(self, text: str, start: int, end: int) -> int:
        """
        Find an appropriate breaking point near the target end position.

        Prioritizes sentence boundaries, then paragraph breaks, then word boundaries.

        Args:
            text: The full text
            start: Start position of current chunk
            end: Target end position

        Returns:
            Adjusted end position at a natural boundary
        """
        # Define the search window (look back up to 20% of chunk size)
        search_start = max(start, end - int(self.chunk_size * 0.2))
        search_text = text[search_start:end]

        # Priority 1: Find sentence boundary (. ! ?)
        sentence_endings = ['.', '!', '?']
        best_pos = -1

        for i in range(len(search_text) - 1, -1, -1):
            char = search_text[i]
            if char in sentence_endings:
                # Check if this looks like end of sentence (followed by space or end)
                next_idx = i + 1
                if next_idx >= len(search_text) or search_text[next_idx].isspace():
                    best_pos = search_start + i + 1
                    break

        if best_pos > start:
            return best_pos

        # Priority 2: Find paragraph break (double newline)
        newline_pos = search_text.rfind('\n\n')
        if newline_pos != -1:
            return search_start + newline_pos + 2

        # Priority 3: Find single newline
        newline_pos = search_text.rfind('\n')
        if newline_pos != -1:
            return search_start + newline_pos + 1

        # Priority 4: Find word boundary (space)
        space_pos = search_text.rfind(' ')
        if space_pos != -1:
            return search_start + space_pos + 1

        # Fallback: Use the original end position
        return end


class MetadataExtractor:
    """
    Generates comprehensive metadata for each processed chunk.

    Metadata includes source file information, chunk positioning,
    document type, and page references where applicable.
    """

    @staticmethod
    def create_chunk_metadata(
        source_file: str,
        chunk_index: int,
        start_pos: int,
        end_pos: int,
        page_number: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Create metadata dictionary for a chunk.

        Args:
            source_file: Path to the source document
            chunk_index: Index of this chunk (0-based)
            start_pos: Starting character position in source
            end_pos: Ending character position in source
            page_number: Page number if applicable

        Returns:
            Dictionary containing chunk metadata
        """
        filename = os.path.basename(source_file)
        _, ext = os.path.splitext(filename)
        doc_type = ext[1:].lower() if ext else 'unknown'

        metadata = {
            "source": filename,
            "document_type": doc_type,
            "chunk_index": chunk_index,
            "start_pos": start_pos,
            "end_pos": end_pos
        }

        if page_number is not None:
            metadata["page_number"] = page_number

        return metadata

    @staticmethod
    def create_chunk_id(source_file: str, chunk_index: int) -> str:
        """
        Create a unique identifier for a chunk.

        Args:
            source_file: Path to the source document
            chunk_index: Index of this chunk

        Returns:
            Unique chunk identifier string
        """
        filename = os.path.basename(source_file)
        return f"{filename}_{chunk_index}"


class DocumentIngestionPipeline:
    """
    Main class that orchestrates the entire document processing workflow.

    This pipeline processes various document formats (PDF, TXT, MD) and
    produces structured chunks with metadata suitable for RAG retrieval.
    """

    # Mapping of file extensions to their processors
    SUPPORTED_EXTENSIONS = {
        '.txt': TextDocumentProcessor,
        '.md': MarkdownDocumentProcessor,
        '.pdf': PDFDocumentProcessor
    }

    def __init__(self, chunk_size: int = 1000, overlap_size: int = 200):
        """
        Initialize the DocumentIngestionPipeline instance.

        Args:
            chunk_size: Target size for text chunks (default: 1000 characters)
            overlap_size: Overlap between chunks (default: 200 characters)
        """
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.chunking_engine = ChunkingEngine(chunk_size, overlap_size)
        self._processors: Dict[str, BaseDocumentProcessor] = {}

    def _get_processor(self, file_path: str) -> BaseDocumentProcessor:
        """
        Get the appropriate processor for a file based on its extension.

        Args:
            file_path: Path to the document

        Returns:
            Document processor instance

        Raises:
            DocumentProcessingError: If file type is not supported
        """
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()

        if ext not in self.SUPPORTED_EXTENSIONS:
            supported = ', '.join(self.SUPPORTED_EXTENSIONS.keys())
            raise DocumentProcessingError(
                f"Unsupported file type: {ext}. Supported types: {supported}"
            )

        # Use cached processor or create new one
        if ext not in self._processors:
            self._processors[ext] = self.SUPPORTED_EXTENSIONS[ext]()

        return self._processors[ext]

    def process_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Process a single document file and return structured chunks.

        Args:
            file_path: Path to the document to process

        Returns:
            List of chunk dictionaries with text, metadata, and chunk_id

        Raises:
            DocumentProcessingError: If file cannot be processed
        """
        # Validate file exists
        if not os.path.exists(file_path):
            raise DocumentProcessingError(f"File not found: {file_path}")

        if not os.path.isfile(file_path):
            raise DocumentProcessingError(f"Path is not a file: {file_path}")

        # Get appropriate processor
        processor = self._get_processor(file_path)

        # Extract text
        text = processor.extract_text(file_path)

        # Get page information
        page_info = processor.get_page_info(file_path)

        # Chunk the text
        raw_chunks = self.chunking_engine.chunk_text(text)

        # Build structured chunks with metadata
        structured_chunks = []
        for i, chunk in enumerate(raw_chunks):
            # Determine page number based on chunk position
            page_number = self._get_page_number(chunk["start_pos"], page_info)

            # Create metadata
            metadata = MetadataExtractor.create_chunk_metadata(
                source_file=file_path,
                chunk_index=i,
                start_pos=chunk["start_pos"],
                end_pos=chunk["end_pos"],
                page_number=page_number
            )

            # Create chunk ID
            chunk_id = MetadataExtractor.create_chunk_id(file_path, i)

            structured_chunks.append({
                "text": chunk["text"],
                "metadata": metadata,
                "chunk_id": chunk_id
            })

        return structured_chunks

    def _get_page_number(
        self,
        position: int,
        page_info: List[Dict[str, Any]]
    ) -> Optional[int]:
        """
        Determine which page a character position falls on.

        Args:
            position: Character position in the text
            page_info: List of page boundary information

        Returns:
            Page number or None if not determinable
        """
        for page in page_info:
            if page["start_pos"] <= position < page["end_pos"]:
                return page["page_number"]

        # If position is at or beyond the last page's end, return last page
        if page_info and position >= page_info[-1]["start_pos"]:
            return page_info[-1]["page_number"]

        return None

    def process_files(self, file_paths: List[str]) -> Dict[str, Any]:
        """
        Process multiple document files.

        Args:
            file_paths: List of paths to documents

        Returns:
            Dictionary with 'chunks' (all processed chunks) and 'errors' (any processing errors)
        """
        all_chunks = []
        errors = []

        for file_path in file_paths:
            try:
                chunks = self.process_file(file_path)
                all_chunks.extend(chunks)
            except DocumentProcessingError as e:
                errors.append({"file": file_path, "error": str(e)})
            except Exception as e:
                errors.append({
                    "file": file_path,
                    "error": f"Unexpected error: {str(e)}"
                })

        return {
            "chunks": all_chunks,
            "errors": errors
        }

    def execute(self, file_paths: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Main entry point for the pipeline.

        Args:
            file_paths: List of document paths to process. If None, returns empty result.

        Returns:
            Dictionary containing processed chunks and any errors encountered
        """
        if file_paths is None or len(file_paths) == 0:
            return {"chunks": [], "errors": []}

        return self.process_files(file_paths)


def create_document_ingestion_pipeline(
    chunk_size: int = 1000,
    overlap_size: int = 200
) -> DocumentIngestionPipeline:
    """
    Factory function for creating DocumentIngestionPipeline instances.

    Args:
        chunk_size: Target size for text chunks (default: 1000)
        overlap_size: Overlap between chunks (default: 200)

    Returns:
        DocumentIngestionPipeline: A new instance of DocumentIngestionPipeline
    """
    return DocumentIngestionPipeline(chunk_size=chunk_size, overlap_size=overlap_size)
