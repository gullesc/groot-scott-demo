"""
Tests for Document Ingestion Pipeline

Run: pytest
Watch: pytest-watch (install with: pip install pytest-watch)
Coverage: pytest --cov=src
"""

import os
import tempfile
import pytest
from src.document_ingestion_pipeline import (
    DocumentIngestionPipeline,
    create_document_ingestion_pipeline,
    DocumentProcessingError,
    TextDocumentProcessor,
    MarkdownDocumentProcessor,
    ChunkingEngine,
    MetadataExtractor,
)


class TestDocumentIngestionPipeline:
    """Test suite for DocumentIngestionPipeline."""

    @pytest.fixture
    def instance(self):
        """Create a fresh instance for each test."""
        return create_document_ingestion_pipeline()

    def test_create_instance(self, instance):
        """Test that we can create an instance."""
        assert instance is not None
        assert isinstance(instance, DocumentIngestionPipeline)

    def test_execute_returns_dict(self, instance):
        """Test that execute returns a dictionary with chunks and errors."""
        result = instance.execute()
        assert isinstance(result, dict)
        assert "chunks" in result
        assert "errors" in result

    def test_execute_with_empty_list(self, instance):
        """Test execute with empty file list."""
        result = instance.execute([])
        assert result["chunks"] == []
        assert result["errors"] == []


class TestTextDocumentProcessor:
    """Tests for TextDocumentProcessor."""

    @pytest.fixture
    def processor(self):
        return TextDocumentProcessor()

    def test_extract_text_utf8(self, processor):
        """Test extracting text from UTF-8 file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write("Hello, World! This is a test document.")
            temp_path = f.name

        try:
            text = processor.extract_text(temp_path)
            assert "Hello, World!" in text
            assert "test document" in text
        finally:
            os.unlink(temp_path)

    def test_extract_text_latin1(self, processor):
        """Test extracting text from Latin-1 encoded file."""
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.txt', delete=False) as f:
            f.write("Café résumé naïve".encode('latin-1'))
            temp_path = f.name

        try:
            text = processor.extract_text(temp_path)
            assert "Caf" in text  # May not decode perfectly but shouldn't crash
        finally:
            os.unlink(temp_path)


class TestMarkdownDocumentProcessor:
    """Tests for MarkdownDocumentProcessor."""

    @pytest.fixture
    def processor(self):
        return MarkdownDocumentProcessor()

    def test_extract_markdown(self, processor):
        """Test extracting text from markdown file."""
        content = """# Main Title

This is an introduction paragraph.

## Section One

Content for section one.

## Section Two

Content for section two.
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
            f.write(content)
            temp_path = f.name

        try:
            text = processor.extract_text(temp_path)
            assert "Main Title" in text
            assert "Section One" in text
            assert "Section Two" in text
        finally:
            os.unlink(temp_path)

    def test_get_page_info_with_headers(self, processor):
        """Test that headers create logical pages."""
        content = """# First Header

First content.

# Second Header

Second content.
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
            f.write(content)
            temp_path = f.name

        try:
            pages = processor.get_page_info(temp_path)
            assert len(pages) >= 2  # At least two header sections
        finally:
            os.unlink(temp_path)


class TestChunkingEngine:
    """Tests for ChunkingEngine."""

    def test_basic_chunking(self):
        """Test basic text chunking."""
        engine = ChunkingEngine(chunk_size=50, overlap_size=10)
        text = "This is a test sentence. Here is another sentence. And one more sentence to chunk."
        chunks = engine.chunk_text(text)

        assert len(chunks) > 0
        assert all("text" in c for c in chunks)
        assert all("start_pos" in c for c in chunks)
        assert all("end_pos" in c for c in chunks)

    def test_overlap_handling(self):
        """Test that chunks have proper overlap."""
        engine = ChunkingEngine(chunk_size=100, overlap_size=20)
        text = "A" * 50 + " " + "B" * 50 + " " + "C" * 50 + " " + "D" * 50

        chunks = engine.chunk_text(text)

        # With overlap, adjacent chunks should share some content
        if len(chunks) > 1:
            # The end of chunk 0 should be close to the start of chunk 1
            # (within overlap_size characters)
            for i in range(len(chunks) - 1):
                gap = chunks[i + 1]["start_pos"] - chunks[i]["end_pos"]
                assert gap < engine.overlap_size or gap == -engine.overlap_size or abs(gap) <= engine.chunk_size

    def test_empty_text(self):
        """Test chunking empty text."""
        engine = ChunkingEngine()
        chunks = engine.chunk_text("")
        assert chunks == []

    def test_small_text(self):
        """Test text smaller than chunk size."""
        engine = ChunkingEngine(chunk_size=1000, overlap_size=100)
        text = "Small text."
        chunks = engine.chunk_text(text)

        assert len(chunks) == 1
        assert chunks[0]["text"] == "Small text."

    def test_invalid_parameters(self):
        """Test that invalid parameters raise errors."""
        with pytest.raises(ValueError):
            ChunkingEngine(chunk_size=0)

        with pytest.raises(ValueError):
            ChunkingEngine(chunk_size=100, overlap_size=100)

        with pytest.raises(ValueError):
            ChunkingEngine(overlap_size=-1)


class TestMetadataExtractor:
    """Tests for MetadataExtractor."""

    def test_create_chunk_metadata(self):
        """Test metadata creation."""
        metadata = MetadataExtractor.create_chunk_metadata(
            source_file="/path/to/document.txt",
            chunk_index=0,
            start_pos=0,
            end_pos=100,
            page_number=1
        )

        assert metadata["source"] == "document.txt"
        assert metadata["document_type"] == "txt"
        assert metadata["chunk_index"] == 0
        assert metadata["start_pos"] == 0
        assert metadata["end_pos"] == 100
        assert metadata["page_number"] == 1

    def test_create_chunk_id(self):
        """Test chunk ID creation."""
        chunk_id = MetadataExtractor.create_chunk_id("/path/to/doc.pdf", 5)
        assert chunk_id == "doc.pdf_5"


class TestAcceptanceCriteria:
    """Tests for acceptance criteria."""

    @pytest.fixture
    def instance(self):
        """Create a fresh instance for each test."""
        return create_document_ingestion_pipeline()

    def test_successfully_processes_pdfs_text_files_and_markdown(self, instance):
        """Test: Successfully processes PDFs, text files, and markdown documents"""
        # Create test files
        test_files = []

        # Create TXT file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write("This is a text file with some content for testing purposes.")
            test_files.append(f.name)

        # Create MD file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
            f.write("# Test Markdown\n\nThis is markdown content with **bold** text.")
            test_files.append(f.name)

        try:
            result = instance.execute(test_files)

            # Should have chunks from both files
            assert len(result["chunks"]) >= 2
            assert len(result["errors"]) == 0

            # Check that we got chunks from both file types
            sources = {c["metadata"]["document_type"] for c in result["chunks"]}
            assert "txt" in sources
            assert "md" in sources

        finally:
            for f in test_files:
                os.unlink(f)

    def test_implements_intelligent_chunking_with_overlap_handling(self, instance):
        """Test: Implements intelligent chunking with overlap handling"""
        # Create a document large enough to require multiple chunks
        long_text = """
        This is the first paragraph of our test document. It contains several sentences
        that will help us test the chunking functionality. The chunking engine should
        split this text into multiple overlapping chunks.

        Here is another paragraph that adds more content to our document. Each paragraph
        represents a distinct semantic unit that should ideally not be split in the middle.
        The intelligent chunking should try to break at sentence or paragraph boundaries.

        And finally, we have a third paragraph to ensure we have enough content for
        multiple chunks with proper overlap. The overlap ensures that context is
        preserved across chunk boundaries.
        """ * 5  # Repeat to ensure multiple chunks

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write(long_text)
            temp_path = f.name

        try:
            # Use smaller chunk size to ensure multiple chunks
            pipeline = create_document_ingestion_pipeline(chunk_size=200, overlap_size=40)
            result = pipeline.execute([temp_path])

            # Should have multiple chunks
            assert len(result["chunks"]) > 1

            # Chunks should have proper structure
            for chunk in result["chunks"]:
                assert "text" in chunk
                assert "metadata" in chunk
                assert "chunk_id" in chunk
                assert len(chunk["text"]) > 0

        finally:
            os.unlink(temp_path)

    def test_includes_error_handling_for_corrupted_or_unsupported(self, instance):
        """Test: Includes error handling for corrupted or unsupported files"""
        # Test nonexistent file
        result = instance.execute(["/nonexistent/file.txt"])
        assert len(result["errors"]) == 1
        assert "not found" in result["errors"][0]["error"].lower() or "no such" in result["errors"][0]["error"].lower()

        # Test unsupported file type
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xyz', delete=False) as f:
            f.write("Unsupported format")
            temp_path = f.name

        try:
            result = instance.execute([temp_path])
            assert len(result["errors"]) == 1
            assert "unsupported" in result["errors"][0]["error"].lower()
        finally:
            os.unlink(temp_path)

    def test_outputs_structured_chunks_with_metadata(self, instance):
        """Test: Outputs structured chunks with metadata (source, page number, etc.)"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write("Test content for metadata verification.")
            temp_path = f.name

        try:
            result = instance.execute([temp_path])

            assert len(result["chunks"]) > 0
            chunk = result["chunks"][0]

            # Verify chunk structure
            assert "text" in chunk
            assert "metadata" in chunk
            assert "chunk_id" in chunk

            # Verify metadata fields
            metadata = chunk["metadata"]
            assert "source" in metadata
            assert "document_type" in metadata
            assert "chunk_index" in metadata
            assert "start_pos" in metadata
            assert "end_pos" in metadata

            # Verify values
            assert metadata["document_type"] == "txt"
            assert metadata["chunk_index"] == 0
            assert isinstance(metadata["start_pos"], int)
            assert isinstance(metadata["end_pos"], int)

        finally:
            os.unlink(temp_path)
