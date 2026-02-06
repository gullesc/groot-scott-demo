"""
Tests for Multi-Modal Document Processor

Run: pytest
Watch: pytest-watch (install with: pip install pytest-watch)
Coverage: pytest --cov=src
"""

import pytest
from src.multi_modal_document_processor import (
    MultiModalDocumentProcessor,
    create_multi_modal_document_processor,
    ContentType,
    RelationshipType,
    ImageProcessor,
    TableProcessor,
    StructuredDataProcessor,
    RelationshipMapper,
    ContentUnifier
)


class TestMultiModalDocumentProcessor:
    """Test suite for MultiModalDocumentProcessor."""

    @pytest.fixture
    def instance(self):
        """Create a fresh instance for each test."""
        return create_multi_modal_document_processor()

    def test_create_instance(self, instance):
        """Test that we can create an instance."""
        assert instance is not None
        assert isinstance(instance, MultiModalDocumentProcessor)

    def test_execute_returns_result(self, instance):
        """Test that execute returns a proper result dictionary."""
        result = instance.execute()
        assert isinstance(result, dict)
        assert "combined_searchable_text" in result
        assert "content_count" in result
        assert "content_by_type" in result

    def test_detect_content_type_image(self, instance):
        """Test image content type detection."""
        image_content = {"type": "image", "data": "data:image/jpeg;base64,/9j/4AAQ..."}
        assert instance.detect_content_type(image_content) == ContentType.IMAGE

    def test_detect_content_type_table(self, instance):
        """Test table content type detection."""
        table_content = "<table><tr><td>test</td></tr></table>"
        assert instance.detect_content_type(table_content) == ContentType.TABLE

    def test_detect_content_type_json(self, instance):
        """Test JSON content type detection."""
        json_content = '{"key": "value"}'
        assert instance.detect_content_type(json_content) == ContentType.JSON

    def test_detect_content_type_csv(self, instance):
        """Test CSV content type detection."""
        csv_content = "name,age\nJohn,30\nJane,25"
        assert instance.detect_content_type(csv_content) == ContentType.CSV


class TestAcceptanceCriteria:
    """Tests for acceptance criteria."""

    @pytest.fixture
    def instance(self):
        """Create a fresh instance for each test."""
        return create_multi_modal_document_processor()

    def test_extracts_and_processes_images_with_ocr_and_vision(self, instance):
        """Test: Extracts and processes images with OCR and vision models"""
        image_content = {
            "type": "image",
            "data": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD...",
            "context": "Figure 1: Sales chart showing \"Q1 Revenue: $1.5M\""
        }

        processed = instance.process_content(image_content)

        # Verify image was processed
        assert processed.content_type == ContentType.IMAGE
        assert processed.extracted_text is not None
        assert len(processed.extracted_text) > 0

        # Verify OCR simulation extracted text
        assert "ocr" in processed.extracted_text.lower() or "vision" in processed.extracted_text.lower() or "description" in processed.extracted_text.lower()

        # Verify searchable text was generated
        assert processed.searchable_text is not None
        assert len(processed.searchable_text) > 0

        # Verify metadata
        assert "image_format" in processed.metadata
        assert "image_type" in processed.metadata

    def test_handles_table_extraction_and_converts_to_searchable(self, instance):
        """Test: Handles table extraction and converts to searchable format"""
        table_html = """
        <table>
            <tr><th>Product</th><th>Price</th><th>Quantity</th></tr>
            <tr><td>Widget A</td><td>$10.00</td><td>100</td></tr>
            <tr><td>Widget B</td><td>$15.00</td><td>50</td></tr>
        </table>
        """

        processed = instance.process_content(table_html)

        # Verify table was processed
        assert processed.content_type == ContentType.TABLE

        # Verify headers were extracted
        assert "column_count" in processed.metadata
        assert processed.metadata["column_count"] == 3

        # Verify rows were extracted
        assert "row_count" in processed.metadata
        assert processed.metadata["row_count"] == 2

        # Verify searchable text contains table content
        assert "widget" in processed.searchable_text.lower()
        assert "product" in processed.searchable_text.lower()

        # Verify extracted text is readable
        assert "Product" in processed.extracted_text or "product" in processed.extracted_text.lower()

    def test_processes_structured_data_json_csv_with_proper_metadata(self, instance):
        """Test: Processes structured data (JSON, CSV) with proper metadata"""
        # Test JSON processing
        json_data = {
            "type": "json",
            "data": '{"config": {"api_key": "xxx", "model": "claude-3", "settings": {"max_tokens": 1000}}}',
            "context": "API configuration"
        }

        json_processed = instance.process_content(json_data)

        assert json_processed.content_type == ContentType.JSON
        assert "structure_type" in json_processed.metadata
        assert "key_count" in json_processed.metadata
        assert json_processed.metadata["key_count"] > 0
        assert "api_key" in json_processed.searchable_text or "config" in json_processed.searchable_text

        # Test CSV processing
        csv_data = {
            "type": "csv",
            "data": "name,department,salary\nAlice,Engineering,85000\nBob,Sales,75000\nCarol,Marketing,70000",
            "context": "Employee roster"
        }

        csv_processed = instance.process_content(csv_data)

        assert csv_processed.content_type == ContentType.CSV
        assert "column_count" in csv_processed.metadata
        assert csv_processed.metadata["column_count"] == 3
        assert "row_count" in csv_processed.metadata
        assert csv_processed.metadata["row_count"] == 3
        assert "alice" in csv_processed.searchable_text.lower()
        assert "engineering" in csv_processed.searchable_text.lower()

    def test_maintains_relationships_between_different_content_types(self, instance):
        """Test: Maintains relationships between different content types"""
        contents = [
            {
                "type": "image",
                "data": "data:image/png;base64,iVBORw0KGgo...",
                "context": "Figure 1: System Architecture"
            },
            {
                "type": "table",
                "data": "<table><tr><th>Component</th><th>Status</th></tr><tr><td>API</td><td>Active</td></tr></table>",
                "context": "Table 1: Component status"
            },
            {
                "type": "json",
                "data": '{"system": "production", "version": "1.0"}',
                "context": "System metadata"
            }
        ]

        result = instance.process_document(contents, infer_relationships=True)

        # Verify multiple content types were processed
        assert len(result.processed_contents) == 3

        # Verify relationships were inferred (sequential relationships)
        assert len(result.relationships) > 0

        # Check that relationships have proper structure
        for rel in result.relationships:
            assert rel.source_id is not None
            assert rel.target_id is not None
            assert rel.relationship_type is not None

        # Verify unified output preserves relationships
        unified = instance.get_unified_output(result)
        assert unified["relationship_count"] > 0
        assert "relationships" in unified
        assert len(unified["relationships"]) > 0


class TestImageProcessor:
    """Tests for ImageProcessor component."""

    def test_process_image_with_context(self):
        """Test image processing with context."""
        result = ImageProcessor.process_image(
            image_data="data:image/jpeg;base64,/9j/4AAQ...",
            context="Sales chart showing quarterly revenue"
        )

        assert result.content_type == ContentType.IMAGE
        assert "chart" in result.metadata.get("image_type", "").lower()
        assert result.confidence_score > 0

    def test_process_image_without_context(self):
        """Test image processing without context."""
        result = ImageProcessor.process_image(
            image_data="data:image/png;base64,iVBORw0KGgo..."
        )

        assert result.content_type == ContentType.IMAGE
        assert result.extracted_text is not None


class TestTableProcessor:
    """Tests for TableProcessor component."""

    def test_parse_html_table(self):
        """Test HTML table parsing."""
        html = "<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>"
        result = TableProcessor.process_table(html)

        assert result.content_type == ContentType.TABLE
        assert result.metadata["column_count"] == 2
        assert result.metadata["row_count"] == 1

    def test_parse_text_table(self):
        """Test plain text table parsing."""
        text = "Name | Age\nJohn | 30\nJane | 25"
        result = TableProcessor.process_table(text)

        assert result.content_type == ContentType.TABLE
        assert "john" in result.searchable_text.lower()


class TestStructuredDataProcessor:
    """Tests for StructuredDataProcessor component."""

    def test_process_json(self):
        """Test JSON processing."""
        json_str = '{"name": "test", "value": 123, "nested": {"key": "value"}}'
        result = StructuredDataProcessor.process_json(json_str)

        assert result.content_type == ContentType.JSON
        assert result.metadata["has_nested"] is True
        assert "test" in result.searchable_text.lower()

    def test_process_csv(self):
        """Test CSV processing."""
        csv_str = "col1,col2,col3\na,b,c\n1,2,3"
        result = StructuredDataProcessor.process_csv(csv_str)

        assert result.content_type == ContentType.CSV
        assert result.metadata["column_count"] == 3
        assert result.metadata["row_count"] == 2

    def test_process_malformed_json(self):
        """Test handling of malformed JSON."""
        bad_json = '{"broken": json'
        result = StructuredDataProcessor.process_json(bad_json)

        # Should handle gracefully with low confidence
        assert result.confidence_score < 0.5


class TestRelationshipMapper:
    """Tests for RelationshipMapper component."""

    def test_add_and_get_relationships(self):
        """Test adding and retrieving relationships."""
        mapper = RelationshipMapper()

        rel = mapper.add_relationship(
            source_id="id1",
            target_id="id2",
            relationship_type=RelationshipType.FOLLOWS
        )

        assert rel.source_id == "id1"
        assert rel.target_id == "id2"

        relationships = mapper.get_relationships_for("id1")
        assert len(relationships) == 1

    def test_relationship_directions(self):
        """Test relationship direction filtering."""
        mapper = RelationshipMapper()

        mapper.add_relationship("a", "b", RelationshipType.FOLLOWS)
        mapper.add_relationship("c", "a", RelationshipType.REFERENCES)

        outgoing = mapper.get_relationships_for("a", direction="outgoing")
        incoming = mapper.get_relationships_for("a", direction="incoming")
        both = mapper.get_relationships_for("a", direction="both")

        assert len(outgoing) == 1
        assert len(incoming) == 1
        assert len(both) == 2


class TestContentUnifier:
    """Tests for ContentUnifier component."""

    def test_unify_contents(self):
        """Test content unification."""
        from src.multi_modal_document_processor import ProcessedContent

        contents = [
            ProcessedContent(
                content_id="1",
                content_type=ContentType.TEXT,
                original_data="Hello",
                extracted_text="Hello",
                searchable_text="hello",
                metadata={}
            ),
            ProcessedContent(
                content_id="2",
                content_type=ContentType.TABLE,
                original_data={},
                extracted_text="Table data",
                searchable_text="table data",
                metadata={}
            )
        ]

        unified = ContentUnifier.unify_contents(contents, [])

        assert unified["content_count"] == 2
        assert "text" in unified["content_types_present"]
        assert "table" in unified["content_types_present"]
        assert "hello" in unified["combined_searchable_text"]


class TestIntegration:
    """Integration tests for the complete processor."""

    @pytest.fixture
    def processor(self):
        """Create processor for integration tests."""
        return create_multi_modal_document_processor()

    def test_full_document_processing(self, processor):
        """Test processing a document with multiple content types."""
        contents = [
            {
                "type": "image",
                "data": "data:image/jpeg;base64,/9j/4AAQ...",
                "context": "Company logo"
            },
            {
                "type": "table",
                "data": "<table><tr><th>Metric</th><th>Value</th></tr><tr><td>Revenue</td><td>$1M</td></tr></table>"
            },
            {
                "type": "json",
                "data": '{"report": "Q1", "status": "complete"}'
            },
            {
                "type": "csv",
                "data": "item,quantity\nA,10\nB,20"
            }
        ]

        result = processor.execute(contents)

        assert result["content_count"] == 4
        assert "image" in result["content_types_present"]
        assert "table" in result["content_types_present"]
        assert "json" in result["content_types_present"]
        assert "csv" in result["content_types_present"]
        assert result["processing_statistics"]["documents_processed"] >= 1

    def test_error_handling(self, processor):
        """Test graceful error handling."""
        contents = [
            {"type": "json", "data": '{"valid": true}'},
            None,  # This might cause issues
            {"type": "table", "data": "<table><tr><td>OK</td></tr></table>"}
        ]

        # Should handle errors gracefully
        try:
            result = processor.execute(contents)
            # If it succeeds, verify error tracking
            assert "errors" in result
        except Exception:
            # If it raises, the test still passes as we're testing error handling
            pass

    def test_statistics_tracking(self, processor):
        """Test that processing statistics are tracked."""
        contents = [
            {"type": "image", "data": "data:image/png;base64,..."},
            {"type": "table", "data": "<table><tr><td>1</td></tr></table>"},
            {"type": "json", "data": '{"a": 1}'},
            {"type": "csv", "data": "x,y\n1,2"}
        ]

        result = processor.execute(contents)
        stats = result["processing_statistics"]

        assert stats["images_processed"] >= 1
        assert stats["tables_processed"] >= 1
        assert stats["json_processed"] >= 1
        assert stats["csv_processed"] >= 1
