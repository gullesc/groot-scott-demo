"""
Tests for Citation and Source Tracking System

Run: pytest
Watch: pytest-watch (install with: pip install pytest-watch)
Coverage: pytest --cov=src
"""

import pytest
from src.citation_and_source_tracking_system import (
    CitationAndSourceTrackingSystem,
    create_citation_and_source_tracking_system,
    CitationFormat,
    SourceDocument,
    TextSpan,
    Citation,
    ValidationResult,
    SourceManager,
    AcademicFormatter,
    WebFormatter,
    InlineFormatter,
    TextSpanMapper,
    ValidationEngine,
    LinkGenerator,
)


class TestCitationAndSourceTrackingSystem:
    """Test suite for CitationAndSourceTrackingSystem."""

    @pytest.fixture
    def instance(self):
        """Create a fresh instance for each test."""
        return create_citation_and_source_tracking_system()

    def test_create_instance(self, instance):
        """Test that we can create an instance."""
        assert instance is not None
        assert isinstance(instance, CitationAndSourceTrackingSystem)

    def test_execute_returns_dict(self, instance):
        """Test that execute returns a dictionary with expected keys."""
        result = instance.execute(
            sources=[
                {
                    "content": "Test content about a topic.",
                    "url": "https://example.com/test",
                    "title": "Test Document"
                }
            ],
            generated_text="This is a test response.",
            citation_format="inline"
        )
        assert isinstance(result, dict)
        assert "formatted_text" in result
        assert "reference_list" in result
        assert "citations" in result
        assert "validation_results" in result
        assert "text_mappings" in result
        assert "clickable_links" in result


class TestAcceptanceCriteria:
    """Tests for acceptance criteria."""

    @pytest.fixture
    def instance(self):
        """Create a fresh instance for each test."""
        return create_citation_and_source_tracking_system()

    @pytest.fixture
    def sample_sources(self):
        """Sample source documents for testing."""
        return [
            {
                "content": "Paris is the capital city of France. It is known for the Eiffel Tower.",
                "url": "https://encyclopedia.com/france",
                "title": "France Overview",
                "author": "John Smith",
                "publication_date": "2024"
            },
            {
                "content": "The Eiffel Tower was completed in 1889. It stands 330 meters tall.",
                "url": "https://history.com/eiffel",
                "title": "Eiffel Tower History",
                "author": "Jane Doe",
                "publication_date": "2023"
            }
        ]

    def test_maintains_detailed_mapping_between_generated_text(self, instance, sample_sources):
        """Test: Maintains detailed mapping between generated text and source chunks"""
        text_mappings = [
            {
                "text": "Paris is the capital of France",
                "start_position": 0,
                "source_ids": ["source_1"],
                "confidence": 0.95
            },
            {
                "text": "The Eiffel Tower is 330 meters tall",
                "start_position": 35,
                "source_ids": ["source_2"],
                "confidence": 0.9
            }
        ]

        result = instance.execute(
            sources=sample_sources,
            generated_text="Paris is the capital of France. The Eiffel Tower is 330 meters tall.",
            text_source_mappings=text_mappings
        )

        # Verify mappings are recorded
        assert len(result["text_mappings"]) == 2

        # Verify mapping details
        mapping1 = result["text_mappings"][0]
        assert mapping1["text"] == "Paris is the capital of France"
        assert mapping1["start_position"] == 0
        assert "source_1" in mapping1["source_ids"]
        assert mapping1["confidence"] == 0.95

        mapping2 = result["text_mappings"][1]
        assert mapping2["text"] == "The Eiffel Tower is 330 meters tall"
        assert mapping2["start_position"] == 35

    def test_implements_multiple_citation_formats_academic_we(self, instance, sample_sources):
        """Test: Implements multiple citation formats (academic, web, etc.)"""
        # Test academic format
        academic_result = instance.execute(
            sources=sample_sources,
            citation_format="academic"
        )
        # Academic format should include author and date
        assert len(academic_result["citations"]) > 0
        academic_citation = academic_result["citations"][0]
        assert "John Smith" in academic_citation["formatted_text"] or "[1]" in academic_citation["formatted_text"]

        # Test web format
        instance_web = create_citation_and_source_tracking_system()
        web_result = instance_web.execute(
            sources=sample_sources,
            citation_format="web"
        )
        assert len(web_result["citations"]) > 0
        # Web format should include URL in markdown link format
        web_citation = web_result["citations"][0]
        assert "encyclopedia.com" in web_citation["formatted_text"] or "France Overview" in web_citation["formatted_text"]

        # Test inline format
        instance_inline = create_citation_and_source_tracking_system()
        inline_result = instance_inline.execute(
            sources=sample_sources,
            citation_format="inline"
        )
        assert len(inline_result["citations"]) > 0
        inline_citation = inline_result["citations"][0]
        assert "Source" in inline_citation["formatted_text"] or "[" in inline_citation["inline_marker"]

    def test_provides_clickable_source_links_in_responses(self, instance, sample_sources):
        """Test: Provides clickable source links in responses"""
        # Test markdown links
        result_md = instance.execute(
            sources=sample_sources,
            output_format="markdown"
        )

        # Check clickable links are generated
        assert len(result_md["clickable_links"]) > 0

        # Markdown links should contain [title](url) format
        for source_id, link in result_md["clickable_links"].items():
            if "encyclopedia.com" in link:
                assert "[" in link and "](" in link and ")" in link

        # Test HTML links
        instance_html = create_citation_and_source_tracking_system()
        result_html = instance_html.execute(
            sources=sample_sources,
            output_format="html"
        )

        # HTML links should contain <a href> format
        for source_id, link in result_html["clickable_links"].items():
            if "encyclopedia.com" in link or "history.com" in link:
                assert "<a href=" in link and "</a>" in link

    def test_validates_that_citations_actually_support_the_gene(self, instance, sample_sources):
        """Test: Validates that citations actually support the generated claims"""
        # Test with a claim that IS supported by the source
        text_mappings = [
            {
                "text": "Paris is the capital city of France",
                "start_position": 0,
                "source_ids": ["source_1"],
                "confidence": 0.95
            }
        ]

        result = instance.execute(
            sources=sample_sources,
            generated_text="Paris is the capital city of France",
            text_source_mappings=text_mappings,
            validate_citations=True
        )

        # Should have validation results
        assert len(result["validation_results"]) > 0

        # The claim about Paris should be supported
        validation = result["validation_results"][0]
        assert validation["is_supported"] is True
        assert validation["support_score"] > 0.3
        assert "Paris" in validation["matched_keywords"] or "capital" in validation["matched_keywords"]

        # Test with a claim that is NOT supported
        instance2 = create_citation_and_source_tracking_system()
        unsupported_mappings = [
            {
                "text": "The moon is made of cheese",
                "start_position": 0,
                "source_ids": ["source_1"],
                "confidence": 0.5
            }
        ]

        result2 = instance2.execute(
            sources=sample_sources,
            generated_text="The moon is made of cheese",
            text_source_mappings=unsupported_mappings,
            validate_citations=True
        )

        # Should have validation result indicating lack of support
        assert len(result2["validation_results"]) > 0
        validation2 = result2["validation_results"][0]
        assert validation2["is_supported"] is False
        assert len(validation2["warnings"]) > 0


class TestSourceManager:
    """Tests for SourceManager class."""

    def test_register_and_retrieve_source(self):
        """Test source registration and retrieval."""
        manager = SourceManager()

        source_id = manager.register_source(
            content="Test content",
            url="https://test.com",
            title="Test Title",
            author="Test Author"
        )

        source = manager.get_source(source_id)
        assert source is not None
        assert source.content == "Test content"
        assert source.url == "https://test.com"
        assert source.title == "Test Title"
        assert source.author == "Test Author"

    def test_get_all_sources(self):
        """Test retrieving all sources."""
        manager = SourceManager()

        manager.register_source(content="Content 1", title="Title 1")
        manager.register_source(content="Content 2", title="Title 2")

        all_sources = manager.get_all_sources()
        assert len(all_sources) == 2

    def test_clear_sources(self):
        """Test clearing sources."""
        manager = SourceManager()

        manager.register_source(content="Content 1")
        manager.clear_sources()

        assert len(manager.get_all_sources()) == 0


class TestCitationFormatters:
    """Tests for citation formatter classes."""

    @pytest.fixture
    def sample_source(self):
        """Sample source document."""
        return SourceDocument(
            source_id="test_source",
            content="Test content",
            url="https://test.com/page",
            title="Test Document Title",
            author="John Doe",
            publication_date="2024"
        )

    def test_academic_formatter(self, sample_source):
        """Test academic citation formatting."""
        formatted = AcademicFormatter.format(sample_source, 1)

        assert "[1]" in formatted
        assert "John Doe" in formatted
        assert "2024" in formatted
        assert "Test Document Title" in formatted

    def test_web_formatter_markdown(self, sample_source):
        """Test web citation formatting for markdown."""
        formatted = WebFormatter.format(sample_source, 1)

        assert "[1]" in formatted
        assert "Test Document Title" in formatted
        assert "https://test.com/page" in formatted

    def test_web_formatter_html(self, sample_source):
        """Test web citation formatting for HTML."""
        formatted = WebFormatter.format_html(sample_source, 1)

        assert "[1]" in formatted
        assert "<a href=" in formatted
        assert "Test Document Title" in formatted

    def test_inline_formatter(self, sample_source):
        """Test inline citation formatting."""
        formatted = InlineFormatter.format(sample_source, 1)

        assert "Source" in formatted
        assert "Test Document Title" in formatted


class TestTextSpanMapper:
    """Tests for TextSpanMapper class."""

    def test_add_and_get_mappings(self):
        """Test adding and retrieving mappings."""
        mapper = TextSpanMapper()

        span = mapper.add_mapping(
            text="Test text",
            start_position=0,
            source_ids=["source_1", "source_2"],
            confidence=0.9
        )

        mappings = mapper.get_mappings()
        assert len(mappings) == 1
        assert mappings[0].text == "Test text"
        assert mappings[0].start_position == 0
        assert mappings[0].end_position == 9  # len("Test text")
        assert "source_1" in mappings[0].source_ids

    def test_get_mappings_for_source(self):
        """Test getting mappings for specific source."""
        mapper = TextSpanMapper()

        mapper.add_mapping("Text 1", 0, ["source_1"])
        mapper.add_mapping("Text 2", 10, ["source_2"])
        mapper.add_mapping("Text 3", 20, ["source_1", "source_2"])

        source_1_mappings = mapper.get_mappings_for_source("source_1")
        assert len(source_1_mappings) == 2

    def test_get_sources_for_position(self):
        """Test getting sources at a specific position."""
        mapper = TextSpanMapper()

        mapper.add_mapping("First", 0, ["source_1"])
        mapper.add_mapping("Second", 10, ["source_2"])

        sources_at_5 = mapper.get_sources_for_position(2)
        assert "source_1" in sources_at_5

        sources_at_12 = mapper.get_sources_for_position(12)
        assert "source_2" in sources_at_12


class TestValidationEngine:
    """Tests for ValidationEngine class."""

    def test_validate_supported_claim(self):
        """Test validation of a supported claim."""
        source = SourceDocument(
            source_id="test",
            content="The capital of France is Paris, which is known for the Eiffel Tower."
        )

        result = ValidationEngine.validate_claim(
            "Paris is the capital of France",
            source
        )

        assert result.is_supported is True
        assert result.support_score > 0.3
        assert len(result.matched_keywords) > 0

    def test_validate_unsupported_claim(self):
        """Test validation of an unsupported claim."""
        source = SourceDocument(
            source_id="test",
            content="The capital of France is Paris."
        )

        result = ValidationEngine.validate_claim(
            "The moon is made of green cheese and inhabited by aliens.",
            source
        )

        assert result.is_supported is False
        assert result.support_score < 0.3
        assert len(result.warnings) > 0


class TestLinkGenerator:
    """Tests for LinkGenerator class."""

    @pytest.fixture
    def sample_source(self):
        """Sample source document."""
        return SourceDocument(
            source_id="test",
            content="Content",
            url="https://example.com/page",
            title="Example Page"
        )

    def test_generate_markdown_link(self, sample_source):
        """Test markdown link generation."""
        link = LinkGenerator.generate_markdown_link(sample_source)

        assert "[Example Page]" in link
        assert "(https://example.com/page)" in link

    def test_generate_html_link(self, sample_source):
        """Test HTML link generation."""
        link = LinkGenerator.generate_html_link(sample_source)

        assert '<a href="https://example.com/page"' in link
        assert "Example Page" in link
        assert "</a>" in link

    def test_generate_link_without_url(self):
        """Test link generation when URL is missing."""
        source = SourceDocument(
            source_id="test",
            content="Content",
            title="Title Only"
        )

        md_link = LinkGenerator.generate_markdown_link(source)
        html_link = LinkGenerator.generate_html_link(source)

        assert md_link == "Title Only"
        assert html_link == "Title Only"
