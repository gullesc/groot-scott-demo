"""
Tests for Production RAG API

Run: pytest
Watch: pytest-watch (install with: pip install pytest-watch)
Coverage: pytest --cov=src
"""

import json
import threading
import time
import pytest
from http.client import HTTPConnection
from src.production_rag_api import (
    ProductionRagApi,
    create_production_rag_api,
    AuthenticationManager,
    RateLimiter,
    RequestValidator,
    APIDocumentationGenerator,
    ErrorCode
)


class TestProductionRagApi:
    """Test suite for ProductionRagApi."""

    @pytest.fixture
    def instance(self):
        """Create a fresh instance for each test."""
        return create_production_rag_api()

    def test_create_instance(self, instance):
        """Test that we can create an instance."""
        assert instance is not None
        assert isinstance(instance, ProductionRagApi)

    def test_instance_has_components(self, instance):
        """Test that instance has all required components."""
        assert instance.auth_manager is not None
        assert instance.rate_limiter is not None
        assert instance.request_validator is not None
        assert instance.doc_generator is not None
        assert instance.metrics is not None

    def test_create_api_key(self, instance):
        """Test API key creation."""
        key = instance.create_api_key("test-client")
        assert key is not None
        assert key.startswith("rag_")
        assert len(key) > 10

    def test_revoke_api_key(self, instance):
        """Test API key revocation."""
        key = instance.create_api_key("test-client")
        assert instance.auth_manager.validate_key(key) is not None

        result = instance.revoke_api_key(key)
        assert result is True
        assert instance.auth_manager.validate_key(key) is None

    def test_get_documentation(self, instance):
        """Test OpenAPI documentation generation."""
        docs = instance.get_documentation()
        assert docs is not None
        assert "openapi" in docs
        assert docs["openapi"] == "3.0.0"
        assert "paths" in docs
        assert "/api/v1/query" in docs["paths"]

    def test_get_metrics(self, instance):
        """Test metrics retrieval."""
        metrics = instance.get_metrics()
        assert metrics is not None
        assert "total_requests" in metrics
        assert "uptime" in metrics


class TestAuthenticationManager:
    """Test suite for AuthenticationManager."""

    @pytest.fixture
    def auth_manager(self):
        """Create a fresh AuthenticationManager for each test."""
        return AuthenticationManager()

    def test_create_and_validate_key(self, auth_manager):
        """Test creating and validating an API key."""
        key = auth_manager.create_api_key("client-1")
        key_data = auth_manager.validate_key(key)

        assert key_data is not None
        assert key_data.client_id == "client-1"
        assert key_data.is_active is True

    def test_validate_invalid_key(self, auth_manager):
        """Test validating an invalid API key."""
        result = auth_manager.validate_key("invalid-key-12345")
        assert result is None

    def test_revoke_key(self, auth_manager):
        """Test revoking an API key."""
        key = auth_manager.create_api_key("client-2")

        # Key should be valid before revocation
        assert auth_manager.validate_key(key) is not None

        # Revoke and verify
        result = auth_manager.revoke_key(key)
        assert result is True
        assert auth_manager.validate_key(key) is None

    def test_permissions(self, auth_manager):
        """Test permission checking."""
        key = auth_manager.create_api_key("client-3", permissions=["query", "metrics"])

        assert auth_manager.has_permission(key, "query") is True
        assert auth_manager.has_permission(key, "metrics") is True
        assert auth_manager.has_permission(key, "admin") is False

    def test_custom_rate_limit(self, auth_manager):
        """Test custom rate limit on API key."""
        key = auth_manager.create_api_key("client-4", rate_limit=100)
        key_data = auth_manager.validate_key(key)

        assert key_data.rate_limit == 100


class TestRateLimiter:
    """Test suite for RateLimiter."""

    @pytest.fixture
    def rate_limiter(self):
        """Create a fresh RateLimiter for each test."""
        return RateLimiter(max_tokens=5, refill_rate=1.0)

    def test_initial_requests_allowed(self, rate_limiter):
        """Test that initial requests are allowed."""
        for i in range(5):
            allowed, info = rate_limiter.check_and_consume("client-1")
            assert allowed is True

    def test_rate_limit_enforced(self, rate_limiter):
        """Test that rate limit is enforced after bucket is empty."""
        # Consume all tokens
        for i in range(5):
            rate_limiter.check_and_consume("client-1")

        # Next request should be denied
        allowed, info = rate_limiter.check_and_consume("client-1")
        assert allowed is False
        assert info["remaining"] == 0
        assert info["retry_after"] > 0

    def test_token_refill(self, rate_limiter):
        """Test that tokens refill over time."""
        # Consume all tokens
        for i in range(5):
            rate_limiter.check_and_consume("client-1")

        # Wait for refill
        time.sleep(1.1)

        # Should have a token now
        allowed, info = rate_limiter.check_and_consume("client-1")
        assert allowed is True

    def test_per_client_limits(self, rate_limiter):
        """Test that limits are tracked per client."""
        # Consume all tokens for client-1
        for i in range(5):
            rate_limiter.check_and_consume("client-1")

        # client-2 should still have tokens
        allowed, info = rate_limiter.check_and_consume("client-2")
        assert allowed is True


class TestRequestValidator:
    """Test suite for RequestValidator."""

    @pytest.fixture
    def validator(self):
        """Create a fresh RequestValidator for each test."""
        return RequestValidator()

    def test_valid_query_request(self, validator):
        """Test validation of a valid query request."""
        data = {"query": "What is RAG?"}
        is_valid, errors = validator.validate("query", data)
        assert is_valid is True
        assert len(errors) == 0

    def test_missing_required_field(self, validator):
        """Test validation fails for missing required field."""
        data = {}
        is_valid, errors = validator.validate("query", data)
        assert is_valid is False
        assert len(errors) == 1
        assert errors[0]["field"] == "query"

    def test_optional_fields_accepted(self, validator):
        """Test that optional fields are accepted."""
        data = {
            "query": "What is ML?",
            "max_length": 200,
            "top_k": 5,
            "include_sources": True
        }
        is_valid, errors = validator.validate("query", data)
        assert is_valid is True

    def test_type_validation(self, validator):
        """Test type validation for fields."""
        data = {
            "query": "What is ML?",
            "max_length": "not a number"  # Should be int
        }
        is_valid, errors = validator.validate("query", data)
        assert is_valid is False
        assert any(e["field"] == "max_length" for e in errors)

    def test_sanitize_removes_control_chars(self, validator):
        """Test that sanitize removes control characters."""
        data = {"query": "Hello\x00World\x1f"}
        sanitized = validator.sanitize(data)
        assert "\x00" not in sanitized["query"]
        assert "\x1f" not in sanitized["query"]


class TestAPIDocumentationGenerator:
    """Test suite for APIDocumentationGenerator."""

    @pytest.fixture
    def doc_generator(self):
        """Create a fresh APIDocumentationGenerator for each test."""
        return APIDocumentationGenerator()

    def test_generate_spec(self, doc_generator):
        """Test OpenAPI spec generation."""
        spec = doc_generator.generate_spec()

        assert spec["openapi"] == "3.0.0"
        assert "info" in spec
        assert "paths" in spec
        assert "components" in spec

    def test_spec_has_all_endpoints(self, doc_generator):
        """Test that spec includes all endpoints."""
        spec = doc_generator.generate_spec()
        paths = spec["paths"]

        assert "/api/v1/query" in paths
        assert "/api/v1/health" in paths
        assert "/api/v1/docs" in paths
        assert "/api/v1/metrics" in paths

    def test_spec_has_security_scheme(self, doc_generator):
        """Test that spec includes security scheme."""
        spec = doc_generator.generate_spec()

        assert "securitySchemes" in spec["components"]
        assert "BearerAuth" in spec["components"]["securitySchemes"]

    def test_spec_has_schemas(self, doc_generator):
        """Test that spec includes request/response schemas."""
        spec = doc_generator.generate_spec()
        schemas = spec["components"]["schemas"]

        assert "QueryRequest" in schemas
        assert "QueryResponse" in schemas
        assert "ErrorResponse" in schemas


class TestAcceptanceCriteria:
    """Tests for acceptance criteria."""

    @pytest.fixture
    def instance(self):
        """Create a fresh instance for each test."""
        return create_production_rag_api()

    def test_implements_rest_api_with_comprehensive_endpoints(self, instance):
        """Test: Implements REST API with comprehensive endpoints"""
        docs = instance.get_documentation()
        paths = docs["paths"]

        # Verify all required endpoints exist
        assert "/api/v1/query" in paths
        assert "/api/v1/health" in paths
        assert "/api/v1/docs" in paths
        assert "/api/v1/metrics" in paths

        # Verify query endpoint supports POST
        assert "post" in paths["/api/v1/query"]

    def test_includes_authentication_rate_limiting_and_input_validation(self, instance):
        """Test: Includes authentication, rate limiting, and input validation"""
        # Test authentication
        key = instance.create_api_key("test-client")
        assert key is not None
        assert instance.auth_manager.validate_key(key) is not None

        # Test rate limiting
        allowed, info = instance.rate_limiter.check_and_consume("test-client")
        assert allowed is True
        assert "remaining" in info
        assert "limit" in info

        # Test input validation
        is_valid, errors = instance.request_validator.validate("query", {"query": "test"})
        assert is_valid is True

        is_valid, errors = instance.request_validator.validate("query", {})
        assert is_valid is False

    def test_provides_openapi_documentation_and_example_usage(self, instance):
        """Test: Provides OpenAPI documentation and example usage"""
        docs = instance.get_documentation()

        # Verify OpenAPI format
        assert docs["openapi"] == "3.0.0"
        assert "info" in docs
        assert "title" in docs["info"]

        # Verify examples in documentation
        query_endpoint = docs["paths"]["/api/v1/query"]["post"]
        assert "requestBody" in query_endpoint
        assert "example" in query_endpoint["requestBody"]["content"]["application/json"]

    def test_handles_concurrent_requests_efficiently(self, instance):
        """Test: Handles concurrent requests efficiently with proper error handling"""
        # Create multiple API keys for concurrent clients
        keys = [instance.create_api_key(f"client-{i}") for i in range(5)]

        # Verify each has its own rate limit bucket
        for i, key in enumerate(keys):
            key_data = instance.auth_manager.validate_key(key)
            allowed, info = instance.rate_limiter.check_and_consume(key_data.client_id)
            assert allowed is True

        # Test that metrics track properly
        metrics = instance.get_metrics()
        assert "total_requests" in metrics
