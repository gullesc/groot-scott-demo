"""
Production RAG API

Deploy your RAG system as a scalable REST API with proper authentication and documentation.
Uses Python's built-in http.server module to implement a production-ready API server
with authentication, rate limiting, input validation, and OpenAPI documentation.

This module demonstrates enterprise-grade API patterns while maintaining educational value
and compatibility with standard library constraints.
"""

import json
import hashlib
import logging
import secrets
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse


# Configure logging for structured output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ErrorCode(Enum):
    """Standard error codes for API responses following RFC 7807."""
    INVALID_REQUEST = "INVALID_REQUEST"
    AUTHENTICATION_REQUIRED = "AUTHENTICATION_REQUIRED"
    INVALID_API_KEY = "INVALID_API_KEY"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    NOT_FOUND = "NOT_FOUND"
    METHOD_NOT_ALLOWED = "METHOD_NOT_ALLOWED"


@dataclass
class APIKey:
    """Represents an API key with associated metadata and permissions."""
    key_hash: str  # We store hash, not the raw key for security
    client_id: str
    created_at: datetime
    is_active: bool = True
    rate_limit: int = 60  # Requests per minute
    permissions: List[str] = field(default_factory=lambda: ["query", "health"])


@dataclass
class RateLimitEntry:
    """Token bucket rate limit tracking entry."""
    tokens: float
    last_update: float
    max_tokens: int
    refill_rate: float  # Tokens per second


class AuthenticationManager:
    """
    Manages API key authentication for the RAG API.

    Uses secure hashing to store API keys and provides methods for
    key validation, creation, and management. Keys are stored in memory
    for demonstration purposes but could be backed by a persistent store.

    Security Features:
    - API keys are hashed using SHA-256 before storage
    - Keys are never logged or exposed in responses
    - Supports key rotation and revocation
    """

    def __init__(self):
        """Initialize the authentication manager with empty key store."""
        self._keys: Dict[str, APIKey] = {}  # Keyed by key_hash
        self._lock = threading.Lock()

    def _hash_key(self, api_key: str) -> str:
        """
        Securely hash an API key for storage and comparison.

        Args:
            api_key: The raw API key string

        Returns:
            SHA-256 hash of the API key
        """
        return hashlib.sha256(api_key.encode()).hexdigest()

    def create_api_key(self, client_id: str, rate_limit: int = 60,
                       permissions: Optional[List[str]] = None) -> str:
        """
        Create a new API key for a client.

        Args:
            client_id: Unique identifier for the client
            rate_limit: Maximum requests per minute (default: 60)
            permissions: List of allowed operations (default: ["query", "health"])

        Returns:
            The raw API key (only returned once, store securely!)
        """
        raw_key = f"rag_{secrets.token_urlsafe(32)}"
        key_hash = self._hash_key(raw_key)

        api_key = APIKey(
            key_hash=key_hash,
            client_id=client_id,
            created_at=datetime.now(),
            rate_limit=rate_limit,
            permissions=permissions or ["query", "health"]
        )

        with self._lock:
            self._keys[key_hash] = api_key

        logger.info(f"Created API key for client: {client_id}")
        return raw_key

    def validate_key(self, api_key: str) -> Optional[APIKey]:
        """
        Validate an API key and return its metadata.

        Args:
            api_key: The raw API key to validate

        Returns:
            APIKey object if valid and active, None otherwise
        """
        key_hash = self._hash_key(api_key)

        with self._lock:
            key_data = self._keys.get(key_hash)

        if key_data and key_data.is_active:
            return key_data
        return None

    def revoke_key(self, api_key: str) -> bool:
        """
        Revoke an API key, making it no longer valid.

        Args:
            api_key: The raw API key to revoke

        Returns:
            True if key was found and revoked, False otherwise
        """
        key_hash = self._hash_key(api_key)

        with self._lock:
            if key_hash in self._keys:
                self._keys[key_hash].is_active = False
                logger.info(f"Revoked API key for client: {self._keys[key_hash].client_id}")
                return True
        return False

    def has_permission(self, api_key: str, permission: str) -> bool:
        """
        Check if an API key has a specific permission.

        Args:
            api_key: The raw API key
            permission: The permission to check

        Returns:
            True if key has the permission, False otherwise
        """
        key_data = self.validate_key(api_key)
        if key_data:
            return permission in key_data.permissions
        return False


class RateLimiter:
    """
    Token bucket rate limiter for API request throttling.

    Implements the token bucket algorithm which allows bursting up to the bucket
    capacity while maintaining a steady average rate. This is more flexible than
    a simple counter-based approach and provides better user experience.

    Algorithm:
    - Each client has a bucket that can hold max_tokens
    - Tokens are added at refill_rate per second
    - Each request consumes one token
    - If no tokens available, request is rejected
    """

    def __init__(self, max_tokens: int = 60, refill_rate: float = 1.0):
        """
        Initialize the rate limiter.

        Args:
            max_tokens: Maximum tokens in bucket (burst capacity)
            refill_rate: Tokens added per second (sustained rate)
        """
        self._buckets: Dict[str, RateLimitEntry] = {}
        self._lock = threading.Lock()
        self.default_max_tokens = max_tokens
        self.default_refill_rate = refill_rate

    def _get_bucket(self, client_id: str, max_tokens: Optional[int] = None) -> RateLimitEntry:
        """Get or create a rate limit bucket for a client."""
        current_time = time.time()
        max_tokens = max_tokens or self.default_max_tokens

        with self._lock:
            if client_id not in self._buckets:
                self._buckets[client_id] = RateLimitEntry(
                    tokens=float(max_tokens),
                    last_update=current_time,
                    max_tokens=max_tokens,
                    refill_rate=self.default_refill_rate
                )
            return self._buckets[client_id]

    def check_and_consume(self, client_id: str, max_tokens: Optional[int] = None) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if a request is allowed and consume a token if so.

        Args:
            client_id: Unique identifier for the client
            max_tokens: Optional custom token limit for this client

        Returns:
            Tuple of (allowed: bool, rate_limit_info: dict)
        """
        current_time = time.time()
        bucket = self._get_bucket(client_id, max_tokens)

        with self._lock:
            # Refill tokens based on time elapsed
            elapsed = current_time - bucket.last_update
            bucket.tokens = min(
                bucket.max_tokens,
                bucket.tokens + (elapsed * bucket.refill_rate)
            )
            bucket.last_update = current_time

            # Check if we have a token available
            if bucket.tokens >= 1.0:
                bucket.tokens -= 1.0
                allowed = True
            else:
                allowed = False

            # Calculate retry after (time until 1 token is available)
            retry_after = max(0, (1.0 - bucket.tokens) / bucket.refill_rate) if not allowed else 0

            return allowed, {
                "remaining": int(bucket.tokens),
                "limit": bucket.max_tokens,
                "reset": int(current_time + (bucket.max_tokens - bucket.tokens) / bucket.refill_rate),
                "retry_after": int(retry_after) + 1 if retry_after > 0 else 0
            }


class RequestValidator:
    """
    Validates incoming API requests for proper format and security.

    Provides comprehensive validation including:
    - Required field checking
    - Type validation
    - Length constraints
    - Input sanitization for security
    """

    # Schema for query requests
    QUERY_SCHEMA = {
        "query": {"type": str, "required": True, "min_length": 1, "max_length": 10000},
        "max_length": {"type": int, "required": False, "min_value": 1, "max_value": 5000},
        "top_k": {"type": int, "required": False, "min_value": 1, "max_value": 100},
        "filters": {"type": dict, "required": False},
        "include_sources": {"type": bool, "required": False}
    }

    def __init__(self):
        """Initialize the request validator."""
        self._schemas: Dict[str, Dict] = {
            "query": self.QUERY_SCHEMA
        }

    def validate(self, request_type: str, data: Dict[str, Any]) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Validate request data against a schema.

        Args:
            request_type: Type of request (e.g., "query")
            data: Request data to validate

        Returns:
            Tuple of (is_valid: bool, errors: list of error dicts)
        """
        schema = self._schemas.get(request_type)
        if not schema:
            return True, []  # No schema, assume valid

        errors = []

        for field_name, rules in schema.items():
            value = data.get(field_name)

            # Check required fields
            if rules.get("required") and value is None:
                errors.append({
                    "field": field_name,
                    "error": "required",
                    "message": f"Field '{field_name}' is required"
                })
                continue

            if value is None:
                continue  # Optional field not provided

            # Type checking
            expected_type = rules.get("type")
            if expected_type and not isinstance(value, expected_type):
                errors.append({
                    "field": field_name,
                    "error": "invalid_type",
                    "message": f"Field '{field_name}' must be of type {expected_type.__name__}"
                })
                continue

            # String constraints
            if isinstance(value, str):
                if "min_length" in rules and len(value) < rules["min_length"]:
                    errors.append({
                        "field": field_name,
                        "error": "min_length",
                        "message": f"Field '{field_name}' must be at least {rules['min_length']} characters"
                    })
                if "max_length" in rules and len(value) > rules["max_length"]:
                    errors.append({
                        "field": field_name,
                        "error": "max_length",
                        "message": f"Field '{field_name}' must be at most {rules['max_length']} characters"
                    })

            # Numeric constraints
            if isinstance(value, (int, float)):
                if "min_value" in rules and value < rules["min_value"]:
                    errors.append({
                        "field": field_name,
                        "error": "min_value",
                        "message": f"Field '{field_name}' must be at least {rules['min_value']}"
                    })
                if "max_value" in rules and value > rules["max_value"]:
                    errors.append({
                        "field": field_name,
                        "error": "max_value",
                        "message": f"Field '{field_name}' must be at most {rules['max_value']}"
                    })

        return len(errors) == 0, errors

    def sanitize(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize request data to prevent injection attacks.

        Args:
            data: Request data to sanitize

        Returns:
            Sanitized data dictionary
        """
        sanitized = {}
        for key, value in data.items():
            if isinstance(value, str):
                # Basic sanitization - remove control characters
                sanitized[key] = ''.join(char for char in value if ord(char) >= 32 or char in '\n\r\t')
            elif isinstance(value, dict):
                sanitized[key] = self.sanitize(value)
            else:
                sanitized[key] = value
        return sanitized


class APIDocumentationGenerator:
    """
    Generates OpenAPI 3.0 specification for the RAG API.

    Creates comprehensive API documentation including:
    - Endpoint schemas
    - Request/response examples
    - Authentication details
    - Error codes
    """

    def __init__(self, title: str = "Production RAG API", version: str = "1.0.0"):
        """
        Initialize the documentation generator.

        Args:
            title: API title
            version: API version string
        """
        self.title = title
        self.version = version

    def generate_spec(self, base_url: str = "http://localhost:8080") -> Dict[str, Any]:
        """
        Generate complete OpenAPI 3.0 specification.

        Args:
            base_url: Base URL for the API server

        Returns:
            OpenAPI specification dictionary
        """
        return {
            "openapi": "3.0.0",
            "info": {
                "title": self.title,
                "version": self.version,
                "description": "A production-ready REST API for RAG (Retrieval-Augmented Generation) queries"
            },
            "servers": [{"url": base_url}],
            "paths": {
                "/api/v1/query": {
                    "post": {
                        "summary": "Execute RAG Query",
                        "description": "Submit a query to the RAG system and receive generated response with sources",
                        "security": [{"BearerAuth": []}],
                        "requestBody": {
                            "required": True,
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/QueryRequest"},
                                    "example": {
                                        "query": "What are the benefits of RAG systems?",
                                        "max_length": 200,
                                        "include_sources": True
                                    }
                                }
                            }
                        },
                        "responses": {
                            "200": {
                                "description": "Successful query response",
                                "content": {
                                    "application/json": {
                                        "schema": {"$ref": "#/components/schemas/QueryResponse"}
                                    }
                                }
                            },
                            "400": {"description": "Invalid request"},
                            "401": {"description": "Authentication required"},
                            "429": {"description": "Rate limit exceeded"},
                            "500": {"description": "Internal server error"}
                        }
                    }
                },
                "/api/v1/health": {
                    "get": {
                        "summary": "Health Check",
                        "description": "Check API server health and status",
                        "responses": {
                            "200": {
                                "description": "Server is healthy",
                                "content": {
                                    "application/json": {
                                        "schema": {"$ref": "#/components/schemas/HealthResponse"}
                                    }
                                }
                            }
                        }
                    }
                },
                "/api/v1/docs": {
                    "get": {
                        "summary": "API Documentation",
                        "description": "Get OpenAPI specification",
                        "responses": {
                            "200": {
                                "description": "OpenAPI specification",
                                "content": {
                                    "application/json": {
                                        "schema": {"type": "object"}
                                    }
                                }
                            }
                        }
                    }
                },
                "/api/v1/metrics": {
                    "get": {
                        "summary": "API Metrics",
                        "description": "Get API usage metrics and statistics",
                        "security": [{"BearerAuth": []}],
                        "responses": {
                            "200": {
                                "description": "Usage metrics",
                                "content": {
                                    "application/json": {
                                        "schema": {"$ref": "#/components/schemas/MetricsResponse"}
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "components": {
                "securitySchemes": {
                    "BearerAuth": {
                        "type": "http",
                        "scheme": "bearer",
                        "description": "API key authentication using Bearer token"
                    }
                },
                "schemas": {
                    "QueryRequest": {
                        "type": "object",
                        "required": ["query"],
                        "properties": {
                            "query": {"type": "string", "description": "The query text"},
                            "max_length": {"type": "integer", "description": "Maximum response length"},
                            "top_k": {"type": "integer", "description": "Number of sources to retrieve"},
                            "include_sources": {"type": "boolean", "description": "Include source documents"}
                        }
                    },
                    "QueryResponse": {
                        "type": "object",
                        "properties": {
                            "response": {"type": "string"},
                            "sources": {"type": "array", "items": {"type": "object"}},
                            "metadata": {"type": "object"},
                            "request_id": {"type": "string"}
                        }
                    },
                    "HealthResponse": {
                        "type": "object",
                        "properties": {
                            "status": {"type": "string"},
                            "uptime": {"type": "number"},
                            "version": {"type": "string"}
                        }
                    },
                    "MetricsResponse": {
                        "type": "object",
                        "properties": {
                            "total_requests": {"type": "integer"},
                            "average_response_time": {"type": "number"},
                            "requests_per_minute": {"type": "number"}
                        }
                    },
                    "ErrorResponse": {
                        "type": "object",
                        "properties": {
                            "error": {
                                "type": "object",
                                "properties": {
                                    "code": {"type": "string"},
                                    "message": {"type": "string"},
                                    "details": {"type": "object"}
                                }
                            },
                            "request_id": {"type": "string"}
                        }
                    }
                }
            }
        }


@dataclass
class RequestMetrics:
    """Tracks metrics for API requests."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_response_time: float = 0.0
    requests_by_endpoint: Dict[str, int] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)

    @property
    def average_response_time(self) -> float:
        """Calculate average response time across all requests."""
        if self.total_requests == 0:
            return 0.0
        return self.total_response_time / self.total_requests

    @property
    def uptime(self) -> float:
        """Calculate server uptime in seconds."""
        return time.time() - self.start_time

    @property
    def requests_per_minute(self) -> float:
        """Calculate average requests per minute."""
        if self.uptime == 0:
            return 0.0
        return (self.total_requests / self.uptime) * 60


class RAGRequestHandler(BaseHTTPRequestHandler):
    """
    HTTP request handler for the RAG API.

    Handles all API endpoints with proper authentication, rate limiting,
    and error handling. Uses class variables to share state across
    handler instances (one per request).
    """

    # Shared state across handler instances (set by ProductionRagApi)
    auth_manager: AuthenticationManager = None
    rate_limiter: RateLimiter = None
    request_validator: RequestValidator = None
    doc_generator: APIDocumentationGenerator = None
    metrics: RequestMetrics = None
    rag_processor: Optional[Callable] = None

    # Suppress default logging
    def log_message(self, format: str, *args) -> None:
        """Override to use structured logging instead of stderr."""
        logger.debug(f"{self.address_string()} - {format % args}")

    def _send_json_response(self, status_code: int, data: Dict[str, Any],
                           headers: Optional[Dict[str, str]] = None) -> None:
        """
        Send a JSON response with appropriate headers.

        Args:
            status_code: HTTP status code
            data: Response data dictionary
            headers: Additional headers to include
        """
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("X-Request-ID", data.get("request_id", str(uuid.uuid4())))

        if headers:
            for key, value in headers.items():
                self.send_header(key, value)

        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def _send_error_response(self, status_code: int, error_code: ErrorCode,
                            message: str, details: Optional[Dict] = None,
                            headers: Optional[Dict[str, str]] = None) -> None:
        """
        Send an error response following RFC 7807 format.

        Args:
            status_code: HTTP status code
            error_code: Error code enum value
            message: Human-readable error message
            details: Additional error details
            headers: Additional headers to include
        """
        request_id = str(uuid.uuid4())
        response = {
            "error": {
                "code": error_code.value,
                "message": message,
            },
            "request_id": request_id
        }
        if details:
            response["error"]["details"] = details

        if self.metrics:
            self.metrics.failed_requests += 1

        self._send_json_response(status_code, response, headers)

    def _extract_api_key(self) -> Optional[str]:
        """Extract API key from Authorization header."""
        auth_header = self.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            return auth_header[7:]
        return None

    def _check_authentication(self, required_permission: str = "query") -> Optional[APIKey]:
        """
        Validate authentication and permissions.

        Returns:
            APIKey if authenticated with proper permissions, None otherwise
        """
        api_key = self._extract_api_key()

        if not api_key:
            self._send_error_response(
                401,
                ErrorCode.AUTHENTICATION_REQUIRED,
                "Authentication required. Provide API key in Authorization header."
            )
            return None

        key_data = self.auth_manager.validate_key(api_key)

        if not key_data:
            self._send_error_response(
                401,
                ErrorCode.INVALID_API_KEY,
                "Invalid or revoked API key"
            )
            return None

        if required_permission not in key_data.permissions:
            self._send_error_response(
                403,
                ErrorCode.AUTHENTICATION_REQUIRED,
                f"API key lacks required permission: {required_permission}"
            )
            return None

        return key_data

    def _check_rate_limit(self, key_data: APIKey) -> bool:
        """
        Check rate limit for the authenticated client.

        Returns:
            True if request is allowed, False if rate limited
        """
        allowed, rate_info = self.rate_limiter.check_and_consume(
            key_data.client_id,
            key_data.rate_limit
        )

        if not allowed:
            self._send_error_response(
                429,
                ErrorCode.RATE_LIMIT_EXCEEDED,
                f"Rate limit exceeded. Retry after {rate_info['retry_after']} seconds.",
                {"retry_after": rate_info["retry_after"]},
                {
                    "X-RateLimit-Limit": str(rate_info["limit"]),
                    "X-RateLimit-Remaining": str(rate_info["remaining"]),
                    "X-RateLimit-Reset": str(rate_info["reset"]),
                    "Retry-After": str(rate_info["retry_after"])
                }
            )
            return False

        return True

    def _read_request_body(self) -> Optional[Dict[str, Any]]:
        """Read and parse JSON request body."""
        content_length = int(self.headers.get("Content-Length", 0))

        if content_length == 0:
            self._send_error_response(
                400,
                ErrorCode.INVALID_REQUEST,
                "Request body is required"
            )
            return None

        try:
            body = self.rfile.read(content_length)
            data = json.loads(body.decode("utf-8"))
            return data
        except json.JSONDecodeError as e:
            self._send_error_response(
                400,
                ErrorCode.INVALID_REQUEST,
                f"Invalid JSON in request body: {str(e)}"
            )
            return None

    def _route_request(self, method: str, path: str) -> None:
        """Route request to appropriate handler based on path."""
        # Track request
        start_time = time.time()
        if self.metrics:
            self.metrics.total_requests += 1
            endpoint = path.split("?")[0]
            self.metrics.requests_by_endpoint[endpoint] = \
                self.metrics.requests_by_endpoint.get(endpoint, 0) + 1

        # Route to handlers
        if path == "/api/v1/health":
            self._handle_health()
        elif path == "/api/v1/docs":
            self._handle_docs()
        elif path == "/api/v1/metrics":
            self._handle_metrics()
        elif path == "/api/v1/query" and method == "POST":
            self._handle_query()
        elif path.startswith("/api/v1/"):
            self._send_error_response(
                404,
                ErrorCode.NOT_FOUND,
                f"Endpoint not found: {path}"
            )
        else:
            self._send_error_response(
                404,
                ErrorCode.NOT_FOUND,
                "Not found"
            )

        # Track response time
        if self.metrics:
            self.metrics.total_response_time += time.time() - start_time

    def _handle_health(self) -> None:
        """Handle health check endpoint."""
        response = {
            "status": "healthy",
            "uptime": self.metrics.uptime if self.metrics else 0,
            "version": "1.0.0",
            "request_id": str(uuid.uuid4())
        }
        if self.metrics:
            self.metrics.successful_requests += 1
        self._send_json_response(200, response)

    def _handle_docs(self) -> None:
        """Handle API documentation endpoint."""
        spec = self.doc_generator.generate_spec()
        spec["request_id"] = str(uuid.uuid4())
        if self.metrics:
            self.metrics.successful_requests += 1
        self._send_json_response(200, spec)

    def _handle_metrics(self) -> None:
        """Handle metrics endpoint (requires authentication)."""
        key_data = self._check_authentication("metrics")
        if not key_data:
            return

        if not self._check_rate_limit(key_data):
            return

        response = {
            "total_requests": self.metrics.total_requests if self.metrics else 0,
            "successful_requests": self.metrics.successful_requests if self.metrics else 0,
            "failed_requests": self.metrics.failed_requests if self.metrics else 0,
            "average_response_time": self.metrics.average_response_time if self.metrics else 0,
            "requests_per_minute": self.metrics.requests_per_minute if self.metrics else 0,
            "uptime": self.metrics.uptime if self.metrics else 0,
            "requests_by_endpoint": self.metrics.requests_by_endpoint if self.metrics else {},
            "request_id": str(uuid.uuid4())
        }
        if self.metrics:
            self.metrics.successful_requests += 1
        self._send_json_response(200, response)

    def _handle_query(self) -> None:
        """Handle RAG query endpoint."""
        # Check authentication
        key_data = self._check_authentication("query")
        if not key_data:
            return

        # Check rate limit
        if not self._check_rate_limit(key_data):
            return

        # Read and validate request body
        data = self._read_request_body()
        if data is None:
            return

        # Validate request
        is_valid, errors = self.request_validator.validate("query", data)
        if not is_valid:
            self._send_error_response(
                400,
                ErrorCode.VALIDATION_ERROR,
                "Request validation failed",
                {"errors": errors}
            )
            return

        # Sanitize input
        data = self.request_validator.sanitize(data)

        # Process query
        try:
            request_id = str(uuid.uuid4())
            query_start = time.time()

            # Use custom processor if provided, otherwise return simulated response
            if self.rag_processor:
                result = self.rag_processor(data)
            else:
                # Simulated RAG response for demonstration
                result = {
                    "response": f"This is a simulated response to: {data['query']}",
                    "sources": [
                        {"doc_id": "doc1", "score": 0.85, "snippet": "Relevant excerpt..."},
                        {"doc_id": "doc2", "score": 0.72, "snippet": "Another excerpt..."}
                    ] if data.get("include_sources", True) else []
                }

            query_time = time.time() - query_start

            response = {
                **result,
                "metadata": {
                    "query_time": round(query_time, 3),
                    "model": "rag-v1",
                    "client_id": key_data.client_id
                },
                "request_id": request_id
            }

            if self.metrics:
                self.metrics.successful_requests += 1

            self._send_json_response(200, response)

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            self._send_error_response(
                500,
                ErrorCode.INTERNAL_ERROR,
                "An error occurred processing your request"
            )

    def do_GET(self) -> None:
        """Handle GET requests."""
        self._route_request("GET", self.path)

    def do_POST(self) -> None:
        """Handle POST requests."""
        self._route_request("POST", self.path)

    def do_OPTIONS(self) -> None:
        """Handle OPTIONS requests for CORS preflight."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.end_headers()


class ThreadedHTTPServer(HTTPServer):
    """HTTP server that handles requests in separate threads."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._shutdown_event = threading.Event()

    def process_request(self, request, client_address):
        """Process each request in a new thread."""
        thread = threading.Thread(
            target=self.process_request_thread,
            args=(request, client_address)
        )
        thread.daemon = True
        thread.start()

    def process_request_thread(self, request, client_address):
        """Process a single request in a thread."""
        try:
            self.finish_request(request, client_address)
        except Exception as e:
            self.handle_error(request, client_address)
        finally:
            self.shutdown_request(request)


class ProductionRagApi:
    """
    ProductionRagApi - Production-ready REST API for RAG systems.

    This class provides a complete REST API implementation with:
    - API key authentication
    - Token bucket rate limiting
    - Request validation and sanitization
    - OpenAPI documentation
    - Concurrent request handling
    - Structured logging and metrics

    Example usage:
        api = ProductionRagApi(host="localhost", port=8080)
        api_key = api.create_api_key("my-client")
        api.execute()  # Starts the server

    For testing:
        curl -X POST http://localhost:8080/api/v1/query \\
            -H "Authorization: Bearer <api_key>" \\
            -H "Content-Type: application/json" \\
            -d '{"query": "What is RAG?"}'
    """

    def __init__(self, host: str = "localhost", port: int = 8080,
                 rag_processor: Optional[Callable] = None):
        """
        Initialize the ProductionRagApi instance.

        Args:
            host: Host address to bind to (default: localhost)
            port: Port number to listen on (default: 8080)
            rag_processor: Optional callable for processing RAG queries
        """
        self.host = host
        self.port = port
        self.rag_processor = rag_processor

        # Initialize components
        self.auth_manager = AuthenticationManager()
        self.rate_limiter = RateLimiter()
        self.request_validator = RequestValidator()
        self.doc_generator = APIDocumentationGenerator()
        self.metrics = RequestMetrics()

        # Server instance (created on execute)
        self._server: Optional[ThreadedHTTPServer] = None
        self._server_thread: Optional[threading.Thread] = None

    def create_api_key(self, client_id: str, rate_limit: int = 60,
                       permissions: Optional[List[str]] = None) -> str:
        """
        Create a new API key for a client.

        Args:
            client_id: Unique identifier for the client
            rate_limit: Maximum requests per minute
            permissions: List of allowed operations

        Returns:
            The raw API key (store securely!)
        """
        return self.auth_manager.create_api_key(client_id, rate_limit, permissions)

    def revoke_api_key(self, api_key: str) -> bool:
        """
        Revoke an API key.

        Args:
            api_key: The API key to revoke

        Returns:
            True if revoked successfully
        """
        return self.auth_manager.revoke_key(api_key)

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current API metrics.

        Returns:
            Dictionary containing metrics data
        """
        return {
            "total_requests": self.metrics.total_requests,
            "successful_requests": self.metrics.successful_requests,
            "failed_requests": self.metrics.failed_requests,
            "average_response_time": self.metrics.average_response_time,
            "requests_per_minute": self.metrics.requests_per_minute,
            "uptime": self.metrics.uptime,
            "requests_by_endpoint": self.metrics.requests_by_endpoint
        }

    def get_documentation(self) -> Dict[str, Any]:
        """
        Get OpenAPI documentation.

        Returns:
            OpenAPI specification dictionary
        """
        return self.doc_generator.generate_spec(f"http://{self.host}:{self.port}")

    def _configure_handler(self) -> None:
        """Configure the request handler with shared state."""
        RAGRequestHandler.auth_manager = self.auth_manager
        RAGRequestHandler.rate_limiter = self.rate_limiter
        RAGRequestHandler.request_validator = self.request_validator
        RAGRequestHandler.doc_generator = self.doc_generator
        RAGRequestHandler.metrics = self.metrics
        RAGRequestHandler.rag_processor = self.rag_processor

    def execute(self, blocking: bool = True) -> None:
        """
        Start the API server.

        Args:
            blocking: If True, block until server is stopped.
                     If False, run server in background thread.

        This starts the HTTP server and begins accepting requests.
        The server handles concurrent requests using threading.
        """
        self._configure_handler()

        self._server = ThreadedHTTPServer(
            (self.host, self.port),
            RAGRequestHandler
        )

        logger.info(f"Starting Production RAG API on http://{self.host}:{self.port}")
        logger.info("Endpoints available:")
        logger.info("  POST /api/v1/query  - RAG query (requires auth)")
        logger.info("  GET  /api/v1/health - Health check")
        logger.info("  GET  /api/v1/docs   - API documentation")
        logger.info("  GET  /api/v1/metrics - Usage metrics (requires auth)")

        if blocking:
            try:
                self._server.serve_forever()
            except KeyboardInterrupt:
                logger.info("Server shutdown requested")
                self._server.shutdown()
        else:
            self._server_thread = threading.Thread(target=self._server.serve_forever)
            self._server_thread.daemon = True
            self._server_thread.start()

    def stop(self) -> None:
        """Stop the API server."""
        if self._server:
            logger.info("Stopping API server...")
            self._server.shutdown()
            self._server = None

    def is_running(self) -> bool:
        """Check if the server is currently running."""
        return self._server is not None


def create_production_rag_api(host: str = "localhost", port: int = 8080,
                              rag_processor: Optional[Callable] = None) -> ProductionRagApi:
    """
    Factory function for creating ProductionRagApi instances.

    Args:
        host: Host address to bind to
        port: Port number to listen on
        rag_processor: Optional callable for processing RAG queries

    Returns:
        ProductionRagApi: A new instance of ProductionRagApi
    """
    return ProductionRagApi(host=host, port=port, rag_processor=rag_processor)


# =============================================================================
# Default RAG Processor using Conversational Interface (Phase 4)
# =============================================================================

def create_default_rag_processor():
    """
    Create a default RAG processor using the Phase 4 Conversational RAG Interface.

    This provides the same RAG processing used by the web UI, ensuring
    consistency between the standalone API and the Flask-based interface.
    """
    from src.conversational_rag_interface import create_conversational_rag_interface

    # Create the conversational interface
    interface = create_conversational_rag_interface()

    # Load sample knowledge (same as web_ui.py)
    knowledge_items = [
        "RAG (Retrieval-Augmented Generation) is a technique that combines information retrieval with text generation. It allows language models to access external knowledge bases to provide more accurate and up-to-date responses.",
        "The main benefits of RAG systems include: 1) Access to current information beyond the model's training data, 2) Reduced hallucination through grounded responses, 3) More cost-effective than fine-tuning for many use cases, 4) Easy to update knowledge without retraining.",
        "Vector embeddings are numerical representations of text that capture semantic meaning. Similar concepts have similar embeddings, enabling semantic search rather than just keyword matching.",
        "Hybrid retrieval combines multiple search strategies: semantic similarity using embeddings, keyword matching using TF-IDF, and metadata filtering. This approach often outperforms single-strategy retrieval.",
        "Fine-tuning involves training a model on specific data to adapt it for particular tasks. While powerful, it's more expensive and requires technical expertise compared to RAG approaches.",
        "Query expansion improves retrieval by adding synonyms and related terms to user queries. For example, 'ML' might be expanded to include 'machine learning' and 'artificial intelligence'.",
        "Conversation memory in RAG systems allows the model to maintain context across multiple turns, enabling natural follow-up questions and references to earlier parts of the conversation.",
        "Multi-modal RAG extends traditional text-based RAG to handle images, tables, and structured data like JSON and CSV files.",
        "Re-ranking in retrieval systems reorders initial search results using additional criteria like document freshness, quality scores, or relevance to the specific query context.",
    ]

    for item in knowledge_items:
        interface.add_knowledge(item)

    def processor(data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a RAG query using the conversational interface."""
        query = data.get('query', '')
        result = interface.chat(query)

        return {
            'response': result['response'],
            'sources': result.get('retrieved_context', []),
            'query_analysis': result.get('query_analysis', {}),
            'conversation_turn': result.get('conversation_turn', 0)
        }

    return processor


# =============================================================================
# Standalone execution with real RAG processing
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Production RAG API - Standalone Server")
    print("  Using Phase 4 Conversational RAG Interface")
    print("=" * 60)

    # Create API with real RAG processor
    processor = create_default_rag_processor()
    api = ProductionRagApi(host="127.0.0.1", port=8080, rag_processor=processor)

    # Create a demo API key
    demo_key = api.create_api_key(
        client_id="demo-user",
        rate_limit=60,
        permissions=["query", "health", "metrics"]
    )

    print(f"\n  Demo API Key (for testing):")
    print(f"  {demo_key}")
    print("\n  Endpoints:")
    print("  - POST /api/v1/query   (requires auth)")
    print("  - GET  /api/v1/health  (no auth)")
    print("  - GET  /api/v1/docs    (no auth)")
    print("  - GET  /api/v1/metrics (requires auth)")
    print("\n  Example curl:")
    print(f'  curl -X POST http://127.0.0.1:8080/api/v1/query \\')
    print(f'    -H "Authorization: Bearer {demo_key}" \\')
    print(f'    -H "Content-Type: application/json" \\')
    print(f'    -d \'{{"query": "What is RAG?"}}\'')
    print("\n" + "=" * 60 + "\n")

    # Start the server
    api.execute(blocking=True)
