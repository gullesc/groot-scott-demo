#!/usr/bin/env python3
"""
Conversational RAG Interface - Web UI

A Flask-based web interface integrating Phase 4 Conversational RAG with
Phase 5 Production API features (authentication, rate limiting, metrics).

Usage:
    pip install flask
    python3 web_ui.py

Then open http://127.0.0.1:5001 in your browser.
"""

from flask import Flask, render_template_string, request, jsonify, session, g
import secrets
import time
from functools import wraps
from src.conversational_rag_interface import create_conversational_rag_interface
from src.production_rag_api import (
    AuthenticationManager,
    RateLimiter,
    RequestValidator,
    RequestMetrics,
    APIDocumentationGenerator,
    ErrorCode
)

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# Store interfaces per session
interfaces = {}

# Production API components (shared across all requests)
auth_manager = AuthenticationManager()
rate_limiter = RateLimiter(max_tokens=60, refill_rate=1.0)  # 60 requests/minute
request_validator = RequestValidator()
metrics = RequestMetrics()
doc_generator = APIDocumentationGenerator(
    title="Conversational RAG API",
    version="1.0.0"
)

# Create a demo API key on startup
demo_api_key = auth_manager.create_api_key(
    client_id="demo-user",
    rate_limit=60,
    permissions=["query", "health", "metrics"]
)
print(f"\n{'='*50}")
print("  Demo API Key (for testing authenticated endpoints):")
print(f"  {demo_api_key}")
print(f"{'='*50}\n")

# HTML Template with embedded CSS and JavaScript
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Conversational RAG Interface</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #e4e4e4;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            display: flex;
            gap: 20px;
            height: 100vh;
        }

        .chat-section {
            flex: 2;
            display: flex;
            flex-direction: column;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 16px;
            overflow: hidden;
        }

        .sidebar {
            flex: 1;
            display: flex;
            flex-direction: column;
            gap: 15px;
        }

        .header {
            padding: 20px;
            background: rgba(255, 255, 255, 0.1);
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }

        .header h1 {
            font-size: 1.5rem;
            color: #00d4ff;
            margin-bottom: 5px;
        }

        .header p {
            font-size: 0.85rem;
            color: #888;
        }

        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 15px;
        }

        .message {
            max-width: 85%;
            padding: 12px 16px;
            border-radius: 12px;
            line-height: 1.5;
        }

        .message.user {
            align-self: flex-end;
            background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%);
            color: #fff;
        }

        .message.assistant {
            align-self: flex-start;
            background: rgba(255, 255, 255, 0.1);
        }

        .message-meta {
            font-size: 0.75rem;
            color: #888;
            margin-top: 8px;
            padding-top: 8px;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
        }

        .message.user .message-meta {
            color: rgba(255, 255, 255, 0.7);
            border-top-color: rgba(255, 255, 255, 0.2);
        }

        .input-area {
            padding: 20px;
            background: rgba(255, 255, 255, 0.05);
            border-top: 1px solid rgba(255, 255, 255, 0.1);
        }

        .input-wrapper {
            display: flex;
            gap: 10px;
        }

        #user-input {
            flex: 1;
            padding: 12px 16px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.05);
            color: #fff;
            font-size: 1rem;
            outline: none;
            transition: border-color 0.3s;
        }

        #user-input:focus {
            border-color: #00d4ff;
        }

        #user-input::placeholder {
            color: #666;
        }

        button {
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%);
            color: #fff;
            font-size: 1rem;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }

        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 212, 255, 0.3);
        }

        button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }

        .card {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            padding: 15px;
        }

        .card h3 {
            font-size: 0.9rem;
            color: #00d4ff;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .card-content {
            font-size: 0.85rem;
            color: #aaa;
        }

        .stat-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
        }

        .stat-item {
            text-align: center;
            padding: 10px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 8px;
        }

        .stat-value {
            font-size: 1.5rem;
            font-weight: bold;
            color: #00d4ff;
        }

        .stat-label {
            font-size: 0.75rem;
            color: #888;
        }

        .topic-tag {
            display: inline-block;
            padding: 4px 8px;
            background: rgba(0, 212, 255, 0.2);
            color: #00d4ff;
            border-radius: 4px;
            font-size: 0.75rem;
            margin: 2px;
        }

        .btn-secondary {
            background: rgba(255, 255, 255, 0.1);
            width: 100%;
            margin-top: 10px;
        }

        .btn-secondary:hover {
            background: rgba(255, 255, 255, 0.2);
            box-shadow: none;
        }

        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }

        .loading.active {
            display: block;
        }

        .loading-dots {
            display: inline-block;
        }

        .loading-dots span {
            animation: blink 1.4s infinite;
            animation-fill-mode: both;
        }

        .loading-dots span:nth-child(2) { animation-delay: 0.2s; }
        .loading-dots span:nth-child(3) { animation-delay: 0.4s; }

        @keyframes blink {
            0%, 80%, 100% { opacity: 0; }
            40% { opacity: 1; }
        }

        .welcome-message {
            text-align: center;
            padding: 40px;
            color: #666;
        }

        .welcome-message h2 {
            color: #00d4ff;
            margin-bottom: 10px;
        }

        @media (max-width: 900px) {
            .container {
                flex-direction: column;
                height: auto;
                min-height: 100vh;
            }

            .chat-section {
                min-height: 60vh;
            }

            .sidebar {
                flex-direction: row;
                flex-wrap: wrap;
            }

            .card {
                flex: 1;
                min-width: 200px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="chat-section">
            <div class="header">
                <h1>Conversational RAG Interface</h1>
                <p>Phase 4 + 5 Demo - Production API with Auth, Rate Limiting & Metrics</p>
            </div>

            <div class="chat-messages" id="chat-messages">
                <div class="welcome-message">
                    <h2>Welcome!</h2>
                    <p>Ask me anything about RAG systems, machine learning, or any topic.</p>
                    <p>I'll remember our conversation context for follow-up questions.</p>
                </div>
            </div>

            <div class="loading" id="loading">
                <div class="loading-dots">
                    Thinking<span>.</span><span>.</span><span>.</span>
                </div>
            </div>

            <div class="input-area">
                <div class="input-wrapper">
                    <input type="text" id="user-input" placeholder="Type your message..." autocomplete="off">
                    <button onclick="sendMessage()" id="send-btn">Send</button>
                </div>
            </div>
        </div>

        <div class="sidebar">
            <div class="card">
                <h3>Statistics</h3>
                <div class="stat-grid">
                    <div class="stat-item">
                        <div class="stat-value" id="stat-turns">0</div>
                        <div class="stat-label">Turns</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="stat-followups">0</div>
                        <div class="stat-label">Follow-ups</div>
                    </div>
                </div>
            </div>

            <div class="card">
                <h3>Topics Discussed</h3>
                <div class="card-content" id="topics-list">
                    <em>No topics yet</em>
                </div>
            </div>

            <div class="card">
                <h3>Summary</h3>
                <div class="card-content" id="summary-text">
                    <em>Start chatting to see a summary</em>
                </div>
                <button class="btn-secondary" onclick="getSummary()">Refresh Summary</button>
            </div>

            <div class="card">
                <h3>Actions</h3>
                <button class="btn-secondary" onclick="clearHistory()">Clear Conversation</button>
            </div>

            <div class="card" style="background: rgba(0, 212, 255, 0.1); border: 1px solid rgba(0, 212, 255, 0.3);">
                <h3>🔐 Production API (Phase 5)</h3>
                <div class="stat-grid">
                    <div class="stat-item">
                        <div class="stat-value" id="api-requests">0</div>
                        <div class="stat-label">Total Requests</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="api-response-time">0</div>
                        <div class="stat-label">Avg Time (ms)</div>
                    </div>
                </div>
                <div style="margin-top: 10px; font-size: 0.8rem; color: #888;">
                    <div>✓ API Key Auth (SHA-256)</div>
                    <div>✓ Token Bucket Rate Limiting</div>
                    <div>✓ Request Validation</div>
                    <div>✓ OpenAPI Documentation</div>
                </div>
                <button class="btn-secondary" onclick="generateApiKey()" style="margin-top: 10px;">Generate API Key</button>
                <button class="btn-secondary" onclick="showApiDocs()">View API Docs</button>
            </div>

            <div class="card" id="api-key-display" style="display: none; background: rgba(0, 255, 100, 0.1); border: 1px solid rgba(0, 255, 100, 0.3);">
                <h3>🔑 Your API Key</h3>
                <div style="font-family: monospace; font-size: 0.7rem; word-break: break-all; padding: 8px; background: rgba(0,0,0,0.3); border-radius: 4px; margin-bottom: 10px;">
                    <span id="generated-api-key"></span>
                </div>
                <div style="font-size: 0.75rem; color: #888;">
                    Use with: <code>Authorization: Bearer &lt;key&gt;</code>
                </div>
                <button class="btn-secondary" onclick="testApiKey()">Test API Key</button>
            </div>
        </div>
    </div>

    <script>
        const chatMessages = document.getElementById('chat-messages');
        const userInput = document.getElementById('user-input');
        const sendBtn = document.getElementById('send-btn');
        const loading = document.getElementById('loading');

        // Send on Enter key
        userInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });

        function addMessage(content, role, meta = null) {
            // Remove welcome message if present
            const welcome = chatMessages.querySelector('.welcome-message');
            if (welcome) welcome.remove();

            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${role}`;

            let html = content;
            if (meta) {
                html += `<div class="message-meta">${meta}</div>`;
            }

            messageDiv.innerHTML = html;
            chatMessages.appendChild(messageDiv);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }

        async function sendMessage() {
            const message = userInput.value.trim();
            if (!message) return;

            // Add user message
            addMessage(message, 'user');
            userInput.value = '';

            // Show loading
            loading.classList.add('active');
            sendBtn.disabled = true;

            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message })
                });

                const data = await response.json();

                if (data.error) {
                    addMessage('Error: ' + data.error, 'assistant');
                } else {
                    const meta = `Type: ${data.query_type}${data.is_follow_up ? ' (follow-up)' : ''} | Turn: ${data.turn}`;
                    addMessage(data.response, 'assistant', meta);
                    updateStats(data.stats);
                    updateTopics(data.topics);
                }
            } catch (error) {
                addMessage('Error: Could not connect to server', 'assistant');
            }

            loading.classList.remove('active');
            sendBtn.disabled = false;
            userInput.focus();
        }

        function updateStats(stats) {
            document.getElementById('stat-turns').textContent = stats.total_turns || 0;
            document.getElementById('stat-followups').textContent = stats.follow_up_queries || 0;
        }

        function updateTopics(topics) {
            const topicsList = document.getElementById('topics-list');
            if (topics && topics.length > 0) {
                topicsList.innerHTML = topics.map(t =>
                    `<span class="topic-tag">${t}</span>`
                ).join('');
            }
        }

        async function getSummary() {
            try {
                const response = await fetch('/summary');
                const data = await response.json();

                document.getElementById('summary-text').innerHTML = data.summary_text || '<em>No summary available</em>';

                if (data.main_topics && data.main_topics.length > 0) {
                    updateTopics(data.main_topics);
                }
            } catch (error) {
                console.error('Error fetching summary:', error);
            }
        }

        async function clearHistory() {
            if (!confirm('Clear conversation history?')) return;

            try {
                await fetch('/clear', { method: 'POST' });

                // Reset UI
                chatMessages.innerHTML = `
                    <div class="welcome-message">
                        <h2>Welcome!</h2>
                        <p>Ask me anything about RAG systems, machine learning, or any topic.</p>
                        <p>I'll remember our conversation context for follow-up questions.</p>
                    </div>
                `;
                document.getElementById('stat-turns').textContent = '0';
                document.getElementById('stat-followups').textContent = '0';
                document.getElementById('topics-list').innerHTML = '<em>No topics yet</em>';
                document.getElementById('summary-text').innerHTML = '<em>Start chatting to see a summary</em>';
            } catch (error) {
                console.error('Error clearing history:', error);
            }
        }

        // Focus input on load
        userInput.focus();

        // =============================================
        // Production API Features (Phase 5)
        // =============================================

        let currentApiKey = null;

        // Refresh API metrics every 5 seconds
        async function refreshApiMetrics() {
            try {
                const response = await fetch('/ui-metrics');
                const data = await response.json();
                document.getElementById('api-requests').textContent = data.total_requests || 0;
                document.getElementById('api-response-time').textContent = data.average_response_time || 0;
            } catch (error) {
                console.error('Error fetching metrics:', error);
            }
        }

        // Start metrics refresh
        setInterval(refreshApiMetrics, 5000);
        refreshApiMetrics();

        async function generateApiKey() {
            try {
                const response = await fetch('/api/v1/keys', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ client_id: 'web-ui-user-' + Date.now() })
                });
                const data = await response.json();

                currentApiKey = data.api_key;
                document.getElementById('generated-api-key').textContent = data.api_key;
                document.getElementById('api-key-display').style.display = 'block';

                alert('API Key generated! This key has 60 requests/minute rate limit.');
            } catch (error) {
                alert('Error generating API key: ' + error.message);
            }
        }

        async function testApiKey() {
            if (!currentApiKey) {
                alert('Generate an API key first!');
                return;
            }

            try {
                const response = await fetch('/api/v1/query', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': 'Bearer ' + currentApiKey
                    },
                    body: JSON.stringify({ query: 'What is RAG?' })
                });

                const data = await response.json();

                if (response.ok) {
                    const rateRemaining = response.headers.get('X-RateLimit-Remaining');
                    alert(`✅ API Key works!\n\nResponse: ${data.response.substring(0, 100)}...\n\nRate limit remaining: ${rateRemaining}`);
                } else {
                    alert(`❌ API Error: ${data.error?.message || 'Unknown error'}`);
                }

                refreshApiMetrics();
            } catch (error) {
                alert('Error testing API key: ' + error.message);
            }
        }

        function showApiDocs() {
            window.open('/api/v1/docs', '_blank');
        }
    </script>
</body>
</html>
'''


def get_interface():
    """Get or create interface for current session."""
    session_id = session.get('session_id')
    if not session_id:
        session_id = secrets.token_hex(16)
        session['session_id'] = session_id

    if session_id not in interfaces:
        interface = create_conversational_rag_interface()
        load_knowledge(interface)
        interfaces[session_id] = interface

    return interfaces[session_id]


def load_knowledge(interface):
    """Load sample knowledge into the interface."""
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


@app.route('/')
def index():
    """Serve the main page."""
    return render_template_string(HTML_TEMPLATE)


@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages (demo mode - no auth required)."""
    start_time = time.time()
    metrics.total_requests += 1
    metrics.requests_by_endpoint['/chat'] = \
        metrics.requests_by_endpoint.get('/chat', 0) + 1

    data = request.json
    message = data.get('message', '').strip()

    if not message:
        metrics.failed_requests += 1
        return jsonify({'error': 'Empty message'})

    interface = get_interface()
    result = interface.chat(message)
    stats = interface.get_statistics()
    summary = interface.get_summary()

    # Track response time
    response_time = time.time() - start_time
    metrics.total_response_time += response_time
    metrics.successful_requests += 1

    return jsonify({
        'response': result['response'],
        'query_type': result['query_analysis']['type'],
        'is_follow_up': result['query_analysis']['is_follow_up'],
        'turn': result['conversation_turn'],
        'stats': stats,
        'topics': summary['main_topics'],
        'response_time_ms': round(response_time * 1000, 1)
    })


@app.route('/summary')
def summary():
    """Get conversation summary."""
    interface = get_interface()
    return jsonify(interface.get_summary())


@app.route('/clear', methods=['POST'])
def clear():
    """Clear conversation history."""
    interface = get_interface()
    interface.clear_history()
    return jsonify({'status': 'ok'})


# =============================================================================
# Production API Endpoints (Phase 5 Integration)
# =============================================================================

def get_api_key_from_header():
    """Extract API key from Authorization header."""
    auth_header = request.headers.get('Authorization', '')
    if auth_header.startswith('Bearer '):
        return auth_header[7:]
    return None


def require_auth(permission='query'):
    """Decorator to require API key authentication."""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            api_key = get_api_key_from_header()

            if not api_key:
                return jsonify({
                    'error': {
                        'code': ErrorCode.AUTHENTICATION_REQUIRED.value,
                        'message': 'API key required. Use Authorization: Bearer <api_key>'
                    }
                }), 401

            key_data = auth_manager.validate_key(api_key)
            if not key_data:
                return jsonify({
                    'error': {
                        'code': ErrorCode.INVALID_API_KEY.value,
                        'message': 'Invalid or revoked API key'
                    }
                }), 401

            if permission not in key_data.permissions:
                return jsonify({
                    'error': {
                        'code': ErrorCode.AUTHENTICATION_REQUIRED.value,
                        'message': f'API key lacks permission: {permission}'
                    }
                }), 403

            # Check rate limit
            allowed, rate_info = rate_limiter.check_and_consume(
                key_data.client_id,
                key_data.rate_limit
            )

            if not allowed:
                response = jsonify({
                    'error': {
                        'code': ErrorCode.RATE_LIMIT_EXCEEDED.value,
                        'message': f"Rate limit exceeded. Retry after {rate_info['retry_after']} seconds"
                    }
                })
                response.headers['X-RateLimit-Limit'] = str(rate_info['limit'])
                response.headers['X-RateLimit-Remaining'] = str(rate_info['remaining'])
                response.headers['Retry-After'] = str(rate_info['retry_after'])
                return response, 429

            # Store key data for use in route
            g.api_key_data = key_data
            g.rate_info = rate_info

            return f(*args, **kwargs)
        return decorated_function
    return decorator


@app.route('/api/v1/health')
def api_health():
    """Health check endpoint (no auth required)."""
    metrics.total_requests += 1
    metrics.successful_requests += 1
    return jsonify({
        'status': 'healthy',
        'uptime': metrics.uptime,
        'version': '1.0.0',
        'components': {
            'conversational_rag': 'active',
            'authentication': 'active',
            'rate_limiting': 'active'
        }
    })


@app.route('/api/v1/docs')
def api_docs():
    """OpenAPI documentation endpoint."""
    metrics.total_requests += 1
    metrics.successful_requests += 1
    spec = doc_generator.generate_spec('http://127.0.0.1:5001')
    return jsonify(spec)


@app.route('/api/v1/metrics')
@require_auth('metrics')
def api_metrics():
    """API metrics endpoint (requires auth)."""
    metrics.total_requests += 1
    metrics.successful_requests += 1
    return jsonify({
        'total_requests': metrics.total_requests,
        'successful_requests': metrics.successful_requests,
        'failed_requests': metrics.failed_requests,
        'average_response_time': metrics.average_response_time,
        'requests_per_minute': metrics.requests_per_minute,
        'uptime': metrics.uptime,
        'requests_by_endpoint': metrics.requests_by_endpoint
    })


@app.route('/api/v1/query', methods=['POST'])
@require_auth('query')
def api_query():
    """
    Production RAG query endpoint with full authentication and validation.

    Requires: Authorization: Bearer <api_key>
    Body: {"query": "your question", "include_sources": true}
    """
    start_time = time.time()
    metrics.total_requests += 1
    metrics.requests_by_endpoint['/api/v1/query'] = \
        metrics.requests_by_endpoint.get('/api/v1/query', 0) + 1

    data = request.json or {}

    # Validate request
    is_valid, errors = request_validator.validate('query', data)
    if not is_valid:
        metrics.failed_requests += 1
        return jsonify({
            'error': {
                'code': ErrorCode.VALIDATION_ERROR.value,
                'message': 'Request validation failed',
                'details': {'errors': errors}
            }
        }), 400

    # Sanitize input
    data = request_validator.sanitize(data)
    query = data.get('query', '').strip()

    if not query:
        metrics.failed_requests += 1
        return jsonify({
            'error': {
                'code': ErrorCode.VALIDATION_ERROR.value,
                'message': 'Query cannot be empty'
            }
        }), 400

    # Process through conversational RAG
    interface = get_interface()
    result = interface.chat(query)

    query_time = time.time() - start_time
    metrics.total_response_time += query_time
    metrics.successful_requests += 1

    # Build response with rate limit headers
    response_data = {
        'response': result['response'],
        'query_analysis': result['query_analysis'],
        'conversation_turn': result['conversation_turn'],
        'sources': result.get('retrieved_context', []),
        'metadata': {
            'query_time': round(query_time, 3),
            'client_id': g.api_key_data.client_id,
            'rate_limit_remaining': g.rate_info['remaining']
        }
    }

    response = jsonify(response_data)
    response.headers['X-RateLimit-Limit'] = str(g.rate_info['limit'])
    response.headers['X-RateLimit-Remaining'] = str(g.rate_info['remaining'])

    return response


@app.route('/api/v1/keys', methods=['POST'])
def api_create_key():
    """
    Create a new API key (for demo purposes - in production, this would be protected).
    Body: {"client_id": "my-app", "rate_limit": 60}
    """
    data = request.json or {}
    client_id = data.get('client_id', f'client-{secrets.token_hex(4)}')
    rate_limit = data.get('rate_limit', 60)

    api_key = auth_manager.create_api_key(
        client_id=client_id,
        rate_limit=rate_limit,
        permissions=['query', 'health', 'metrics']
    )

    return jsonify({
        'api_key': api_key,
        'client_id': client_id,
        'rate_limit': rate_limit,
        'message': 'Store this key securely - it cannot be retrieved again!'
    })


@app.route('/ui-metrics')
def ui_metrics():
    """Get metrics for the UI dashboard."""
    return jsonify({
        'total_requests': metrics.total_requests,
        'successful_requests': metrics.successful_requests,
        'failed_requests': metrics.failed_requests,
        'average_response_time': round(metrics.average_response_time * 1000, 1),  # ms
        'requests_per_minute': round(metrics.requests_per_minute, 2),
        'uptime': round(metrics.uptime, 0)
    })


if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("  Conversational RAG Interface - Web UI")
    print("=" * 50)
    print("\nStarting server...")
    print("Open http://127.0.0.1:5001 in your browser\n")
    app.run(debug=True, host='127.0.0.1', port=5001)
