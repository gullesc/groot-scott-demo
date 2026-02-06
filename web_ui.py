#!/usr/bin/env python3
"""
Conversational RAG Interface - Web UI

A Flask-based web interface for the Phase 4 Conversational RAG system.

Usage:
    pip install flask
    python3 web_ui.py

Then open http://localhost:5000 in your browser.
"""

from flask import Flask, render_template_string, request, jsonify, session
import secrets
from src.conversational_rag_interface import create_conversational_rag_interface

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# Store interfaces per session
interfaces = {}

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
                <p>Phase 4 Demo - Multi-turn conversations with context awareness</p>
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
    """Handle chat messages."""
    data = request.json
    message = data.get('message', '').strip()

    if not message:
        return jsonify({'error': 'Empty message'})

    interface = get_interface()
    result = interface.chat(message)
    stats = interface.get_statistics()
    summary = interface.get_summary()

    return jsonify({
        'response': result['response'],
        'query_type': result['query_analysis']['type'],
        'is_follow_up': result['query_analysis']['is_follow_up'],
        'turn': result['conversation_turn'],
        'stats': stats,
        'topics': summary['main_topics']
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


if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("  Conversational RAG Interface - Web UI")
    print("=" * 50)
    print("\nStarting server...")
    print("Open http://localhost:5000 in your browser\n")
    app.run(debug=True, port=5000)
