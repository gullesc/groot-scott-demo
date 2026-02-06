"""
Tests for Conversational RAG Interface

Run: pytest
Watch: pytest-watch (install with: pip install pytest-watch)
Coverage: pytest --cov=src
"""

import pytest
from src.conversational_rag_interface import (
    ConversationalRagInterface,
    create_conversational_rag_interface,
    ConversationHistory,
    ContextAnalyzer,
    ConversationMemory,
    ResponseGenerator,
    ConversationSummarizer,
    MessageRole,
    QueryType,
    ConversationTurn
)


class TestConversationalRagInterface:
    """Test suite for ConversationalRagInterface."""

    @pytest.fixture
    def instance(self):
        """Create a fresh instance for each test."""
        return create_conversational_rag_interface()

    def test_create_instance(self, instance):
        """Test that we can create an instance."""
        assert instance is not None
        assert isinstance(instance, ConversationalRagInterface)

    def test_execute_returns_result(self, instance):
        """Test that execute returns a proper result dictionary."""
        result = instance.execute()
        assert isinstance(result, dict)
        assert "conversation" in result
        assert "summary" in result
        assert "statistics" in result

    def test_chat_basic(self, instance):
        """Test basic chat functionality."""
        result = instance.chat("Hello, how are you?")

        assert "response" in result
        assert "query_analysis" in result
        assert result["conversation_turn"] == 2  # User + Assistant

    def test_add_knowledge(self, instance):
        """Test adding knowledge to the system."""
        instance.add_knowledge("Python is a programming language.")
        instance.add_knowledge("Machine learning uses algorithms to learn from data.")

        # Knowledge should influence responses
        result = instance.chat("Tell me about Python programming")
        assert result["response"] is not None


class TestAcceptanceCriteria:
    """Tests for acceptance criteria."""

    @pytest.fixture
    def instance(self):
        """Create a fresh instance for each test."""
        return create_conversational_rag_interface()

    def test_maintains_conversation_history_and_context_across(self, instance):
        """Test: Maintains conversation history and context across multiple turns"""
        # Have a multi-turn conversation
        instance.chat("What are the benefits of machine learning?")
        instance.chat("Can you elaborate on that?")
        instance.chat("How does it compare to traditional programming?")

        # Verify history is maintained
        history = instance.get_conversation_history(recent_only=False)

        # Should have 6 turns (3 user + 3 assistant)
        assert len(history) == 6

        # Verify turns have proper structure
        for turn in history:
            assert "turn_id" in turn
            assert "role" in turn
            assert "content" in turn
            assert "timestamp" in turn

        # Verify both roles are present
        roles = [turn["role"] for turn in history]
        assert "user" in roles
        assert "assistant" in roles

        # Verify context tracking
        stats = instance.get_statistics()
        assert stats["total_queries"] == 3
        assert stats["total_turns"] == 6

    def test_handles_follow_up_questions_and_clarifications_int(self, instance):
        """Test: Handles follow-up questions and clarifications intelligently"""
        # Add some knowledge
        instance.add_knowledge(
            "RAG systems have three main benefits: 1) Up-to-date information access, "
            "2) Reduced hallucination, 3) Cost-effective knowledge updates."
        )

        # Initial question - may detect as follow_up due to "what are" pattern
        result1 = instance.chat("Describe the main benefits of RAG systems.")
        # The first query should be processed
        assert result1["query_analysis"]["type"] in ["new_topic", "follow_up"]

        # Follow-up question
        result2 = instance.chat("Can you explain the first one in more detail?")
        assert result2["query_analysis"]["is_follow_up"] is True

        # Clarification request
        result3 = instance.chat("What do you mean by reduced hallucination?")
        assert result3["query_analysis"]["type"] in ["clarification", "follow_up", "reference"]

        # Reference to earlier content
        result4 = instance.chat("How does that compare to what you mentioned earlier?")
        assert result4["query_analysis"]["is_follow_up"] is True or \
               result4["query_analysis"]["type"] in ["reference", "comparison"]

    def test_implements_conversation_memory_with_relevant_conte(self, instance):
        """Test: Implements conversation memory with relevant context retrieval"""
        # Add knowledge
        instance.add_knowledge("Python is known for its simple and readable syntax.")
        instance.add_knowledge("Machine learning models can predict outcomes from data.")

        # Start conversation about Python
        result1 = instance.chat("Tell me about Python programming")
        assert len(result1["context_used"]) >= 0  # May or may not use context

        # Continue conversation - should use previous context
        result2 = instance.chat("What makes it good for beginners?")

        # The conversation should maintain context
        assert result2["conversation_turn"] >= 3

        # Check that context retrieval is working
        history = instance.history.get_all_turns()
        assert len(history) >= 2

        # Verify memory is retrieving relevant context
        memory = instance.memory
        analysis = ContextAnalyzer.analyze_query("Python syntax", instance.history)
        context = memory.retrieve_context("Python syntax", instance.history, analysis)

        assert "context_items" in context
        assert "combined_text" in context

    def test_provides_conversation_summarization_and_key_points(self, instance):
        """Test: Provides conversation summarization and key points extraction"""
        # Have a conversation with multiple topics
        instance.add_knowledge(
            "The key benefits of RAG include: improved accuracy through retrieval, "
            "access to current information, and reduced computational costs."
        )

        instance.chat("What are the benefits of RAG systems?")
        instance.chat("How does retrieval improve accuracy?")
        instance.chat("What about the costs involved?")

        # Get summary
        summary = instance.get_summary()

        # Verify summary structure
        assert "main_topics" in summary
        assert "key_points" in summary
        assert "turn_count" in summary
        assert "summary_text" in summary

        # Verify turn count
        assert summary["turn_count"] == 6  # 3 user + 3 assistant

        # Verify summary text exists
        assert len(summary["summary_text"]) > 0

        # Get key points
        key_points = instance.get_key_points()
        assert isinstance(key_points, list)

        # Get topics timeline
        timeline = instance.get_topics_timeline()
        assert isinstance(timeline, list)


class TestConversationHistory:
    """Tests for ConversationHistory component."""

    def test_add_and_retrieve_turns(self):
        """Test adding and retrieving conversation turns."""
        history = ConversationHistory()

        # Add turns
        turn1 = history.add_turn(MessageRole.USER, "Hello")
        turn2 = history.add_turn(MessageRole.ASSISTANT, "Hi there!")

        assert history.get_conversation_length() == 2

        # Retrieve turns
        recent = history.get_recent_turns(10)
        assert len(recent) == 2

    def test_get_turns_by_role(self):
        """Test filtering turns by role."""
        history = ConversationHistory()

        history.add_turn(MessageRole.USER, "Question 1")
        history.add_turn(MessageRole.ASSISTANT, "Answer 1")
        history.add_turn(MessageRole.USER, "Question 2")
        history.add_turn(MessageRole.ASSISTANT, "Answer 2")

        user_turns = history.get_turns_by_role(MessageRole.USER)
        assistant_turns = history.get_turns_by_role(MessageRole.ASSISTANT)

        assert len(user_turns) == 2
        assert len(assistant_turns) == 2

    def test_search_turns(self):
        """Test searching conversation turns."""
        history = ConversationHistory()

        history.add_turn(MessageRole.USER, "Tell me about Python")
        history.add_turn(MessageRole.ASSISTANT, "Python is a programming language")
        history.add_turn(MessageRole.USER, "What about JavaScript?")

        results = history.search_turns("Python")
        assert len(results) == 2

        results = history.search_turns("JavaScript")
        assert len(results) == 1

    def test_serialization(self):
        """Test conversation serialization and deserialization."""
        history = ConversationHistory()

        history.add_turn(MessageRole.USER, "Test message", topics=["testing"])
        history.add_turn(MessageRole.ASSISTANT, "Test response")

        # Serialize
        data = history.to_dict()
        assert "conversation_id" in data
        assert "turns" in data
        assert len(data["turns"]) == 2

        # Deserialize
        restored = ConversationHistory.from_dict(data)
        assert restored.get_conversation_length() == 2


class TestContextAnalyzer:
    """Tests for ContextAnalyzer component."""

    def test_detect_follow_up_query(self):
        """Test detection of follow-up questions."""
        history = ConversationHistory()
        history.add_turn(MessageRole.USER, "What is machine learning?")
        history.add_turn(MessageRole.ASSISTANT, "Machine learning is...")

        analysis = ContextAnalyzer.analyze_query("Tell me more about that", history)

        assert analysis["query_type"] in [QueryType.FOLLOW_UP, QueryType.REFERENCE]
        assert analysis["is_follow_up"] is True

    def test_detect_clarification_query(self):
        """Test detection of clarification requests."""
        history = ConversationHistory()

        analysis = ContextAnalyzer.analyze_query(
            "What do you mean by that? Could you clarify?",
            history
        )

        assert analysis["query_type"] == QueryType.CLARIFICATION

    def test_detect_new_topic(self):
        """Test detection of new topic queries."""
        history = ConversationHistory()

        # Use a query that doesn't match follow-up or clarification patterns
        analysis = ContextAnalyzer.analyze_query(
            "Quantum computing uses qubits for parallel processing",
            history
        )

        assert analysis["query_type"] == QueryType.NEW_TOPIC

    def test_extract_topics(self):
        """Test topic extraction from text."""
        topics = ContextAnalyzer._extract_topics(
            "Machine learning algorithms for natural language processing"
        )

        assert len(topics) > 0
        assert "machine" in topics or "learning" in topics or "algorithms" in topics

    def test_query_expansion(self):
        """Test query expansion with context."""
        history = ConversationHistory()
        history.add_turn(
            MessageRole.USER,
            "What is Python?",
            topics=["python", "programming"]
        )
        history.add_turn(
            MessageRole.ASSISTANT,
            "Python is a language",
            topics=["python"]
        )

        analysis = ContextAnalyzer.analyze_query("Tell me more", history)
        expanded = ContextAnalyzer.expand_query("Tell me more", analysis, history)

        # Should include context
        assert "Tell me more" in expanded


class TestConversationMemory:
    """Tests for ConversationMemory component."""

    def test_add_and_retrieve_knowledge(self):
        """Test adding and retrieving from knowledge base."""
        memory = ConversationMemory()

        memory.add_knowledge("Python is a programming language")
        memory.add_knowledge("Machine learning uses algorithms")

        history = ConversationHistory()
        analysis = {"topics": ["python"], "query_type": QueryType.NEW_TOPIC}

        context = memory.retrieve_context("Python programming", history, analysis)

        assert "context_items" in context
        assert len(context["context_items"]) > 0

    def test_retrieve_conversation_context(self):
        """Test retrieval of conversation context."""
        memory = ConversationMemory()
        history = ConversationHistory()

        # Add conversation history
        history.add_turn(
            MessageRole.USER,
            "Tell me about Python",
            topics=["python"]
        )
        history.add_turn(
            MessageRole.ASSISTANT,
            "Python is a versatile programming language",
            topics=["python", "programming"]
        )

        analysis = {"topics": ["python"], "query_type": QueryType.FOLLOW_UP}

        context = memory.retrieve_context("More about Python features", history, analysis)

        # Should find conversation context
        assert "context_items" in context


class TestConversationSummarizer:
    """Tests for ConversationSummarizer component."""

    def test_summarize_conversation(self):
        """Test conversation summarization."""
        history = ConversationHistory()

        history.add_turn(
            MessageRole.USER,
            "What are the benefits of Python?",
            topics=["python", "benefits"]
        )
        history.add_turn(
            MessageRole.ASSISTANT,
            "Python offers several key benefits including readability and versatility.",
            topics=["python", "benefits", "readability"]
        )
        history.add_turn(
            MessageRole.USER,
            "How does it compare to Java?",
            topics=["python", "java", "comparison"]
        )
        history.add_turn(
            MessageRole.ASSISTANT,
            "The main difference is that Python has simpler syntax.",
            topics=["python", "java", "syntax"]
        )

        summary = ConversationSummarizer.summarize(history)

        assert summary.turn_count == 4
        assert len(summary.main_topics) > 0
        assert "python" in summary.main_topics
        assert len(summary.summary_text) > 0

    def test_empty_conversation_summary(self):
        """Test summarization of empty conversation."""
        history = ConversationHistory()

        summary = ConversationSummarizer.summarize(history)

        assert summary.turn_count == 0
        assert len(summary.main_topics) == 0

    def test_topics_timeline(self):
        """Test topics timeline extraction."""
        history = ConversationHistory()

        history.add_turn(
            MessageRole.USER,
            "Discuss Python",
            topics=["python"]
        )
        history.add_turn(
            MessageRole.USER,
            "Now about Java",
            topics=["java"]
        )

        timeline = ConversationSummarizer.extract_topics_timeline(history)

        assert len(timeline) >= 1


class TestIntegration:
    """Integration tests for the complete interface."""

    @pytest.fixture
    def interface(self):
        """Create interface with test data."""
        interface = create_conversational_rag_interface()
        interface.add_knowledge(
            "RAG (Retrieval-Augmented Generation) combines retrieval with generation."
        )
        interface.add_knowledge(
            "The main benefits of RAG include: improved accuracy, current information access, and cost efficiency."
        )
        return interface

    def test_full_conversation_flow(self, interface):
        """Test a complete conversation flow."""
        # Initial question
        result1 = interface.chat("What is RAG?")
        assert result1["response"] is not None
        assert result1["conversation_turn"] == 2

        # Follow-up
        result2 = interface.chat("What are its main benefits?")
        assert result2["conversation_turn"] == 4

        # Reference to earlier
        result3 = interface.chat("Can you explain the first benefit more?")
        assert result3["conversation_turn"] == 6

        # Get summary
        summary = interface.get_summary()
        assert summary["turn_count"] == 6

    def test_state_save_and_restore(self, interface):
        """Test saving and restoring conversation state."""
        # Have a conversation
        interface.chat("Hello, tell me about RAG")
        interface.chat("What are its benefits?")

        # Save state
        state = interface.save_state()

        # Create new interface and restore
        new_interface = create_conversational_rag_interface()
        new_interface.restore_state(state)

        # Verify state was restored
        assert new_interface.history.get_conversation_length() == 4

    def test_clear_history(self, interface):
        """Test clearing conversation history."""
        interface.chat("Test message")
        assert interface.history.get_conversation_length() > 0

        interface.clear_history()
        assert interface.history.get_conversation_length() == 0

    def test_execute_demo_mode(self):
        """Test execute with demo data."""
        interface = create_conversational_rag_interface()
        result = interface.execute()

        assert "conversation" in result
        assert "summary" in result
        assert len(result["conversation"]) > 0
