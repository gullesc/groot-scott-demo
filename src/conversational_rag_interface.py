"""
Conversational RAG Interface

A chat-like interface that maintains conversation context across multiple turns,
handles follow-up questions intelligently, and provides conversation summarization.
This module implements conversation memory, context-aware retrieval, and intelligent
response generation using only the Python standard library.
"""

import re
import uuid
import json
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import Counter


class MessageRole(Enum):
    """Role of a message in the conversation."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class QueryType(Enum):
    """Types of user queries in conversation context."""
    NEW_TOPIC = "new_topic"           # Fresh topic, no context needed
    FOLLOW_UP = "follow_up"           # Continues previous topic
    CLARIFICATION = "clarification"   # Asks for clarification
    REFERENCE = "reference"           # References earlier conversation
    COMPARISON = "comparison"         # Compares topics from conversation


@dataclass
class ConversationTurn:
    """Represents a single turn in the conversation."""
    turn_id: str
    role: MessageRole
    content: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    topics: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    context_used: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "turn_id": self.turn_id,
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "topics": self.topics,
            "entities": self.entities,
            "context_used": self.context_used
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationTurn":
        """Create from dictionary."""
        return cls(
            turn_id=data["turn_id"],
            role=MessageRole(data["role"]),
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {}),
            topics=data.get("topics", []),
            entities=data.get("entities", []),
            context_used=data.get("context_used", [])
        )


@dataclass
class ConversationSummary:
    """Summary of a conversation."""
    main_topics: List[str]
    key_points: List[str]
    entities_mentioned: List[str]
    turn_count: int
    summary_text: str


# ============================================================================
# CONVERSATION HISTORY
# ============================================================================

class ConversationHistory:
    """
    Manages dialogue history and turn tracking.

    Stores and retrieves conversation turns with timestamps,
    topics, and metadata for context management.
    """

    def __init__(self, max_turns: int = 100):
        """
        Initialize conversation history.

        Args:
            max_turns: Maximum turns to keep in history
        """
        self._turns: List[ConversationTurn] = []
        self._max_turns = max_turns
        self._conversation_id = str(uuid.uuid4())
        self._created_at = datetime.now()

    def add_turn(
        self,
        role: MessageRole,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        topics: Optional[List[str]] = None,
        entities: Optional[List[str]] = None,
        context_used: Optional[List[str]] = None
    ) -> ConversationTurn:
        """
        Add a new turn to the conversation.

        Args:
            role: Who sent the message
            content: Message content
            metadata: Additional metadata
            topics: Topics discussed in this turn
            entities: Entities mentioned
            context_used: Context sources used

        Returns:
            The created ConversationTurn
        """
        turn = ConversationTurn(
            turn_id=str(uuid.uuid4()),
            role=role,
            content=content,
            timestamp=datetime.now(),
            metadata=metadata or {},
            topics=topics or [],
            entities=entities or [],
            context_used=context_used or []
        )

        self._turns.append(turn)

        # Trim if exceeds max
        if len(self._turns) > self._max_turns:
            self._turns = self._turns[-self._max_turns:]

        return turn

    def get_recent_turns(self, count: int = 10) -> List[ConversationTurn]:
        """Get most recent turns."""
        return self._turns[-count:] if self._turns else []

    def get_all_turns(self) -> List[ConversationTurn]:
        """Get all turns in the conversation."""
        return self._turns.copy()

    def get_turn_by_id(self, turn_id: str) -> Optional[ConversationTurn]:
        """Get a specific turn by ID."""
        for turn in self._turns:
            if turn.turn_id == turn_id:
                return turn
        return None

    def get_turns_by_role(self, role: MessageRole) -> List[ConversationTurn]:
        """Get all turns from a specific role."""
        return [t for t in self._turns if t.role == role]

    def search_turns(self, query: str) -> List[ConversationTurn]:
        """Search turns for matching content."""
        query_lower = query.lower()
        return [
            t for t in self._turns
            if query_lower in t.content.lower()
        ]

    def get_conversation_length(self) -> int:
        """Get number of turns."""
        return len(self._turns)

    def get_last_user_turn(self) -> Optional[ConversationTurn]:
        """Get the most recent user message."""
        for turn in reversed(self._turns):
            if turn.role == MessageRole.USER:
                return turn
        return None

    def get_last_assistant_turn(self) -> Optional[ConversationTurn]:
        """Get the most recent assistant message."""
        for turn in reversed(self._turns):
            if turn.role == MessageRole.ASSISTANT:
                return turn
        return None

    def clear(self) -> None:
        """Clear all conversation history."""
        self._turns.clear()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize conversation to dictionary."""
        return {
            "conversation_id": self._conversation_id,
            "created_at": self._created_at.isoformat(),
            "turns": [t.to_dict() for t in self._turns]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationHistory":
        """Restore conversation from dictionary."""
        history = cls()
        history._conversation_id = data["conversation_id"]
        history._created_at = datetime.fromisoformat(data["created_at"])
        history._turns = [ConversationTurn.from_dict(t) for t in data["turns"]]
        return history


# ============================================================================
# CONTEXT ANALYZER
# ============================================================================

class ContextAnalyzer:
    """
    Analyzes queries for references to previous conversation.

    Detects follow-up questions, references to earlier topics,
    and expands abbreviated queries using conversation context.
    """

    # Patterns indicating follow-up questions
    FOLLOW_UP_PATTERNS = [
        r"^(what|how|why|can you|could you|tell me more|explain|elaborate)",
        r"^(and|also|additionally|furthermore|moreover)",
        r"(about (that|this|it|them))",
        r"^(yes|no|right|exactly|correct)",
        r"(the (same|previous|last|earlier))",
    ]

    # Patterns indicating references to earlier content
    REFERENCE_PATTERNS = [
        r"\b(that|this|it|they|them|those|these)\b",
        r"\b(the (first|second|third|last|previous) (one|point|item))\b",
        r"\b(as (you|we) (mentioned|discussed|said|talked about))\b",
        r"\b(earlier|before|previously|above)\b",
        r"\bthe (other|another) (one|option|approach)\b",
    ]

    # Patterns indicating clarification requests
    CLARIFICATION_PATTERNS = [
        r"\b(what do you mean|clarify|explain|could you be more specific)\b",
        r"\b(i (don't|do not) understand)\b",
        r"\b(what (exactly|specifically))\b",
        r"\bcan you rephrase\b",
    ]

    # Patterns indicating comparison requests
    COMPARISON_PATTERNS = [
        r"\b(compare|versus|vs|difference between|similarities)\b",
        r"\b(how does .+ compare to)\b",
        r"\b(which (one|is better|should i))\b",
    ]

    @classmethod
    def analyze_query(
        cls,
        query: str,
        conversation_history: ConversationHistory
    ) -> Dict[str, Any]:
        """
        Analyze a query in the context of conversation history.

        Args:
            query: The user's query
            conversation_history: Previous conversation turns

        Returns:
            Analysis results including query type and references
        """
        query_lower = query.lower().strip()

        # Detect query type
        query_type = cls._detect_query_type(query_lower)

        # Find references to previous content
        references = cls._find_references(query_lower, conversation_history)

        # Extract topics and entities from query
        topics = cls._extract_topics(query)
        entities = cls._extract_entities(query)

        # Check if query needs context expansion
        needs_expansion = cls._needs_context_expansion(query_lower)

        # Get relevant context from history
        relevant_context = cls._get_relevant_context(
            query_lower, conversation_history, topics
        )

        return {
            "query_type": query_type,
            "references": references,
            "topics": topics,
            "entities": entities,
            "needs_expansion": needs_expansion,
            "relevant_context": relevant_context,
            "is_follow_up": query_type in (QueryType.FOLLOW_UP, QueryType.CLARIFICATION)
        }

    @classmethod
    def _detect_query_type(cls, query: str) -> QueryType:
        """Detect the type of query."""
        # Check for clarification
        for pattern in cls.CLARIFICATION_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                return QueryType.CLARIFICATION

        # Check for comparison
        for pattern in cls.COMPARISON_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                return QueryType.COMPARISON

        # Check for follow-up
        for pattern in cls.FOLLOW_UP_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                return QueryType.FOLLOW_UP

        # Check for references
        for pattern in cls.REFERENCE_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                return QueryType.REFERENCE

        return QueryType.NEW_TOPIC

    @classmethod
    def _find_references(
        cls,
        query: str,
        history: ConversationHistory
    ) -> List[Dict[str, Any]]:
        """Find references to previous conversation content."""
        references = []

        # Look for pronouns and demonstratives
        pronouns = re.findall(r'\b(it|that|this|they|them|those|these)\b', query)

        if pronouns:
            # Try to resolve references from recent turns
            recent_turns = history.get_recent_turns(5)
            for turn in reversed(recent_turns):
                if turn.topics:
                    references.append({
                        "type": "topic_reference",
                        "turn_id": turn.turn_id,
                        "topics": turn.topics,
                        "confidence": 0.7
                    })
                    break

        # Look for explicit references
        if re.search(r'(earlier|before|previously|you (mentioned|said))', query):
            for turn in history.get_turns_by_role(MessageRole.ASSISTANT)[-3:]:
                references.append({
                    "type": "explicit_reference",
                    "turn_id": turn.turn_id,
                    "content_preview": turn.content[:100],
                    "confidence": 0.8
                })

        return references

    @classmethod
    def _extract_topics(cls, text: str) -> List[str]:
        """Extract topic keywords from text."""
        # Simple keyword extraction based on word frequency and length
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())

        # Filter common words
        stop_words = {
            'that', 'this', 'what', 'when', 'where', 'which', 'with',
            'have', 'from', 'about', 'would', 'could', 'should', 'there',
            'their', 'more', 'some', 'other', 'very', 'just', 'also'
        }

        topics = [w for w in words if w not in stop_words]

        # Return unique topics
        return list(dict.fromkeys(topics))[:5]

    @classmethod
    def _extract_entities(cls, text: str) -> List[str]:
        """Extract named entities from text."""
        # Simple pattern matching for entities (capitalized words)
        entities = re.findall(r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b', text)
        return list(set(entities))[:5]

    @classmethod
    def _needs_context_expansion(cls, query: str) -> bool:
        """Check if query is too short or ambiguous."""
        words = query.split()
        if len(words) <= 3:
            return True

        # Check for high pronoun ratio
        pronouns = ['it', 'that', 'this', 'they', 'them', 'those', 'these', 'what']
        pronoun_count = sum(1 for w in words if w in pronouns)
        if pronoun_count / len(words) > 0.3:
            return True

        return False

    @classmethod
    def _get_relevant_context(
        cls,
        query: str,
        history: ConversationHistory,
        topics: List[str]
    ) -> List[Dict[str, Any]]:
        """Get relevant context from conversation history."""
        relevant = []

        for turn in history.get_recent_turns(10):
            relevance_score = 0.0

            # Check topic overlap
            topic_overlap = len(set(topics) & set(turn.topics))
            if topic_overlap > 0:
                relevance_score += 0.3 * topic_overlap

            # Check content similarity
            turn_words = set(turn.content.lower().split())
            query_words = set(query.split())
            word_overlap = len(turn_words & query_words)
            if word_overlap > 0:
                relevance_score += 0.1 * word_overlap

            if relevance_score > 0.1:
                relevant.append({
                    "turn_id": turn.turn_id,
                    "role": turn.role.value,
                    "content": turn.content[:200],
                    "relevance_score": relevance_score,
                    "topics": turn.topics
                })

        # Sort by relevance and return top results
        relevant.sort(key=lambda x: x["relevance_score"], reverse=True)
        return relevant[:3]

    @classmethod
    def expand_query(
        cls,
        query: str,
        analysis: Dict[str, Any],
        history: ConversationHistory
    ) -> str:
        """
        Expand a query with context from conversation history.

        Args:
            query: Original query
            analysis: Query analysis results
            history: Conversation history

        Returns:
            Expanded query with context
        """
        if not analysis["needs_expansion"]:
            return query

        # Get recent topics for context
        recent_topics = []
        for turn in history.get_recent_turns(3):
            recent_topics.extend(turn.topics)

        if recent_topics:
            # Add topic context to query
            unique_topics = list(dict.fromkeys(recent_topics))[:3]
            context_str = ", ".join(unique_topics)
            return f"{query} (in context of: {context_str})"

        return query


# ============================================================================
# CONVERSATION MEMORY
# ============================================================================

class ConversationMemory:
    """
    Retrieves relevant context from conversation history and knowledge base.

    Combines conversation context with external knowledge for
    comprehensive response generation.
    """

    def __init__(self, max_context_length: int = 2000):
        """
        Initialize conversation memory.

        Args:
            max_context_length: Maximum characters for context
        """
        self._max_context_length = max_context_length
        self._knowledge_base: List[Dict[str, Any]] = []

    def add_knowledge(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add content to the knowledge base."""
        self._knowledge_base.append({
            "id": str(uuid.uuid4()),
            "content": content,
            "metadata": metadata or {}
        })

    def retrieve_context(
        self,
        query: str,
        conversation_history: ConversationHistory,
        query_analysis: Dict[str, Any],
        max_items: int = 5
    ) -> Dict[str, Any]:
        """
        Retrieve relevant context from both conversation and knowledge base.

        Args:
            query: Current query
            conversation_history: Conversation history
            query_analysis: Analysis of the query
            max_items: Maximum context items to return

        Returns:
            Retrieved context with sources
        """
        context_items = []

        # Get conversation context
        conv_context = self._retrieve_conversation_context(
            query, conversation_history, query_analysis
        )
        context_items.extend(conv_context)

        # Get knowledge base context
        kb_context = self._retrieve_knowledge_context(query)
        context_items.extend(kb_context)

        # Sort by relevance and limit
        context_items.sort(key=lambda x: x["relevance"], reverse=True)
        context_items = context_items[:max_items]

        # Build combined context
        combined_text = self._build_context_text(context_items)

        return {
            "context_items": context_items,
            "combined_text": combined_text,
            "sources": [item["source"] for item in context_items],
            "total_items": len(context_items)
        }

    def _retrieve_conversation_context(
        self,
        query: str,
        history: ConversationHistory,
        analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant context from conversation history."""
        items = []
        # Use regex to extract words, stripping punctuation
        query_words = set(re.findall(r'\b[a-zA-Z]{2,}\b', query.lower()))

        for turn in history.get_recent_turns(10):
            # Calculate relevance
            turn_words = set(re.findall(r'\b[a-zA-Z]{2,}\b', turn.content.lower()))
            word_overlap = len(query_words & turn_words)
            topic_overlap = len(set(analysis.get("topics", [])) & set(turn.topics))

            relevance = (word_overlap * 0.1) + (topic_overlap * 0.3)

            # Boost recent turns
            recency_boost = 0.1 if turn == history.get_last_assistant_turn() else 0

            if relevance > 0 or recency_boost > 0:
                items.append({
                    "source": f"conversation:{turn.turn_id}",
                    "content": turn.content,
                    "relevance": relevance + recency_boost,
                    "type": "conversation",
                    "metadata": {"role": turn.role.value}
                })

        return items

    def _retrieve_knowledge_context(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve relevant context from knowledge base."""
        items = []
        # Use regex to extract words, stripping punctuation
        query_words = set(re.findall(r'\b[a-zA-Z]{2,}\b', query.lower()))

        for kb_item in self._knowledge_base:
            content_words = set(re.findall(r'\b[a-zA-Z]{2,}\b', kb_item["content"].lower()))
            word_overlap = len(query_words & content_words)

            if word_overlap > 0:
                relevance = word_overlap * 0.15
                items.append({
                    "source": f"knowledge:{kb_item['id']}",
                    "content": kb_item["content"],
                    "relevance": relevance,
                    "type": "knowledge",
                    "metadata": kb_item.get("metadata", {})
                })

        return items

    def _build_context_text(self, items: List[Dict[str, Any]]) -> str:
        """Build combined context text from items."""
        parts = []
        total_length = 0

        for item in items:
            content = item["content"]
            if total_length + len(content) > self._max_context_length:
                # Truncate if needed
                remaining = self._max_context_length - total_length
                if remaining > 100:
                    content = content[:remaining] + "..."
                else:
                    break

            parts.append(f"[{item['type'].upper()}]: {content}")
            total_length += len(content)

        return "\n\n".join(parts)


# ============================================================================
# RESPONSE GENERATOR
# ============================================================================

class ResponseGenerator:
    """
    Generates context-aware responses.

    Creates responses that acknowledge conversation history
    and maintain dialogue continuity.
    """

    @classmethod
    def generate_response(
        cls,
        query: str,
        query_analysis: Dict[str, Any],
        context: Dict[str, Any],
        conversation_history: ConversationHistory
    ) -> Dict[str, Any]:
        """
        Generate a context-aware response.

        Args:
            query: User query
            query_analysis: Analysis of the query
            context: Retrieved context
            conversation_history: Conversation history

        Returns:
            Generated response with metadata
        """
        # Determine response style based on query type
        query_type = query_analysis["query_type"]

        # Build response prefix based on query type
        prefix = cls._get_response_prefix(query_type, query_analysis)

        # Generate main response content
        main_content = cls._generate_main_content(
            query, context, query_type, query_analysis
        )

        # Add continuity elements
        continuity = cls._add_continuity(
            query_analysis, conversation_history
        )

        # Combine response
        response_text = f"{prefix}{main_content}{continuity}"

        # Extract topics from response
        topics = ContextAnalyzer._extract_topics(response_text)

        return {
            "response": response_text,
            "query_type": query_type.value,
            "topics": topics,
            "context_used": context.get("sources", []),
            "is_follow_up_response": query_analysis.get("is_follow_up", False)
        }

    @classmethod
    def _get_response_prefix(
        cls,
        query_type: QueryType,
        analysis: Dict[str, Any]
    ) -> str:
        """Get appropriate response prefix based on query type."""
        if query_type == QueryType.FOLLOW_UP:
            return "Continuing from our discussion, "
        elif query_type == QueryType.CLARIFICATION:
            return "To clarify, "
        elif query_type == QueryType.REFERENCE:
            return "Regarding what we discussed earlier, "
        elif query_type == QueryType.COMPARISON:
            return "Comparing these options, "
        return ""

    @classmethod
    def _generate_main_content(
        cls,
        query: str,
        context: Dict[str, Any],
        query_type: QueryType,
        query_analysis: Dict[str, Any]
    ) -> str:
        """
        Generate main response content based on query and context.

        In a real system, this would call an LLM. For educational purposes,
        we simulate intelligent response generation by:
        1. Finding the most relevant context based on query keywords
        2. Extracting relevant portions from that context
        3. Formatting appropriately for the query type
        """
        context_items = context.get("context_items", [])
        query_topics = query_analysis.get("topics", [])
        query_lower = query.lower()

        # Extract query keywords for matching
        query_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', query_lower))

        # Find the best matching context item
        best_item = cls._find_best_context_match(context_items, query_words, query_topics)

        if best_item:
            # Extract the most relevant portion from the best item
            relevant_text = cls._extract_relevant_portion(
                best_item["content"], query_words, query_lower
            )
            return cls._format_response(relevant_text, query_type, query)
        else:
            # No context found - generate a helpful response
            return cls._generate_no_context_response(query, query_type)

    # Stop words to ignore when matching queries to content
    STOP_WORDS = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
        'dare', 'ought', 'used', 'what', 'who', 'whom', 'which', 'when',
        'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more',
        'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
        'same', 'so', 'than', 'too', 'very', 'just', 'also', 'now', 'here',
        'there', 'then', 'once', 'about', 'after', 'before', 'above', 'below',
        'between', 'into', 'through', 'during', 'under', 'again', 'further',
        'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
        'she', 'her', 'hers', 'it', 'its', 'they', 'them', 'their', 'theirs',
        'this', 'that', 'these', 'those', 'i', 'me', 'my', 'mine', 'we', 'us',
        'our', 'ours', 'know', 'tell', 'explain', 'describe', 'give', 'show',
        'continuing', 'discussion', 'mentioned', 'earlier', 'regarding'
    }

    @classmethod
    def _find_best_context_match(
        cls,
        context_items: List[Dict[str, Any]],
        query_words: Set[str],
        query_topics: List[str]
    ) -> Optional[Dict[str, Any]]:
        """Find the context item that best matches the query."""
        if not context_items:
            return None

        # Filter out stop words from query for better matching
        meaningful_query_words = query_words - cls.STOP_WORDS

        best_score = 0
        best_item = None

        for item in context_items:
            content_lower = item["content"].lower()
            content_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', content_lower))
            meaningful_content_words = content_words - cls.STOP_WORDS

            # Calculate match score using meaningful words only
            word_overlap = len(meaningful_query_words & meaningful_content_words)
            topic_matches = sum(1 for t in query_topics if t.lower() in content_lower)

            # Knowledge items get a strong boost over conversation context
            # Conversation context should only be used if highly relevant
            if item.get("type") == "knowledge":
                type_boost = 3.0  # Strong preference for knowledge base
            else:
                type_boost = 0.5  # Conversation context needs to be very relevant

            score = (word_overlap * 0.3 + topic_matches * 0.5) * type_boost

            if score > best_score:
                best_score = score
                best_item = item

        # Only return a match if there's actual relevance - don't fall back to unrelated content
        return best_item if best_score > 0 else None

    # Prefixes to strip from content to avoid duplication
    RESPONSE_PREFIXES = [
        "Continuing from our discussion, ",
        "To clarify, ",
        "Regarding what we discussed earlier, ",
        "Comparing these options, ",
        "As mentioned: ",
        "Let me explain: ",
        "Here's what I found for comparison: ",
    ]

    @classmethod
    def _extract_relevant_portion(
        cls,
        content: str,
        query_words: Set[str],
        query_lower: str
    ) -> str:
        """Extract the most relevant portion of content based on query."""
        # Strip any existing response prefixes to avoid duplication
        cleaned_content = content
        for prefix in cls.RESPONSE_PREFIXES:
            if cleaned_content.startswith(prefix):
                cleaned_content = cleaned_content[len(prefix):]

        # Split content into sentences
        sentences = re.split(r'(?<=[.!?])\s+', cleaned_content)

        if not sentences:
            return content[:300]

        # Score each sentence by relevance to query
        scored_sentences = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            sentence_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', sentence_lower))

            # Calculate relevance score
            word_overlap = len(query_words & sentence_words)

            # Check for specific query terms
            specific_matches = sum(1 for w in query_words if w in sentence_lower and len(w) > 4)

            score = word_overlap + specific_matches * 2
            scored_sentences.append((sentence, score))

        # Sort by score and take top relevant sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)

        # Build response from most relevant sentences
        relevant_parts = []
        total_length = 0
        max_length = 400

        for sentence, score in scored_sentences:
            if total_length + len(sentence) > max_length:
                break
            relevant_parts.append(sentence.strip())
            total_length += len(sentence)

        if relevant_parts:
            return " ".join(relevant_parts)

        # Fallback to first part of content
        return content[:300] + "..." if len(content) > 300 else content

    @classmethod
    def _format_response(
        cls,
        relevant_text: str,
        query_type: QueryType,
        query: str
    ) -> str:
        """Format the response based on query type."""
        # Clean up the text
        relevant_text = relevant_text.strip()

        if query_type == QueryType.COMPARISON:
            return f"Here's what I found for comparison: {relevant_text}"
        elif query_type == QueryType.CLARIFICATION:
            return f"Let me explain: {relevant_text}"
        elif query_type == QueryType.FOLLOW_UP:
            return relevant_text
        elif query_type == QueryType.REFERENCE:
            return f"As mentioned: {relevant_text}"
        else:
            # NEW_TOPIC - provide informative response
            return relevant_text

    @classmethod
    def _generate_no_context_response(
        cls,
        query: str,
        query_type: QueryType
    ) -> str:
        """Generate a helpful response when no context is available."""
        query_topics = ContextAnalyzer._extract_topics(query)

        if query_topics:
            topic_str = ", ".join(query_topics[:2])
            return f"I don't have specific information about {topic_str} in my knowledge base. Could you provide more context or ask about a related topic?"
        else:
            return "I'd be happy to help, but I don't have relevant information in my current knowledge base. Could you provide more details about what you're looking for?"

    @classmethod
    def _add_continuity(
        cls,
        analysis: Dict[str, Any],
        history: ConversationHistory
    ) -> str:
        """Add continuity elements to maintain conversation flow."""
        if analysis.get("is_follow_up") and history.get_conversation_length() > 2:
            return " Would you like me to elaborate on any specific aspect?"
        return ""


# ============================================================================
# CONVERSATION SUMMARIZER
# ============================================================================

class ConversationSummarizer:
    """
    Creates conversation summaries and extracts key points.

    Analyzes conversation to identify main topics, important points,
    and overall conversation arc.
    """

    @classmethod
    def summarize(cls, history: ConversationHistory) -> ConversationSummary:
        """
        Generate a summary of the conversation.

        Args:
            history: Conversation history to summarize

        Returns:
            ConversationSummary with topics and key points
        """
        turns = history.get_all_turns()

        if not turns:
            return ConversationSummary(
                main_topics=[],
                key_points=[],
                entities_mentioned=[],
                turn_count=0,
                summary_text="No conversation to summarize."
            )

        # Extract all topics
        all_topics = []
        all_entities = []
        for turn in turns:
            all_topics.extend(turn.topics)
            all_entities.extend(turn.entities)

        # Find main topics (most frequent)
        topic_counts = Counter(all_topics)
        main_topics = [t for t, _ in topic_counts.most_common(5)]

        # Find key entities
        entity_counts = Counter(all_entities)
        main_entities = [e for e, _ in entity_counts.most_common(5)]

        # Extract key points from assistant responses
        key_points = cls._extract_key_points(turns)

        # Generate summary text
        summary_text = cls._generate_summary_text(
            turns, main_topics, key_points
        )

        return ConversationSummary(
            main_topics=main_topics,
            key_points=key_points,
            entities_mentioned=main_entities,
            turn_count=len(turns),
            summary_text=summary_text
        )

    @classmethod
    def _extract_key_points(cls, turns: List[ConversationTurn]) -> List[str]:
        """Extract key points from conversation turns."""
        key_points = []

        for turn in turns:
            if turn.role == MessageRole.ASSISTANT:
                # Extract sentences that might be key points
                sentences = re.split(r'[.!?]', turn.content)
                for sentence in sentences:
                    sentence = sentence.strip()
                    if len(sentence) > 20 and len(sentence) < 200:
                        # Check for key point indicators
                        if any(indicator in sentence.lower() for indicator in [
                            'important', 'key', 'main', 'primary',
                            'first', 'second', 'third',
                            'benefit', 'advantage', 'reason'
                        ]):
                            key_points.append(sentence)

        return key_points[:5]  # Limit to 5 key points

    @classmethod
    def _generate_summary_text(
        cls,
        turns: List[ConversationTurn],
        main_topics: List[str],
        key_points: List[str]
    ) -> str:
        """Generate a textual summary of the conversation."""
        parts = []

        # Add topic summary
        if main_topics:
            topics_str = ", ".join(main_topics[:3])
            parts.append(f"This conversation covered topics including: {topics_str}.")

        # Add turn count
        user_turns = sum(1 for t in turns if t.role == MessageRole.USER)
        parts.append(f"The discussion spanned {user_turns} user messages.")

        # Add key points summary
        if key_points:
            parts.append(f"Key points discussed: {key_points[0][:100]}...")

        return " ".join(parts) if parts else "Brief conversation with minimal content."

    @classmethod
    def extract_topics_timeline(
        cls,
        history: ConversationHistory
    ) -> List[Dict[str, Any]]:
        """
        Extract a timeline of topics discussed.

        Args:
            history: Conversation history

        Returns:
            List of topic transitions with timestamps
        """
        timeline = []
        current_topics = set()

        for turn in history.get_all_turns():
            new_topics = set(turn.topics) - current_topics

            if new_topics:
                timeline.append({
                    "timestamp": turn.timestamp.isoformat(),
                    "turn_id": turn.turn_id,
                    "new_topics": list(new_topics),
                    "role": turn.role.value
                })
                current_topics.update(new_topics)

        return timeline


# ============================================================================
# MAIN CONVERSATIONAL RAG INTERFACE
# ============================================================================

class ConversationalRagInterface:
    """
    Conversational RAG Interface

    A chat-like interface that maintains conversation context across
    multiple turns, handles follow-up questions intelligently, and
    provides conversation summarization and key points extraction.
    """

    def __init__(self, max_history_turns: int = 50):
        """
        Initialize the conversational RAG interface.

        Args:
            max_history_turns: Maximum conversation turns to maintain
        """
        self.history = ConversationHistory(max_turns=max_history_turns)
        self.context_analyzer = ContextAnalyzer()
        self.memory = ConversationMemory()
        self.response_generator = ResponseGenerator()
        self.summarizer = ConversationSummarizer()

        # Statistics
        self._stats = {
            "total_queries": 0,
            "follow_up_queries": 0,
            "clarification_queries": 0,
            "new_topic_queries": 0
        }

    def add_knowledge(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add content to the knowledge base.

        Args:
            content: Knowledge content
            metadata: Optional metadata
        """
        self.memory.add_knowledge(content, metadata)

    def chat(
        self,
        user_message: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a user message and generate a response.

        Args:
            user_message: User's message
            metadata: Optional message metadata

        Returns:
            Response with context and metadata
        """
        # Analyze the query
        analysis = self.context_analyzer.analyze_query(
            user_message, self.history
        )

        # Update statistics
        self._stats["total_queries"] += 1
        if analysis["query_type"] == QueryType.FOLLOW_UP:
            self._stats["follow_up_queries"] += 1
        elif analysis["query_type"] == QueryType.CLARIFICATION:
            self._stats["clarification_queries"] += 1
        elif analysis["query_type"] == QueryType.NEW_TOPIC:
            self._stats["new_topic_queries"] += 1

        # Expand query if needed
        expanded_query = self.context_analyzer.expand_query(
            user_message, analysis, self.history
        )

        # Retrieve context
        context = self.memory.retrieve_context(
            expanded_query, self.history, analysis
        )

        # Generate response
        response_data = self.response_generator.generate_response(
            user_message, analysis, context, self.history
        )

        # Add user turn to history
        user_turn = self.history.add_turn(
            role=MessageRole.USER,
            content=user_message,
            metadata=metadata,
            topics=analysis["topics"],
            entities=analysis["entities"]
        )

        # Add assistant turn to history
        assistant_turn = self.history.add_turn(
            role=MessageRole.ASSISTANT,
            content=response_data["response"],
            topics=response_data["topics"],
            context_used=response_data["context_used"]
        )

        return {
            "response": response_data["response"],
            "query_analysis": {
                "type": analysis["query_type"].value,
                "is_follow_up": analysis["is_follow_up"],
                "topics": analysis["topics"],
                "references": analysis["references"]
            },
            "context_used": context["sources"],
            "conversation_turn": self.history.get_conversation_length(),
            "user_turn_id": user_turn.turn_id,
            "assistant_turn_id": assistant_turn.turn_id
        }

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current conversation.

        Returns:
            Conversation summary
        """
        summary = self.summarizer.summarize(self.history)
        return {
            "main_topics": summary.main_topics,
            "key_points": summary.key_points,
            "entities_mentioned": summary.entities_mentioned,
            "turn_count": summary.turn_count,
            "summary_text": summary.summary_text
        }

    def get_key_points(self) -> List[str]:
        """
        Extract key points from the conversation.

        Returns:
            List of key points
        """
        summary = self.summarizer.summarize(self.history)
        return summary.key_points

    def get_topics_timeline(self) -> List[Dict[str, Any]]:
        """
        Get timeline of topics discussed.

        Returns:
            Topic timeline
        """
        return self.summarizer.extract_topics_timeline(self.history)

    def get_conversation_history(
        self,
        recent_only: bool = True,
        count: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get conversation history.

        Args:
            recent_only: Only get recent turns
            count: Number of turns if recent_only

        Returns:
            List of conversation turns
        """
        if recent_only:
            turns = self.history.get_recent_turns(count)
        else:
            turns = self.history.get_all_turns()

        return [t.to_dict() for t in turns]

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.history.clear()
        self._stats = {
            "total_queries": 0,
            "follow_up_queries": 0,
            "clarification_queries": 0,
            "new_topic_queries": 0
        }

    def save_state(self) -> Dict[str, Any]:
        """
        Save conversation state for later restoration.

        Returns:
            Serialized conversation state
        """
        return {
            "history": self.history.to_dict(),
            "stats": self._stats
        }

    def restore_state(self, state: Dict[str, Any]) -> None:
        """
        Restore conversation from saved state.

        Args:
            state: Previously saved state
        """
        self.history = ConversationHistory.from_dict(state["history"])
        self._stats = state.get("stats", self._stats)

    def get_statistics(self) -> Dict[str, Any]:
        """Get conversation statistics."""
        return {
            **self._stats,
            "total_turns": self.history.get_conversation_length(),
            "user_messages": len(self.history.get_turns_by_role(MessageRole.USER)),
            "assistant_messages": len(self.history.get_turns_by_role(MessageRole.ASSISTANT))
        }

    def execute(
        self,
        messages: Optional[List[str]] = None,
        knowledge: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Main entry point for the conversational RAG interface.

        Args:
            messages: List of user messages to process
            knowledge: Knowledge base content to add

        Returns:
            Conversation results with summary
        """
        # Add knowledge if provided
        if knowledge:
            for content in knowledge:
                self.add_knowledge(content)

        # Use demo messages if none provided
        if messages is None:
            messages = [
                "What are the main benefits of RAG systems?",
                "Can you explain the first one in more detail?",
                "How does this compare to fine-tuning?",
                "What about the cost considerations?"
            ]

        # Add demo knowledge
        if not knowledge:
            demo_knowledge = [
                "RAG (Retrieval-Augmented Generation) systems offer several key benefits: 1) Access to up-to-date information beyond the model's training data, 2) Reduced hallucination through grounded responses, 3) More cost-effective than fine-tuning for many use cases.",
                "Fine-tuning involves training the model on specific data, which is more expensive and requires more technical expertise. RAG provides a lighter-weight alternative that can be updated in real-time.",
                "Cost considerations for RAG include: storage costs for the knowledge base, compute costs for retrieval operations, and API costs for the generation step."
            ]
            for content in demo_knowledge:
                self.add_knowledge(content)

        # Process each message
        responses = []
        for message in messages:
            result = self.chat(message)
            responses.append({
                "user_message": message,
                "assistant_response": result["response"],
                "query_type": result["query_analysis"]["type"],
                "is_follow_up": result["query_analysis"]["is_follow_up"]
            })

        # Get summary
        summary = self.get_summary()

        return {
            "conversation": responses,
            "summary": summary,
            "statistics": self.get_statistics(),
            "topics_timeline": self.get_topics_timeline()
        }


def create_conversational_rag_interface() -> ConversationalRagInterface:
    """
    Factory function for creating ConversationalRagInterface instances.

    Returns:
        ConversationalRagInterface: A new instance of ConversationalRagInterface
    """
    return ConversationalRagInterface()
