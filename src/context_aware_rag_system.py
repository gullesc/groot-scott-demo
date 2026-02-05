"""
Context-Aware RAG System

An intelligent system that adapts prompts based on retrieval quality and query complexity.
This module implements dynamic prompt strategy selection based on confidence scores,
query classification, and context management for optimal RAG responses.
"""

from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass
import re


class QueryType(Enum):
    """Types of queries for classification."""
    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    CREATIVE = "creative"


class RetrievalQuality(Enum):
    """Quality levels for retrieval assessment."""
    EXCELLENT = "excellent"
    GOOD = "good"
    POOR = "poor"


class ConfidenceLevel(Enum):
    """Confidence levels for responses."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class PromptStrategy(Enum):
    """Available prompt strategies."""
    DIRECT_ANSWER = "direct_answer"
    CAUTIOUS_REASONING = "cautious_reasoning"
    ANALYTICAL_DEEP_DIVE = "analytical_deep_dive"
    CREATIVE_SYNTHESIS = "creative_synthesis"
    INSUFFICIENT_CONTEXT = "insufficient_context"


@dataclass
class RetrievedContext:
    """Represents a retrieved context chunk with metadata."""
    content: str
    confidence_score: float
    source: str


# ============================================================================
# QUERY CLASSIFICATION PATTERNS
# ============================================================================

# Keywords indicating factual queries
FACTUAL_KEYWORDS = [
    "what is", "what are", "who is", "who are", "when did", "when was",
    "where is", "where are", "how many", "how much", "define", "definition",
    "capital of", "name of", "list", "which", "does", "is there", "are there"
]

# Keywords indicating analytical queries
ANALYTICAL_KEYWORDS = [
    "why", "how does", "how do", "explain", "analyze", "compare", "contrast",
    "difference between", "relationship between", "impact of", "effect of",
    "implications", "consequences", "evaluate", "assess", "examine", "investigate",
    "what causes", "what factors", "advantages", "disadvantages", "pros and cons"
]

# Keywords indicating creative queries
CREATIVE_KEYWORDS = [
    "imagine", "create", "design", "suggest", "recommend", "propose",
    "what if", "could you", "would you", "brainstorm", "generate", "invent",
    "come up with", "think of", "possibilities", "alternatives", "ideas for",
    "future", "predict", "forecast", "speculate"
]

# ============================================================================
# PROMPT TEMPLATES FOR DIFFERENT STRATEGIES
# ============================================================================

DIRECT_ANSWER_TEMPLATE = """Based on the retrieved information, provide a direct and accurate answer.

## Context
{context}

## Question
{question}

## Instructions
- Answer directly using information from the provided context
- Cite sources using [Source: source_name] format
- If information is definitive, state it confidently
- Keep the response focused and concise

## Response"""

CAUTIOUS_REASONING_TEMPLATE = """The available context has limited relevance. Proceed with appropriate caution.

## Available Context (Limited Confidence)
{context}

## Question
{question}

## Instructions
- Acknowledge the limitations in the available information
- Provide what information IS available with appropriate caveats
- Clearly indicate uncertainty where it exists
- Cite sources even when confidence is low
- Suggest what additional information might help

## Response (Note: Limited context available)"""

ANALYTICAL_DEEP_DIVE_TEMPLATE = """Provide a thorough analytical response based on the available information.

## Context
{context}

## Question
{question}

## Instructions
- Analyze the information systematically
- Identify patterns, relationships, and implications
- Draw logical conclusions supported by the context
- Consider multiple perspectives where applicable
- Cite sources for each major point
- Acknowledge limitations in the analysis

## Analysis"""

CREATIVE_SYNTHESIS_TEMPLATE = """Use the available context to inform a creative response.

## Context
{context}

## Question
{question}

## Instructions
- Use the context as a foundation for your response
- Build upon the information creatively while staying grounded
- Clearly distinguish between context-based facts and creative extrapolation
- Cite sources for factual elements
- Indicate when you're going beyond the provided information

## Creative Response"""

INSUFFICIENT_CONTEXT_TEMPLATE = """The retrieval system found insufficient relevant information for this query.

## Available Context
{context}

## Question
{question}

## Instructions
- Acknowledge the lack of sufficient relevant context
- Share whatever limited information is available
- Do NOT fabricate information
- Suggest how the user might refine their query
- Indicate what additional context would be helpful

## Response (Insufficient Context)"""


class RetrievalAnalyzer:
    """Analyzes retrieval confidence scores and determines quality."""

    # Thresholds for quality assessment
    EXCELLENT_THRESHOLD = 0.8
    GOOD_THRESHOLD = 0.5
    MIN_CONTEXTS_FOR_EXCELLENT = 1

    @classmethod
    def analyze_confidence(cls, contexts: List[RetrievedContext]) -> Dict[str, Any]:
        """
        Analyze confidence scores from retrieved contexts.

        Returns:
            Dictionary with analysis results including quality assessment,
            average confidence, and recommendations.
        """
        if not contexts:
            return {
                "quality": RetrievalQuality.POOR,
                "avg_confidence": 0.0,
                "max_confidence": 0.0,
                "context_count": 0,
                "high_confidence_count": 0,
                "recommendation": "insufficient_context"
            }

        scores = [c.confidence_score for c in contexts]
        avg_confidence = sum(scores) / len(scores)
        max_confidence = max(scores)
        high_confidence_count = sum(1 for s in scores if s >= cls.EXCELLENT_THRESHOLD)

        # Determine quality based on multiple factors
        if (avg_confidence >= cls.EXCELLENT_THRESHOLD or
            (max_confidence >= 0.9 and high_confidence_count >= cls.MIN_CONTEXTS_FOR_EXCELLENT)):
            quality = RetrievalQuality.EXCELLENT
            recommendation = "direct_answer"
        elif avg_confidence >= cls.GOOD_THRESHOLD:
            quality = RetrievalQuality.GOOD
            recommendation = "moderate_confidence"
        else:
            quality = RetrievalQuality.POOR
            recommendation = "cautious_approach"

        return {
            "quality": quality,
            "avg_confidence": avg_confidence,
            "max_confidence": max_confidence,
            "context_count": len(contexts),
            "high_confidence_count": high_confidence_count,
            "recommendation": recommendation
        }


class QueryClassifier:
    """Classifies queries into factual, analytical, or creative categories."""

    @classmethod
    def classify(cls, query: str) -> QueryType:
        """
        Classify a query based on keywords and patterns.

        Args:
            query: The user's query string

        Returns:
            QueryType indicating the classification
        """
        query_lower = query.lower().strip()

        # Count keyword matches for each type
        factual_score = cls._count_matches(query_lower, FACTUAL_KEYWORDS)
        analytical_score = cls._count_matches(query_lower, ANALYTICAL_KEYWORDS)
        creative_score = cls._count_matches(query_lower, CREATIVE_KEYWORDS)

        # Determine classification based on highest score
        if creative_score > factual_score and creative_score > analytical_score:
            return QueryType.CREATIVE
        elif analytical_score > factual_score:
            return QueryType.ANALYTICAL
        else:
            # Default to factual for straightforward queries
            return QueryType.FACTUAL

    @staticmethod
    def _count_matches(query: str, keywords: List[str]) -> int:
        """Count how many keywords match in the query."""
        return sum(1 for keyword in keywords if keyword in query)


class ContextManager:
    """Manages context truncation and organization."""

    DEFAULT_MAX_LENGTH = 4000  # Characters

    @classmethod
    def process_contexts(
        cls,
        contexts: List[RetrievedContext],
        max_length: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Process and potentially truncate contexts.

        Args:
            contexts: List of retrieved contexts
            max_length: Maximum total character length for context

        Returns:
            Dictionary with processed contexts and metadata
        """
        if max_length is None:
            max_length = cls.DEFAULT_MAX_LENGTH

        if not contexts:
            return {
                "contexts": [],
                "total_length": 0,
                "was_truncated": False,
                "contexts_used": 0,
                "contexts_dropped": 0
            }

        # Sort by confidence (highest first)
        sorted_contexts = sorted(contexts, key=lambda c: c.confidence_score, reverse=True)

        processed = []
        total_length = 0
        contexts_dropped = 0

        for ctx in sorted_contexts:
            content_length = len(ctx.content)

            if total_length + content_length <= max_length:
                processed.append(ctx)
                total_length += content_length
            else:
                # Try to include a truncated version if there's space
                remaining_space = max_length - total_length
                if remaining_space > 100:  # Only truncate if meaningful space remains
                    truncated_content = ctx.content[:remaining_space - 20] + "... [truncated]"
                    processed.append(RetrievedContext(
                        content=truncated_content,
                        confidence_score=ctx.confidence_score,
                        source=ctx.source
                    ))
                    total_length = max_length
                    contexts_dropped += 0.5  # Partial drop
                else:
                    contexts_dropped += 1

        return {
            "contexts": processed,
            "total_length": total_length,
            "was_truncated": contexts_dropped > 0,
            "contexts_used": len(processed),
            "contexts_dropped": int(contexts_dropped)
        }


class PromptTemplateManager:
    """Manages prompt template selection based on analysis results."""

    TEMPLATES = {
        PromptStrategy.DIRECT_ANSWER: DIRECT_ANSWER_TEMPLATE,
        PromptStrategy.CAUTIOUS_REASONING: CAUTIOUS_REASONING_TEMPLATE,
        PromptStrategy.ANALYTICAL_DEEP_DIVE: ANALYTICAL_DEEP_DIVE_TEMPLATE,
        PromptStrategy.CREATIVE_SYNTHESIS: CREATIVE_SYNTHESIS_TEMPLATE,
        PromptStrategy.INSUFFICIENT_CONTEXT: INSUFFICIENT_CONTEXT_TEMPLATE,
    }

    @classmethod
    def select_strategy(
        cls,
        query_type: QueryType,
        retrieval_quality: RetrievalQuality
    ) -> PromptStrategy:
        """
        Select appropriate prompt strategy based on query type and retrieval quality.

        Args:
            query_type: The classified query type
            retrieval_quality: The assessed retrieval quality

        Returns:
            The selected prompt strategy
        """
        # Handle poor quality retrievals
        if retrieval_quality == RetrievalQuality.POOR:
            return PromptStrategy.CAUTIOUS_REASONING

        # Handle query types with good/excellent retrieval
        if query_type == QueryType.CREATIVE:
            return PromptStrategy.CREATIVE_SYNTHESIS
        elif query_type == QueryType.ANALYTICAL:
            return PromptStrategy.ANALYTICAL_DEEP_DIVE
        else:
            # Factual queries with good retrieval
            if retrieval_quality == RetrievalQuality.EXCELLENT:
                return PromptStrategy.DIRECT_ANSWER
            else:
                return PromptStrategy.CAUTIOUS_REASONING

    @classmethod
    def get_template(cls, strategy: PromptStrategy) -> str:
        """Get the template for a given strategy."""
        return cls.TEMPLATES.get(strategy, DIRECT_ANSWER_TEMPLATE)


class ResponseGenerator:
    """Generates final responses with proper citations and confidence indicators."""

    @classmethod
    def format_context_for_prompt(cls, contexts: List[RetrievedContext]) -> str:
        """Format contexts for insertion into prompt template."""
        if not contexts:
            return "[No relevant context available]"

        formatted_parts = []
        for i, ctx in enumerate(contexts, 1):
            formatted_parts.append(
                f"[{i}] Source: {ctx.source}\n"
                f"Confidence: {ctx.confidence_score:.2f}\n"
                f"Content: {ctx.content}\n"
            )

        return "\n".join(formatted_parts)

    @classmethod
    def generate_response(
        cls,
        query: str,
        contexts: List[RetrievedContext],
        template: str,
        retrieval_analysis: Dict[str, Any],
        query_type: QueryType,
        strategy: PromptStrategy
    ) -> Dict[str, Any]:
        """
        Generate a complete response with metadata.

        Args:
            query: The user's query
            contexts: Processed context list
            template: The selected prompt template
            retrieval_analysis: Analysis of retrieval quality
            query_type: The classified query type
            strategy: The selected prompt strategy

        Returns:
            Complete response dictionary
        """
        # Format context for prompt
        formatted_context = cls.format_context_for_prompt(contexts)

        # Build the prompt
        prompt = template.format(context=formatted_context, question=query)

        # Determine overall confidence level
        avg_confidence = retrieval_analysis.get("avg_confidence", 0)
        if avg_confidence >= 0.8:
            confidence_level = ConfidenceLevel.HIGH.value
        elif avg_confidence >= 0.5:
            confidence_level = ConfidenceLevel.MEDIUM.value
        else:
            confidence_level = ConfidenceLevel.LOW.value

        # Extract sources used
        sources_used = [ctx.source for ctx in contexts] if contexts else []

        # Generate simulated response based on context
        # In a real system, this would be sent to Claude for completion
        response_text = cls._generate_simulated_response(
            query, contexts, retrieval_analysis, confidence_level
        )

        return {
            "response": response_text,
            "confidence_level": confidence_level,
            "retrieval_quality": retrieval_analysis["quality"].value,
            "query_type": query_type.value,
            "sources_used": sources_used,
            "prompt_strategy": strategy.value,
            "prompt": prompt
        }

    @classmethod
    def _generate_simulated_response(
        cls,
        query: str,
        contexts: List[RetrievedContext],
        analysis: Dict[str, Any],
        confidence_level: str
    ) -> str:
        """
        Generate a simulated response for demonstration purposes.

        In a production system, the prompt would be sent to Claude for completion.
        This method creates a representative response structure.
        """
        if not contexts:
            return (
                "I was unable to find relevant information to answer your question. "
                "Please try rephrasing your query or providing additional context."
            )

        # Build response based on confidence level
        if confidence_level == "low":
            prefix = "Based on limited available information, "
            suffix = (
                f" However, I have low confidence in providing a comprehensive "
                f"response due to insufficient retrieval results."
            )
        elif confidence_level == "medium":
            prefix = "Based on the available context, "
            suffix = ""
        else:
            prefix = ""
            suffix = ""

        # Use content from highest confidence context
        best_context = max(contexts, key=lambda c: c.confidence_score)
        content_summary = best_context.content[:200]
        if len(best_context.content) > 200:
            content_summary += "..."

        # Build citations
        citations = " ".join(f"[Source: {ctx.source}]" for ctx in contexts[:3])

        return f"{prefix}{content_summary} {citations}{suffix}"


class ContextAwareRagSystem:
    """
    Context-Aware RAG System

    An intelligent retrieval-augmented generation system that dynamically adapts
    its prompting strategy based on retrieval quality and query complexity.
    """

    def __init__(self):
        """Initialize the ContextAwareRagSystem instance."""
        self.retrieval_analyzer = RetrievalAnalyzer()
        self.query_classifier = QueryClassifier()
        self.context_manager = ContextManager()
        self.template_manager = PromptTemplateManager()
        self.response_generator = ResponseGenerator()

    def analyze_retrieval(self, contexts: List[RetrievedContext]) -> Dict[str, Any]:
        """Analyze retrieval confidence scores."""
        return self.retrieval_analyzer.analyze_confidence(contexts)

    def classify_query(self, query: str) -> QueryType:
        """Classify the query type."""
        return self.query_classifier.classify(query)

    def process_contexts(
        self,
        contexts: List[RetrievedContext],
        max_length: Optional[int] = None
    ) -> Dict[str, Any]:
        """Process and manage context length."""
        return self.context_manager.process_contexts(contexts, max_length)

    def select_strategy(
        self,
        query_type: QueryType,
        retrieval_quality: RetrievalQuality
    ) -> PromptStrategy:
        """Select the appropriate prompt strategy."""
        return self.template_manager.select_strategy(query_type, retrieval_quality)

    def execute(
        self,
        query: str = "",
        retrieved_contexts: Optional[List[Dict[str, Any]]] = None,
        max_context_length: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Main entry point for the context-aware RAG system.

        Args:
            query: User's question or request
            retrieved_contexts: List of dictionaries containing:
                - content: Retrieved text content
                - confidence_score: Retrieval confidence (0.0-1.0)
                - source: Source identifier or URL
            max_context_length: Optional maximum character limit for context

        Returns:
            Dictionary containing:
                - response: Generated answer
                - confidence_level: Overall confidence ("high", "medium", "low")
                - retrieval_quality: Assessment ("excellent", "good", "poor")
                - query_type: Classified query category
                - sources_used: List of source identifiers cited
                - prompt_strategy: Which prompting approach was used
        """
        # Convert input contexts to RetrievedContext objects
        contexts = []
        if retrieved_contexts:
            for ctx in retrieved_contexts:
                contexts.append(RetrievedContext(
                    content=ctx.get("content", ""),
                    confidence_score=ctx.get("confidence_score", 0.5),
                    source=ctx.get("source", "unknown")
                ))

        # Step 1: Analyze retrieval quality
        retrieval_analysis = self.analyze_retrieval(contexts)

        # Step 2: Classify the query
        query_type = self.classify_query(query)

        # Step 3: Process and manage context length
        context_result = self.process_contexts(contexts, max_context_length)
        processed_contexts = context_result["contexts"]

        # Step 4: Select prompt strategy
        strategy = self.select_strategy(query_type, retrieval_analysis["quality"])

        # Step 5: Get appropriate template
        template = self.template_manager.get_template(strategy)

        # Step 6: Generate response
        result = self.response_generator.generate_response(
            query=query,
            contexts=processed_contexts,
            template=template,
            retrieval_analysis=retrieval_analysis,
            query_type=query_type,
            strategy=strategy
        )

        # Remove prompt from final output (keep for debugging if needed)
        del result["prompt"]

        return result


def create_context_aware_rag_system() -> ContextAwareRagSystem:
    """
    Factory function for creating ContextAwareRagSystem instances.

    Returns:
        ContextAwareRagSystem: A new instance of ContextAwareRagSystem
    """
    return ContextAwareRagSystem()
