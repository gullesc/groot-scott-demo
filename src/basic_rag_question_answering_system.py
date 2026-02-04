"""
Basic RAG Question-Answering System

Create a simple RAG system that can answer questions about your processed documents using Claude.

This module implements a Retrieval-Augmented Generation system that combines document
retrieval with Claude's generation capabilities. It uses TF-IDF for similarity matching
and provides answers with source citations.
"""

import json
import math
import os
import re
import urllib.error
import urllib.request
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple


class RAGError(Exception):
    """Custom exception for RAG system errors."""
    pass


class TextProcessor:
    """
    Handles text preprocessing for the RAG system.

    Implements tokenization, normalization, and stopword removal
    using only Python standard library.
    """

    # Common English stopwords that don't carry much meaning for retrieval
    STOPWORDS = {
        'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
        'this', 'that', 'these', 'those', 'it', 'its', 'they', 'them', 'their',
        'he', 'she', 'him', 'her', 'his', 'hers', 'we', 'us', 'our', 'you',
        'your', 'i', 'me', 'my', 'who', 'what', 'which', 'when', 'where',
        'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most',
        'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
        'so', 'than', 'too', 'very', 'just', 'also', 'now', 'here', 'there',
        'then', 'if', 'about', 'into', 'through', 'during', 'before', 'after',
        'above', 'below', 'between', 'under', 'again', 'further', 'once'
    }

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """
        Split text into tokens (words).

        Converts to lowercase and extracts alphanumeric sequences.

        Args:
            text: Input text string

        Returns:
            List of tokens
        """
        # Convert to lowercase
        text = text.lower()

        # Extract words (alphanumeric sequences)
        tokens = re.findall(r'\b[a-z0-9]+\b', text)

        return tokens

    @classmethod
    def preprocess(cls, text: str, remove_stopwords: bool = True) -> List[str]:
        """
        Full preprocessing pipeline: tokenize, normalize, and optionally remove stopwords.

        Args:
            text: Input text string
            remove_stopwords: Whether to filter out common stopwords

        Returns:
            List of preprocessed tokens
        """
        tokens = cls.tokenize(text)

        if remove_stopwords:
            tokens = [t for t in tokens if t not in cls.STOPWORDS]

        return tokens


class TFIDFCalculator:
    """
    Implements TF-IDF (Term Frequency-Inverse Document Frequency) from scratch.

    TF-IDF is a numerical statistic that reflects how important a word is
    to a document in a collection. It's commonly used in information retrieval.
    """

    def __init__(self):
        """Initialize the TF-IDF calculator."""
        self.document_frequencies: Dict[str, int] = {}
        self.num_documents: int = 0
        self.vocabulary: set = set()

    def fit(self, documents: List[List[str]]) -> None:
        """
        Calculate document frequencies from a corpus of tokenized documents.

        Args:
            documents: List of tokenized documents (each is a list of tokens)
        """
        self.num_documents = len(documents)
        self.document_frequencies = {}
        self.vocabulary = set()

        for doc in documents:
            # Get unique terms in this document
            unique_terms = set(doc)
            self.vocabulary.update(unique_terms)

            # Count document frequency for each term
            for term in unique_terms:
                self.document_frequencies[term] = self.document_frequencies.get(term, 0) + 1

    def calculate_tf(self, tokens: List[str]) -> Dict[str, float]:
        """
        Calculate term frequency for a document.

        TF(t) = (Number of times term t appears in document) / (Total terms in document)

        Args:
            tokens: List of tokens in the document

        Returns:
            Dictionary mapping terms to their TF values
        """
        if not tokens:
            return {}

        term_counts = Counter(tokens)
        total_terms = len(tokens)

        return {term: count / total_terms for term, count in term_counts.items()}

    def calculate_idf(self, term: str) -> float:
        """
        Calculate inverse document frequency for a term.

        IDF(t) = log(Total documents / (Documents containing term t + 1))

        The +1 in the denominator prevents division by zero for unknown terms.

        Args:
            term: The term to calculate IDF for

        Returns:
            IDF value for the term
        """
        doc_freq = self.document_frequencies.get(term, 0)
        # Add 1 to prevent division by zero and smooth the calculation
        return math.log((self.num_documents + 1) / (doc_freq + 1))

    def calculate_tfidf(self, tokens: List[str]) -> Dict[str, float]:
        """
        Calculate TF-IDF vector for a document.

        TF-IDF(t) = TF(t) * IDF(t)

        Args:
            tokens: List of tokens in the document

        Returns:
            Dictionary mapping terms to their TF-IDF values
        """
        tf = self.calculate_tf(tokens)
        tfidf = {}

        for term, tf_value in tf.items():
            idf_value = self.calculate_idf(term)
            tfidf[term] = tf_value * idf_value

        return tfidf


class SimilarityCalculator:
    """
    Calculates similarity between text vectors using cosine similarity.

    Cosine similarity measures the angle between two vectors, making it
    robust to document length variations.
    """

    @staticmethod
    def cosine_similarity(vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        """
        Calculate cosine similarity between two sparse vectors.

        cos(θ) = (A · B) / (||A|| * ||B||)

        Args:
            vec1: First vector as {term: weight}
            vec2: Second vector as {term: weight}

        Returns:
            Cosine similarity score between 0 and 1
        """
        if not vec1 or not vec2:
            return 0.0

        # Find common terms for dot product
        common_terms = set(vec1.keys()) & set(vec2.keys())

        if not common_terms:
            return 0.0

        # Calculate dot product
        dot_product = sum(vec1[term] * vec2[term] for term in common_terms)

        # Calculate magnitudes
        magnitude1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
        magnitude2 = math.sqrt(sum(v ** 2 for v in vec2.values()))

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)


class ContextFormatter:
    """
    Formats retrieved document chunks for Claude API calls.

    Structures the context with clear boundaries and source attribution
    to help Claude generate accurate, well-grounded responses.
    """

    @staticmethod
    def format_context(
        chunks: List[Dict[str, Any]],
        scores: List[float]
    ) -> str:
        """
        Format retrieved chunks into a context string for Claude.

        Args:
            chunks: List of chunk dictionaries with 'text' and 'source' keys
            scores: Relevance scores for each chunk

        Returns:
            Formatted context string
        """
        if not chunks:
            return "No relevant context found in the documents."

        context_parts = []
        context_parts.append("Here are the relevant excerpts from the documents:\n")

        for i, (chunk, score) in enumerate(zip(chunks, scores), 1):
            source = chunk.get('source', 'Unknown source')
            chunk_id = chunk.get('chunk_id', i)
            text = chunk.get('text', '')

            context_parts.append(f"--- Excerpt {i} (Source: {source}, Relevance: {score:.2f}) ---")
            context_parts.append(text)
            context_parts.append("")  # Empty line between excerpts

        return '\n'.join(context_parts)

    @staticmethod
    def format_prompt(question: str, context: str) -> str:
        """
        Create the full prompt for Claude with question and context.

        Args:
            question: User's question
            context: Formatted context from retrieved documents

        Returns:
            Complete prompt string
        """
        prompt = f"""Based on the following context from the documents, please answer the question.
If the context doesn't contain enough information to answer the question, say so clearly.
Always cite your sources by mentioning which excerpt(s) you used.

{context}

Question: {question}

Please provide a clear, accurate answer based on the context above, citing the relevant sources."""

        return prompt


class SourceTracker:
    """
    Manages source citations and chunk metadata for transparency.

    Ensures answers can be traced back to specific document chunks.
    """

    @staticmethod
    def format_citations(chunks: List[Dict[str, Any]]) -> List[str]:
        """
        Format source citations for retrieved chunks.

        Args:
            chunks: List of chunk dictionaries

        Returns:
            List of formatted citation strings
        """
        citations = []

        for chunk in chunks:
            source = chunk.get('source', 'Unknown')
            chunk_id = chunk.get('chunk_id', 'N/A')

            # Include page number if available
            metadata = chunk.get('metadata', {})
            page = metadata.get('page_number')

            if page:
                citation = f"{source} (chunk {chunk_id}, page {page})"
            else:
                citation = f"{source} (chunk {chunk_id})"

            citations.append(citation)

        return citations


class ClaudeAPIClient:
    """
    Client for interacting with the Anthropic Claude API.

    Uses urllib from the standard library for HTTP requests.
    """

    API_URL = "https://api.anthropic.com/v1/messages"
    DEFAULT_MODEL = "claude-sonnet-4-20250514"

    def __init__(self, api_key: str, model: Optional[str] = None):
        """
        Initialize the API client.

        Args:
            api_key: Anthropic API key
            model: Model to use (default: claude-sonnet-4-20250514)
        """
        if not api_key:
            raise RAGError("API key is required")

        self.api_key = api_key
        self.model = model or self.DEFAULT_MODEL

    def generate_response(
        self,
        prompt: str,
        max_tokens: int = 1024
    ) -> str:
        """
        Send a prompt to Claude and get a response.

        Args:
            prompt: The prompt to send
            max_tokens: Maximum tokens in the response

        Returns:
            Generated response text

        Raises:
            RAGError: If API call fails
        """
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }

        data = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }

        try:
            request_data = json.dumps(data).encode('utf-8')
            request = urllib.request.Request(
                self.API_URL,
                data=request_data,
                headers=headers,
                method='POST'
            )

            with urllib.request.urlopen(request, timeout=60) as response:
                response_data = json.loads(response.read().decode('utf-8'))

            # Extract the text from Claude's response
            if 'content' in response_data and response_data['content']:
                return response_data['content'][0].get('text', '')

            return ''

        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8') if e.fp else ''
            raise RAGError(f"API request failed (HTTP {e.code}): {error_body}")

        except urllib.error.URLError as e:
            raise RAGError(f"Network error: {str(e.reason)}")

        except json.JSONDecodeError as e:
            raise RAGError(f"Failed to parse API response: {str(e)}")

        except Exception as e:
            raise RAGError(f"Unexpected error calling API: {str(e)}")


class BasicRagQuestionAnsweringSystem:
    """
    Main RAG system class that orchestrates the complete question-answering pipeline.

    Combines document retrieval (using TF-IDF similarity) with Claude's generation
    capabilities to provide accurate, source-cited answers.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_chunks: int = 3,
        similarity_threshold: float = 0.1
    ):
        """
        Initialize the RAG system.

        Args:
            api_key: Anthropic API key (can also be set via ANTHROPIC_API_KEY env var)
            max_chunks: Maximum number of chunks to retrieve (default: 3)
            similarity_threshold: Minimum similarity score to include a chunk (default: 0.1)
        """
        # Get API key from parameter or environment
        self.api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')

        self.max_chunks = max_chunks
        self.similarity_threshold = similarity_threshold

        # Initialize components
        self.text_processor = TextProcessor()
        self.tfidf_calculator = TFIDFCalculator()
        self.similarity_calculator = SimilarityCalculator()
        self.context_formatter = ContextFormatter()
        self.source_tracker = SourceTracker()

        # Document storage
        self.documents: List[Dict[str, Any]] = []
        self.document_tokens: List[List[str]] = []
        self.document_vectors: List[Dict[str, float]] = []

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add documents to the RAG system and build the index.

        Args:
            documents: List of document chunks, each with 'text' and metadata
        """
        self.documents = documents
        self.document_tokens = []
        self.document_vectors = []

        # Tokenize all documents
        for doc in documents:
            text = doc.get('text', '')
            tokens = self.text_processor.preprocess(text)
            self.document_tokens.append(tokens)

        # Fit TF-IDF on the corpus
        self.tfidf_calculator.fit(self.document_tokens)

        # Calculate TF-IDF vectors for all documents
        for tokens in self.document_tokens:
            vector = self.tfidf_calculator.calculate_tfidf(tokens)
            self.document_vectors.append(vector)

    def retrieve(
        self,
        question: str,
        max_chunks: Optional[int] = None
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """
        Retrieve the most relevant document chunks for a question.

        Args:
            question: User's question
            max_chunks: Override for maximum chunks to retrieve

        Returns:
            Tuple of (retrieved chunks, similarity scores)
        """
        if not self.documents:
            return [], []

        max_k = max_chunks or self.max_chunks

        # Preprocess the question
        question_tokens = self.text_processor.preprocess(question)

        if not question_tokens:
            return [], []

        # Calculate TF-IDF vector for the question
        question_vector = self.tfidf_calculator.calculate_tfidf(question_tokens)

        # Calculate similarity with all documents
        similarities = []
        for i, doc_vector in enumerate(self.document_vectors):
            score = self.similarity_calculator.cosine_similarity(
                question_vector,
                doc_vector
            )
            similarities.append((i, score))

        # Sort by similarity score (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Filter by threshold and limit
        retrieved_chunks = []
        scores = []

        for idx, score in similarities[:max_k]:
            if score >= self.similarity_threshold:
                retrieved_chunks.append(self.documents[idx])
                scores.append(score)

        return retrieved_chunks, scores

    def generate_answer(
        self,
        question: str,
        chunks: List[Dict[str, Any]],
        scores: List[float]
    ) -> str:
        """
        Generate an answer using Claude based on retrieved context.

        Args:
            question: User's question
            chunks: Retrieved document chunks
            scores: Relevance scores for chunks

        Returns:
            Generated answer text
        """
        if not self.api_key:
            raise RAGError(
                "API key not provided. Set via constructor or ANTHROPIC_API_KEY environment variable."
            )

        if not chunks:
            return "I cannot find relevant information in the provided documents to answer your question."

        # Format context and prompt
        context = self.context_formatter.format_context(chunks, scores)
        prompt = self.context_formatter.format_prompt(question, context)

        # Call Claude API
        client = ClaudeAPIClient(self.api_key)
        response = client.generate_response(prompt)

        return response

    def ask(self, question: str) -> Dict[str, Any]:
        """
        Process a question and return an answer with sources.

        This is the main method for using the RAG system.

        Args:
            question: User's natural language question

        Returns:
            Dictionary containing:
                - answer: Generated response
                - sources: List of source citations
                - confidence_scores: Similarity scores
                - retrieved_chunks: The chunks used as context
        """
        # Retrieve relevant chunks
        chunks, scores = self.retrieve(question)

        # Generate answer
        try:
            answer = self.generate_answer(question, chunks, scores)
        except RAGError as e:
            answer = f"Error generating answer: {str(e)}"

        # Format sources
        sources = self.source_tracker.format_citations(chunks)

        # Extract just the text from chunks for output
        retrieved_texts = [c.get('text', '')[:200] + '...' if len(c.get('text', '')) > 200
                          else c.get('text', '') for c in chunks]

        return {
            "answer": answer,
            "sources": sources,
            "confidence_scores": scores,
            "retrieved_chunks": retrieved_texts
        }

    def execute(
        self,
        question: Optional[str] = None,
        documents: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Main entry point for the RAG pipeline.

        Args:
            question: User's question (required)
            documents: Optional list of documents to add before querying

        Returns:
            Dictionary with answer, sources, confidence scores, and retrieved chunks
        """
        if documents:
            self.add_documents(documents)

        if not question:
            raise RAGError("A question is required")

        return self.ask(question)


def create_basic_rag_question_answering_system(
    api_key: Optional[str] = None,
    max_chunks: int = 3,
    similarity_threshold: float = 0.1
) -> BasicRagQuestionAnsweringSystem:
    """
    Factory function for creating BasicRagQuestionAnsweringSystem instances.

    Args:
        api_key: Anthropic API key (optional, can use env var)
        max_chunks: Maximum chunks to retrieve (default: 3)
        similarity_threshold: Minimum similarity score (default: 0.1)

    Returns:
        BasicRagQuestionAnsweringSystem: A new instance
    """
    return BasicRagQuestionAnsweringSystem(
        api_key=api_key,
        max_chunks=max_chunks,
        similarity_threshold=similarity_threshold
    )
