"""
Semantic RAG System

Upgrade your RAG system to use semantic search instead of keyword matching.
This implementation uses TF-IDF vectors as embeddings for semantic similarity search,
demonstrating the concepts of embedding-based retrieval using only the Python standard library.
"""

import math
import re
from collections import Counter
from typing import Dict, List, Optional, Any, Tuple, Set


class SemanticRagSystem:
    """
    Semantic RAG system using TF-IDF embeddings for similarity search.

    This class provides:
    - Document indexing with TF-IDF vectors
    - Cosine similarity-based semantic search
    - Result ranking and filtering
    - Comparison with keyword-based search
    """

    def __init__(self, similarity_threshold: float = 0.0, top_k: int = 5):
        """
        Initialize the SemanticRagSystem instance.

        Args:
            similarity_threshold: Minimum similarity score for results (default: 0.0)
            top_k: Number of results to return by default (default: 5)
        """
        # Document storage
        self._documents: List[str] = []
        self._doc_metadata: List[Dict[str, Any]] = []

        # Vocabulary and IDF values
        self._vocabulary: List[str] = []
        self._vocab_index: Dict[str, int] = {}
        self._idf_values: Dict[str, float] = {}

        # TF-IDF vectors for documents
        self._doc_vectors: List[List[float]] = []

        # Configuration
        self._similarity_threshold = similarity_threshold
        self._top_k = top_k

        # Stopwords for text preprocessing
        self._stopwords: Set[str] = {
            "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "must", "shall", "can", "need",
            "this", "that", "these", "those", "i", "you", "he", "she", "it", "we",
            "they", "what", "which", "who", "whom", "when", "where", "why", "how",
            "all", "each", "every", "both", "few", "more", "most", "other", "some",
            "such", "no", "not", "only", "same", "so", "than", "too", "very", "just"
        }

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize and normalize text.

        Args:
            text: Input text string

        Returns:
            List of normalized tokens
        """
        # Convert to lowercase
        text = text.lower()

        # Remove punctuation and split into words
        words = re.findall(r'\b[a-z]+\b', text)

        # Remove stopwords and short tokens
        tokens = [w for w in words if w not in self._stopwords and len(w) > 1]

        return tokens

    def _build_vocabulary(self, documents: List[str]) -> None:
        """
        Build vocabulary from documents and calculate IDF values.

        Args:
            documents: List of document texts
        """
        # Count document frequency for each term
        doc_freq: Counter = Counter()
        all_terms: Set[str] = set()

        for doc in documents:
            tokens = set(self._tokenize(doc))
            all_terms.update(tokens)
            for token in tokens:
                doc_freq[token] += 1

        # Build vocabulary (sorted for consistency)
        self._vocabulary = sorted(all_terms)
        self._vocab_index = {term: idx for idx, term in enumerate(self._vocabulary)}

        # Calculate IDF values: log(N / df) where N is total documents
        n_docs = len(documents)
        self._idf_values = {}
        for term, df in doc_freq.items():
            # Add 1 to avoid division by zero and log(0)
            self._idf_values[term] = math.log((n_docs + 1) / (df + 1)) + 1

    def _compute_tf_idf_vector(self, text: str) -> List[float]:
        """
        Compute TF-IDF vector for a text.

        Args:
            text: Input text

        Returns:
            TF-IDF vector (list of floats)
        """
        tokens = self._tokenize(text)

        # Calculate term frequencies
        tf = Counter(tokens)
        total_terms = len(tokens) if tokens else 1

        # Build TF-IDF vector
        vector = [0.0] * len(self._vocabulary)

        for term, count in tf.items():
            if term in self._vocab_index:
                idx = self._vocab_index[term]
                # TF = count / total terms in document
                term_freq = count / total_terms
                # TF-IDF = TF * IDF
                vector[idx] = term_freq * self._idf_values.get(term, 1.0)

        return vector

    def _normalize_vector(self, vector: List[float]) -> List[float]:
        """
        Normalize a vector to unit length.

        Args:
            vector: Input vector

        Returns:
            Normalized vector
        """
        magnitude = math.sqrt(sum(v * v for v in vector))
        if magnitude == 0:
            return vector
        return [v / magnitude for v in vector]

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Similarity score between -1 and 1
        """
        if len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        mag1 = math.sqrt(sum(a * a for a in vec1))
        mag2 = math.sqrt(sum(b * b for b in vec2))

        if mag1 == 0 or mag2 == 0:
            return 0.0

        return dot_product / (mag1 * mag2)

    def index_documents(
        self,
        documents: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> int:
        """
        Index documents for semantic search.

        Args:
            documents: List of document texts to index
            metadata: Optional list of metadata dicts for each document

        Returns:
            Number of documents indexed
        """
        if not documents:
            return 0

        self._documents = documents
        self._doc_metadata = metadata or [{} for _ in documents]

        # Build vocabulary and IDF values
        self._build_vocabulary(documents)

        # Compute TF-IDF vectors for all documents
        self._doc_vectors = []
        for doc in documents:
            vector = self._compute_tf_idf_vector(doc)
            # Normalize for cosine similarity
            normalized = self._normalize_vector(vector)
            self._doc_vectors.append(normalized)

        return len(documents)

    def get_query_embedding(self, query: str) -> List[float]:
        """
        Generate embedding for a query.

        Args:
            query: Query text

        Returns:
            TF-IDF vector for the query
        """
        vector = self._compute_tf_idf_vector(query)
        return self._normalize_vector(vector)

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search for a query.

        Args:
            query: Query text
            top_k: Number of results to return (default: instance setting)
            similarity_threshold: Minimum similarity (default: instance setting)

        Returns:
            List of results with document, score, and metadata
        """
        if not self._documents or not query:
            return []

        top_k = top_k or self._top_k
        threshold = similarity_threshold if similarity_threshold is not None else self._similarity_threshold

        # Generate query embedding
        query_vector = self.get_query_embedding(query)

        # Calculate similarities
        similarities: List[Tuple[int, float]] = []
        for idx, doc_vector in enumerate(self._doc_vectors):
            similarity = self._cosine_similarity(query_vector, doc_vector)
            if similarity >= threshold:
                similarities.append((idx, similarity))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Take top-k results
        results = []
        for idx, score in similarities[:top_k]:
            results.append({
                "document": self._documents[idx],
                "score": score,
                "index": idx,
                "metadata": self._doc_metadata[idx]
            })

        return results

    def _keyword_search(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Perform simple keyword-based search for comparison.

        Args:
            query: Query text
            top_k: Number of results to return

        Returns:
            List of results with document and match count
        """
        if not self._documents or not query:
            return []

        top_k = top_k or self._top_k
        query_tokens = set(self._tokenize(query))

        # Score documents by keyword matches
        scores: List[Tuple[int, int]] = []
        for idx, doc in enumerate(self._documents):
            doc_tokens = set(self._tokenize(doc))
            # Count matching keywords
            matches = len(query_tokens & doc_tokens)
            if matches > 0:
                scores.append((idx, matches))

        # Sort by match count (descending)
        scores.sort(key=lambda x: x[1], reverse=True)

        # Format results
        results = []
        for idx, match_count in scores[:top_k]:
            results.append({
                "document": self._documents[idx],
                "match_count": match_count,
                "index": idx,
                "metadata": self._doc_metadata[idx]
            })

        return results

    def compare_with_keyword_search(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Compare semantic search with keyword search for a query.

        Args:
            query: Query text
            top_k: Number of results for each method

        Returns:
            Comparison results showing both methods' outputs
        """
        top_k = top_k or self._top_k

        semantic_results = self.search(query, top_k=top_k)
        keyword_results = self._keyword_search(query, top_k=top_k)

        # Extract document indices for comparison
        semantic_indices = {r["index"] for r in semantic_results}
        keyword_indices = {r["index"] for r in keyword_results}

        # Find documents only found by each method
        semantic_only = semantic_indices - keyword_indices
        keyword_only = keyword_indices - semantic_indices
        overlap = semantic_indices & keyword_indices

        return {
            "query": query,
            "semantic_results": semantic_results,
            "keyword_results": keyword_results,
            "analysis": {
                "semantic_found": len(semantic_results),
                "keyword_found": len(keyword_results),
                "overlap_count": len(overlap),
                "semantic_only_count": len(semantic_only),
                "keyword_only_count": len(keyword_only),
                "semantic_only_indices": list(semantic_only),
                "keyword_only_indices": list(keyword_only)
            }
        }

    def get_document_embedding(self, index: int) -> Optional[List[float]]:
        """
        Get the embedding vector for a document by index.

        Args:
            index: Document index

        Returns:
            Document embedding vector or None if index invalid
        """
        if 0 <= index < len(self._doc_vectors):
            return self._doc_vectors[index]
        return None

    def get_vocabulary_size(self) -> int:
        """Return the size of the vocabulary."""
        return len(self._vocabulary)

    def get_document_count(self) -> int:
        """Return the number of indexed documents."""
        return len(self._documents)

    def execute(self) -> Dict[str, Any]:
        """
        Main entry point demonstrating complete functionality.

        Returns:
            Dictionary with demonstration results
        """
        # Sample documents demonstrating semantic understanding
        documents = [
            "Machine learning algorithms require large datasets for training models effectively.",
            "Neural networks are inspired by biological brain structures and neurons.",
            "Deep learning uses multiple layers to extract hierarchical features from data.",
            "Python is a popular programming language for data science applications.",
            "Artificial intelligence systems can learn patterns from examples.",
            "Natural language processing helps computers understand human text.",
            "Computer vision enables machines to interpret visual information.",
            "Supervised learning needs labeled data to train predictive models."
        ]

        # Index documents
        indexed_count = self.index_documents(documents)

        # Semantic search - should find related documents even without exact matches
        query1 = "AI training data requirements"
        semantic_results1 = self.search(query1, top_k=3)

        # Compare with keyword search
        comparison = self.compare_with_keyword_search(query1, top_k=3)

        # Another query showing semantic understanding
        query2 = "brain-inspired computing"
        semantic_results2 = self.search(query2, top_k=3)

        return {
            "indexed_documents": indexed_count,
            "vocabulary_size": self.get_vocabulary_size(),
            "query1": query1,
            "semantic_results1": semantic_results1,
            "comparison": comparison,
            "query2": query2,
            "semantic_results2": semantic_results2
        }


def create_semantic_rag_system() -> SemanticRagSystem:
    """
    Factory function for creating SemanticRagSystem instances.

    Returns:
        SemanticRagSystem: A new instance of SemanticRagSystem
    """
    return SemanticRagSystem()
