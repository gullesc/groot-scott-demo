"""
Vector Database Implementation

Set up a vector database (Chroma or Pinecone) and implement document embedding storage and retrieval.
This implementation uses only the Python standard library, providing an in-memory vector store
with cosine similarity search, metadata filtering, and performance benchmarking.
"""

import math
import time
import uuid
from typing import Dict, List, Optional, Any, Tuple


class VectorDatabaseImplementation:
    """
    In-memory vector database for storing and retrieving document embeddings.

    This class provides:
    - Embedding storage with metadata
    - Cosine similarity-based search
    - Configurable top-k results
    - Metadata filtering
    - Performance benchmarking
    """

    def __init__(self):
        """Initialize the VectorDatabaseImplementation instance."""
        # Storage for embeddings: id -> embedding vector
        self._embeddings: Dict[str, List[float]] = {}
        # Storage for metadata: id -> metadata dict
        self._metadata: Dict[str, Dict[str, Any]] = {}
        # Storage for original text: id -> text
        self._texts: Dict[str, str] = {}
        # Performance tracking
        self._query_times: List[float] = []
        self._store_times: List[float] = []
        # Expected embedding dimension (set on first store)
        self._embedding_dim: Optional[int] = None

    def store_embedding(
        self,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
        text: str = "",
        doc_id: Optional[str] = None
    ) -> str:
        """
        Store a document embedding with associated metadata.

        Args:
            embedding: Dense vector representation of the document
            metadata: Dictionary containing document metadata (doc_id, source, etc.)
            text: Original text content of the document chunk
            doc_id: Optional unique identifier; auto-generated if not provided

        Returns:
            The ID of the stored embedding
        """
        start_time = time.time()

        # Validate embedding
        if not embedding:
            raise ValueError("Embedding cannot be empty")

        # Set or validate embedding dimension
        if self._embedding_dim is None:
            self._embedding_dim = len(embedding)
        elif len(embedding) != self._embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self._embedding_dim}, got {len(embedding)}"
            )

        # Generate ID if not provided
        if doc_id is None:
            doc_id = str(uuid.uuid4())

        # Store the embedding and associated data
        self._embeddings[doc_id] = embedding
        self._metadata[doc_id] = metadata or {}
        self._texts[doc_id] = text

        # Track performance
        self._store_times.append(time.time() - start_time)

        return doc_id

    def store_embeddings_batch(
        self,
        items: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Store multiple embeddings in a batch operation.

        Args:
            items: List of dicts with keys: embedding, metadata (optional), text (optional), doc_id (optional)

        Returns:
            List of IDs for the stored embeddings
        """
        ids = []
        for item in items:
            doc_id = self.store_embedding(
                embedding=item["embedding"],
                metadata=item.get("metadata"),
                text=item.get("text", ""),
                doc_id=item.get("doc_id")
            )
            ids.append(doc_id)
        return ids

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.

        Cosine similarity = dot(a, b) / (||a|| * ||b||)

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Similarity score between 0 and 1 (for non-negative vectors)
        """
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have the same dimension")

        # Calculate dot product
        dot_product = sum(a * b for a, b in zip(vec1, vec2))

        # Calculate magnitudes
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))

        # Avoid division by zero
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

    def _matches_filter(self, doc_id: str, filters: Dict[str, Any]) -> bool:
        """
        Check if a document's metadata matches the given filters.

        Args:
            doc_id: Document ID to check
            filters: Key-value pairs that must match in metadata

        Returns:
            True if document matches all filters, False otherwise
        """
        if not filters:
            return True

        doc_metadata = self._metadata.get(doc_id, {})

        for key, value in filters.items():
            if key not in doc_metadata:
                return False
            if doc_metadata[key] != value:
                return False

        return True

    def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        similarity_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Find the most similar documents to a query embedding.

        Args:
            query_embedding: Query vector to search against
            top_k: Number of results to return (default: 5, max: 100)
            filters: Metadata filters to apply
            similarity_threshold: Minimum similarity score (default: 0.0)

        Returns:
            List of results with text, score, metadata, and id
        """
        start_time = time.time()

        # Validate parameters
        top_k = min(max(1, top_k), 100)  # Clamp between 1 and 100

        if not query_embedding:
            return []

        # Calculate similarities for all documents
        similarities: List[Tuple[str, float]] = []

        for doc_id, embedding in self._embeddings.items():
            # Apply metadata filters
            if not self._matches_filter(doc_id, filters or {}):
                continue

            # Calculate similarity
            similarity = self._cosine_similarity(query_embedding, embedding)

            # Apply threshold
            if similarity >= similarity_threshold:
                similarities.append((doc_id, similarity))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Take top-k results
        top_results = similarities[:top_k]

        # Format results
        results = []
        for doc_id, score in top_results:
            results.append({
                "id": doc_id,
                "text": self._texts.get(doc_id, ""),
                "score": score,
                "metadata": self._metadata.get(doc_id, {})
            })

        # Track performance
        self._query_times.append(time.time() - start_time)

        return results

    def get_embedding(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific embedding by ID.

        Args:
            doc_id: The document ID to retrieve

        Returns:
            Dictionary with embedding, metadata, and text, or None if not found
        """
        if doc_id not in self._embeddings:
            return None

        return {
            "id": doc_id,
            "embedding": self._embeddings[doc_id],
            "metadata": self._metadata.get(doc_id, {}),
            "text": self._texts.get(doc_id, "")
        }

    def delete_embedding(self, doc_id: str) -> bool:
        """
        Delete an embedding by ID.

        Args:
            doc_id: The document ID to delete

        Returns:
            True if deleted, False if not found
        """
        if doc_id not in self._embeddings:
            return False

        del self._embeddings[doc_id]
        self._metadata.pop(doc_id, None)
        self._texts.pop(doc_id, None)

        return True

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance benchmarks for the vector database.

        Returns:
            Dictionary containing performance metrics:
            - avg_query_time: Average query response time in seconds
            - avg_store_time: Average store operation time in seconds
            - total_embeddings: Number of stored embeddings
            - embedding_dimension: Dimension of stored vectors
            - total_queries: Number of queries executed
            - index_size_mb: Estimated memory usage in MB
        """
        # Calculate average query time
        avg_query_time = (
            sum(self._query_times) / len(self._query_times)
            if self._query_times else 0.0
        )

        # Calculate average store time
        avg_store_time = (
            sum(self._store_times) / len(self._store_times)
            if self._store_times else 0.0
        )

        # Estimate index size (rough estimate: 8 bytes per float + overhead)
        total_floats = sum(len(emb) for emb in self._embeddings.values())
        estimated_size_mb = (total_floats * 8) / (1024 * 1024)

        return {
            "avg_query_time": avg_query_time,
            "avg_store_time": avg_store_time,
            "total_embeddings": len(self._embeddings),
            "embedding_dimension": self._embedding_dim,
            "total_queries": len(self._query_times),
            "index_size_mb": estimated_size_mb
        }

    def clear(self) -> None:
        """Clear all stored embeddings and reset performance metrics."""
        self._embeddings.clear()
        self._metadata.clear()
        self._texts.clear()
        self._query_times.clear()
        self._store_times.clear()
        self._embedding_dim = None

    def count(self) -> int:
        """Return the number of stored embeddings."""
        return len(self._embeddings)

    def execute(self) -> Dict[str, Any]:
        """
        Main entry point demonstrating complete functionality.

        Returns:
            Dictionary with demonstration results and metrics
        """
        # Create sample embeddings (simulating 8-dimensional vectors for demo)
        sample_docs = [
            {
                "embedding": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                "metadata": {"doc_id": "doc1", "source": "manual.pdf", "chunk_index": 0},
                "text": "RAG systems combine retrieval with generation for better answers."
            },
            {
                "embedding": [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85],
                "metadata": {"doc_id": "doc1", "source": "manual.pdf", "chunk_index": 1},
                "text": "Vector databases store embeddings for semantic search."
            },
            {
                "embedding": [0.9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                "metadata": {"doc_id": "doc2", "source": "guide.pdf", "chunk_index": 0},
                "text": "Machine learning models require training data."
            },
            {
                "embedding": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                "metadata": {"doc_id": "doc2", "source": "guide.pdf", "chunk_index": 1},
                "text": "Neural networks learn patterns from examples."
            },
            {
                "embedding": [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
                "metadata": {"doc_id": "doc3", "source": "tutorial.pdf", "chunk_index": 0},
                "text": "Python is popular for data science and AI applications."
            }
        ]

        # Store embeddings
        stored_ids = self.store_embeddings_batch(sample_docs)

        # Perform similarity search
        query_embedding = [0.12, 0.22, 0.32, 0.42, 0.52, 0.62, 0.72, 0.82]
        search_results = self.similarity_search(
            query_embedding=query_embedding,
            top_k=3
        )

        # Search with metadata filter
        filtered_results = self.similarity_search(
            query_embedding=query_embedding,
            top_k=3,
            filters={"source": "manual.pdf"}
        )

        # Get performance metrics
        metrics = self.get_performance_metrics()

        return {
            "stored_count": len(stored_ids),
            "search_results": search_results,
            "filtered_results": filtered_results,
            "performance_metrics": metrics
        }


def create_vector_database_implementation() -> VectorDatabaseImplementation:
    """
    Factory function for creating VectorDatabaseImplementation instances.

    Returns:
        VectorDatabaseImplementation: A new instance of VectorDatabaseImplementation
    """
    return VectorDatabaseImplementation()
