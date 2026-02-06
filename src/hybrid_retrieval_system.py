"""
Hybrid Retrieval System

A sophisticated retrieval system that combines multiple search strategies
for improved accuracy and relevance in RAG applications. This module implements
semantic similarity, keyword matching, metadata filtering, query expansion,
re-ranking, and explainable scoring using only the Python standard library.
"""

import math
import re
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict


class RetrievalStrategy(Enum):
    """Available retrieval strategies."""
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    METADATA = "metadata"


@dataclass
class Document:
    """Represents a document in the retrieval system."""
    doc_id: str
    content: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class RetrievalResult:
    """Result from a retrieval operation with scoring details."""
    document: Document
    final_score: float
    score_breakdown: Dict[str, float]
    explanation: str
    strategy_contributions: Dict[str, float]
    query_terms_matched: List[str] = field(default_factory=list)
    metadata_matches: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryExpansion:
    """Result of query expansion with synonyms and related terms."""
    original_query: str
    expanded_terms: List[str]
    synonyms_used: Dict[str, List[str]]
    expansion_method: str


# ============================================================================
# SYNONYM DICTIONARY
# ============================================================================

# Built-in synonym dictionary for educational purposes
SYNONYM_DICTIONARY = {
    # Technical terms
    "machine learning": ["ml", "artificial intelligence", "ai", "deep learning"],
    "ml": ["machine learning", "artificial intelligence"],
    "ai": ["artificial intelligence", "machine learning", "ml"],
    "database": ["db", "data store", "storage"],
    "api": ["application programming interface", "endpoint", "interface"],
    "server": ["backend", "host", "service"],
    "client": ["frontend", "user", "consumer"],
    "error": ["bug", "issue", "problem", "fault", "exception"],
    "bug": ["error", "issue", "defect", "problem"],
    "feature": ["functionality", "capability", "function"],
    "performance": ["speed", "efficiency", "optimization"],
    "security": ["protection", "safety", "authentication", "authorization"],

    # Common verbs
    "create": ["make", "build", "generate", "construct"],
    "delete": ["remove", "erase", "destroy", "eliminate"],
    "update": ["modify", "change", "edit", "alter"],
    "find": ["search", "locate", "discover", "retrieve"],
    "show": ["display", "present", "render", "view"],

    # Business terms
    "revenue": ["income", "earnings", "sales"],
    "cost": ["expense", "price", "expenditure"],
    "customer": ["client", "user", "consumer", "buyer"],
    "product": ["item", "goods", "merchandise"],

    # RAG-specific
    "retrieval": ["search", "fetch", "query", "lookup"],
    "embedding": ["vector", "representation", "encoding"],
    "document": ["doc", "file", "text", "content"],
    "chunk": ["segment", "piece", "fragment", "section"],
}


# ============================================================================
# QUERY PROCESSOR
# ============================================================================

class QueryProcessor:
    """
    Handles query expansion, synonym replacement, and normalization.

    Transforms user queries to improve retrieval coverage by adding
    related terms and handling vocabulary mismatch.
    """

    def __init__(self, synonym_dict: Optional[Dict[str, List[str]]] = None):
        """Initialize with optional custom synonym dictionary."""
        self.synonym_dict = synonym_dict or SYNONYM_DICTIONARY

    def normalize_query(self, query: str) -> str:
        """
        Normalize a query string.

        Args:
            query: Raw query string

        Returns:
            Normalized query
        """
        # Convert to lowercase
        normalized = query.lower().strip()
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        # Remove special characters but keep alphanumeric and spaces
        normalized = re.sub(r'[^\w\s]', '', normalized)
        return normalized

    def tokenize(self, text: str) -> List[str]:
        """Split text into tokens."""
        return text.lower().split()

    def expand_query(
        self,
        query: str,
        max_synonyms_per_term: int = 3
    ) -> QueryExpansion:
        """
        Expand a query with synonyms and related terms.

        Args:
            query: Original query string
            max_synonyms_per_term: Maximum synonyms to add per term

        Returns:
            QueryExpansion with expanded terms
        """
        normalized = self.normalize_query(query)
        tokens = self.tokenize(normalized)

        expanded_terms = list(tokens)  # Start with original terms
        synonyms_used = {}

        # Check for multi-word phrases first
        for phrase, synonyms in self.synonym_dict.items():
            if phrase in normalized:
                phrase_synonyms = synonyms[:max_synonyms_per_term]
                expanded_terms.extend(phrase_synonyms)
                synonyms_used[phrase] = phrase_synonyms

        # Check individual tokens
        for token in tokens:
            if token in self.synonym_dict and token not in synonyms_used:
                token_synonyms = self.synonym_dict[token][:max_synonyms_per_term]
                expanded_terms.extend(token_synonyms)
                synonyms_used[token] = token_synonyms

        # Remove duplicates while preserving order
        seen = set()
        unique_terms = []
        for term in expanded_terms:
            term_lower = term.lower()
            if term_lower not in seen:
                seen.add(term_lower)
                unique_terms.append(term)

        return QueryExpansion(
            original_query=query,
            expanded_terms=unique_terms,
            synonyms_used=synonyms_used,
            expansion_method="synonym_dictionary"
        )

    def get_query_ngrams(self, query: str, n: int = 2) -> List[str]:
        """Generate n-grams from query."""
        tokens = self.tokenize(self.normalize_query(query))
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngrams.append(' '.join(tokens[i:i + n]))
        return ngrams


# ============================================================================
# SEMANTIC RETRIEVER
# ============================================================================

class SemanticRetriever:
    """
    Implements vector-based semantic similarity search.

    Uses cosine similarity between query embeddings and document embeddings
    to find semantically similar documents.
    """

    @staticmethod
    def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity score (0 to 1 for non-negative vectors)
        """
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

    @staticmethod
    def simple_embedding(text: str, dim: int = 64) -> List[float]:
        """
        Generate a simple embedding from text.

        This is a simplified embedding for educational purposes.
        In production, use proper embedding models.

        Args:
            text: Text to embed
            dim: Embedding dimension

        Returns:
            Simple embedding vector
        """
        # Use character and word statistics to create a simple embedding
        text_lower = text.lower()
        words = text_lower.split()

        embedding = [0.0] * dim

        # Character frequency features
        for i, char in enumerate(text_lower[:dim // 2]):
            embedding[i % (dim // 2)] += ord(char) / 1000.0

        # Word-based features
        for i, word in enumerate(words):
            idx = (hash(word) % (dim // 2)) + (dim // 2)
            embedding[idx] += len(word) / 10.0

        # Normalize
        magnitude = math.sqrt(sum(x * x for x in embedding))
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]

        return embedding

    def retrieve(
        self,
        query_embedding: List[float],
        documents: List[Document],
        top_k: int = 10
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve documents by semantic similarity.

        Args:
            query_embedding: Query vector
            documents: Documents to search
            top_k: Number of results

        Returns:
            List of (document, score) tuples
        """
        results = []

        for doc in documents:
            if doc.embedding:
                score = self.cosine_similarity(query_embedding, doc.embedding)
                results.append((doc, score))

        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


# ============================================================================
# KEYWORD RETRIEVER (TF-IDF)
# ============================================================================

class KeywordRetriever:
    """
    Implements TF-IDF based keyword matching.

    Provides keyword-based retrieval using term frequency-inverse document
    frequency scoring.
    """

    def __init__(self):
        """Initialize the keyword retriever."""
        self._idf_cache: Dict[str, float] = {}
        self._doc_count: int = 0

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        # Simple tokenization: lowercase and split on non-alphanumeric
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens

    def _compute_tf(self, tokens: List[str]) -> Dict[str, float]:
        """Compute term frequency for tokens."""
        tf = defaultdict(int)
        for token in tokens:
            tf[token] += 1

        # Normalize by document length
        total = len(tokens) if tokens else 1
        return {term: count / total for term, count in tf.items()}

    def _compute_idf(self, term: str, documents: List[Document]) -> float:
        """Compute inverse document frequency for a term."""
        if term in self._idf_cache and self._doc_count == len(documents):
            return self._idf_cache[term]

        doc_count = sum(
            1 for doc in documents
            if term in self._tokenize(doc.content)
        )

        if doc_count == 0:
            return 0.0

        idf = math.log(len(documents) / doc_count) + 1
        self._idf_cache[term] = idf
        self._doc_count = len(documents)

        return idf

    def compute_tfidf_score(
        self,
        query_tokens: List[str],
        document: Document,
        documents: List[Document]
    ) -> Tuple[float, List[str]]:
        """
        Compute TF-IDF score for a document given query terms.

        Args:
            query_tokens: Query terms
            document: Document to score
            documents: All documents (for IDF calculation)

        Returns:
            Tuple of (score, matched_terms)
        """
        doc_tokens = self._tokenize(document.content)
        doc_tf = self._compute_tf(doc_tokens)

        score = 0.0
        matched_terms = []

        for term in query_tokens:
            if term in doc_tf:
                tf = doc_tf[term]
                idf = self._compute_idf(term, documents)
                term_score = tf * idf
                score += term_score
                matched_terms.append(term)

        return score, matched_terms

    def retrieve(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 10
    ) -> List[Tuple[Document, float, List[str]]]:
        """
        Retrieve documents by keyword matching.

        Args:
            query: Search query
            documents: Documents to search
            top_k: Number of results

        Returns:
            List of (document, score, matched_terms) tuples
        """
        query_tokens = self._tokenize(query)
        results = []

        for doc in documents:
            score, matched = self.compute_tfidf_score(query_tokens, doc, documents)
            if score > 0:
                results.append((doc, score, matched))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


# ============================================================================
# METADATA FILTER
# ============================================================================

class MetadataFilter:
    """
    Handles metadata-based filtering and scoring.

    Filters documents based on metadata constraints and contributes
    to relevance scoring based on metadata matches.
    """

    @staticmethod
    def matches_filter(
        document: Document,
        filters: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if document metadata matches filters.

        Args:
            document: Document to check
            filters: Key-value filter constraints

        Returns:
            Tuple of (matches, matched_fields)
        """
        if not filters:
            return True, {}

        matched_fields = {}

        for key, value in filters.items():
            doc_value = document.metadata.get(key)

            if doc_value is None:
                return False, {}

            # Handle different comparison types
            if isinstance(value, dict):
                # Range filters: {"min": 0, "max": 100}
                if "min" in value and doc_value < value["min"]:
                    return False, {}
                if "max" in value and doc_value > value["max"]:
                    return False, {}
                matched_fields[key] = doc_value
            elif isinstance(value, list):
                # List membership: value in list
                if doc_value not in value:
                    return False, {}
                matched_fields[key] = doc_value
            else:
                # Exact match
                if doc_value != value:
                    return False, {}
                matched_fields[key] = doc_value

        return True, matched_fields

    @staticmethod
    def compute_metadata_score(
        document: Document,
        scoring_fields: Dict[str, float]
    ) -> float:
        """
        Compute a metadata-based relevance score.

        Args:
            document: Document to score
            scoring_fields: Fields to score with their weights

        Returns:
            Metadata relevance score
        """
        score = 0.0

        for field, weight in scoring_fields.items():
            if field in document.metadata:
                value = document.metadata[field]

                # Normalize different value types
                if isinstance(value, bool):
                    score += weight if value else 0
                elif isinstance(value, (int, float)):
                    # Normalize numeric values (assuming 0-1 range or use sigmoid)
                    normalized = 1 / (1 + math.exp(-value / 10))
                    score += weight * normalized
                elif isinstance(value, str):
                    # String presence contributes full weight
                    score += weight

        return score


# ============================================================================
# RESULT FUSER
# ============================================================================

class ResultFuser:
    """
    Combines and normalizes scores from multiple retrieval strategies.

    Implements score normalization and weighted combination to create
    a unified ranking from different retrieval approaches.
    """

    @staticmethod
    def normalize_scores(
        scores: List[float],
        method: str = "min_max"
    ) -> List[float]:
        """
        Normalize a list of scores.

        Args:
            scores: Raw scores to normalize
            method: Normalization method ("min_max" or "z_score")

        Returns:
            Normalized scores
        """
        if not scores:
            return []

        if method == "min_max":
            min_score = min(scores)
            max_score = max(scores)
            if max_score == min_score:
                return [0.5] * len(scores)
            return [(s - min_score) / (max_score - min_score) for s in scores]

        elif method == "z_score":
            mean = sum(scores) / len(scores)
            std = math.sqrt(sum((s - mean) ** 2 for s in scores) / len(scores))
            if std == 0:
                return [0.0] * len(scores)
            # Convert z-scores to 0-1 range using sigmoid
            z_scores = [(s - mean) / std for s in scores]
            return [1 / (1 + math.exp(-z)) for z in z_scores]

        return scores

    @staticmethod
    def fuse_scores(
        strategy_scores: Dict[str, float],
        weights: Dict[str, float]
    ) -> Tuple[float, Dict[str, float]]:
        """
        Combine scores from multiple strategies.

        Args:
            strategy_scores: Scores from each strategy
            weights: Weight for each strategy

        Returns:
            Tuple of (final_score, contributions)
        """
        total_weight = sum(weights.values())
        if total_weight == 0:
            return 0.0, {}

        final_score = 0.0
        contributions = {}

        for strategy, score in strategy_scores.items():
            weight = weights.get(strategy, 0.0)
            contribution = (score * weight) / total_weight
            final_score += contribution
            contributions[strategy] = contribution

        return final_score, contributions


# ============================================================================
# RE-RANKER
# ============================================================================

class ReRanker:
    """
    Post-retrieval re-ranking using additional criteria.

    Implements re-ranking algorithms that consider factors like
    document freshness, metadata quality, and retrieval confidence.
    """

    def __init__(
        self,
        freshness_weight: float = 0.1,
        quality_weight: float = 0.1,
        diversity_weight: float = 0.05
    ):
        """Initialize re-ranker with configurable weights."""
        self.freshness_weight = freshness_weight
        self.quality_weight = quality_weight
        self.diversity_weight = diversity_weight

    def compute_freshness_boost(self, document: Document) -> float:
        """
        Compute freshness boost based on document timestamp.

        Args:
            document: Document to evaluate

        Returns:
            Freshness boost (0 to 1)
        """
        timestamp = document.metadata.get("timestamp")
        if timestamp is None:
            return 0.5  # Neutral if no timestamp

        # Simple decay based on age (assuming timestamp is unix timestamp)
        try:
            import time
            age_days = (time.time() - float(timestamp)) / (24 * 3600)
            # Exponential decay with half-life of 30 days
            return math.exp(-age_days / 30)
        except (ValueError, TypeError):
            return 0.5

    def compute_quality_score(self, document: Document) -> float:
        """
        Compute quality score based on document metadata.

        Args:
            document: Document to evaluate

        Returns:
            Quality score (0 to 1)
        """
        quality_signals = []

        # Check for quality indicators in metadata
        if "quality_score" in document.metadata:
            quality_signals.append(float(document.metadata["quality_score"]))

        if "word_count" in document.metadata:
            # Prefer documents with substantial content
            word_count = document.metadata["word_count"]
            if word_count > 100:
                quality_signals.append(min(word_count / 500, 1.0))

        if "source_reliability" in document.metadata:
            quality_signals.append(float(document.metadata["source_reliability"]))

        if not quality_signals:
            return 0.5

        return sum(quality_signals) / len(quality_signals)

    def rerank(
        self,
        results: List[Tuple[Document, float]],
        original_scores: Dict[str, float]
    ) -> List[Tuple[Document, float, Dict[str, float]]]:
        """
        Re-rank results using additional criteria.

        Args:
            results: Initial ranked results (document, score)
            original_scores: Original scores by strategy

        Returns:
            Re-ranked results with adjustment details
        """
        reranked = []

        for doc, base_score in results:
            adjustments = {}

            # Freshness adjustment
            freshness = self.compute_freshness_boost(doc)
            freshness_adj = (freshness - 0.5) * self.freshness_weight
            adjustments["freshness"] = freshness_adj

            # Quality adjustment
            quality = self.compute_quality_score(doc)
            quality_adj = (quality - 0.5) * self.quality_weight
            adjustments["quality"] = quality_adj

            # Final adjusted score
            adjusted_score = base_score + sum(adjustments.values())
            adjusted_score = max(0.0, min(1.0, adjusted_score))

            reranked.append((doc, adjusted_score, adjustments))

        # Sort by adjusted score
        reranked.sort(key=lambda x: x[1], reverse=True)

        return reranked


# ============================================================================
# EXPLANATION GENERATOR
# ============================================================================

class ExplanationGenerator:
    """
    Creates human-readable explanations for retrieval decisions.

    Generates detailed explanations showing score breakdown, strategy
    contributions, and reasoning for document retrieval and ranking.
    """

    @staticmethod
    def generate_explanation(
        document: Document,
        score_breakdown: Dict[str, float],
        strategy_contributions: Dict[str, float],
        query_terms_matched: List[str],
        metadata_matches: Dict[str, Any],
        rerank_adjustments: Dict[str, float]
    ) -> str:
        """
        Generate explanation for a retrieval result.

        Args:
            document: Retrieved document
            score_breakdown: Scores from each component
            strategy_contributions: Weighted contributions
            query_terms_matched: Terms that matched
            metadata_matches: Metadata fields that matched
            rerank_adjustments: Re-ranking adjustments applied

        Returns:
            Human-readable explanation string
        """
        parts = []

        # Overall summary
        total_score = sum(strategy_contributions.values())
        parts.append(f"Retrieved with score {total_score:.3f}")

        # Strategy breakdown
        strategy_parts = []
        for strategy, contribution in strategy_contributions.items():
            if contribution > 0:
                percentage = (contribution / total_score * 100) if total_score > 0 else 0
                strategy_parts.append(f"{strategy}: {contribution:.3f} ({percentage:.1f}%)")

        if strategy_parts:
            parts.append("Strategy contributions: " + ", ".join(strategy_parts))

        # Matched terms
        if query_terms_matched:
            parts.append(f"Matched terms: {', '.join(query_terms_matched[:5])}")
            if len(query_terms_matched) > 5:
                parts.append(f"  ... and {len(query_terms_matched) - 5} more")

        # Metadata matches
        if metadata_matches:
            meta_str = ", ".join(f"{k}={v}" for k, v in list(metadata_matches.items())[:3])
            parts.append(f"Metadata matches: {meta_str}")

        # Re-ranking adjustments
        if rerank_adjustments:
            adj_parts = []
            for adj_type, value in rerank_adjustments.items():
                if abs(value) > 0.001:
                    sign = "+" if value > 0 else ""
                    adj_parts.append(f"{adj_type}: {sign}{value:.3f}")
            if adj_parts:
                parts.append("Re-ranking adjustments: " + ", ".join(adj_parts))

        return " | ".join(parts)


# ============================================================================
# MAIN HYBRID RETRIEVAL SYSTEM
# ============================================================================

class HybridRetrievalSystem:
    """
    Hybrid Retrieval System

    A sophisticated document retrieval system that combines multiple search
    strategies including semantic similarity, keyword matching, and metadata
    filtering. Provides query expansion, re-ranking, and explainable scoring.
    """

    def __init__(
        self,
        semantic_weight: float = 0.4,
        keyword_weight: float = 0.4,
        metadata_weight: float = 0.2,
        enable_reranking: bool = True
    ):
        """
        Initialize the hybrid retrieval system.

        Args:
            semantic_weight: Weight for semantic similarity (0-1)
            keyword_weight: Weight for keyword matching (0-1)
            metadata_weight: Weight for metadata scoring (0-1)
            enable_reranking: Whether to apply re-ranking
        """
        self.query_processor = QueryProcessor()
        self.semantic_retriever = SemanticRetriever()
        self.keyword_retriever = KeywordRetriever()
        self.metadata_filter = MetadataFilter()
        self.result_fuser = ResultFuser()
        self.reranker = ReRanker()
        self.explanation_generator = ExplanationGenerator()

        self.weights = {
            "semantic": semantic_weight,
            "keyword": keyword_weight,
            "metadata": metadata_weight
        }

        self.enable_reranking = enable_reranking

        # Document storage
        self._documents: List[Document] = []

    def add_document(
        self,
        doc_id: str,
        content: str,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Document:
        """
        Add a document to the retrieval system.

        Args:
            doc_id: Unique document identifier
            content: Document text content
            embedding: Pre-computed embedding (auto-generated if None)
            metadata: Document metadata

        Returns:
            Added Document
        """
        # Generate embedding if not provided
        if embedding is None:
            embedding = self.semantic_retriever.simple_embedding(content)

        doc = Document(
            doc_id=doc_id,
            content=content,
            embedding=embedding,
            metadata=metadata or {}
        )

        self._documents.append(doc)
        return doc

    def add_documents(self, documents: List[Dict[str, Any]]) -> List[Document]:
        """
        Add multiple documents.

        Args:
            documents: List of document dictionaries

        Returns:
            List of added Documents
        """
        added = []
        for doc_dict in documents:
            doc = self.add_document(
                doc_id=doc_dict.get("doc_id", str(len(self._documents))),
                content=doc_dict.get("content", ""),
                embedding=doc_dict.get("embedding"),
                metadata=doc_dict.get("metadata")
            )
            added.append(doc)
        return added

    def expand_query(self, query: str) -> QueryExpansion:
        """
        Expand a query with synonyms and related terms.

        Args:
            query: Original query

        Returns:
            QueryExpansion result
        """
        return self.query_processor.expand_query(query)

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        metadata_scoring_fields: Optional[Dict[str, float]] = None,
        expand_query: bool = True
    ) -> List[RetrievalResult]:
        """
        Retrieve documents using hybrid strategies.

        Args:
            query: Search query
            top_k: Number of results to return
            filters: Metadata filters to apply
            metadata_scoring_fields: Fields to use for metadata scoring
            expand_query: Whether to expand query with synonyms

        Returns:
            List of RetrievalResult with explanations
        """
        if not self._documents:
            return []

        # Query expansion
        if expand_query:
            expansion = self.query_processor.expand_query(query)
            expanded_query = " ".join(expansion.expanded_terms)
        else:
            expansion = QueryExpansion(query, [query], {}, "none")
            expanded_query = query

        # Filter documents by metadata
        filtered_docs = []
        metadata_match_info = {}

        for doc in self._documents:
            matches, matched_fields = self.metadata_filter.matches_filter(doc, filters or {})
            if matches:
                filtered_docs.append(doc)
                metadata_match_info[doc.doc_id] = matched_fields

        if not filtered_docs:
            return []

        # Semantic retrieval
        query_embedding = self.semantic_retriever.simple_embedding(expanded_query)
        semantic_results = self.semantic_retriever.retrieve(
            query_embedding, filtered_docs, top_k * 2
        )
        semantic_scores = {doc.doc_id: score for doc, score in semantic_results}

        # Keyword retrieval
        keyword_results = self.keyword_retriever.retrieve(
            expanded_query, filtered_docs, top_k * 2
        )
        keyword_scores = {}
        keyword_matches = {}
        for doc, score, matched in keyword_results:
            keyword_scores[doc.doc_id] = score
            keyword_matches[doc.doc_id] = matched

        # Normalize scores
        if semantic_scores:
            sem_list = list(semantic_scores.values())
            sem_normalized = self.result_fuser.normalize_scores(sem_list)
            semantic_scores = dict(zip(semantic_scores.keys(), sem_normalized))

        if keyword_scores:
            kw_list = list(keyword_scores.values())
            kw_normalized = self.result_fuser.normalize_scores(kw_list)
            keyword_scores = dict(zip(keyword_scores.keys(), kw_normalized))

        # Combine scores for all candidate documents
        all_doc_ids = set(semantic_scores.keys()) | set(keyword_scores.keys())
        combined_results = []

        for doc_id in all_doc_ids:
            doc = next(d for d in filtered_docs if d.doc_id == doc_id)

            # Get individual scores
            sem_score = semantic_scores.get(doc_id, 0.0)
            kw_score = keyword_scores.get(doc_id, 0.0)

            # Metadata scoring
            meta_score = 0.0
            if metadata_scoring_fields:
                meta_score = self.metadata_filter.compute_metadata_score(
                    doc, metadata_scoring_fields
                )

            # Fuse scores
            strategy_scores = {
                "semantic": sem_score,
                "keyword": kw_score,
                "metadata": meta_score
            }

            final_score, contributions = self.result_fuser.fuse_scores(
                strategy_scores, self.weights
            )

            combined_results.append((doc, final_score, strategy_scores, contributions))

        # Sort by combined score
        combined_results.sort(key=lambda x: x[1], reverse=True)

        # Re-ranking
        rerank_adjustments = {}
        if self.enable_reranking:
            rerank_input = [(doc, score) for doc, score, _, _ in combined_results]
            reranked = self.reranker.rerank(rerank_input, {})
            rerank_map = {doc.doc_id: (score, adj) for doc, score, adj in reranked}

            # Update with reranked scores
            new_results = []
            for doc, old_score, strategy_scores, contributions in combined_results:
                if doc.doc_id in rerank_map:
                    new_score, adjustments = rerank_map[doc.doc_id]
                    rerank_adjustments[doc.doc_id] = adjustments
                    new_results.append((doc, new_score, strategy_scores, contributions))
                else:
                    new_results.append((doc, old_score, strategy_scores, contributions))

            combined_results = new_results
            combined_results.sort(key=lambda x: x[1], reverse=True)

        # Build final results with explanations
        final_results = []
        for doc, final_score, strategy_scores, contributions in combined_results[:top_k]:
            explanation = self.explanation_generator.generate_explanation(
                document=doc,
                score_breakdown=strategy_scores,
                strategy_contributions=contributions,
                query_terms_matched=keyword_matches.get(doc.doc_id, []),
                metadata_matches=metadata_match_info.get(doc.doc_id, {}),
                rerank_adjustments=rerank_adjustments.get(doc.doc_id, {})
            )

            result = RetrievalResult(
                document=doc,
                final_score=final_score,
                score_breakdown=strategy_scores,
                explanation=explanation,
                strategy_contributions=contributions,
                query_terms_matched=keyword_matches.get(doc.doc_id, []),
                metadata_matches=metadata_match_info.get(doc.doc_id, {})
            )
            final_results.append(result)

        return final_results

    def get_document_count(self) -> int:
        """Get number of documents in the system."""
        return len(self._documents)

    def clear(self) -> None:
        """Clear all documents."""
        self._documents.clear()

    def execute(
        self,
        query: str = "",
        documents: Optional[List[Dict[str, Any]]] = None,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Main entry point for the hybrid retrieval system.

        Args:
            query: Search query (uses demo query if empty)
            documents: Documents to add (uses demo docs if None)
            top_k: Number of results
            filters: Metadata filters

        Returns:
            Retrieval results with explanations
        """
        # Clear previous documents if new ones provided
        if documents is not None:
            self.clear()
            self.add_documents(documents)

        # Use demo data if no documents
        if not self._documents:
            demo_docs = [
                {
                    "doc_id": "doc1",
                    "content": "Machine learning models require training data and validation sets for proper evaluation.",
                    "metadata": {"category": "ml", "year": 2023}
                },
                {
                    "doc_id": "doc2",
                    "content": "RAG systems combine retrieval with generation for better answers to user questions.",
                    "metadata": {"category": "rag", "year": 2024}
                },
                {
                    "doc_id": "doc3",
                    "content": "Vector databases store embeddings for semantic search and similarity matching.",
                    "metadata": {"category": "database", "year": 2023}
                },
                {
                    "doc_id": "doc4",
                    "content": "Neural networks learn patterns from examples through backpropagation and gradient descent.",
                    "metadata": {"category": "ml", "year": 2022}
                },
                {
                    "doc_id": "doc5",
                    "content": "API endpoints provide interfaces for client applications to interact with server services.",
                    "metadata": {"category": "api", "year": 2024}
                }
            ]
            self.add_documents(demo_docs)

        # Use demo query if empty
        if not query:
            query = "machine learning model training"

        # Perform retrieval
        results = self.retrieve(query, top_k=top_k, filters=filters, expand_query=True)

        # Format output
        query_expansion = self.expand_query(query)

        return {
            "query": query,
            "expanded_query": " ".join(query_expansion.expanded_terms),
            "synonyms_used": query_expansion.synonyms_used,
            "total_documents": self.get_document_count(),
            "results_count": len(results),
            "results": [
                {
                    "doc_id": r.document.doc_id,
                    "content": r.document.content[:200] + "..." if len(r.document.content) > 200 else r.document.content,
                    "score": r.final_score,
                    "score_breakdown": r.score_breakdown,
                    "strategy_contributions": r.strategy_contributions,
                    "matched_terms": r.query_terms_matched,
                    "explanation": r.explanation
                }
                for r in results
            ]
        }


def create_hybrid_retrieval_system() -> HybridRetrievalSystem:
    """
    Factory function for creating HybridRetrievalSystem instances.

    Returns:
        HybridRetrievalSystem: A new instance of HybridRetrievalSystem
    """
    return HybridRetrievalSystem()
