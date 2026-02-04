# Implementation Plan: Semantic RAG System

## Approach
The implementation will create a simple but effective semantic search system using TF-IDF vectors as embeddings, which can be computed using only the Python standard library. While not as sophisticated as transformer-based embeddings, TF-IDF provides a solid foundation for understanding semantic search concepts and demonstrates clear improvements over keyword matching.

The system will use a document-term matrix approach where each document is represented as a vector in high-dimensional space. Cosine similarity will measure the angle between query and document vectors, with smaller angles indicating higher semantic similarity. This approach is educational, interpretable, and sufficient to demonstrate the core concepts of semantic search in RAG systems.

## Architecture

### Key Components
- **SemanticRAGSystem**: Main class coordinating the semantic search functionality
- **VectorIndex**: In-memory storage for document embeddings and metadata
- **EmbeddingGenerator**: Creates TF-IDF vector representations of text
- **SimilarityCalculator**: Computes cosine similarity between vectors
- **ResultRanker**: Sorts and filters search results based on relevance scores
- **PerformanceComparator**: Demonstrates improvements over keyword-based search

### Data Flow
1. Documents are preprocessed (tokenization, normalization)
2. TF-IDF vectors are computed for each document and stored in the vector index
3. Query text is converted to the same vector representation
4. Cosine similarity is calculated between query vector and all document vectors
5. Results are ranked by similarity score and filtered by threshold
6. Top-k most relevant documents are returned with scores

## Implementation Steps

### Step 1: Text Preprocessing
- Implement tokenization, lowercasing, and basic text normalization
- Handle punctuation removal and stop word filtering
- Create vocabulary mapping for consistent vector dimensions

### Step 2: TF-IDF Vector Generation
- Calculate term frequency for each document
- Compute inverse document frequency across the corpus
- Generate normalized TF-IDF vectors for documents and queries

### Step 3: Vector Index Creation
- Build in-memory storage for document embeddings
- Maintain mapping between vectors and original document content
- Support efficient similarity computation across all stored vectors

### Step 4: Similarity Search Implementation
- Implement cosine similarity calculation using dot product and norms
- Support batch similarity computation for multiple documents
- Apply relevance thresholds to filter low-quality matches

### Step 5: Result Processing
- Rank results by similarity scores in descending order
- Format output with document content and relevance scores
- Implement top-k result selection

## Key Decisions

**TF-IDF over Word Counts**: TF-IDF provides better semantic representation than simple word counts by considering both local term importance and global term rarity, making it ideal for educational purposes.

**Cosine Similarity Metric**: Cosine similarity is chosen over Euclidean distance because it focuses on vector direction rather than magnitude, making it more suitable for text similarity where document length varies.

**In-Memory Storage**: Using in-memory vector storage keeps the implementation simple while demonstrating core concepts, though real systems would use persistent vector databases.

## Testing Strategy
- Tests are already provided in `tests/test_semantic_rag_system.py`
- Run with: `python3 -m pytest -v`
- Tests verify each acceptance criterion
- Include tests for edge cases like empty queries and single-document corpora

## Edge Cases
- Empty query strings or document lists
- Single document in corpus (IDF calculation edge case)
- All documents having identical content
- Queries with no vocabulary overlap with documents
- Very short documents with minimal text content