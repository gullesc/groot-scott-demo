# Tasks: Basic RAG Question-Answering System

## Prerequisites
- [ ] Read spec.md to understand requirements
- [ ] Read plan.md to understand the approach
- [ ] Run tests to see them fail: `python3 -m pytest -v`

## Implementation Tasks

- [ ] **Task 1**: Implement core class initialization and document storage
  - File: `src/basic_rag_question_answering_system.py`
  - Details: Set up the BasicRAGQuestionAnsweringSystem class with proper initialization, document storage, and parameter validation for API key and configuration options.

- [ ] **Task 2**: Build text preprocessing pipeline
  - File: `src/basic_rag_question_answering_system.py`
  - Details: Implement tokenization, text normalization, and basic stopword removal using only standard library string methods and regular expressions.

- [ ] **Task 3**: Create TF-IDF calculation system
  - File: `src/basic_rag_question_answering_system.py`
  - Details: Build term frequency and inverse document frequency calculations from scratch using dictionaries and math library, creating vector representations for documents and queries.

- [ ] **Task 4**: Implement similarity scoring and retrieval ranking
  - File: `src/basic_rag_question_answering_system.py`
  - Details: Calculate cosine similarity between question and document vectors, then rank and select the most relevant chunks based on similarity scores.

- [ ] **Task 5**: Develop context formatting for Claude API
  - File: `src/basic_rag_question_answering_system.py`
  - Details: Structure retrieved document chunks into well-formatted prompts with clear context boundaries and source attribution for optimal Claude processing.

- [ ] **Task 6**: Integrate Claude API calls with error handling
  - File: `src/basic_rag_question_answering_system.py`
  - Details: Implement HTTP requests to Anthropic Claude API using urllib from standard library, with proper error handling and response processing.

- [ ] **Task 7**: Build comprehensive source citation system
  - File: `src/basic_rag_question_answering_system.py`
  - Details: Create source tracking that maps generated answers back to specific document chunks and formats proper citations for transparency.

- [ ] **Task 8**: Implement the main execute() method with full pipeline
  - File: `src/basic_rag_question_answering_system.py`
  - Details: Integrate all components into the execute() method that orchestrates the complete RAG workflow from question input to cited answer output.

## Verification
- [ ] All tests pass: `python3 -m pytest -v`
- [ ] Code follows project constitution
- [ ] No external dependencies added
- [ ] Code includes helpful comments