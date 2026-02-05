# Feature Spec: Context-Aware RAG System

## Overview
The Context-Aware RAG System is an intelligent retrieval-augmented generation implementation that dynamically adapts its prompting strategy based on the quality and quantity of retrieved information. This system represents a sophisticated approach to RAG that goes beyond simple context concatenation, instead analyzing retrieval confidence scores and query complexity to select the most appropriate prompt engineering techniques.

In traditional RAG systems, a single prompt template is often used regardless of whether the retrieval system found highly relevant documents or struggled to find matching content. This context-aware system addresses this limitation by implementing multiple prompt strategies that can handle varying scenarios: from high-confidence retrievals with abundant context to low-confidence situations with sparse information. The system also incorporates query classification to determine whether a question requires factual lookup, analytical reasoning, or creative synthesis.

This implementation is particularly valuable for educational purposes as it demonstrates advanced prompt engineering concepts including few-shot prompting, chain-of-thought reasoning, and dynamic context management—all essential skills for building production-ready RAG applications with Claude.

## Requirements

### Functional Requirements
1. **Retrieval Analysis**: The system must analyze confidence scores from retrieved documents and categorize retrieval quality as high, medium, or low confidence
2. **Adaptive Prompting**: Based on retrieval quality, the system must select from different prompt strategies (direct answering for high confidence, cautious reasoning for low confidence)
3. **Context Management**: The system must intelligently truncate or organize large amounts of retrieved context while preserving the most relevant information
4. **Query Classification**: Implement automatic classification of user queries into categories (factual, analytical, creative) to select appropriate prompt templates
5. **Citation Integration**: Generate responses that include proper citations and source references based on the prompt strategy
6. **Confidence Reporting**: Provide confidence indicators in responses that reflect both retrieval quality and answer certainty

### Non-Functional Requirements
- Must use only the Python standard library
- Must work with the existing project structure
- Code must be educational and well-commented
- Must demonstrate prompt engineering best practices
- Should handle edge cases gracefully

## Interface

### Input
The system accepts a dictionary containing:
- `query`: User's question or request (string)
- `retrieved_contexts`: List of dictionaries, each containing:
  - `content`: Retrieved text content (string)
  - `confidence_score`: Retrieval confidence from 0.0 to 1.0 (float)
  - `source`: Source identifier or URL (string)
- `max_context_length`: Optional maximum character limit for context (integer)

### Output
The system returns a dictionary containing:
- `response`: Generated answer incorporating retrieved information (string)
- `confidence_level`: Overall confidence in the response ("high", "medium", "low") (string)
- `retrieval_quality`: Assessment of retrieval results ("excellent", "good", "poor") (string)
- `query_type`: Classified query category ("factual", "analytical", "creative") (string)
- `sources_used`: List of source identifiers that were cited (list)
- `prompt_strategy`: Which prompting approach was used (string)

## Acceptance Criteria
- [ ] Analyzes retrieval confidence scores to adjust prompt strategy
- [ ] Handles cases with too much or too little retrieved context
- [ ] Implements query classification to select appropriate prompt templates
- [ ] Includes confidence indicators in responses

## Examples

### Example 1: High-Confidence Factual Query
**Input:**
```python
{
    "query": "What is the capital of France?",
    "retrieved_contexts": [
        {
            "content": "Paris is the capital and most populous city of France.",
            "confidence_score": 0.95,
            "source": "encyclopedia.com/france"
        }
    ]
}
```

**Output:**
```python
{
    "response": "The capital of France is Paris. [Source: encyclopedia.com/france]",
    "confidence_level": "high",
    "retrieval_quality": "excellent",
    "query_type": "factual",
    "sources_used": ["encyclopedia.com/france"],
    "prompt_strategy": "direct_answer"
}
```

### Example 2: Low-Confidence Analytical Query
**Input:**
```python
{
    "query": "How will AI impact healthcare in the next decade?",
    "retrieved_contexts": [
        {
            "content": "AI applications in medical imaging have shown promise.",
            "confidence_score": 0.3,
            "source": "tech_blog.com"
        }
    ]
}
```

**Output:**
```python
{
    "response": "Based on limited available information, AI appears to have potential applications in healthcare, particularly in medical imaging [Source: tech_blog.com]. However, I have low confidence in providing a comprehensive analysis of AI's impact over the next decade due to insufficient retrieval results.",
    "confidence_level": "low",
    "retrieval_quality": "poor",
    "query_type": "analytical",
    "sources_used": ["tech_blog.com"],
    "prompt_strategy": "cautious_reasoning"
}
```

## Dependencies
- Source file: `src/context_aware_rag_system.py`
- Test file: `tests/test_context_aware_rag_system.py`