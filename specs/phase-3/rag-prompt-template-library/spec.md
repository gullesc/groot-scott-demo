# Feature Spec: RAG Prompt Template Library

## Overview
The RAG Prompt Template Library is a comprehensive system for generating sophisticated prompts that optimize Retrieval-Augmented Generation workflows with Anthropic Claude. This library addresses the critical challenge of effectively presenting retrieved context to language models while ensuring accurate, well-cited responses across different domains and query types.

By providing structured, reusable prompt templates, this feature enables developers to implement consistent prompt engineering best practices without starting from scratch for each use case. The library incorporates advanced techniques like few-shot prompting, chain-of-thought reasoning, and dynamic context management to maximize the quality and reliability of RAG-based responses.

The system is particularly valuable for educational purposes, as it demonstrates how different prompt structures affect model behavior and provides a foundation for understanding the nuances of prompt engineering in retrieval-augmented scenarios.

## Requirements

### Functional Requirements
1. **Template Categories**: Implement distinct prompt templates for factual Q&A, summarization, comparison, and analysis query types
2. **Dynamic Context Integration**: Provide mechanisms to dynamically insert retrieved documents with proper formatting and organization
3. **Domain Specialization**: Include specialized templates optimized for technical documentation, legal documents, and academic papers
4. **Confidence-Based Fallbacks**: Implement fallback prompt strategies for scenarios with low-confidence retrievals or insufficient context
5. **Citation Management**: Ensure all templates encourage proper source attribution and reduce hallucination through explicit citation instructions
6. **Template Customization**: Allow for parameter-based customization of templates for specific use cases

### Non-Functional Requirements
- Must use only the Python standard library
- Must work with the existing project structure
- Code must be educational and well-commented
- Templates must be easily extensible and maintainable
- Implementation should demonstrate prompt engineering principles clearly

## Interface

### Input
- Query type (factual_qa, summarization, comparison, analysis)
- User question or request
- Retrieved document chunks (list of dictionaries with content and metadata)
- Document domain type (technical, legal, academic, general)
- Confidence scores for retrieved documents
- Optional customization parameters (temperature suggestions, specific instructions)

### Output
- Complete prompt string formatted for Anthropic Claude
- Structured prompt with clear sections for context, instructions, and query
- Appropriate fallback prompts when retrieval confidence is low
- Citation formatting instructions embedded in the prompt

## Acceptance Criteria
- [ ] Includes templates for factual Q&A, summarization, comparison, and analysis
- [ ] Implements dynamic context insertion with proper formatting
- [ ] Provides templates for different document types (technical, legal, academic)
- [ ] Includes fallback prompts for low-confidence retrievals

## Examples

### Factual Q&A Template
```
Input: query_type="factual_qa", question="What is photosynthesis?", documents=[{content: "Photosynthesis is...", source: "biology_textbook.pdf"}]
Output: "You are a helpful assistant that answers questions based on provided context..."
```

### Summarization Template
```
Input: query_type="summarization", question="Summarize the key findings", documents=[multiple research papers]
Output: "Please provide a comprehensive summary of the following documents..."
```

### Low Confidence Fallback
```
Input: confidence_scores=[0.2, 0.15], question="Complex technical question"
Output: "Based on limited relevant context, I should acknowledge uncertainty..."
```

## Dependencies
- Source file: `src/rag_prompt_template_library.py`
- Test file: `tests/test_rag_prompt_template_library.py`