# Tasks: RAG Prompt Template Library

## Prerequisites
- [ ] Read spec.md to understand requirements
- [ ] Read plan.md to understand the approach
- [ ] Run tests to see them fail: `python3 -m pytest -v`

## Implementation Tasks

- [ ] **Task 1**: Implement Base Template System
  - File: `src/rag_prompt_template_library.py`
  - Details: Create template constants for factual_qa, summarization, comparison, and analysis query types with placeholder variables for dynamic content insertion.

- [ ] **Task 2**: Build Context Formatting Functions
  - File: `src/rag_prompt_template_library.py`
  - Details: Implement functions to format retrieved documents into clean, numbered context sections with proper source attribution and metadata handling.

- [ ] **Task 3**: Implement Domain-Specific Template Variations
  - File: `src/rag_prompt_template_library.py`
  - Details: Create specialized prompt modifications for technical, legal, and academic document types that adjust language and citation requirements appropriately.

- [ ] **Task 4**: Add Confidence-Based Fallback Logic
  - File: `src/rag_prompt_template_library.py`
  - Details: Implement fallback prompt templates and logic for low-confidence retrievals, including appropriate uncertainty acknowledgments and limited-context responses.

- [ ] **Task 5**: Integrate Citation Management
  - File: `src/rag_prompt_template_library.py`
  - Details: Embed explicit citation instructions in all templates and implement functions to format source references consistently across different document types.

- [ ] **Task 6**: Complete Main Execute Method
  - File: `src/rag_prompt_template_library.py`
  - Details: Implement the main execute() method that orchestrates template selection, context formatting, and final prompt assembly based on input parameters.

## Verification
- [ ] All tests pass: `python3 -m pytest -v`
- [ ] Code follows project constitution
- [ ] No external dependencies added
- [ ] Code includes helpful comments