"""Prompt templates and messaging constants"""

# Base code-aware prompt template
BASE_CODE_PROMPT_TEMPLATE = """You are a software engineer. Use the code context to answer the question accurately.

Context:
{context}

Question: {question}

Instructions:
- Answer based on the actual code provided
- Use clear markdown formatting with code blocks
- Reference specific files and functions
- Keep response focused and practical

"""

# Enhanced prompt templates for different query types
ENHANCED_PROMPT_TEMPLATES = {
    "how_to": "Please provide step-by-step instructions with code examples.",
    "what_is": "Please provide a clear explanation with examples from the codebase.",
    "debugging": "Please analyze the code for potential issues and suggest fixes.",
    "implementation": "Please provide implementation details with relevant code patterns from the codebase.",
    "architectural": "Please provide an architectural overview showing how components interact.",
    "detailed": "Please provide a comprehensive answer with code examples and explanations.",
    "code_related": "- Focus on code structure and functions\n",
    "test_related": "- Pay attention to test files and patterns\n",
    "config_related": "- Focus on configuration and settings\n",
}

# Cross-repo architectural ranking prompt
ARCHITECTURAL_RANKING_PROMPT = """You are ranking code snippets that best answer the user's query, prioritizing FUNCTION definitions, calls, and architectural patterns.
Query: {query}
Snippets (index: [repo_name] content preview):
{items}

Consider cross-repo relationships and architectural flow when ranking. Return a JSON list of the top indices in order of relevance (e.g., [3,1,0])."""

# Fallback messages
FALLBACK_MESSAGES = {
    "no_results": """## No Results Found

I couldn't find relevant information in the codebase to answer your question.

**Suggestions:**
- Try rephrasing your question
- Use more specific terms
- Check if the code you're looking for exists in the indexed repositories""",
    
    "search_results_header": "## Search Results\n\nBased on the codebase analysis, I found relevant information but couldn't generate a complete AI response.\n\n",
    
    "next_steps": """### Next Steps

- Review the files mentioned above for detailed implementation
- Try rephrasing your question for better AI analysis
- Ask more specific questions about particular functions or modules""",
    
    "stream_error": "I couldn't generate a proper response. Please try rephrasing your question."
}

# Context enhancement templates
CONTEXT_TEMPLATES = {
    "file_header": """FILE: {rel_path}
REPOSITORY: {repo_name}
LANGUAGE: {language}
PATH: {rel_path}

""",
    "structure_header": "STRUCTURE:\n{structure_info}\n\n"
}

# Query analysis instructions
QUERY_ANALYSIS_PROMPTS = {
    "context_with_languages": "Context: Looking at code in {languages} from {total_documents} files",
    "language_boost": " language:{language}",
    "test_boost": " test",
    "module_boost": " module:{module_name}"
}

# Stream response templates
STREAM_MESSAGES = {
    "starting": "Initializing search and analysis...",
    "analyzing": "Analyzing query and preparing search...",
    "searching": "Searching through codebase...",
    "generating": "Generating comprehensive answer...",
    "completed": "Answer generation completed successfully",
    "error": "An error occurred: {error}"
}
