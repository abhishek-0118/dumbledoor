"""Simplified and dynamic prompt templates"""

# Main dynamic prompt template
DYNAMIC_CODE_PROMPT_TEMPLATE = """You are an expert software engineer and technical documentation specialist with deep understanding of complex systems.

## Context from Codebase:
{context}

## User Question:
{question}

## Instructions:
1. **Analyze the context thoroughly** - Look for patterns, connections, and relevant information across all provided code sections
2. **Provide comprehensive, insightful explanations** - Connect different parts of the code to give complete understanding
3. **Use actual code snippets** from the context to illustrate your points with precise explanations
4. **Create clear, well-formatted responses** using proper markdown syntax with detailed structure
5. **Add ASCII diagrams** when they help explain architecture, data flow, processes, or relationships
6. **Be thorough in your analysis** - Even if information seems scattered, piece together the full picture from available context
7. **Focus on practical understanding** - Explain not just what the code does, but how it fits into the larger system

## Response Format:
- Use **clear headings** (##, ###) to organize complex topics
- Use `code blocks` code snippets
- Use ``technical terms`` for function names, variables, file paths etc.
- Use bullet points and numbered lists for processes and multiple items
- Create ASCII diagrams for complex concepts:
  ```
  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
  │  Component  │───▶│   Process   │───▶│   Result    │
  └─────────────┘    └─────────────┘    └─────────────┘
  ```
- Include relevant file paths and function signatures
- Show relationships between different code sections

## Response Guidelines:
- **Be thorough**: Extract maximum insight from the provided context
- **Be analytical**: Explain the reasoning behind code design and implementation choices
- **Be comprehensive**: Cover all relevant aspects found in the context, connecting related pieces
- **Be practical**: Focus on real-world usage, implementation details, and system behavior
- **Be clear**: Use examples and step-by-step explanations
- **Create diagrams**: Visual representations help understand complex systems and flows

## Special Focus Areas:
- **System Architecture**: How components interact and fit together
- **Data Flow**: How information moves through the system
- **Business Logic**: The underlying processes and rules
- **Integration Points**: How different services or modules connect
- **Error Handling**: How the system deals with edge cases
- **Configuration**: How the system is configured and customized

Provide a detailed, insightful analysis based on the codebase context."""

FALLBACK_MESSAGES = {
    "no_results": """##  No Relevant Information Found

I couldn't find information in the indexed codebase that matches your query.

### 🔧 Suggestions:
1. **Try broader search terms** - Use more general keywords
2. **Check spelling** - Ensure technical terms are spelled correctly  
3. **Use different terminology** - Try synonyms or related concepts
4. **Be more specific** - Add context about the type of code or functionality you're looking for

### Tips for Better Results:
- Ask about specific **functions**, **classes**, or **modules**
- Mention the **programming language** or **framework**
- Describe what you're trying to **accomplish** or **understand**
""",
    
    "error": """##  Search Error

I encountered an issue while searching the codebase. Please try:

1. **Rephrasing your question** with different keywords
2. **Breaking down complex queries** into smaller, specific questions  
3. **Being more specific** about what you're looking for

If the issue persists, there may be a technical problem with the search system."""
}

# ASCII diagram templates removed - were unused dead code

# Dynamic context enhancement
def create_dynamic_context(repositories: list, languages: list, total_docs: int) -> str:
    """Create dynamic context information based on search results"""
    context_info = []
    
    if repositories:
        if len(repositories) == 1:
            context_info.append(f"**Repository**: {repositories[0]}")
        else:
            context_info.append(f"**Repositories**: {', '.join(repositories)}")
    
    if languages:
        if len(languages) == 1:
            context_info.append(f"**Language**: {languages[0]}")
        else:
            context_info.append(f"**Languages**: {', '.join(languages)}")
    
    if total_docs > 0:
        context_info.append(f"**Documents analyzed**: {total_docs}")
    
    if context_info:
        return "\n\n---\n\n### 📊 Context Information\n\n" + "\n".join(f"- {info}" for info in context_info)
    
    return ""

# Dynamic prompt creation
def create_dynamic_prompt(context: str, question: str, context_summary: dict = None) -> str:
    """Create a dynamic prompt with context-aware enhancements"""
    base_prompt = DYNAMIC_CODE_PROMPT_TEMPLATE.format(context=context, question=question)
    
    # Add dynamic context information if available
    if context_summary:
        repos = context_summary.get("repositories", [])
        languages = context_summary.get("languages", [])
        total_docs = context_summary.get("total_documents", 0)
        
        dynamic_context = create_dynamic_context(repos, languages, total_docs)
        if dynamic_context:
            base_prompt += dynamic_context
    
    return base_prompt