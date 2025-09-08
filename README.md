# Enhanced Repository Indexing and Search Service

A powerful code search and analysis tool that provides comprehensive, context-aware answers about your codebase using advanced AI and vector embeddings.

## Features

### Enhanced Search Capabilities
- **Context-Aware Search**: Finds relevant code with rich context and relationships
- **Code Structure Understanding**: Analyzes classes, functions, imports, and module relationships
- **Multi-Language Support**: Python, JavaScript, TypeScript, Go, Java, and more
- **Intelligent Reranking**: Cross-encoder models for better result relevance
- **Related Context Discovery**: Automatically finds related code from same files/modules

### 🧠 Smart Code Analysis
- **Query Understanding**: Analyzes intent (how-to, debugging, implementation, architecture)
- **Code-Aware Chunking**: Preserves function/class boundaries for better context
- **Metadata Enrichment**: File types, languages, test detection, configuration files
- **Architectural Insights**: Understands project structure and component relationships

### ⚡ Advanced Operations
- **Force Cleanup**: `--force-cleanup` - Remove all embeddings and indexes
- **Force Re-indexing**: `--force-index` - Clean up and re-index all repositories
- **Test Search**: `--test-search "query"` - Test search functionality
- **Enhanced API**: More comprehensive responses with context summaries

## Configuration

### Basic Setup
```yaml
# config/local.yaml or config/prod.yaml
indexing:
    local_paths: ["./repos/your-repo"]  # Path to your repositories
    chunk_size: 1500                   # Increased for better context
    chunk_overlap: 200                 # Better continuity between chunks
    max_file_mb: 2.0                   # Support larger files

retrieval:
    top_k: 20                          # More comprehensive results
    use_reranker: true                 # Enable cross-encoder reranking
    include_file_context: true         # Include related file context
    boost_same_language: true          # Prioritize same programming language
```

## Quick Start

### Step 1: Setup Repository
```bash
# Copy your repository to the repos directory
cp -r /path/to/your-repo ./repos/

# Update config/local.yaml with the correct path
# indexing:
#   local_paths: ["./repos/your-repo"]
```

### Step 2: Index Your Code
```bash
# Clean start with enhanced indexing
python -m app.cli --force-index --env local

# Or just index normally
python -m app.cli --index --env local
```

### Step 3: Test Search
```bash
# Test the enhanced search functionality
python -m app.cli --test-search "How to implement authentication?" --env local

# Start the API server
python -m app.cli --serve --env local
```

### Step 4: Query Your Codebase
```bash
# Example API queries
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{
       "q": "How does the authentication system work?",
       "k": 15,
       "detailed_response": true,
       "include_context": true
     }'
```

## Enhanced API Response

The API now returns much richer information:

```json
{
  "answer": "Comprehensive answer with code examples and explanations...",
  "sources": [
    {
      "repo": "my-app",
      "path": "app/auth/github.py", 
      "language": "python",
      "module_name": "app.auth.github",
      "is_test": false,
      "is_config": false,
      "preview": "Enhanced preview with file context...",
      "relevance_score": 0.95
    }
  ],
  "context_summary": {
    "total_documents": 15,
    "languages": ["python", "javascript"],
    "file_types": [".py", ".js", ".md"],
    "modules": ["app.auth", "app.api", "app.config"],
    "test_files": 3,
    "config_files": 2
  },
  "query_analysis": {
    "is_how_to": true,
    "code_related": true,
    "mentions_specific_tech": ["python"]
  },
  "total_sources_found": 25
}
```

## Advanced Usage

### Environment Management
```bash
# Production indexing with GitHub repositories
python -m app.cli --env prod --index

# Force cleanup (removes all indexes and embeddings)
python -m app.cli --force-cleanup --env local

# Complete re-indexing
python -m app.cli --force-index --env prod
```

### Docker Deployment
```bash
# Local development
docker compose up -d

# Production deployment  
docker compose --profile prod up -d

# Rebuild with latest changes
docker compose up --build

# View logs
docker compose logs -f

# Stop services
docker compose down
```

### Testing Enhanced Features
```bash
# Run comprehensive search test
python test_enhanced_search.py

# Test specific queries
python -m app.cli --test-search "Show me database configuration patterns"
python -m app.cli --test-search "How to write unit tests for API endpoints"
python -m app.cli --test-search "What is the project architecture"
```

## Enhanced Features in Detail

### 1. Code-Aware Indexing
- **Smart Chunking**: Preserves function, class, and logical boundaries
- **Rich Metadata**: Language detection, test/config file identification
- **Structure Analysis**: Extracts imports, exports, classes, functions
- **Context Headers**: Each chunk includes file context and structure info

### 2. Intelligent Search
- **Query Analysis**: Understands what you're looking for (how-to, debugging, etc.)
- **Multi-Stage Retrieval**: Vector search + reranking + related context
- **Language Awareness**: Boosts results in relevant programming languages
- **Context Expansion**: Finds related code from same files/modules

### 3. Comprehensive Responses
- **Detailed Answers**: Step-by-step explanations with code examples
- **Source Attribution**: Rich metadata about where answers come from
- **Context Summaries**: Overview of what was found in your codebase
- **Fallback Handling**: Graceful degradation when AI services are unavailable

### 4. Developer Experience
- **Test Commands**: Easy way to test search functionality
- **Force Operations**: Clean slate re-indexing capabilities
- **Rich Logging**: Detailed information about indexing and search processes
- **Configuration Validation**: Clear error messages for setup issues

## Configuration Options

### Indexing Configuration
```yaml
indexing:
  chunk_size: 1500              # Size of text chunks (increased for context)
  chunk_overlap: 200            # Overlap between chunks
  max_file_mb: 2.0             # Maximum file size to index
  include_globs: ["**/*"]       # Files to include
  exclude_globs:                # Files to exclude
    - "**/.git/**"
    - "**/node_modules/**"
    - "**/dist/**"
    - "**/*.min.js"
```

### Retrieval Configuration
```yaml
retrieval:
  top_k: 20                     # Number of results to return
  use_reranker: true            # Enable cross-encoder reranking
  similarity_threshold: 0.4     # Minimum similarity score
  max_context_docs: 30          # Maximum documents for context
  include_file_context: true    # Include related file context
  boost_same_language: true     # Boost same language results
```

## Troubleshooting

### Common Issues
1. **No results found**: Check if repositories are properly indexed
2. **Poor quality answers**: Try `--force-index` to rebuild with enhanced features
3. **Slow performance**: Reduce `top_k` or disable reranking temporarily
4. **Memory issues**: Reduce `chunk_size` or `max_file_mb`

### Debug Commands
```bash
# Check indexing status
python -m app.cli --test-search "test query" --env local

# Verify configuration
python -c "from app.config.loader import load_config; print(load_config())"

# Test without reranking
# Set use_reranker: false in config
```

## Environment Variables
```bash
# Required for some embedding providers
export OPENAI_API_KEY="your-key"
export GOOGLE_API_KEY="your-key"

# For GitHub integration
export GITHUB_TOKEN="your-token"

# Gemini for chat responses
export GEMINI_API_KEY="your-key"
```