"""System-wide constants and configuration values"""

# Search and Retrieval Constants
SEARCH_CONFIG = {
    # Content limits
    "MAX_EMBEDDING_CONTENT_LENGTH": 500,  # For similarity calculations
    
    # Scoring and similarity boosts
    "SIMILARITY_BOOST_TEST": 0.1,      # Boost for test file matches
    "SIMILARITY_BOOST_CONFIG": 0.1,    # Boost for config file matches  
    "SIMILARITY_BOOST_LANGUAGE": 0.05, # Boost for language matches
}

# Token and Cost Management
TOKEN_CONFIG = {
    # Context limits per model - Increased for better coverage
    "MAX_CONTEXT_TOKENS": {
        "gpt-4": 8000,
        "gpt-4o": 8000,
        "gpt-4o-mini": 6000,
        "gpt-3.5": 4000,
        "gemini": 4000,
        "default": 4000
    },
    
    # Tokenizer mappings
    "TOKENIZER_MODELS": {
        "gpt-4": "gpt-4",
        "gpt-3.5": "gpt-3.5-turbo", 
        "default": "cl100k_base"
    },
    
    # Token estimation
    "WORD_TO_TOKEN_RATIO": 1.3,
    "MINIMUM_REMAINING_TOKENS": 100,  # Only truncate if we have meaningful space
    "TRUNCATION_SUFFIX": "...",
}

# Model Configuration - Removed unused reranking configs

# Embedding Configuration
EMBEDDING_CONFIG = {
    # Embedding models and costs
    "EMBEDDING_COSTS": {
        "openai": {
            "text-embedding-ada-002": 0.0001,
            "text-embedding-3-small": 0.00002,
            "text-embedding-3-large": 0.00013,
        },
        "google": {
            "models/embedding-001": 0.00001,
            "models/text-embedding-004": 0.00001,
        },
        "huggingface": {
            "default": 0.0  # Usually free for local/self-hosted models
        }
    },
    
    # Default costs
    "DEFAULT_EMBEDDING_COSTS": {
        "openai": 0.0001,
        "google": 0.00001,
        "gemini": 0.00001,
        "huggingface": 0.0
    },
}

# API and Server Configuration  
SERVER_CONFIG = {
    # CORS origins
    "CORS_ORIGINS": [
        "http://localhost:3000", 
        "https://jarvis.orangehealth.dev",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001"
    ],
    
    # SSE headers
    "SSE_HEADERS": {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "Cache-Control",
        "Content-Type": "text/event-stream"
    },
    
    # GitHub API
    "GITHUB_API": {
        "BASE_URL": "https://api.github.com",
        "PER_PAGE": 100,
        "TIMEOUT": 30,
        "MAX_RETRIES": 3,
    },
    
    # Rate limiting
    "DEFAULT_K": 30,          # Increased default number of documents to retrieve
    "DEFAULT_ALPHA": 0.3,     # Default alpha for hybrid search
    "MAX_PREVIEW_LENGTH": 1200,  # Longer preview for better context
    "RELEVANCE_SCORE_DECAY": 0.03,  # Lower decay for more relevant results
}

# Logging and Monitoring
LOGGING_CONFIG = {
    "FORMAT": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "DATE_FORMAT": "%Y-%m-%d %H:%M:%S",
    "MAX_LOG_SIZE": "10MB",
    "BACKUP_COUNT": 5,
    "COST_PRECISION": 6,  # Decimal places for cost logging
}

# Environment Variables
ENV_VARS = {
    "OPENAI_KEY": "OPENAI_API_KEY",
    "GEMINI_KEY": "GEMINI_API_KEY", 
    "GOOGLE_KEY": "GOOGLE_API_KEY",
    "GITHUB_TOKEN": "GITHUB_TOKEN",
    "APP_ENV": "APP_ENV",
    "CONFIG_PATH": "APP_CONFIG_PATH",
    "ACCESS_TOKEN_EXPIRE_MINUTES": "ACCESS_TOKEN_EXPIRE_MINUTES",
    "JWT_SECRET": "JWT_SECRET",
    "JWT_ALGORITHM": "HS256",
    "DEFAULT_TOKEN_EXPIRE": 1440,  # 24 hours in minutes
}

# File Processing
FILE_PROCESSING = {
    # Default indexing patterns  
    "DEFAULT_INCLUDE_GLOBS": ["**/*"],
    "DEFAULT_EXCLUDE_GLOBS": [
        "**/.git/**", "**/.github/**", "**/node_modules/**", "**/dist/**", 
        "**/build/**", "**/*.min.js", "**/*.png", "**/*.jpg", "**/*.jpeg",
        "**/*.pdf", "**/__pycache__/**", "**/venv/**", "**/env/**", 
        "**/.venv/**", "**/target/**", "**/bin/**", "**/.DS_Store", "**/Thumbs.db"
    ],
    
    # Text splitting
    "TEXT_SEPARATORS": [
        "\n\n", "\nclass ", "\ndef ", "\n@", "\nif ", "\nfor ", 
        "\nwhile ", "\ntry:", "\nwith ", "\n", " ", ""
    ],
    
    # File size limits
    "MAX_FILE_SIZE_MB": 2.0,
    "CHUNK_SIZE": 1500,
    "CHUNK_OVERLAP": 200,
    "BATCH_SIZE": 2000,
}

# Cache Configuration
CACHE_CONFIG = {
    "EMBEDDING_CACHE_TTL": 3600,      # 1 hour
    "QUERY_CACHE_TTL": 300,           # 5 minutes  
    "MAX_CACHE_SIZE": 1000,
    "CLEANUP_INTERVAL": 1800,         # 30 minutes
    "ENABLE_EMBEDDING_CACHE": True,
    "ENABLE_QUERY_CACHE": True,
}

# Response Quality Settings
RESPONSE_CONFIG = {
    "MAX_CONTEXT_SUMMARY_MODULES": 10,  # Limit context summary modules
    "FALLBACK_SNIPPET_COUNT": 3,        # Number of fallback code snippets
    "MAX_FALLBACK_SNIPPET_LENGTH": 300, # Max length of each snippet
    "PROGRAMMING_LANGUAGES": [
        "python", "javascript", "typescript", "java", "go", "rust", "cpp", 
        "c", "csharp", "php", "ruby", "swift", "kotlin", "scala", "sql", 
        "bash", "shell", "yaml", "json", "xml", "html", "css"
    ],
    
    # ASCII diagram settings
    "ENABLE_ASCII_DIAGRAMS": True,
    "MAX_DIAGRAM_WIDTH": 80,
    "DIAGRAM_STYLES": ["simple", "box", "flowchart"],
}

# Query Analysis Keywords
QUERY_ANALYSIS = {
    "HOW_TO_KEYWORDS": [
        "how to", "how do", "how can", "steps to", "tutorial", "guide", 
        "instructions", "implement", "create", "build", "setup", "configure"
    ],
    
    "WHAT_IS_KEYWORDS": [
        "what is", "what does", "explain", "definition", "meaning", "purpose",
        "overview", "introduction", "about"
    ],
    
    "DEBUGGING_KEYWORDS": [
        "debug", "error", "issue", "problem", "bug", "fix", "troubleshoot", 
        "not working", "broken", "fails", "exception", "crash"
    ],
    
    "ARCHITECTURAL_KEYWORDS": [
        "architecture", "structure", "design", "pattern", "flow", "system", 
        "overview", "diagram", "relationship", "component", "module"
    ],
    
    "CODE_KEYWORDS": [
        "function", "class", "method", "variable", "code", "implementation", 
        "syntax", "algorithm", "logic", "script"
    ],
    
    "TEST_KEYWORDS": [
        "test", "testing", "spec", "unit test", "integration", "coverage",
        "mock", "assert", "validate"
    ],
    
    "CONFIG_KEYWORDS": [
        "config", "configuration", "settings", "environment", "env", "setup",
        "yaml", "json", "ini", "properties"
    ]
}

# Technology Keywords for Context Boosting
TECH_KEYWORDS = [
    # Languages
    "python", "javascript", "typescript", "java", "go", "rust", "cpp", "c++",
    "csharp", "c#", "php", "ruby", "swift", "kotlin", "scala", "clojure",
    
    # Frameworks
    "react", "vue", "angular", "django", "flask", "fastapi", "express", 
    "spring", "laravel", "rails", "nextjs", "nuxt", "gatsby",
    
    # Databases
    "mysql", "postgresql", "mongodb", "redis", "elasticsearch", "sqlite",
    "cassandra", "dynamodb", "firebase",
    
    # Cloud & DevOps
    "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "ansible",
    "jenkins", "gitlab", "github", "circleci",
    
    # Tools & Libraries
    "numpy", "pandas", "scikit", "tensorflow", "pytorch", "langchain",
    "openai", "gemini", "chromadb", "pinecone", "weaviate"
]
