"""Default values and configuration constants"""

# Default configuration values
DEFAULT_VALUES = {
    # Indexing defaults
    "chunk_size": 1500,
    "chunk_overlap": 200,
    "batch_size": 2000,
    "max_file_mb": 2.0,
    "max_retries": 3,
    "timeout": 30,
    
    # Retrieval defaults
    "top_k": 20,
    "alpha_hybrid": 0.3,
    "similarity_threshold": 0.4,
    "max_context_docs": 8,
    "max_context_tokens": 4000,
    "aggressive_context_tokens": 1200,
    
    # Embedding defaults
    "min_dimension": 1024,
    "normalize": True,
    "max_batch_size": 32,
    
    # Server defaults
    "host": "0.0.0.0",
    "port": 8000,
    "log_level": "INFO",
    
    # Chat defaults
    "temperature": 0.1,
    "max_tokens": 1024,
    
    # Cost estimation
    "token_estimation_multiplier": 1.3,  # Rough word to token ratio
}

# CORS configuration
CORS_ORIGINS = [
    "http://localhost:3000", 
    "http://127.0.0.1:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3001"
]

# Server-Sent Events headers
SERVER_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Cache-Control",
    "Content-Type": "text/event-stream"
}

# Text splitter separators (in order of preference) - USED by indexer.py
TEXT_SPLITTER_SEPARATORS = [
    "\n\n",      # Paragraph breaks
    "\nclass ",  # Class definitions
    "\ndef ",    # Function definitions
    "\n@",       # Decorators
    "\nif ",     # Control structures
    "\nfor ",
    "\nwhile ",
    "\ntry:",
    "\nwith ",
    "\n",        # Line breaks
    " ",         # Word breaks
    ""
]

# Default environment variable names - USED by chat.py
DEFAULT_ENV_VARS = {
    "openai_key": "OPENAI_API_KEY",
    "gemini_key": "GEMINI_API_KEY",
    "google_key": "GOOGLE_API_KEY",
    "github_token": "GITHUB_TOKEN",
    "app_env": "APP_ENV",
    "config_path": "APP_CONFIG_PATH",
}

# Dead code removed: DEFAULT_INDEXING_PATTERNS, GITHUB_API_CONFIG, LARGE_SCALE_CONFIG, CACHE_CONFIG, LOGGING_CONFIG were unused
