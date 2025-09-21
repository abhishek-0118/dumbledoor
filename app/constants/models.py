"""Model configurations, costs, and dimensions"""

# Dead code removed: COMMON_EMBEDDING_DIMENSIONS and MODEL_TOKEN_LIMITS were unused

# Cost per 1K tokens for chat models (USD)
MODEL_COSTS = {
    "openai": {
        "gpt-4": 0.03,
        "gpt-4-turbo": 0.01,
        "gpt-3.5-turbo": 0.0015,
        "gpt-3.5-turbo-16k": 0.003,
    },
    "gemini": {
        "gemini-1.5-flash": 0.0002,
        "gemini-1.5-pro": 0.0035,
        "gemini-pro": 0.0005,
    }
}

# Default costs for unknown models
DEFAULT_MODEL_COSTS = {
    "openai": 0.002,
    "gemini": 0.0005,
    "default": 0.001
}

# Embedding costs per 1K tokens (USD)
EMBEDDING_COSTS = {
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
}

# Default embedding costs
DEFAULT_EMBEDDING_COSTS = {
    "openai": 0.0001,
    "google": 0.00001,
    "gemini": 0.00001,
    "huggingface": 0.0
}

# Dead code removed: MODEL_ALIASES and SUPPORTED_PROVIDERS were unused

# Maximum dimension limit for optimization
MAX_DIMENSION_LIMIT = 1024

# Tokenizer models for different providers
TOKENIZER_MODELS = {
    "openai": {
        "gpt-4": "gpt-4",
        "gpt-3.5": "gpt-3.5-turbo",
        "default": "cl100k_base"
    },
    "gemini": {
        "default": "cl100k_base"
    },
    "default": "cl100k_base"
}
