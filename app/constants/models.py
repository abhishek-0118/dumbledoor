"""Model configurations, costs, and dimensions"""

# Common embedding dimensions (for reference only)
# Actual dimensions should be configured in config files
COMMON_EMBEDDING_DIMENSIONS = {
    "text-embedding-3-large": 3072,  # Full dimension (we cap at 1024)
    "text-embedding-3-small": 1536,
    "text-embedding-ada-002": 1536,
    "models/embedding-001": 768,
    "models/text-embedding-004": 768,
}

# Token limits for different models
MODEL_TOKEN_LIMITS = {
    "gpt-4": 8192,
    "gpt-4-turbo": 128000,
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-16k": 16384,
    "gemini-1.5-flash": 1048576,
    "gemini-1.5-pro": 2097152,
    "gemini-pro": 32768,
}

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

# Note: Fallback embedding models removed as we only support OpenAI and Google providers

# Model aliases for easier configuration
MODEL_ALIASES = {
    "gpt4": "gpt-4",
    "gpt4-turbo": "gpt-4-turbo",
    "gpt35": "gpt-3.5-turbo",
    "flash": "gemini-1.5-flash",
    "pro": "gemini-1.5-pro",
}

# Supported providers
SUPPORTED_PROVIDERS = ["openai", "gemini", "google"]

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
