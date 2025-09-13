import logging
import os
from typing import Any, Optional
import tiktoken
from ..config.models import ChatConfig

logger = logging.getLogger("app.core.chat")


def create_chat_llm(chat_config: ChatConfig, current_method: str) -> Any:
    """Create a chat LLM based on the current method and configuration"""
    
    # Validate current_method matches chat provider
    if current_method != chat_config.provider:
        logger.warning(f"current_method '{current_method}' doesn't match chat provider '{chat_config.provider}'. Using provider setting.")
        current_method = chat_config.provider
    
    if current_method.lower() == "gemini":
        return _create_gemini_chat(chat_config)
    elif current_method.lower() == "openai":
        return _create_openai_chat(chat_config)
    else:
        raise ValueError(f"Unsupported chat method: {current_method}. Supported methods: gemini, openai")


def _create_gemini_chat(chat_config: ChatConfig) -> Any:
    """Create Gemini chat LLM"""
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        # Get API key
        api_key = None
        if chat_config.api_key_env:
            api_key = os.getenv(chat_config.api_key_env)
        
        # Fallback to common environment variables
        if not api_key:
            api_key = (
                os.getenv("GEMINI_API_KEY")
                or os.getenv("GOOGLE_API_KEY")
                or os.getenv("GOOGLE_GENAI_API_KEY")
            )
        
        if not api_key:
            raise ValueError(f"Gemini API key not found. Set {chat_config.api_key_env} or GEMINI_API_KEY environment variable.")
        
        logger.info(f"Creating Gemini chat LLM with model: {chat_config.model}")
        
        # Build kwargs
        kwargs = {
            "model": chat_config.model,
            "google_api_key": api_key,
            "temperature": chat_config.temperature,
        }
        
        if chat_config.max_tokens:
            kwargs["max_output_tokens"] = chat_config.max_tokens
        
        if chat_config.additional_kwargs:
            kwargs.update(chat_config.additional_kwargs)
        
        return ChatGoogleGenerativeAI(**kwargs)
        
    except ImportError:
        logger.error("langchain_google_genai not installed. Install with: pip install langchain-google-genai")
        raise
    except Exception as e:
        logger.error(f"Failed to create Gemini chat LLM: {e}")
        raise


def _create_openai_chat(chat_config: ChatConfig) -> Any:
    """Create OpenAI chat LLM"""
    try:
        from langchain_openai import ChatOpenAI
        
        # Get API key
        api_key = None
        if chat_config.api_key_env:
            api_key = os.getenv(chat_config.api_key_env)
        
        # Fallback to common environment variables
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            raise ValueError(f"OpenAI API key not found. Set {chat_config.api_key_env} or OPENAI_API_KEY environment variable.")
        
        logger.info(f"Creating OpenAI chat LLM with model: {chat_config.model}")
        
        # Build kwargs
        kwargs = {
            "model": chat_config.model,
            "openai_api_key": api_key,
            "temperature": chat_config.temperature,
        }
        
        if chat_config.max_tokens:
            kwargs["max_tokens"] = chat_config.max_tokens
        
        if chat_config.additional_kwargs:
            kwargs.update(chat_config.additional_kwargs)
        
        return ChatOpenAI(**kwargs)
        
    except ImportError:
        logger.error("langchain_openai not installed. Install with: pip install langchain-openai")
        raise
    except Exception as e:
        logger.error(f"Failed to create OpenAI chat LLM: {e}")
        raise


def estimate_token_cost(text: str, model: str, provider: str) -> dict:
    """Estimate token usage and cost for the given text and model"""
    
    try:
        # Get appropriate tokenizer
        if provider.lower() == "openai":
            if "gpt-4" in model.lower():
                encoding = tiktoken.encoding_for_model("gpt-4")
            elif "gpt-3.5" in model.lower():
                encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
            else:
                # Default to cl100k_base for most OpenAI models
                encoding = tiktoken.get_encoding("cl100k_base")
        elif provider.lower() == "gemini":
            # Gemini uses similar tokenization to OpenAI for estimation
            encoding = tiktoken.get_encoding("cl100k_base")
        else:
            # Fallback encoding
            encoding = tiktoken.get_encoding("cl100k_base")
        
        tokens = encoding.encode(text)
        token_count = len(tokens)
        
        # Estimate cost based on provider and model
        cost_per_1k_tokens = _get_cost_per_1k_tokens(model, provider)
        estimated_cost = (token_count / 1000) * cost_per_1k_tokens
        
        return {
            "token_count": token_count,
            "estimated_cost_usd": round(estimated_cost, 6),
            "cost_per_1k_tokens": cost_per_1k_tokens,
            "model": model,
            "provider": provider
        }
        
    except Exception as e:
        logger.warning(f"Failed to estimate token cost: {e}")
        return {
            "token_count": len(text.split()) * 1.3,  # Rough estimation
            "estimated_cost_usd": 0.0,
            "cost_per_1k_tokens": 0.0,
            "model": model,
            "provider": provider,
            "error": str(e)
        }


def _get_cost_per_1k_tokens(model: str, provider: str) -> float:
    """Get cost per 1K tokens for different models and providers"""
    
    # OpenAI pricing (as of 2024)
    openai_costs = {
        "gpt-4": 0.03,  # Input tokens
        "gpt-4-turbo": 0.01,
        "gpt-3.5-turbo": 0.0015,
        "gpt-3.5-turbo-16k": 0.003,
    }
    
    # Gemini pricing (estimated, as pricing varies)
    gemini_costs = {
        "gemini-1.5-flash": 0.0002,
        "gemini-1.5-pro": 0.0035,
        "gemini-pro": 0.0005,
    }
    
    if provider.lower() == "openai":
        for model_key, cost in openai_costs.items():
            if model_key in model.lower():
                return cost
        return 0.002  # Default OpenAI cost
    
    elif provider.lower() == "gemini":
        for model_key, cost in gemini_costs.items():
            if model_key in model.lower():
                return cost
        return 0.0005  # Default Gemini cost
    
    return 0.001  # Default cost


def estimate_embeddings_cost(text: str, provider: str, model: str) -> dict:
    """Estimate cost for embeddings"""
    
    try:
        # Use tiktoken for token estimation
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)
        token_count = len(tokens)
        
        # Embedding costs (per 1K tokens)
        embedding_costs = {
            "openai": {
                "text-embedding-ada-002": 0.0001,
                "text-embedding-3-small": 0.00002,
                "text-embedding-3-large": 0.00013,
            },
            "google": {
                "models/embedding-001": 0.00001,  # Estimated
                "models/text-embedding-004": 0.00001,
            },
            "huggingface": {
                "default": 0.0  # Usually free for local/self-hosted models
            }
        }
        
        cost_per_1k = 0.0
        if provider.lower() == "openai":
            costs = embedding_costs.get("openai", {})
            for model_key, cost in costs.items():
                if model_key in model.lower():
                    cost_per_1k = cost
                    break
            if cost_per_1k == 0.0:
                cost_per_1k = 0.0001  # Default OpenAI embedding cost
        
        elif provider.lower() in ["google", "gemini"]:
            costs = embedding_costs.get("google", {})
            for model_key, cost in costs.items():
                if model_key in model.lower():
                    cost_per_1k = cost
                    break
            if cost_per_1k == 0.0:
                cost_per_1k = 0.00001  # Default Google embedding cost
        
        elif provider.lower() == "huggingface":
            cost_per_1k = 0.0  # Free for most HuggingFace models
        
        estimated_cost = (token_count / 1000) * cost_per_1k
        
        return {
            "token_count": token_count,
            "estimated_cost_usd": round(estimated_cost, 8),
            "cost_per_1k_tokens": cost_per_1k,
            "model": model,
            "provider": provider
        }
        
    except Exception as e:
        logger.warning(f"Failed to estimate embedding cost: {e}")
        return {
            "token_count": len(text.split()) * 1.3,  # Rough estimation
            "estimated_cost_usd": 0.0,
            "cost_per_1k_tokens": 0.0,
            "model": model,
            "provider": provider,
            "error": str(e)
        }
