import logging
import os
from typing import Tuple, Any
try:
	from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
	from langchain_community.embeddings import HuggingFaceEmbeddings
from ..config.models import EmbeddingConfig

logger = logging.getLogger("app.core.embeddings")


def create_embedding_fn(cfg: EmbeddingConfig) -> Tuple[Any, int]:
	try:
		encode_kwargs = cfg.encode_kwargs or {}
		if cfg.normalize:
			encode_kwargs["normalize_embeddings"] = cfg.normalize
		
		if cfg.api_provider == "openai":
			return _create_openai_embeddings(cfg, encode_kwargs)
		elif cfg.api_provider == "google":
			return _create_google_embeddings(cfg, encode_kwargs)
		else:
			return _create_huggingface_embeddings(cfg, encode_kwargs)
	
	except Exception as e:
		logger.error(f"Failed to create embedding function: {e}")
		return _create_fallback_embeddings()


def _create_huggingface_embeddings(cfg: EmbeddingConfig, encode_kwargs: dict) -> Tuple[HuggingFaceEmbeddings, int]:
	logger.info(f"Creating HuggingFace embeddings with model: {cfg.model_name}")
	
	# Check for GPU availability and configure device
	import torch
	device = "cuda" if torch.cuda.is_available() else "cpu"
	if "device" in encode_kwargs:
		device = encode_kwargs["device"]
	
	logger.info(f"Using device: {device}")
	
	# Enhanced encode_kwargs for better performance
	enhanced_kwargs = {
		"normalize_embeddings": encode_kwargs.get("normalize_embeddings", True),
		"batch_size": encode_kwargs.get("batch_size", 32),
		"device": device,
		"show_progress_bar": True,
	}
	
	emb = HuggingFaceEmbeddings(
		model_name=cfg.model_name,
		model_kwargs={"device": device},
		encode_kwargs=enhanced_kwargs,
	)
	
	if cfg.model_dimension:
		dim = cfg.model_dimension
		logger.info(f"Using configured dimension: {dim}")
	else:
		try:
			dim = len(emb.embed_query("test"))
			logger.info(f"Detected embedding dimension: {dim}")
		except Exception as e:
			logger.warning(f"Could not detect dimension: {e}, using min_dimension")
			dim = cfg.min_dimension
	
	if dim < cfg.min_dimension:
		logger.warning(f"Embedding dim {dim} < min {cfg.min_dimension}. Consider using a higher-dim model.")
	else:
		logger.info(f"Embedding dimension OK: {dim} >= {cfg.min_dimension}")
	
	return emb, dim


def _create_openai_embeddings(cfg: EmbeddingConfig, encode_kwargs: dict) -> Tuple[Any, int]:
	try:
		from langchain_openai import OpenAIEmbeddings
		
		api_key = None
		if cfg.api_key_env:
			api_key = os.getenv(cfg.api_key_env)
		
		if not api_key:
			raise ValueError(f"OpenAI API key not found in environment variable: {cfg.api_key_env}")
		
		logger.info(f"Creating OpenAI embeddings with model: {cfg.model_name}")
		
		emb = OpenAIEmbeddings(
			model=cfg.model_name,
			openai_api_key=api_key,
		)
		
		model_dims = {
			"text-embedding-ada-002": 1536,
			"text-embedding-3-small": 1536,
			"text-embedding-3-large": 3072,
		}
		
		dim = cfg.model_dimension or model_dims.get(cfg.model_name, cfg.min_dimension)
		logger.info(f"Using OpenAI embedding dimension: {dim}")
		
		return emb, dim
		
	except ImportError:
		logger.error("langchain_openai not installed. Install with: pip install langchain-openai")
		raise
	except Exception as e:
		logger.error(f"Failed to create OpenAI embeddings: {e}")
		raise


def _create_google_embeddings(cfg: EmbeddingConfig, encode_kwargs: dict) -> Tuple[Any, int]:
	try:
		from langchain_google_genai import GoogleGenerativeAIEmbeddings
		
		api_key = None
		if cfg.api_key_env:
			api_key = os.getenv(cfg.api_key_env)
		
		if not api_key:
			raise ValueError(f"Google API key not found in environment variable: {cfg.api_key_env}")
		
		logger.info(f"Creating Google embeddings with model: {cfg.model_name}")
		
		emb = GoogleGenerativeAIEmbeddings(
			model=cfg.model_name,
			google_api_key=api_key,
		)
		
		model_dims = {
			"models/embedding-001": 768,
			"models/text-embedding-004": 768,
		}
		
		dim = cfg.model_dimension or model_dims.get(cfg.model_name, cfg.min_dimension)
		logger.info(f"Using Google embedding dimension: {dim}")
		
		return emb, dim
		
	except ImportError:
		logger.error("langchain_google_genai not installed. Install with: pip install langchain-google-genai")
		raise
	except Exception as e:
		logger.error(f"Failed to create Google embeddings: {e}")
		raise


def _create_fallback_embeddings() -> Tuple[HuggingFaceEmbeddings, int]:
	logger.warning("Creating fallback HuggingFace embeddings")
	
	fallback_models = [
		"sentence-transformers/all-MiniLM-L6-v2",
		"sentence-transformers/all-mpnet-base-v2",
		"sentence-transformers/paraphrase-MiniLM-L6-v2"
	]
	
	for model_name in fallback_models:
		try:
			emb = HuggingFaceEmbeddings(
				model_name=model_name,
				encode_kwargs={"normalize_embeddings": True},
			)
			dim = len(emb.embed_query("test"))
			logger.info(f"Using fallback model {model_name} with dimension: {dim}")
			return emb, dim
		except Exception as e:
			logger.warning(f"Fallback model {model_name} failed: {e}")
			continue
	
	raise RuntimeError("All embedding models failed to load")
