import logging
import os
from typing import Tuple, Any
try:
	from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
	from langchain_community.embeddings import HuggingFaceEmbeddings
from ..config.models import EmbeddingConfig
from .chat import estimate_embeddings_cost

logger = logging.getLogger("app.core.embeddings")


def create_embedding_fn(cfg: EmbeddingConfig) -> Tuple[Any, int]:
	try:
		encode_kwargs = cfg.encode_kwargs or {}
		if cfg.normalize:
			encode_kwargs["normalize_embeddings"] = cfg.normalize
		
		# Log cost estimation for embedding creation
		test_text = "This is a test string for embedding cost estimation."
		cost_estimate = estimate_embeddings_cost(test_text, cfg.api_provider or "huggingface", cfg.model_name)
		logger.info(f"Embedding cost estimate per test query: {cost_estimate}")
		
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
	
	# Add optimizations for faster embeddings
	optimized_kwargs = {
		"normalize_embeddings": cfg.normalize,
		"show_progress": False,  # Disable progress bars for speed
		**encode_kwargs
	}
	
	# Add model kwargs for better performance
	model_kwargs = {
		"device": "cpu",  # Force CPU usage as requested (no GPU logic)
		"torch_dtype": "float32",  # Use float32 for balance of speed and accuracy
	}
	
	emb = HuggingFaceEmbeddings(
		model_name=cfg.model_name,
		encode_kwargs=optimized_kwargs,
		model_kwargs=model_kwargs,
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
	
	# Enforce 1024 dimension limit for optimization
	if dim > 1024:
		logger.warning(f"Dimension {dim} exceeds 1024 limit. Capping at 1024 for performance.")
		dim = 1024
	
	if dim < cfg.min_dimension:
		logger.warning(f"Embedding dim {dim} < min {cfg.min_dimension}. Consider using a higher-dim model.")
	else:
		logger.info(f"Embedding dimension OK: {dim} (max 1024 for optimization)")
	
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
		
		# Enforce 1024 dimension limit for optimization
		if dim > 1024:
			logger.warning(f"OpenAI dimension {dim} exceeds 1024 limit. Capping at 1024 for performance.")
			dim = 1024
		
		logger.info(f"Using OpenAI embedding dimension: {dim} (max 1024 for optimization)")
		
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
		
		# Enforce 1024 dimension limit for optimization
		if dim > 1024:
			logger.warning(f"Google dimension {dim} exceeds 1024 limit. Capping at 1024 for performance.")
			dim = 1024
		
		logger.info(f"Using Google embedding dimension: {dim} (max 1024 for optimization)")
		
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


class OptimizedEmbeddingWrapper:
	"""Wrapper for embeddings with optimizations like batching and caching"""
	
	def __init__(self, embedding_fn, max_batch_size: int = 32):
		self.embedding_fn = embedding_fn
		self.max_batch_size = max_batch_size
		self._cache = {}
	
	def embed_query(self, text: str) -> list:
		"""Embed a single query with caching"""
		if text in self._cache:
			return self._cache[text]
		
		result = self.embedding_fn.embed_query(text)
		self._cache[text] = result
		return result
	
	def embed_documents(self, texts: list) -> list:
		"""Embed documents with batching and caching"""
		results = []
		uncached_texts = []
		uncached_indices = []
		
		# Check cache first
		for i, text in enumerate(texts):
			if text in self._cache:
				results.append((i, self._cache[text]))
			else:
				uncached_texts.append(text)
				uncached_indices.append(i)
		
		# Process uncached texts in batches
		if uncached_texts:
			for i in range(0, len(uncached_texts), self.max_batch_size):
				batch = uncached_texts[i:i + self.max_batch_size]
				batch_embeddings = self.embedding_fn.embed_documents(batch)
				
				# Cache results
				for j, embedding in enumerate(batch_embeddings):
					text_idx = i + j
					if text_idx < len(uncached_texts):
						text = uncached_texts[text_idx]
						self._cache[text] = embedding
						original_idx = uncached_indices[text_idx]
						results.append((original_idx, embedding))
		
		# Sort by original order
		results.sort(key=lambda x: x[0])
		return [embedding for _, embedding in results]
	
	def clear_cache(self):
		"""Clear the embedding cache"""
		self._cache.clear()


def create_optimized_embedding_fn(cfg: EmbeddingConfig, enable_optimizations: bool = True) -> Tuple[Any, int]:
	"""Create an optimized embedding function with batching and caching"""
	base_emb, dim = create_embedding_fn(cfg)
	
	if enable_optimizations:
		# Wrap with optimizations
		optimized_emb = OptimizedEmbeddingWrapper(base_emb, max_batch_size=32)
		logger.info("Created optimized embedding function with caching and batching")
		return optimized_emb, dim
	else:
		return base_emb, dim
