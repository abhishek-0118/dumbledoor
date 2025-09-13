import os
from pathlib import Path
from typing import Optional
import yaml
from .models import AppConfig
from dotenv import load_dotenv

load_dotenv()

def _read_yaml(path: Optional[str]) -> dict:
	if not path:
		return {}
	p = Path(path)
	if not p.exists():
		raise FileNotFoundError(f"Configuration file not found: {path}")
	with p.open() as f:
		content = yaml.safe_load(f)
		if content is None:
			raise ValueError(f"Configuration file is empty: {path}")
		return content

def _get_config_path(env: str) -> str:
	if env.lower() == "local":
		return "config/local.yaml"
	elif env.lower() == "prod":
		return "config/prod.yaml"
	else:
		raise ValueError(f"Unknown environment: {env}. Use 'local' or 'prod'")

def load_config(env: str = None) -> AppConfig:
	app_env = env or os.getenv("APP_ENV", "local")
	base_path = os.getenv("APP_CONFIG_PATH", _get_config_path(app_env))
	
	try:
		base_config = _read_yaml(base_path)
	except FileNotFoundError:
		raise FileNotFoundError(f"Configuration file not found: {base_path}. Ensure the file exists.")
	except Exception as e:
		raise ValueError(f"Failed to load configuration from {base_path}: {e}")
	
	emb_path = os.getenv("EMBEDDING_CONFIG_PATH")
	emb_config = {}
	if emb_path:
		try:
			emb_config = _read_yaml(emb_path)
		except FileNotFoundError:
			print(f"Warning: Embedding config file not found: {emb_path}")
		except Exception as e:
			print(f"Warning: Failed to load embedding config from {emb_path}: {e}")
	
	merged_config = {**base_config}
	if emb_config:
		merged_config.setdefault("embedding", {})
		merged_config["embedding"].update(emb_config.get("embedding", emb_config))
	
	merged_config["app_env"] = app_env
	
	try:
		cfg = AppConfig(**merged_config)
	except Exception as e:
		raise ValueError(f"Invalid configuration: {e}")
	
	_apply_env_overrides(cfg)
	
	return cfg

def _apply_env_overrides(cfg: AppConfig) -> None:
	if os.getenv("CHROMA_DIR"):
		cfg.backend.chroma_dir = os.getenv("CHROMA_DIR")
	if os.getenv("COLLECTION_NAME"):
		cfg.backend.collection_name = os.getenv("COLLECTION_NAME")
	
	if os.getenv("LOCAL_REPO_ROOT"):
		cfg.indexing.local_repo_root = os.getenv("LOCAL_REPO_ROOT")
	if os.getenv("CHUNK_SIZE"):
		cfg.indexing.chunk_size = int(os.getenv("CHUNK_SIZE"))
	if os.getenv("CHUNK_OVERLAP"):
		cfg.indexing.chunk_overlap = int(os.getenv("CHUNK_OVERLAP"))
	if os.getenv("BATCH_SIZE"):
		cfg.indexing.batch_size = int(os.getenv("BATCH_SIZE"))
	if os.getenv("MAX_FILE_MB"):
		cfg.indexing.max_file_mb = float(os.getenv("MAX_FILE_MB"))
	if os.getenv("GITHUB_TOKEN_ENV"):
		cfg.indexing.github_token_env = os.getenv("GITHUB_TOKEN_ENV")
	
	if os.getenv("EMBEDDING_MODEL"):
		cfg.embedding.model_name = os.getenv("EMBEDDING_MODEL")
	if os.getenv("EMBEDDING_NORMALIZE"):
		cfg.embedding.normalize = os.getenv("EMBEDDING_NORMALIZE").lower() == "true"
	if os.getenv("EMBEDDING_MIN_DIM"):
		cfg.embedding.min_dimension = int(os.getenv("EMBEDDING_MIN_DIM"))
	if os.getenv("EMBEDDING_API_PROVIDER"):
		cfg.embedding.api_provider = os.getenv("EMBEDDING_API_PROVIDER")
	if os.getenv("EMBEDDING_API_KEY_ENV"):
		cfg.embedding.api_key_env = os.getenv("EMBEDDING_API_KEY_ENV")
	if os.getenv("EMBEDDING_MODEL_DIM"):
		cfg.embedding.model_dimension = int(os.getenv("EMBEDDING_MODEL_DIM"))
	
	if os.getenv("TOP_K"):
		cfg.retrieval.top_k = int(os.getenv("TOP_K"))
	if os.getenv("ALPHA_HYBRID"):
		cfg.retrieval.alpha_hybrid = float(os.getenv("ALPHA_HYBRID"))
	if os.getenv("USE_RERANKER"):
		cfg.retrieval.use_reranker = os.getenv("USE_RERANKER").lower() == "true"
	if os.getenv("CROSS_ENCODER_MODEL"):
		cfg.retrieval.cross_encoder_model = os.getenv("CROSS_ENCODER_MODEL")
	
	if os.getenv("BIND_HOST"):
		cfg.server.host = os.getenv("BIND_HOST")
	if os.getenv("BIND_PORT"):
		cfg.server.port = int(os.getenv("BIND_PORT"))
	
	# Chat configuration overrides
	if os.getenv("CHAT_PROVIDER"):
		cfg.chat.provider = os.getenv("CHAT_PROVIDER")
	if os.getenv("CHAT_MODEL"):
		cfg.chat.model = os.getenv("CHAT_MODEL")
	if os.getenv("CHAT_TEMPERATURE"):
		cfg.chat.temperature = float(os.getenv("CHAT_TEMPERATURE"))
	if os.getenv("CHAT_API_KEY_ENV"):
		cfg.chat.api_key_env = os.getenv("CHAT_API_KEY_ENV")
	if os.getenv("CHAT_MAX_TOKENS"):
		cfg.chat.max_tokens = int(os.getenv("CHAT_MAX_TOKENS"))
	if os.getenv("CURRENT_METHOD"):
		cfg.current_method = os.getenv("CURRENT_METHOD")
	
	# Legacy support for existing environment variables
	if os.getenv("GEMINI_CHAT_MODEL"):
		cfg.chat.model = os.getenv("GEMINI_CHAT_MODEL")
		cfg.chat.provider = "gemini"
		cfg.current_method = "gemini"
