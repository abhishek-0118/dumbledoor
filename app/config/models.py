from pydantic import BaseModel, Field
from typing import Optional, List, Union
import os


class EmbeddingConfig(BaseModel):
	model_name: str
	normalize: bool
	min_dimension: int
	api_provider: Optional[str] = None
	api_key_env: Optional[str] = None
	model_dimension: Optional[int] = None
	encode_kwargs: Optional[dict] = None

class BackendConfig(BaseModel):
	chroma_dir: str
	collection_name: str

class IndexingConfig(BaseModel):
	local_repo_root: str
	local_paths: List[str] = Field(default_factory=list)
	repo_urls: List[str] = Field(default_factory=list)
	include_globs: List[str]
	exclude_globs: List[str]
	max_file_mb: float
	chunk_size: int
	chunk_overlap: int
	batch_size: int
	github_token_env: Optional[str] = None
	github_auth_required: bool = False
	enable_incremental: bool = False
	parallelism: int = 1
	cache_embeddings: bool = False
	compression_enabled: bool = False
	max_retries: int = 3
	timeout: int = 30
	force_reindex: bool = True

class RetrievalConfig(BaseModel):
	top_k: int
	alpha_hybrid: float
	use_reranker: bool
	cross_encoder_model: str
	similarity_threshold: float = 0.5
	max_context_docs: int = 30
	max_context_tokens: int = 4000
	include_file_context: bool = True
	boost_same_language: bool = True
	enable_caching: bool = False
	cache_ttl: int = 300

class ServerConfig(BaseModel):
	host: str
	port: int

class ChatConfig(BaseModel):
	provider: str  # "gemini", "openai", etc.
	model: str
	temperature: float = 0.1
	api_key_env: Optional[str] = None
	max_tokens: Optional[int] = None
	additional_kwargs: Optional[dict] = None

class LoggingConfig(BaseModel):
	level: str = "INFO"
	file: str = "./logs/app.log"
	enable_cost_tracking: bool = False
	cost_log_file: str = "./logs/cost_history.log"
	log_token_usage: bool = False
	max_log_size: str = "10MB"
	backup_count: int = 5

class AppConfig(BaseModel):
	embedding: EmbeddingConfig
	backend: BackendConfig
	indexing: IndexingConfig
	retrieval: RetrievalConfig
	server: ServerConfig
	chat: ChatConfig
	current_method: str  # "gemini", "openai"
	app_env: str
	logging: Optional[LoggingConfig] = None
