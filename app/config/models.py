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

class MongoDBConfig(BaseModel):
	connection_string: str = "mongodb://localhost:27017"
	database_name: str = "starkfoundation"
	max_pool_size: int = 10
	min_pool_size: int = 1
	max_idle_time_ms: int = 30000
	connect_timeout_ms: int = 10000
	server_selection_timeout_ms: int = 5000

class AuthConfig(BaseModel):
	google_client_id: Optional[str] = None
	google_client_secret: Optional[str] = None
	google_redirect_uri: str = "http://localhost:8000/auth/google/callback"
	jwt_secret_key: Optional[str] = None
	jwt_algorithm: str = "HS256"
	jwt_expire_hours: int = 24
	session_expire_hours: int = 168  # 7 days
	require_email_verification: bool = False

class ConversationConfig(BaseModel):
	max_token_limit: int = 4000
	max_messages_per_session: int = 20
	enable_summarization: bool = True
	summary_trigger_ratio: float = 0.8  # Trigger summary when 80% of token limit is reached
	keep_recent_messages: int = 8
	enable_concept_tracking: bool = True
	max_tracked_concepts: int = 20
	max_tracked_files: int = 15

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
	mongodb: MongoDBConfig
	auth: AuthConfig
	conversation: ConversationConfig
	current_method: str  # "gemini", "openai"
	app_env: str
	logging: Optional[LoggingConfig] = None
