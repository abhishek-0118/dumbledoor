import logging
import os
from typing import Optional, List, Dict, Any
from langchain.schema import Document
from ..config.models import AppConfig
from .fast_retrieval import create_fast_retriever
from .openai_reranker import openai_rerank_documents, embedding_similarity_rerank, lightweight_rerank

logger = logging.getLogger("app.search.retrieval")


def build_retriever(store, cfg: AppConfig, repo_prefix: Optional[str], k: int, alpha: float):
	# Increase search_k to get more candidates for reranking
	search_k = max(k * 3, 20)  # Get 3x more candidates or at least 20
	search_kwargs = {"k": search_k}
	
	if repo_prefix:
		search_kwargs["filter"] = {
			"$or": [
				{"repo": {"$like": f"{repo_prefix}%"}},
				{"repo_name": {"$like": f"{repo_prefix}%"}},
				{"repo_folder": {"$like": f"{repo_prefix}%"}},
			]
		}
	
	vector_ret = store.as_retriever(search_kwargs=search_kwargs)
	return vector_ret


def enhanced_search(store, query: str, cfg: AppConfig, repo_prefix: Optional[str] = None, 
                   k: int = 12, include_related: bool = True) -> List[Document]:
	"""Enhanced search with fast retrieval optimizations"""
	
	# Use fast retriever if caching is enabled
	if getattr(cfg.retrieval, 'enable_caching', False):
		fast_retriever = create_fast_retriever(store, cfg)
		primary_docs = fast_retriever.retrieve(
			query, repo_prefix, k, cfg.retrieval.alpha_hybrid
		)
	else:
		retriever = build_retriever(store, cfg, repo_prefix, k, cfg.retrieval.alpha_hybrid)
		primary_docs = retriever.get_relevant_documents(query)
	
	# Apply reranking only if not using fast retrieval (which has lightweight ranking)
	if not getattr(cfg.retrieval, 'enable_caching', False):
		reranked_docs = smart_rerank(primary_docs, query, k, cfg)
	else:
		reranked_docs = primary_docs[:k]
	
	if not include_related:
		return reranked_docs
	
	# Find related documents from same files/modules
	related_docs = find_related_context(store, reranked_docs, cfg, k // 3)
	
	# Combine and deduplicate
	all_docs = reranked_docs + related_docs
	seen_content = set()
	unique_docs = []
	
	for doc in all_docs:
		content_hash = hash(doc.page_content[:200])  # Use first 200 chars for deduplication
		if content_hash not in seen_content:
			seen_content.add(content_hash)
			unique_docs.append(doc)
	
	return unique_docs[:k * 2]  # Return up to 2x the requested amount for better context


def find_related_context(store, primary_docs: List[Document], cfg: AppConfig, max_related: int) -> List[Document]:
	"""Find related context from the same files or modules as primary results"""
	related_docs = []
	
	for doc in primary_docs[:5]:  # Only check top 5 primary docs
		metadata = doc.metadata or {}
		repo = metadata.get("repo", "")
		path = metadata.get("path", "")
		module_name = metadata.get("module_name", "")
		
		if not path:
			continue
		
		# Search for documents from the same file
		try:
			file_filter = {
				"$and": [
					{"repo": {"$eq": repo}},
					{"path": {"$eq": path}}
				]
			}
			
			file_docs = store.similarity_search(
				doc.page_content,
				k=3,
				filter=file_filter
			)
			
			# Add file context docs that aren't already in primary results
			for file_doc in file_docs:
				if file_doc.page_content != doc.page_content:
					related_docs.append(file_doc)
			
		except Exception as e:
			logger.debug(f"Failed to find related context for {path}: {e}")
	
	return related_docs[:max_related]


def smart_rerank(docs: List[Document], query: str, top_k: int, cfg: AppConfig) -> List[Document]:
	"""Smart reranking using the best available method"""
	if not cfg.retrieval.use_reranker or not docs:
		return docs[:top_k]
	
	# Choose reranking strategy based on configuration and performance needs
	rerank_strategy = getattr(cfg.retrieval, 'rerank_strategy', 'lightweight')
	
	if rerank_strategy == 'openai_llm':
		# Use OpenAI LLM for high-quality reranking (slower, more expensive)
		return openai_rerank_documents(docs, query, top_k, cfg)
	
	elif rerank_strategy == 'openai_embedding':
		# Use OpenAI embeddings for similarity-based reranking (fast, accurate)
		return embedding_similarity_rerank(docs, query, top_k, cfg)
	
	else:  # 'lightweight' or fallback
		# Use fast heuristic-based reranking (fastest, good enough)
		return lightweight_rerank(docs, query, top_k)


# Legacy function for backward compatibility (now redirects to smart_rerank)
def apply_cross_encoder_rerank(docs: List[Document], query: str, top_k: int, cfg: AppConfig) -> List[Document]:
	"""Legacy cross-encoder function - now uses smart reranking"""
	logger.info("Using smart reranking instead of cross-encoder (dependency removed)")
	return smart_rerank(docs, query, top_k, cfg)


def create_context_summary(docs: List[Document]) -> Dict[str, Any]:
	"""Create a summary of the retrieved context for better LLM understanding"""
	if not docs:
		return {}
	
	# Analyze the retrieved documents
	repos = set()
	languages = set()
	file_types = set()
	modules = set()
	test_files = 0
	config_files = 0
	
	for doc in docs:
		metadata = doc.metadata or {}
		if metadata.get("repo"):
			repos.add(metadata["repo"])
		if metadata.get("language"):
			languages.add(metadata["language"])
		if metadata.get("file_type"):
			file_types.add(metadata["file_type"])
		if metadata.get("module_name"):
			modules.add(metadata["module_name"])
		if metadata.get("is_test"):
			test_files += 1
		if metadata.get("is_config"):
			config_files += 1
	
	return {
		"total_documents": len(docs),
		"repositories": list(repos),
		"languages": list(languages),
		"file_types": list(file_types),
		"modules": list(modules)[:10],  # Limit to 10 modules
		"test_files": test_files,
		"config_files": config_files,
		"has_diverse_context": len(file_types) > 1,
	}
