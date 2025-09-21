import logging
from typing import Optional, List, Dict, Any
from langchain.schema import Document
from ..config.models import AppConfig
from .openai_reranker import embedding_similarity_rerank
from ..constants.system import SEARCH_CONFIG

logger = logging.getLogger("app.search.retrieval")


def build_retriever(store, repo_prefix: Optional[str] = None, k: int = 12):
	"""Build a simple retriever with optional repository filtering"""
	search_kwargs = {"k": k}
	
	if repo_prefix:
		search_kwargs["filter"] = {
			"$or": [
				{"repo": {"$eq": repo_prefix}},
				{"repo_name": {"$eq": repo_prefix}},
				{"repo_folder": {"$eq": repo_prefix}},
			]
		}
	
	return store.as_retriever(search_kwargs=search_kwargs)


def rag_search(store, query: str, cfg: AppConfig, repo_prefix: Optional[str] = None, k: int = 25) -> List[Document]:
	"""Comprehensive RAG search: Query -> Embeddings -> Broad Search -> Smart Rerank -> Return"""
	try:
		# Step 1: Cast a wide net - get many more candidates for comprehensive coverage
		search_multiplier = 4  # Get 4x more candidates than final k
		initial_candidates = k * search_multiplier
		
		retriever = build_retriever(store, repo_prefix, initial_candidates)
		docs = retriever.invoke(query)
		
		if not docs:
			logger.warning(f"No documents found for query: {query[:50]}...")
			return []
		
		logger.info(f"Retrieved {len(docs)} initial candidates for query: {query[:100]}")
		
		# Step 2: Apply multiple search strategies for comprehensive results
		
		# Strategy 1: Direct similarity search
		primary_docs = docs[:initial_candidates // 2]
		
		# Strategy 2: Keyword-enhanced search (for queries like "trigger report generation")
		query_terms = query.lower().split()
		keyword_docs = []
		
		# Find documents that contain query keywords for better recall
		for doc in docs:
			content_lower = doc.page_content.lower()
			keyword_matches = sum(1 for term in query_terms if term in content_lower)
			if keyword_matches >= len(query_terms) // 2:  # At least half the keywords match
				keyword_docs.append(doc)
		
		# Combine strategies
		combined_docs = primary_docs + keyword_docs
		
		# Remove duplicates while preserving order
		seen_content = set()
		unique_docs = []
		for doc in combined_docs:
			content_hash = hash(doc.page_content[:200])  # Hash first 200 chars for dedup
			if content_hash not in seen_content:
				seen_content.add(content_hash)
				unique_docs.append(doc)
		
		logger.info(f"After deduplication: {len(unique_docs)} unique candidates")
		
		# Step 3: Smart reranking with more lenient approach
		if cfg.retrieval.use_reranker and len(unique_docs) > k:
			reranked_docs = embedding_similarity_rerank(unique_docs, query, k, cfg)
			logger.info(f"Reranked to top {len(reranked_docs)} documents")
			return reranked_docs
		else:
			return unique_docs[:k]
		
	except Exception as e:
		logger.error(f"RAG search failed: {e}")
		return []


def create_context_from_documents(docs: List[Document]) -> str:
	"""Create formatted context string from retrieved documents"""
	if not docs:
		return "No relevant information found in the codebase."
	
	context_parts = []
	for i, doc in enumerate(docs, 1):
		metadata = doc.metadata or {}
		repo = metadata.get("repo", "Unknown")
		path = metadata.get("path", "Unknown")
		language = metadata.get("language", "")
		
		header = f"\n--- Document {i} ---\n"
		header += f"Repository: {repo}\n"
		header += f"File: {path}\n"
		if language:
			header += f"Language: {language}\n"
		header += "\n"
		
		context_parts.append(header + doc.page_content)
	
	return "\n".join(context_parts)


def get_context_summary(docs: List[Document]) -> Dict[str, Any]:
	"""Create simple summary of retrieved documents"""
	if not docs:
		return {"total_documents": 0, "repositories": [], "languages": []}
	
	repos = set()
	languages = set()
	
	for doc in docs:
		metadata = doc.metadata or {}
		if metadata.get("repo"):
			repos.add(metadata["repo"])
		if metadata.get("language"):
			languages.add(metadata["language"])
	
	return {
		"total_documents": len(docs),
		"repositories": list(repos),
		"languages": list(languages)
	}


# Legacy function for backward compatibility
def apply_cross_encoder_rerank(docs: List[Document], query: str, top_k: int, cfg: AppConfig) -> List[Document]:
	"""Legacy cross-encoder function - now uses embedding similarity rerank"""
	logger.info("Using embedding similarity reranking (simplified)")
	if cfg.retrieval.use_reranker:
		return embedding_similarity_rerank(docs, query, top_k, cfg)
	return docs[:top_k]


# Legacy function - simplified for compatibility
def create_context_summary(docs: List[Document]) -> Dict[str, Any]:
	"""Create simple context summary for compatibility"""
	return get_context_summary(docs)


# Legacy function - redirects to new RAG search
def enhanced_search(store, query: str, cfg: AppConfig, repo_prefix: Optional[str] = None, 
                   k: int = 12, include_related: bool = True, query_analysis: Optional[Dict[str, Any]] = None) -> List[Document]:
	"""Legacy enhanced search function - now uses simplified RAG search"""
	logger.info("Using simplified RAG search (legacy enhanced_search)")
	return rag_search(store, query, cfg, repo_prefix, k)


# Main RAG interface function
def perform_rag_search(store, query: str, cfg: AppConfig, repo_prefix: Optional[str] = None, k: int = 25) -> Dict[str, Any]:
	"""Main RAG search function: Query -> Embeddings -> Comprehensive Search -> Response"""
	try:
		logger.info(f"Starting comprehensive RAG search for query: {query[:100]}...")
		
		# Perform comprehensive RAG search with multiple strategies
		docs = rag_search(store, query, cfg, repo_prefix, k)
		
		# Be more permissive - try different search strategies if first attempt yields few results
		if len(docs) < k // 2:
			logger.info(f"Initial search yielded only {len(docs)} documents, trying broader search...")
			
			# Try with broader parameters and no repository filter
			broader_docs = rag_search(store, query, cfg, None, k * 2)  # Remove repo filter, get more docs
			
			# Combine results
			combined_docs = docs + broader_docs
			
			# Deduplicate
			seen_content = set()
			unique_docs = []
			for doc in combined_docs:
				content_hash = hash(doc.page_content[:200])
				if content_hash not in seen_content:
					seen_content.add(content_hash)
					unique_docs.append(doc)
			
			docs = unique_docs[:k]
			logger.info(f"Broader search yielded {len(docs)} documents total")
		
		if not docs:
			logger.warning("No documents retrieved even with comprehensive search")
			return {
				"context": "I couldn't find specific information about this in the indexed codebase. This might be because the relevant code hasn't been indexed yet, or the query needs different keywords.",
				"sources": [],
				"summary": {"total_documents": 0, "repositories": [], "languages": []}
			}
		
		# Create comprehensive context and summary
		context = create_context_from_documents(docs)
		summary = get_context_summary(docs)
		
		# Create detailed source info for frontend
		sources = []
		for doc in docs:
			metadata = doc.metadata or {}
			sources.append({
				"repo": metadata.get("repo"),
				"path": metadata.get("path"),
				"language": metadata.get("language"),
				"preview": doc.page_content[:400] + "..." if len(doc.page_content) > 400 else doc.page_content  # Longer preview
			})
		
		logger.info(f"Comprehensive RAG search completed: {len(docs)} documents, {len(summary.get('repositories', []))} repositories")
		
		return {
			"context": context,
			"sources": sources,
			"summary": summary
		}
		
	except Exception as e:
		logger.error(f"RAG search failed: {e}")
		logger.exception("Full error details:")
		return {
			"context": "I encountered a technical issue while searching the codebase. Please try rephrasing your query or contact support if this persists.",
			"sources": [],
			"summary": {"total_documents": 0, "repositories": [], "languages": []}
		}