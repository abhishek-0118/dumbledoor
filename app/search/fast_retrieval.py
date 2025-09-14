"""Fast retrieval optimizations for large scale deployment"""

import hashlib
import json
import time
from typing import List, Dict, Any, Optional
from langchain.schema import Document
from functools import lru_cache
import logging
from ..constants import CACHE_CONFIG

logger = logging.getLogger("app.search.fast_retrieval")


class QueryCache:
    """LRU cache for query results with TTL"""
    
    def __init__(self, max_size: int = 1000, ttl: int = 300):
        self.max_size = max_size
        self.ttl = ttl
        self._cache = {}
        self._timestamps = {}
    
    def _hash_query(self, query: str, filters: Dict = None, k: int = 10) -> str:
        """Create hash key for query"""
        cache_key = {
            "query": query.lower().strip(),
            "filters": filters or {},
            "k": k
        }
        return hashlib.md5(json.dumps(cache_key, sort_keys=True).encode()).hexdigest()
    
    def get(self, query: str, filters: Dict = None, k: int = 10) -> Optional[List[Document]]:
        """Get cached results if valid"""
        key = self._hash_query(query, filters, k)
        
        if key not in self._cache:
            return None
        
        # Check TTL
        if time.time() - self._timestamps[key] > self.ttl:
            del self._cache[key]
            del self._timestamps[key]
            return None
        
        logger.debug(f"Cache hit for query: {query[:50]}...")
        return self._cache[key]
    
    def put(self, query: str, results: List[Document], filters: Dict = None, k: int = 10):
        """Cache query results"""
        key = self._hash_query(query, filters, k)
        
        # Evict oldest if at capacity
        if len(self._cache) >= self.max_size:
            oldest_key = min(self._timestamps.keys(), key=self._timestamps.get)
            del self._cache[oldest_key]
            del self._timestamps[oldest_key]
        
        self._cache[key] = results
        self._timestamps[key] = time.time()
        logger.debug(f"Cached results for query: {query[:50]}...")
    
    def clear(self):
        """Clear all cached results"""
        self._cache.clear()
        self._timestamps.clear()


class FastRetriever:
    """Optimized retriever with caching and batching"""
    
    def __init__(self, store, cfg, enable_cache: bool = True):
        self.store = store
        self.cfg = cfg
        self.enable_cache = enable_cache
        
        # Initialize cache if enabled
        if enable_cache:
            cache_config = getattr(cfg.retrieval, 'cache_config', CACHE_CONFIG)
            self.query_cache = QueryCache(
                max_size=cache_config.get('max_cache_size', 1000),
                ttl=getattr(cfg.retrieval, 'cache_ttl', cache_config['query_cache_ttl'])
            )
        else:
            self.query_cache = None
        
        # Pre-compiled filters for common queries
        self._common_filters = {}
    
    def retrieve(self, query: str, repo_filter: Optional[str] = None, 
                k: int = 10, alpha: float = 0.3) -> List[Document]:
        """Fast retrieval with caching and optimizations"""
        
        # Build filter
        search_filter = None
        if repo_filter:
            search_filter = {
                "$or": [
                    {"repo": {"$like": f"{repo_filter}%"}},
                    {"repo_name": {"$like": f"{repo_filter}%"}},
                    {"repo_folder": {"$like": f"{repo_filter}%"}},
                ]
            }
        
        # Check cache first
        if self.query_cache:
            cached_results = self.query_cache.get(query, search_filter, k)
            if cached_results:
                return cached_results
        
        # Perform search with optimizations
        start_time = time.time()
        
        # Use larger search_k for better recall, then trim
        search_k = min(k * 3, 50)  # Cap to prevent excessive computation
        
        try:
            # Direct similarity search (fastest path)
            docs = self.store.similarity_search(
                query, 
                k=search_k,
                filter=search_filter
            )
            
            # Apply lightweight post-processing
            filtered_docs = self._post_process_results(docs, query, k)
            
        except Exception as e:
            logger.warning(f"Fast search failed, falling back to standard retrieval: {e}")
            # Fallback to standard retrieval
            retriever = self.store.as_retriever(
                search_kwargs={"k": search_k, "filter": search_filter}
            )
            docs = retriever.get_relevant_documents(query)
            filtered_docs = docs[:k]
        
        search_time = time.time() - start_time
        logger.debug(f"Retrieved {len(filtered_docs)} docs in {search_time:.3f}s")
        
        # Cache results
        if self.query_cache:
            self.query_cache.put(query, filtered_docs, search_filter, k)
        
        return filtered_docs
    
    def _post_process_results(self, docs: List[Document], query: str, k: int) -> List[Document]:
        """Lightweight post-processing for relevance"""
        
        # Score documents based on simple heuristics
        scored_docs = []
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        for doc in docs:
            score = 0.0
            content_lower = doc.page_content.lower()
            
            # Simple relevance scoring
            for word in query_words:
                if word in content_lower:
                    score += content_lower.count(word)
            
            # Boost for metadata relevance
            metadata = doc.metadata or {}
            if metadata.get('language') and any(lang in query_lower for lang in ['python', 'javascript', 'go']):
                if metadata['language'] in query_lower:
                    score += 2
            
            # Boost for test files if query mentions testing
            if 'test' in query_lower and metadata.get('is_test'):
                score += 1
            
            scored_docs.append((score, doc))
        
        # Sort by score and return top k
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored_docs[:k]]
    
    def batch_retrieve(self, queries: List[str], **kwargs) -> Dict[str, List[Document]]:
        """Batch retrieval for multiple queries"""
        results = {}
        
        # Group uncached queries
        uncached_queries = []
        if self.query_cache:
            for query in queries:
                cached = self.query_cache.get(query, kwargs.get('repo_filter'), kwargs.get('k', 10))
                if cached:
                    results[query] = cached
                else:
                    uncached_queries.append(query)
        else:
            uncached_queries = queries
        
        # Process uncached queries
        for query in uncached_queries:
            results[query] = self.retrieve(query, **kwargs)
        
        return results
    
    def warm_cache(self, common_queries: List[str]):
        """Pre-warm cache with common queries"""
        if not self.query_cache:
            return
        
        logger.info(f"Warming cache with {len(common_queries)} queries")
        for query in common_queries:
            try:
                self.retrieve(query, k=5)  # Use smaller k for warming
            except Exception as e:
                logger.warning(f"Failed to warm cache for query '{query}': {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        if not self.query_cache:
            return {"cache_enabled": False}
        
        return {
            "cache_enabled": True,
            "cache_size": len(self.query_cache._cache),
            "max_size": self.query_cache.max_size,
            "ttl": self.query_cache.ttl
        }


@lru_cache(maxsize=100)
def get_query_embedding_cached(query: str, embedding_fn) -> List[float]:
    """Cache query embeddings for repeated queries"""
    try:
        return embedding_fn.embed_query(query)
    except Exception as e:
        logger.warning(f"Failed to embed query: {e}")
        return []


def create_fast_retriever(store, cfg) -> FastRetriever:
    """Factory function to create fast retriever"""
    enable_cache = getattr(cfg.retrieval, 'enable_caching', False)
    return FastRetriever(store, cfg, enable_cache=enable_cache)
