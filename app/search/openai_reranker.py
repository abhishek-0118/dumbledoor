"""Embedding-based reranking for better relevance"""

import logging
from typing import List
from langchain.schema import Document
from ..config.models import AppConfig
from ..constants.system import SEARCH_CONFIG

logger = logging.getLogger("app.search.openai_reranker")


def embedding_similarity_rerank(docs: List[Document], query: str, top_k: int, cfg: AppConfig) -> List[Document]:
    """Fast reranking using OpenAI embeddings similarity"""
    
    if not docs or len(docs) <= top_k:
        return docs[:top_k]
    
    try:
        from ..core.embeddings import create_embedding_fn
        
        # Get embedding function
        embedding_fn, _ = create_embedding_fn(cfg.embedding)
        
        # Get query embedding
        query_embedding = embedding_fn.embed_query(query)
        
        # Score documents by similarity
        scored_docs = []
        for doc in docs:
            try:
                doc_embedding = embedding_fn.embed_query(doc.page_content[:SEARCH_CONFIG["MAX_EMBEDDING_CONTENT_LENGTH"]])
                similarity = _cosine_similarity(query_embedding, doc_embedding)
                
                # Apply metadata boosts
                metadata = doc.metadata or {}
                boost = 0
                
                if "test" in query.lower() and metadata.get("is_test"):
                    boost += SEARCH_CONFIG["SIMILARITY_BOOST_TEST"]
                if "config" in query.lower() and metadata.get("is_config"):
                    boost += SEARCH_CONFIG["SIMILARITY_BOOST_CONFIG"]
                if metadata.get("language") and metadata["language"] in query.lower():
                    boost += SEARCH_CONFIG["SIMILARITY_BOOST_LANGUAGE"]
                
                final_score = similarity + boost
                scored_docs.append((doc, final_score))
                
            except Exception as e:
                logger.debug(f"Failed to embed document: {e}")
                scored_docs.append((doc, 0.0))
        
        # Sort by score and return top_k
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        reranked_docs = [doc for doc, _ in scored_docs[:top_k]]
        
        logger.info(f"Embedding similarity reranked {len(docs)} -> {len(reranked_docs)} documents")
        return reranked_docs
        
    except Exception as e:
        logger.warning(f"Embedding similarity reranking failed: {e}. Using original order.")
        return docs[:top_k]




def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    
    try:
        import math
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(a * a for a in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    except Exception:
        return 0.0
