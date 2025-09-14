"""OpenAI-based reranking for better cost control and performance"""

import logging
from typing import List, Dict, Any, Tuple
from langchain.schema import Document
from ..config.models import AppConfig
from ..core.chat import create_chat_llm
import json

logger = logging.getLogger("app.search.openai_reranker")


def openai_rerank_documents(docs: List[Document], query: str, top_k: int, cfg: AppConfig) -> List[Document]:
    """Use OpenAI to rerank documents based on relevance"""
    
    if not docs or len(docs) <= top_k:
        return docs[:top_k]
    
    try:
        # Use a smaller, faster model for reranking
        rerank_cfg = _get_reranking_config(cfg)
        llm = create_chat_llm(rerank_cfg, cfg.current_method)
        
        # Prepare documents for ranking with metadata
        doc_summaries = []
        for i, doc in enumerate(docs[:min(20, len(docs))]):  # Limit to top 20 for efficiency
            metadata = doc.metadata or {}
            
            # Create concise document summary
            summary = {
                "index": i,
                "content_preview": doc.page_content[:300],  # First 300 chars
                "file_path": metadata.get("path", "unknown"),
                "language": metadata.get("language", "unknown"),
                "is_test": metadata.get("is_test", False),
                "is_config": metadata.get("is_config", False)
            }
            doc_summaries.append(summary)
        
        # Create ranking prompt
        ranking_prompt = _create_ranking_prompt(query, doc_summaries, top_k)
        
        # Get ranking from OpenAI
        response = llm.invoke(ranking_prompt)
        ranked_indices = _parse_ranking_response(response, len(doc_summaries))
        
        # Return reranked documents
        reranked_docs = []
        for idx in ranked_indices[:top_k]:
            if 0 <= idx < len(docs):
                reranked_docs.append(docs[idx])
        
        # Fill remaining slots with original order if needed
        while len(reranked_docs) < top_k and len(reranked_docs) < len(docs):
            for doc in docs:
                if doc not in reranked_docs and len(reranked_docs) < top_k:
                    reranked_docs.append(doc)
        
        logger.info(f"OpenAI reranked {len(docs)} -> {len(reranked_docs)} documents")
        return reranked_docs
        
    except Exception as e:
        logger.warning(f"OpenAI reranking failed: {e}. Using original order.")
        return docs[:top_k]


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
                doc_embedding = embedding_fn.embed_query(doc.page_content[:500])  # First 500 chars
                similarity = _cosine_similarity(query_embedding, doc_embedding)
                
                # Apply metadata boosts
                metadata = doc.metadata or {}
                boost = 0
                
                if "test" in query.lower() and metadata.get("is_test"):
                    boost += 0.1
                if "config" in query.lower() and metadata.get("is_config"):
                    boost += 0.1
                if metadata.get("language") and metadata["language"] in query.lower():
                    boost += 0.05
                
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


def lightweight_rerank(docs: List[Document], query: str, top_k: int) -> List[Document]:
    """Ultra-fast lightweight reranking using simple heuristics"""
    
    if not docs or len(docs) <= top_k:
        return docs[:top_k]
    
    query_lower = query.lower()
    query_words = set(query_lower.split())
    
    scored_docs = []
    for doc in docs:
        score = 0.0
        content_lower = doc.page_content.lower()
        
        # Word match scoring
        for word in query_words:
            if len(word) > 2:  # Skip short words
                score += content_lower.count(word) * len(word)
        
        # Metadata boosts
        metadata = doc.metadata or {}
        if "test" in query_lower and metadata.get("is_test"):
            score += 10
        if "config" in query_lower and metadata.get("is_config"):
            score += 10
        if metadata.get("language") and metadata["language"] in query_lower:
            score += 5
        
        # File type preferences
        file_type = metadata.get("file_type", "")
        if file_type in [".py", ".js", ".ts", ".go"] and any(lang in query_lower for lang in ["python", "javascript", "typescript", "go"]):
            score += 8
        
        scored_docs.append((doc, score))
    
    # Sort by score
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    reranked_docs = [doc for doc, _ in scored_docs[:top_k]]
    
    return reranked_docs


def _get_reranking_config(cfg: AppConfig) -> Any:
    """Get optimized config for reranking (cheaper model)"""
    from ..config.models import ChatConfig
    
    # Use cheaper model for reranking
    rerank_config = ChatConfig(
        provider=cfg.chat.provider,
        model="gpt-3.5-turbo" if cfg.chat.provider == "openai" else "gemini-1.5-flash",
        temperature=0.0,  # Deterministic for ranking
        api_key_env=cfg.chat.api_key_env,
        max_tokens=150  # Short response needed
    )
    
    return rerank_config


def _create_ranking_prompt(query: str, doc_summaries: List[Dict], top_k: int) -> str:
    """Create prompt for document ranking"""
    
    docs_text = ""
    for doc in doc_summaries:
        docs_text += f"""
Document {doc['index']}:
File: {doc['file_path']}
Language: {doc['language']}
Content: {doc['content_preview']}
---"""
    
    prompt = f"""You are a code search expert. Rank these documents by relevance to the query.

Query: "{query}"

Documents:{docs_text}

Instructions:
- Consider code relevance, file types, and context
- Prioritize functional code over configuration when appropriate  
- Return ONLY a JSON list of document indices in order of relevance
- Example: [2, 0, 4, 1, 3]

Return the top {min(top_k, len(doc_summaries))} most relevant document indices:"""
    
    return prompt


def _parse_ranking_response(response: str, max_docs: int) -> List[int]:
    """Parse OpenAI ranking response"""
    
    try:
        # Extract JSON from response
        response_text = str(response)
        
        # Find JSON array in response
        start = response_text.find('[')
        end = response_text.rfind(']') + 1
        
        if start >= 0 and end > start:
            json_str = response_text[start:end]
            indices = json.loads(json_str)
            
            # Validate indices
            valid_indices = []
            for idx in indices:
                if isinstance(idx, int) and 0 <= idx < max_docs:
                    if idx not in valid_indices:  # Avoid duplicates
                        valid_indices.append(idx)
            
            return valid_indices if valid_indices else list(range(max_docs))
    
    except Exception as e:
        logger.debug(f"Failed to parse ranking response: {e}")
    
    # Fallback to original order
    return list(range(max_docs))


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
