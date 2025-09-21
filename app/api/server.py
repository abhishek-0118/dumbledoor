import logging
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, Query, Depends
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import json
from pydantic import BaseModel
from ..config.loader import load_config
from ..indexing.indexer import RepoIndexer
from ..search.retrieval import perform_rag_search
from ..core.chat import create_chat_llm, estimate_token_cost
from pathlib import Path
from ..repos.github import clone_or_pull
from ..constants import CORS_ORIGINS, SERVER_HEADERS
from ..constants.prompts import create_dynamic_prompt, FALLBACK_MESSAGES
from ..constants.system import SERVER_CONFIG
from ..utils import setup_cost_tracking
from ..utils.response_enhancer import response_enhancer

logger = logging.getLogger("app.api.server")


class AskIn(BaseModel): 
    q: str
    k: int = SERVER_CONFIG["DEFAULT_K"]  # Increased default for more context
    alpha: float = SERVER_CONFIG["DEFAULT_ALPHA"]
    repo: Optional[str] = None
    architectural: bool = False
    include_context: bool = True
    detailed_response: bool = True


class SourceInfo(BaseModel):
    repo: Optional[str]
    repo_name: Optional[str] 
    path: Optional[str]
    file_type: Optional[str]
    language: Optional[str]
    module_name: Optional[str]
    is_test: bool = False
    is_config: bool = False
    preview: str
    relevance_score: Optional[float] = None


class AskOut(BaseModel):
    answer: str
    sources: List[SourceInfo]
    context_summary: Dict[str, Any]
    query_analysis: Dict[str, Any]
    total_sources_found: int


_indexer: Optional[RepoIndexer] = None
_cfg = None
_cost_tracker = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _indexer, _cfg, _cost_tracker
    _cfg = load_config()
    _indexer = RepoIndexer(_cfg)
    
    # Connect to MongoDB
    from ..db.mongodb import connect_to_mongo, close_mongo_connection
    await connect_to_mongo()
    
    # Setup cost tracking if enabled
    if _cfg.logging and _cfg.logging.enable_cost_tracking:
        _cost_tracker = setup_cost_tracking(
            enable_tracking=True,
            cost_log_file=_cfg.logging.cost_log_file
        )
        logger.info("Cost tracking enabled")
    else:
        _cost_tracker = None
    try:
        root = Path(_cfg.indexing.local_repo_root).resolve()
        root.mkdir(parents=True, exist_ok=True)
        if _cfg.indexing.repo_urls:
            logger.info(f"Cloning and indexing repos from URLs: {len(_cfg.indexing.repo_urls)}")
            _indexer.index_repo_urls(_cfg.indexing.repo_urls)
        elif _cfg.indexing.local_paths:
            logger.info(f"Indexing configured local paths: {len(_cfg.indexing.local_paths)}")
            _indexer.index_local_paths(_cfg.indexing.local_paths)
        else:
            logger.info(f"Indexing all repos under local root: {root}")
            _indexer.index_local_root(root)
    except Exception as e:
        logger.exception(f"Indexing on startup failed: {e}")
    yield
    
    # Close MongoDB connection
    await close_mongo_connection()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
from .auth_routes import router as auth_router
from .chat_routes import router as chat_router
from ..auth.dependencies import get_optional_current_user, get_current_user_flexible, get_current_user_required
from ..models.user import User
from ..services.user_service import user_service

app.include_router(auth_router)
app.include_router(chat_router)

@app.get("/health")
async def health_check():
    total_docs = 0
    if _indexer and hasattr(_indexer.store, 'doc_count'):
        try:
            total_docs = _indexer.store.doc_count
        except Exception as e:
            logger.warning(f"Could not get document count: {e}")
            total_docs = 0
    
    # Determine backend type from store type or collection name
    backend_info = "unknown"
    if _cfg and _cfg.backend:
        if hasattr(_cfg.backend, 'collection_name'):
            backend_info = f"chroma (collection: {_cfg.backend.collection_name})"
        else:
            backend_info = "chroma"
    
    return {
        "status": "ok", 
        "indexer_available": _indexer is not None,
        "total_documents": total_docs,
        "backend_info": backend_info,
        "chroma_dir": _cfg.backend.chroma_dir if _cfg and _cfg.backend else None
    }

@app.get("/test-auth")
async def test_auth(current_user: User = Depends(get_current_user_required)):
    """Test endpoint to verify authentication is working."""
    return {
        "authenticated": True,
        "user_id": str(current_user.id),
        "email": current_user.email,
        "queries_used": current_user.settings.queries_used,
        "query_limit": current_user.settings.query_limit,
        "message": "Authentication successful!"
    }

@app.get("/cost-summary")
async def get_cost_summary():
    if not _cost_tracker:
        return {"error": "Cost tracking not enabled"}
    
    session_summary = _cost_tracker.get_session_summary()
    daily_summary = _cost_tracker.get_daily_summary()
    
    return {
        "session": session_summary,
        "today": daily_summary,
        "recommendations": {
            "total_cost_today": f"${daily_summary['total']:.6f}",
            "token_efficiency": daily_summary["token_count"] / max(daily_summary["request_count"], 1)
        }
    }

@app.get("/repositories")
async def get_repositories():
    """Get list of indexed repositories."""
    if not _cfg or not _cfg.indexing:
        return {"repositories": []}
    
    repos = []
    
    # Get repositories from repo_urls if available
    if _cfg.indexing.repo_urls:
        for repo_url in _cfg.indexing.repo_urls:
            # Extract repo name from URL
            repo_name = repo_url.split('/')[-1].replace('.git', '')
            repos.append({
                "name": repo_name,
                "url": repo_url,
                "type": "github"
            })
    
    # Get repositories from local_paths if available
    if _cfg.indexing.local_paths:
        for local_path in _cfg.indexing.local_paths:
            path_obj = Path(local_path)
            repos.append({
                "name": path_obj.name,
                "url": None,
                "type": "local",
                "path": local_path
            })
    
    return {
        "repositories": repos,
        "total_count": len(repos)
    }

@app.get("/ask/stream")
async def ask_stream(
    q: str = Query(..., description="The question to ask"),
    k: int = Query(SERVER_CONFIG["DEFAULT_K"], description="Number of documents to retrieve"),
    alpha: float = Query(SERVER_CONFIG["DEFAULT_ALPHA"], description="Alpha parameter for retrieval"),
    repo: Optional[str] = Query(None, description="Specific repository to search"),
    architectural: bool = Query(False, description="Focus on architectural aspects"),
    include_context: bool = Query(True, description="Include context in search"),
    detailed_response: bool = Query(True, description="Provide detailed response"),
    current_user: User = Depends(get_current_user_required)
):
    """Stream the answer generation process"""
    # Check query limits for authenticated users
    if current_user.settings.queries_used >= current_user.settings.query_limit:
        from fastapi import HTTPException, status
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Query limit exceeded. You have used {current_user.settings.queries_used}/{current_user.settings.query_limit} queries."
        )
    
    # Increment query count
    await user_service.increment_query_count(str(current_user.id))
    logger.info(f"User {current_user.email} query count incremented to {current_user.settings.queries_used + 1}")
    
    payload = AskIn(
        q=q,
        k=k,
        alpha=alpha,
        repo=repo,
        architectural=architectural,
        include_context=include_context,
        detailed_response=detailed_response
    )
    
    return StreamingResponse(
        stream_answer(payload, current_user),
        media_type="text/event-stream",
        headers=SERVER_HEADERS
    )

@app.post("/ask", response_model=AskOut)
async def ask(payload: AskIn):
    """Simplified RAG-first ask endpoint: Query -> Embeddings -> RAG -> Response -> LLM"""
    
    # Step 1: Perform RAG search
    rag_result = perform_rag_search(_indexer.store, payload.q, _cfg, payload.repo, payload.k)
    
    # Check if RAG found any results
    if not rag_result["sources"]:
        logger.warning(f"No documents found for query: {payload.q}")
        return AskOut(
            answer=FALLBACK_MESSAGES["no_results"], 
            sources=[],
            context_summary=rag_result["summary"],
            query_analysis={},
            total_sources_found=0
        )

    # Step 2: Generate LLM response using RAG context
    try:
        # Create LLM
        llm = create_chat_llm(_cfg.chat, _cfg.current_method)
        
        # Create dynamic prompt using new system
        prompt_text = create_dynamic_prompt(rag_result["context"], payload.q, rag_result["summary"])
        
        # Track costs if enabled
        if _cost_tracker:
            query_cost = _cost_tracker.track_chat_cost(
                provider=_cfg.chat.provider,
                model=_cfg.chat.model,
                input_text=prompt_text,
                query=payload.q,
                context_size=len(rag_result["sources"])
            )
            logger.info(f"Query cost tracking: {query_cost}")
        else:
            query_cost = estimate_token_cost(prompt_text, _cfg.chat.model, _cfg.chat.provider)
            logger.info(f"Query token estimate: {query_cost}")
        
        # Get LLM response
        from langchain.schema import HumanMessage
        messages = [HumanMessage(content=prompt_text)]
        
        response = llm.invoke(messages)
        raw_answer = response.content if hasattr(response, 'content') else str(response)
        
        if not raw_answer or not raw_answer.strip():
            raw_answer = "I cannot provide a proper answer based on the retrieved context. Please rephrase your question."
        
        # Apply simple response enhancement
        answer = response_enhancer.enhance_response(raw_answer, rag_result["summary"])
        
        # Track response costs if enabled
        if _cost_tracker and answer:
            response_cost = _cost_tracker.track_chat_cost(
                provider=_cfg.chat.provider,
                model=_cfg.chat.model,
                input_text="",  # Already tracked in query
                output_text=answer,
                context_size=0
            )
            session_summary = _cost_tracker.get_session_summary()
            logger.info(f"Response cost: {response_cost}, Session total: ${session_summary['session_costs']['total']:.6f}")
        
        # Create source info for response
        sources = []
        for source in rag_result["sources"]:
            source_info = SourceInfo(
                repo=source.get("repo"),
                repo_name=source.get("repo"),
                path=source.get("path"),
                file_type=None,
                language=source.get("language"),
                module_name=None,
                is_test=False,
                is_config=False,
                preview=source.get("preview", ""),
                relevance_score=None
            )
            sources.append(source_info)
        
        return AskOut(
            answer=answer, 
            sources=sources,
            context_summary=rag_result["summary"],
            query_analysis={},
            total_sources_found=len(rag_result["sources"])
        )
        
    except Exception as e:
        logger.error(f"LLM query failed: {e}")
        error_answer = "I encountered an error generating the response. Please try rephrasing your question."
        
        sources = []
        for source in rag_result["sources"]:
            source_info = SourceInfo(
                repo=source.get("repo"),
                repo_name=source.get("repo"),
                path=source.get("path"),
                file_type=None,
                language=source.get("language"),
                module_name=None,
                is_test=False,
                is_config=False,
                preview=source.get("preview", ""),
                relevance_score=None
            )
            sources.append(source_info)
        
        return AskOut(
            answer=error_answer,
            sources=sources,
            context_summary=rag_result["summary"],
            query_analysis={},
            total_sources_found=len(rag_result["sources"])
        )

async def stream_answer(payload: AskIn, current_user: Optional[User] = None):
    """Simplified RAG-first streaming: Query -> Embeddings -> RAG -> Response -> LLM"""
    
    def create_sse_message(data: dict, event: str = "data"):
        """Format data as Server-Sent Event"""
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"
    
    try:
        # Send initial status
        yield create_sse_message({
            "status": "starting",
            "message": "🚀 Starting RAG search..."
        }, "status")
        
        # Step 1: Perform RAG search (Query -> Embeddings -> Search)
        yield create_sse_message({
            "status": "searching",
            "message": "🔍 Searching codebase..."
        }, "status")
        
        rag_result = perform_rag_search(_indexer.store, payload.q, _cfg, payload.repo, payload.k)
        
        # Check if RAG found any results
        if not rag_result["sources"]:
            logger.warning(f"No documents found for query: {payload.q}")
            yield create_sse_message({
                "chunk": FALLBACK_MESSAGES["no_results"],
                "is_final": True,
                "final_content": FALLBACK_MESSAGES["no_results"]
            }, "answer_chunk")
            yield create_sse_message({"status": "completed"}, "completed")
            return
        
        # Send sources and context summary
        yield create_sse_message({
            "sources": rag_result["sources"],
            "context_summary": rag_result["summary"],
            "total_sources_found": len(rag_result["sources"])
        }, "sources")
        
        # Step 2: Generate LLM response
        yield create_sse_message({
            "status": "generating",
            "message": "🤖 Generating response..."
        }, "status")
        
        try:
            # Create LLM
            llm = create_chat_llm(_cfg.chat, _cfg.current_method)
            
            # Create dynamic prompt using new system
            prompt_text = create_dynamic_prompt(rag_result["context"], payload.q, rag_result["summary"])
            
            # Track costs if enabled
            if _cost_tracker:
                query_cost = _cost_tracker.track_chat_cost(
                    provider=_cfg.chat.provider,
                    model=_cfg.chat.model,
                    input_text=prompt_text,
                    query=payload.q,
                    context_size=len(rag_result["sources"])
                )
                yield create_sse_message({
                    "token_estimate": query_cost,
                    "session_total": _cost_tracker.session_costs["total"]
                }, "cost_estimate")
            
            # Stream the LLM response
            from langchain.schema import HumanMessage
            
            full_response = ""
            messages = [HumanMessage(content=prompt_text)]
            
            async for chunk in llm.astream(messages):
                if chunk.content:
                    full_response += chunk.content
                    
                    # Send chunk immediately (no complex enhancement during streaming)
                    yield create_sse_message({
                        "chunk": chunk.content,
                        "is_final": False,
                        "full_content_so_far": full_response
                    }, "answer_chunk")
            
            # Apply simple response enhancement
            if full_response:
                enhanced_response = response_enhancer.enhance_response(full_response, rag_result["summary"])
                
                # Final response tracking
                if _cost_tracker:
                    response_cost = _cost_tracker.track_chat_cost(
                        provider=_cfg.chat.provider,
                        model=_cfg.chat.model,
                        input_text="",
                        output_text=full_response,
                        context_size=0
                    )
                    yield create_sse_message({
                        "response_token_estimate": response_cost,
                        "session_total": _cost_tracker.get_session_summary()["session_costs"]["total"]
                    }, "cost_estimate")
                
                # Send final completion signal
                yield create_sse_message({
                    "chunk": "",
                    "is_final": True,
                    "final_content": enhanced_response
                }, "answer_chunk")
            else:
                # No response generated
                yield create_sse_message({
                    "chunk": "I cannot provide a proper answer based on the retrieved context. Please rephrase your question.",
                    "is_final": True,
                    "final_content": "I cannot provide a proper answer based on the retrieved context. Please rephrase your question."
                }, "answer_chunk")
        
        except Exception as llm_error:
            logger.error(f"LLM error: {llm_error}")
            error_msg = "I encountered an error generating the response. Please try rephrasing your question."
            yield create_sse_message({
                "chunk": error_msg,
                "is_final": True,
                "error": str(llm_error)
            }, "answer_chunk")
        
        # Send completion status
        yield create_sse_message({
            "status": "completed",
            "message": "✅ Response completed"
        }, "completed")
        
    except Exception as e:
        logger.error(f"Streaming failed: {e}")
        yield create_sse_message({
            "error": str(e),
            "status": "error",
            "message": f"Error: {str(e)}"
        }, "error")

# Removed complex truncation function - now handled in RAG search


# Removed complex analysis functions - now using simplified RAG approach

# Legacy function stubs removed - no longer needed

class WebhookIn(BaseModel):
    repo_url: Optional[str] = None
    repo_name: Optional[str] = None
    changed_files: List[str]

@app.post("/webhook/github")
async def webhook(payload: WebhookIn):
    root = Path(_cfg.indexing.local_repo_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if payload.repo_name:
        repo_dir = root / payload.repo_name
    elif payload.repo_url:
        repo_dir = clone_or_pull(payload.repo_url, root, _cfg.indexing.github_token_env)
    else:
        return {"ok": False, "error": "repo_url or repo_name required"}
    _indexer.upsert_changed_files(repo_dir, repo_dir.name, payload.changed_files)
    return {"ok": True}
