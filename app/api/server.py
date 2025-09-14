import logging
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import json
import asyncio
from pydantic import BaseModel
from ..config.loader import load_config
from ..indexing.indexer import RepoIndexer
from ..search.retrieval import build_retriever, apply_cross_encoder_rerank, enhanced_search, create_context_summary
from ..core.chat import create_chat_llm, estimate_token_cost
from langchain.chains import RetrievalQA
from langchain.schema import Document
from pathlib import Path
from ..repos.github import clone_or_pull
import tiktoken
from ..constants import (
    CORS_ORIGINS, SERVER_HEADERS, TECH_KEYWORDS, QUERY_ANALYSIS_KEYWORDS,
    ENHANCED_PROMPT_TEMPLATES, BASE_CODE_PROMPT_TEMPLATE, FALLBACK_MESSAGES,
    STREAM_MESSAGES, PROGRAMMING_LANGUAGES, DEFAULT_VALUES, TOKENIZER_MODELS
)
from ..utils import setup_cost_tracking

logger = logging.getLogger("app.api.server")


class AskIn(BaseModel): 
    q: str
    k: int = 20  # Increased default for more context
    alpha: float = 0.3
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


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    return {"status": "ok", "indexed_repos": len(_indexer.indexed_repos) if _indexer else 0, "total_documents": _indexer.store.doc_count if _indexer else 0}

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

@app.get("/ask/stream")
async def ask_stream(
    q: str = Query(..., description="The question to ask"),
    k: int = Query(20, description="Number of documents to retrieve"),
    alpha: float = Query(0.3, description="Alpha parameter for retrieval"),
    repo: Optional[str] = Query(None, description="Specific repository to search"),
    architectural: bool = Query(False, description="Focus on architectural aspects"),
    include_context: bool = Query(True, description="Include context in search"),
    detailed_response: bool = Query(True, description="Provide detailed response")
):
    """Stream the answer generation process"""
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
        stream_answer(payload),
        media_type="text/event-stream",
        headers=SERVER_HEADERS
    )

@app.post("/ask", response_model=AskOut)
async def ask(payload: AskIn):
    """Enhanced ask endpoint with better code understanding and context"""

    query_analysis = analyze_query(payload.q)
    docs = enhanced_search(
        _indexer.store, 
        payload.q, 
        _cfg, 
        payload.repo, 
        payload.k,
        payload.include_context
    )
    
    # Truncate docs to fit within token limits (very aggressive truncation for OpenAI)
    truncated_docs = truncate_context_for_model(docs, _cfg.chat.model, max_context_tokens=1200)
    context_summary = create_context_summary(truncated_docs)

    ret = build_retriever(_indexer.store, _cfg, payload.repo, len(truncated_docs), payload.alpha)
    
    enhanced_query = create_enhanced_prompt(payload.q, query_analysis, context_summary, payload.detailed_response)
    
    try:
        # Create LLM based on current method configuration
        llm = create_chat_llm(_cfg.chat, _cfg.current_method)
        
        # Track costs with enhanced logging
        if _cost_tracker:
            query_cost = _cost_tracker.track_chat_cost(
                provider=_cfg.chat.provider,
                model=_cfg.chat.model,
                input_text=enhanced_query,
                query=payload.q,
                context_size=len(truncated_docs)
            )
            logger.info(f"Query cost tracking: {query_cost}")
        else:
            query_cost = estimate_token_cost(enhanced_query, _cfg.chat.model, _cfg.chat.provider)
            logger.info(f"Query token estimate: {query_cost}")
        
        # custom QA chain with code-awared prompt
        qa = RetrievalQA.from_chain_type(
            llm=llm, 
            retriever=ret, 
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": create_code_aware_prompt(payload.detailed_response, query_analysis)
            }
        )
        
        res = qa.invoke({"query": enhanced_query})
        answer = res.get("result", "")
        
        # Track response costs
        if _cost_tracker and answer:
            # Update the previous chat cost entry with output tokens
            response_cost = _cost_tracker.track_chat_cost(
                provider=_cfg.chat.provider,
                model=_cfg.chat.model,
                input_text="",  # Already tracked in query
                output_text=answer,
                context_size=0
            )
            session_summary = _cost_tracker.get_session_summary()
            logger.info(f"Response cost: {response_cost}, Session total: ${session_summary['session_costs']['total']:.6f}")
        else:
            response_cost = estimate_token_cost(answer, _cfg.chat.model, _cfg.chat.provider)
            logger.info(f"Response token estimate: {response_cost}")
        
    except Exception as e:
        logger.error(f"LLM query failed: {e}")
        answer = create_fallback_answer(docs, payload.q, query_analysis)
    
    # Enhanced source information
    sources = []
    for i, doc in enumerate(truncated_docs[:payload.k]):
        metadata = doc.metadata or {}
        source_info = SourceInfo(
            repo=metadata.get("repo"),
            repo_name=metadata.get("repo_name", metadata.get("repo")),
            path=metadata.get("path"),
            file_type=metadata.get("file_type"),
            language=metadata.get("language"),
            module_name=metadata.get("module_name"),
            is_test=metadata.get("is_test", False),
            is_config=metadata.get("is_config", False),
            preview=doc.page_content[:600],  # Increased preview length
            relevance_score=1.0 - (i * 0.1)  # Simple relevance scoring
        )
        sources.append(source_info)
    
    return AskOut(
        answer=answer, 
        sources=sources,
        context_summary=context_summary,
        query_analysis=query_analysis,
        total_sources_found=len(docs)
    )

async def stream_answer(payload: AskIn):
    """Stream the answer generation process to the client using proper SSE format"""
    
    def create_sse_message(data: dict, event: str = "data"):
        """Format data as Server-Sent Event"""
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"
    
    try:
        # Send initial status
        yield create_sse_message({
            "status": "starting",
            "message": STREAM_MESSAGES["starting"]
        }, "status")
        
        # Analyze query
        yield create_sse_message({
            "status": "analyzing",
            "message": STREAM_MESSAGES["analyzing"]
        }, "status")
        
        query_analysis = analyze_query(payload.q)
        yield create_sse_message({
            "query_analysis": query_analysis
        }, "analysis")
        
        # Search for documents
        yield create_sse_message({
            "status": "searching",
            "message": STREAM_MESSAGES["searching"]
        }, "status")
        
        docs = enhanced_search(
            _indexer.store, 
            payload.q, 
            _cfg, 
            payload.repo, 
            payload.k,
            payload.include_context
        )
        
        # Truncate docs to fit within token limits (very aggressive truncation for OpenAI)
        truncated_docs = truncate_context_for_model(docs, _cfg.chat.model, max_context_tokens=1200)
        
        # Send context summary
        context_summary = create_context_summary(truncated_docs)
        yield create_sse_message({
            "context_summary": context_summary,
            "total_sources_found": len(docs),
            "docs_after_truncation": len(truncated_docs)
        }, "context")
        
        # Format and send sources
        sources = []
        for i, doc in enumerate(truncated_docs[:payload.k]):
            metadata = doc.metadata or {}
            source_info = SourceInfo(
                repo=metadata.get("repo"),
                repo_name=metadata.get("repo_name", metadata.get("repo")),
                path=metadata.get("path"),
                file_type=metadata.get("file_type"),
                language=metadata.get("language"),
                module_name=metadata.get("module_name"),
                is_test=metadata.get("is_test", False),
                is_config=metadata.get("is_config", False),
                preview=doc.page_content[:800],  # Increased preview for better context
                relevance_score=1.0 - (i * 0.05)  # Better relevance scoring
            )
            sources.append(source_info.dict())
        
        yield create_sse_message({
            "sources": sources
        }, "sources")
        
        # Generate answer with streaming
        yield create_sse_message({
            "status": "generating",
            "message": STREAM_MESSAGES["generating"]
        }, "status")
        
        ret = build_retriever(_indexer.store, _cfg, payload.repo, len(truncated_docs), payload.alpha)
        enhanced_query = create_enhanced_prompt(payload.q, query_analysis, context_summary, payload.detailed_response)
        
        try:
            # Create LLM based on current method configuration
            llm = create_chat_llm(_cfg.chat, _cfg.current_method)
            
            # Track token costs for streaming
            if _cost_tracker:
                query_cost = _cost_tracker.track_chat_cost(
                    provider=_cfg.chat.provider,
                    model=_cfg.chat.model,
                    input_text=enhanced_query,
                    query=payload.q,
                    context_size=len(truncated_docs)
                )
                yield create_sse_message({
                    "token_estimate": query_cost,
                    "session_total": _cost_tracker.session_costs["total"]
                }, "cost_estimate")
            else:
                query_cost = estimate_token_cost(enhanced_query, _cfg.chat.model, _cfg.chat.provider)
                yield create_sse_message({
                    "token_estimate": query_cost
                }, "cost_estimate")
            
            qa = RetrievalQA.from_chain_type(
                llm=llm, 
                retriever=ret, 
                return_source_documents=True,
                chain_type_kwargs={
                    "prompt": create_code_aware_prompt(payload.detailed_response, query_analysis)
                }
            )
            
            # Generate the answer
            res = qa.invoke({"query": enhanced_query})
            answer = res.get("result", "")
            
            # Track response token cost for streaming
            if answer:
                if _cost_tracker:
                    response_cost = _cost_tracker.track_chat_cost(
                        provider=_cfg.chat.provider,
                        model=_cfg.chat.model,
                        input_text="",
                        output_text=answer,
                        context_size=0
                    )
                    session_summary = _cost_tracker.get_session_summary()
                    yield create_sse_message({
                        "response_token_estimate": response_cost,
                        "session_total": session_summary["session_costs"]["total"]
                    }, "cost_estimate")
                else:
                    response_cost = estimate_token_cost(answer, _cfg.chat.model, _cfg.chat.provider)
                    yield create_sse_message({
                        "response_token_estimate": response_cost
                    }, "cost_estimate")
            
            # Stream the answer in chunks with proper markdown formatting
            if answer:
                # Split by sentences for better streaming experience
                sentences = answer.replace('. ', '.|').replace('.\n', '.\n|').split('|')
                
                for i, sentence in enumerate(sentences):
                    if sentence.strip():
                        # Add proper spacing and formatting
                        chunk = sentence
                        if not sentence.endswith('\n') and i < len(sentences) - 1:
                            chunk += ' '
                        
                        yield create_sse_message({
                            "chunk": chunk,
                            "is_final": i == len(sentences) - 1
                        }, "answer_chunk")
                        
                        # Small delay for better UX
                        await asyncio.sleep(0.05)
            else:
                yield create_sse_message({
                    "chunk": FALLBACK_MESSAGES["stream_error"],
                    "is_final": True
                }, "answer_chunk")
        
        except Exception as e:
            logger.error(f"LLM query failed: {e}")
            fallback_answer = create_fallback_answer(docs, payload.q, query_analysis)
            
            # Stream fallback answer
            yield create_sse_message({
                "chunk": fallback_answer,
                "is_final": True
            }, "answer_chunk")
        
        # Send completion status
        yield create_sse_message({
            "status": "completed",
            "message": STREAM_MESSAGES["completed"]
        }, "status")
        
    except Exception as e:
        logger.error(f"Streaming failed: {e}")
        yield create_sse_message({
            "error": str(e),
            "status": "error",
            "message": STREAM_MESSAGES["error"].format(error=str(e))
        }, "error")

def truncate_context_for_model(docs: List[Document], model: str, max_context_tokens: int = 4000) -> List[Document]:
    """Truncate documents to fit within token limits"""
    try:
        if "gpt-4" in model.lower():
            encoding = tiktoken.encoding_for_model("gpt-4")
        elif "gpt-3.5" in model.lower():
            encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        else:
            encoding = tiktoken.get_encoding(TOKENIZER_MODELS.get("default"))
    except KeyError:
        encoding = tiktoken.get_encoding(TOKENIZER_MODELS.get("default"))
    
    truncated_docs = []
    total_tokens = 0
    
    for doc in docs:
        # Count tokens for this document
        doc_tokens = len(encoding.encode(doc.page_content))
        
        if total_tokens + doc_tokens <= max_context_tokens:
            # Add full document
            truncated_docs.append(doc)
            total_tokens += doc_tokens
        else:
            # Truncate document to fit remaining space
            remaining_tokens = max_context_tokens - total_tokens
            if remaining_tokens > 100:  # Only add if we have meaningful space left
                # Truncate content to fit
                content = doc.page_content
                truncated_content = encoding.decode(encoding.encode(content)[:remaining_tokens])
                
                # Create truncated document
                truncated_doc = Document(
                    page_content=truncated_content + "...",
                    metadata=doc.metadata
                )
                truncated_docs.append(truncated_doc)
            break
    
    logger.info(f"Truncated context: {len(docs)} -> {len(truncated_docs)} docs, ~{total_tokens} tokens")
    return truncated_docs


def analyze_query(query: str) -> Dict[str, Any]:
    """Analyze the query to understand what the user is looking for"""
    query_lower = query.lower()
    
    analysis = {}
    for key, phrases in QUERY_ANALYSIS_KEYWORDS.items():
        analysis[key] = any(phrase in query_lower for phrase in phrases)
    analysis["mentions_specific_tech"] = []
    
    for tech in TECH_KEYWORDS:
        if tech in query_lower:
            analysis["mentions_specific_tech"].append(tech)
    
    return analysis

def create_enhanced_prompt(original_query: str, analysis: Dict[str, Any], context_summary: Dict[str, Any], detailed: bool) -> str:
    """Create an enhanced prompt with better context"""
    
    enhanced_parts = [original_query]
    
    # Add context about the codebase
    if context_summary.get("languages"):
        enhanced_parts.append(f"Context: Looking at code in {', '.join(context_summary['languages'])} from {context_summary.get('total_documents', 0)} files")
    
    for key, template in ENHANCED_PROMPT_TEMPLATES.items():
        if key == "detailed" and detailed:
            enhanced_parts.append(template)
        elif key.startswith("is_") and analysis.get(key):
            enhanced_parts.append(template)
            break
    
    return " ".join(enhanced_parts)

def create_code_aware_prompt(detailed: bool, analysis: Dict[str, Any]):
    """Create a code-aware prompt template for the LLM with markdown formatting"""
    from langchain.prompts import PromptTemplate
    
    base_template = BASE_CODE_PROMPT_TEMPLATE
    
    for key in ["code_related", "test_related", "config_related"]:
        if analysis.get(key):
            base_template += ENHANCED_PROMPT_TEMPLATES[key]
    
    if detailed:
        base_template += ENHANCED_PROMPT_TEMPLATES["detailed"]
    
    if analysis.get("is_architectural"):
        base_template += ENHANCED_PROMPT_TEMPLATES["architectural"]
    
    base_template += "\nAnswer:"
    
    return PromptTemplate(
        template=base_template,
        input_variables=["context", "question"]
    )

def create_fallback_answer(docs: List[Document], query: str, analysis: Dict[str, Any]) -> str:
    """Create a fallback answer when LLM fails, formatted as markdown"""
    if not docs:
        return FALLBACK_MESSAGES["no_results"]
    
    # Extract key information from documents
    relevant_files = []
    key_snippets = []
    languages = set()
    
    for doc in docs[:5]:
        metadata = doc.metadata or {}
        if metadata.get("path"):
            repo_name = metadata.get('repo', 'unknown')
            file_path = metadata['path']
            relevant_files.append(f"{repo_name}/{file_path}")
            
            if metadata.get('language'):
                languages.add(metadata['language'])
        
        # Extract a meaningful snippet
        content = doc.page_content
        if len(content) > 300:
            key_snippets.append(content[:300] + "...")
        else:
            key_snippets.append(content)
    
    fallback = FALLBACK_MESSAGES["search_results_header"]
    
    if languages:
        fallback += f"**Languages found:** {', '.join(sorted(languages))}\n\n"
    
    fallback += f"### Relevant Files ({len(relevant_files)} found)\n\n"
    
    for i, file_path in enumerate(relevant_files):
        fallback += f"{i+1}. `{file_path}`\n"
    
    fallback += f"\n### Code Snippets\n\n"
    
    for i, snippet in enumerate(key_snippets[:3]):
        # Try to detect language from file extension or content
        lang = "text"
        if i < len(docs):
            metadata = docs[i].metadata or {}
            file_lang = metadata.get('language', '').lower()
            if file_lang in PROGRAMMING_LANGUAGES:
                lang = file_lang
        
        fallback += f"**Snippet {i+1}:**\n```{lang}\n{snippet}\n```\n\n"
    
    fallback += FALLBACK_MESSAGES["next_steps"]
    
    return fallback
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
