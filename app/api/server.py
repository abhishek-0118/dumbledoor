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


@asynccontextmanager
async def lifespan(app: FastAPI):
	global _indexer, _cfg
	_cfg = load_config()
	_indexer = RepoIndexer(_cfg)
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

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
	"""Health check endpoint"""
	return {"status": "ok", "indexed_repos": len(_indexer.indexed_repos), "total_documents": _indexer.store.doc_count}

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
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control",
            "Content-Type": "text/event-stream"
        }
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
	
	context_summary = create_context_summary(docs)

	ret = build_retriever(_indexer.store, _cfg, payload.repo, len(docs), payload.alpha)
	
	enhanced_query = create_enhanced_prompt(payload.q, query_analysis, context_summary, payload.detailed_response)
	
	try:
		# Create LLM based on current method configuration
		llm = create_chat_llm(_cfg.chat, _cfg.current_method)
		
		# Estimate token cost for the query
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
		
		# Estimate response token cost
		response_cost = estimate_token_cost(answer, _cfg.chat.model, _cfg.chat.provider)
		logger.info(f"Response token estimate: {response_cost}")
		
	except Exception as e:
		logger.error(f"LLM query failed: {e}")
		answer = create_fallback_answer(docs, payload.q, query_analysis)
	
	# Enhanced source information
	sources = []
	for i, doc in enumerate(docs[:payload.k]):
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
            "message": "Initializing search and analysis..."
        }, "status")
        
        # Analyze query
        yield create_sse_message({
            "status": "analyzing",
            "message": "Analyzing query and preparing search..."
        }, "status")
        
        query_analysis = analyze_query(payload.q)
        yield create_sse_message({
            "query_analysis": query_analysis
        }, "analysis")
        
        # Search for documents
        yield create_sse_message({
            "status": "searching",
            "message": "Searching through codebase..."
        }, "status")
        
        docs = enhanced_search(
            _indexer.store, 
            payload.q, 
            _cfg, 
            payload.repo, 
            payload.k,
            payload.include_context
        )
        
        # Send context summary
        context_summary = create_context_summary(docs)
        yield create_sse_message({
            "context_summary": context_summary,
            "total_sources_found": len(docs)
        }, "context")
        
        # Format and send sources
        sources = []
        for i, doc in enumerate(docs[:payload.k]):
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
            "message": "Generating comprehensive answer..."
        }, "status")
        
        ret = build_retriever(_indexer.store, _cfg, payload.repo, len(docs), payload.alpha)
        enhanced_query = create_enhanced_prompt(payload.q, query_analysis, context_summary, payload.detailed_response)
        
        try:
            # Create LLM based on current method configuration
            llm = create_chat_llm(_cfg.chat, _cfg.current_method)
            
            # Estimate token cost for the query
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
            
            # Estimate response token cost
            if answer:
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
                    "chunk": "I couldn't generate a proper response. Please try rephrasing your question.",
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
            "message": "Answer generation completed successfully"
        }, "status")
        
    except Exception as e:
        logger.error(f"Streaming failed: {e}")
        yield create_sse_message({
            "error": str(e),
            "status": "error",
            "message": f"An error occurred: {str(e)}"
        }, "error")

def analyze_query(query: str) -> Dict[str, Any]:
	"""Analyze the query to understand what the user is looking for"""
	query_lower = query.lower()
	
	analysis = {
		"is_how_to": any(phrase in query_lower for phrase in ["how to", "how do", "how can", "how should"]),
		"is_what_is": any(phrase in query_lower for phrase in ["what is", "what does", "what are"]),
		"is_debugging": any(phrase in query_lower for phrase in ["error", "bug", "issue", "problem", "fix", "debug"]),
		"is_implementation": any(phrase in query_lower for phrase in ["implement", "create", "build", "develop", "code"]),
		"is_architectural": any(phrase in query_lower for phrase in ["architecture", "structure", "design", "pattern", "overview"]),
		"mentions_specific_tech": [],
		"code_related": any(phrase in query_lower for phrase in ["function", "class", "method", "variable", "import", "module"]),
		"test_related": any(phrase in query_lower for phrase in ["test", "testing", "unit test", "spec"]),
		"config_related": any(phrase in query_lower for phrase in ["config", "configuration", "settings", "setup"]),
	}
	
	# Detect specific technologies mentioned
	tech_keywords = [
		"python", "javascript", "typescript", "react", "node", "go", "java", 
		"rust", "docker", "kubernetes", "api", "database", "sql", "mongodb",
		"fastapi", "flask", "django", "express", "vue", "angular"
	]
	
	for tech in tech_keywords:
		if tech in query_lower:
			analysis["mentions_specific_tech"].append(tech)
	
	return analysis

def create_enhanced_prompt(original_query: str, analysis: Dict[str, Any], context_summary: Dict[str, Any], detailed: bool) -> str:
	"""Create an enhanced prompt with better context"""
	
	enhanced_parts = [original_query]
	
	# Add context about the codebase
	if context_summary.get("languages"):
		enhanced_parts.append(f"Context: Looking at code in {', '.join(context_summary['languages'])} from {context_summary.get('total_documents', 0)} files")
	
	# Add specific instructions based on query analysis
	if analysis.get("is_how_to"):
		enhanced_parts.append("Please provide step-by-step instructions with code examples.")
	elif analysis.get("is_what_is"):
		enhanced_parts.append("Please provide a clear explanation with examples from the codebase.")
	elif analysis.get("is_debugging"):
		enhanced_parts.append("Please analyze the code for potential issues and suggest fixes.")
	elif analysis.get("is_implementation"):
		enhanced_parts.append("Please provide implementation details with relevant code patterns from the codebase.")
	elif analysis.get("is_architectural"):
		enhanced_parts.append("Please provide an architectural overview showing how components interact.")
	
	if detailed:
		enhanced_parts.append("Please provide a comprehensive answer with code examples and explanations.")
	
	return " ".join(enhanced_parts)

def create_code_aware_prompt(detailed: bool, analysis: Dict[str, Any]):
	"""Create a code-aware prompt template for the LLM with markdown formatting"""
	from langchain.prompts import PromptTemplate
	
	base_template = """You are an expert software engineer and code analyst. Use the following pieces of code and documentation to answer the question comprehensively.

Context from codebase:
{context}

Question: {question}

Instructions:
- Provide specific, actionable answers based on the actual code in the context
- Format your response in clear, well-structured Markdown
- Use proper code blocks with language identifiers (```python, ```go, ```javascript, etc.)
- Include relevant code snippets and examples in properly formatted code blocks
- Use headers (##, ###) to organize your response
- Use bullet points or numbered lists for step-by-step instructions
- Use **bold** for important concepts and *italics* for emphasis
- Explain how different parts of the codebase work together
- If the question is about implementation, show concrete examples with proper syntax highlighting
- If the question is about debugging, analyze the code for potential issues and provide solutions
- Reference specific files, functions, and modules when relevant
- Use blockquotes (>) for important notes or warnings
- Maintain accuracy and don't make assumptions beyond what's in the context

"""
	
	if detailed:
		base_template += """
- Provide a comprehensive explanation with multiple examples
- Explain the underlying concepts and patterns using clear markdown structure
- Show how the code fits into the larger architecture with diagrams if helpful
- Include best practices and potential improvements in organized sections
- Use tables for comparisons when appropriate
"""
	
	if analysis.get("code_related"):
		base_template += "- Focus on code structure, functions, classes, and their relationships with proper markdown formatting\n"
	
	if analysis.get("test_related"):
		base_template += "- Pay special attention to test files and testing patterns, format test examples clearly\n"

	if analysis.get("is_architectural"):
		base_template += "- Provide architectural overview with clear section headers and organized structure\n"
	
	base_template += """
Format your response as clean, readable Markdown. Use appropriate headers, code blocks, and formatting to make the information easy to understand and scan.

Answer:"""
	
	return PromptTemplate(
		template=base_template,
		input_variables=["context", "question"]
	)

def create_fallback_answer(docs: List[Document], query: str, analysis: Dict[str, Any]) -> str:
	"""Create a fallback answer when LLM fails, formatted as markdown"""
	if not docs:
		return "## No Results Found\n\nI couldn't find relevant information in the codebase to answer your question.\n\n**Suggestions:**\n- Try rephrasing your question\n- Use more specific terms\n- Check if the code you're looking for exists in the indexed repositories"
	
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
	
	fallback = f"## Search Results\n\nBased on the codebase analysis, I found relevant information but couldn't generate a complete AI response.\n\n"
	
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
			if file_lang in ['python', 'go', 'javascript', 'typescript', 'java', 'rust', 'cpp', 'c']:
				lang = file_lang
		
		fallback += f"**Snippet {i+1}:**\n```{lang}\n{snippet}\n```\n\n"
	
	fallback += "### Next Steps\n\n"
	fallback += "- Review the files mentioned above for detailed implementation\n"
	fallback += "- Try rephrasing your question for better AI analysis\n"
	fallback += "- Ask more specific questions about particular functions or modules\n"
	
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
