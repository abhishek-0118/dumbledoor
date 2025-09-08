import logging
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any
from fastapi import FastAPI
from pydantic import BaseModel
from ..config.loader import load_config
from ..indexing.indexer import RepoIndexer
from ..search.retrieval import build_retriever, apply_cross_encoder_rerank, enhanced_search, create_context_summary
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


@app.post("/ask", response_model=AskOut)
async def ask(payload: AskIn):
	"""Enhanced ask endpoint with better code understanding and context"""
	
	# Analyze the query for better understanding
	query_analysis = analyze_query(payload.q)
	
	# Enhanced search with more context
	docs = enhanced_search(
		_indexer.store, 
		payload.q, 
		_cfg, 
		payload.repo, 
		payload.k,
		payload.include_context
	)
	
	# Create context summary
	context_summary = create_context_summary(docs)
	
	# Build enhanced retriever for LLM
	ret = build_retriever(_indexer.store, _cfg, payload.repo, len(docs), payload.alpha)
	
	# Prepare enhanced prompt with better context
	enhanced_query = create_enhanced_prompt(payload.q, query_analysis, context_summary, payload.detailed_response)
	
	try:
		from langchain_google_genai import ChatGoogleGenerativeAI
		llm = ChatGoogleGenerativeAI(
			model=_cfg.gemini_chat_model, 
			google_api_key=_cfg.gemini_api_key, 
			temperature=0.1
		)
		
		# Create a custom QA chain with enhanced context
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
	"""Create a code-aware prompt template for the LLM"""
	from langchain.prompts import PromptTemplate
	
	base_template = """You are an expert software engineer and code analyst. Use the following pieces of code and documentation to answer the question comprehensively.

Context from codebase:
{context}

Question: {question}

Instructions:
- Provide specific, actionable answers based on the actual code in the context
- Include relevant code snippets and examples
- Explain how different parts of the codebase work together
- If the question is about implementation, show concrete examples
- If the question is about debugging, analyze the code for potential issues
- Reference specific files, functions, and modules when relevant
- Maintain accuracy and don't make assumptions beyond what's in the context

"""
	
	if detailed:
		base_template += """
- Provide a comprehensive explanation with multiple examples
- Explain the underlying concepts and patterns
- Show how the code fits into the larger architecture
- Include best practices and potential improvements
"""
	
	if analysis.get("code_related"):
		base_template += "- Focus on code structure, functions, classes, and their relationships\n"
	
	if analysis.get("test_related"):
		base_template += "- Pay special attention to test files and testing patterns\n"
	
	base_template += "\nAnswer:"
	
	return PromptTemplate(
		template=base_template,
		input_variables=["context", "question"]
	)


def create_fallback_answer(docs: List[Document], query: str, analysis: Dict[str, Any]) -> str:
	"""Create a fallback answer when LLM fails"""
	if not docs:
		return "I couldn't find relevant information in the codebase to answer your question."
	
	# Extract key information from documents
	relevant_files = []
	key_snippets = []
	
	for doc in docs[:5]:
		metadata = doc.metadata or {}
		if metadata.get("path"):
			relevant_files.append(f"{metadata.get('repo', 'unknown')}/{metadata['path']}")
		
		# Extract a meaningful snippet
		content = doc.page_content
		if len(content) > 200:
			key_snippets.append(content[:200] + "...")
		else:
			key_snippets.append(content)
	
	fallback = f"Based on the codebase analysis, I found relevant information in the following files:\n\n"
	
	for i, file_path in enumerate(relevant_files):
		fallback += f"{i+1}. {file_path}\n"
	
	fallback += f"\nKey code snippets related to your query:\n\n"
	
	for i, snippet in enumerate(key_snippets[:3]):
		fallback += f"```\n{snippet}\n```\n\n"
	
	fallback += "For a more detailed analysis, please try rephrasing your question or check the individual files mentioned above."
	
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
