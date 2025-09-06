import logging
from contextlib import asynccontextmanager
from typing import Optional, List, Dict
from fastapi import FastAPI
from pydantic import BaseModel
from ..config.loader import load_config
from ..indexing.indexer import RepoIndexer
from ..search.retrieval import build_retriever, apply_cross_encoder_rerank
from langchain.chains import RetrievalQA
from langchain.schema import Document
from pathlib import Path
from ..repos.github import clone_or_pull

logger = logging.getLogger("app.api.server")


class AskIn(BaseModel): 
	q: str
	k: int = 12
	alpha: float = 0.3
	repo: Optional[str] = None
	architectural: bool = False


class AskOut(BaseModel):
	answer: str
	sources: List[Dict]


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
	ret = build_retriever(_indexer.store, _cfg, payload.repo, payload.k, payload.alpha)
	docs = ret.get_relevant_documents(payload.q)
	docs = apply_cross_encoder_rerank(docs, payload.q, payload.k, _cfg)
	from langchain_google_genai import ChatGoogleGenerativeAI
	llm = ChatGoogleGenerativeAI(model=_cfg.gemini_chat_model, google_api_key=_cfg.gemini_api_key, temperature=0.1)
	qa = RetrievalQA.from_chain_type(llm=llm, retriever=ret, return_source_documents=True)
	res = qa.invoke({"query": payload.q})
	answer = res.get("result", "")
	srcs = []
	for d in docs[:payload.k]:
		md = d.metadata or {}
		srcs.append({
			"repo": md.get("repo"),
			"repo_name": md.get("repo_name", md.get("repo")),
			"path": md.get("path"),
			"file_type": md.get("file_type"),
			"preview": d.page_content[:500]
		})
	return AskOut(answer=answer, sources=srcs)


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
