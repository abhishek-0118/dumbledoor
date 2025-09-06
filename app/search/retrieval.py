import logging
import os
from typing import Optional, List
from langchain.schema import Document
from ..config.models import AppConfig

logger = logging.getLogger("app.search.retrieval")


def build_retriever(store, cfg: AppConfig, repo_prefix: Optional[str], k: int, alpha: float):
	search_k = k
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


def apply_cross_encoder_rerank(docs: List[Document], query: str, top_k: int, cfg: AppConfig) -> List[Document]:
	if not cfg.retrieval.use_reranker:
		return docs[:top_k]
	try:
		from sentence_transformers import CrossEncoder
		ce = CrossEncoder(cfg.retrieval.cross_encoder_model)
		scores = ce.predict([(query, d.page_content) for d in docs])
		ranked = sorted(zip(docs, scores), key=lambda x: float(x[1]), reverse=True)
		return [d for d,_ in ranked[:top_k]]
	except Exception:
		return docs[:top_k]
