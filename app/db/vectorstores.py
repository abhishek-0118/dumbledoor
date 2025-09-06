from typing import Any
try:
	from langchain_chroma import Chroma
except ImportError:
	from langchain_community.vectorstores import Chroma
from ..config.models import BackendConfig


def create_vectorstore(cfg: BackendConfig, embedding_fn: Any, dim: int):
	collection_name = f"{cfg.collection_name}_{dim}"
	return Chroma(
		collection_name=collection_name,
		embedding_function=embedding_fn,
		persist_directory=cfg.chroma_dir,
	)
