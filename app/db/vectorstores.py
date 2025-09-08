from typing import Any
from pathlib import Path
try:
	from langchain_chroma import Chroma
except ImportError:
	from langchain_community.vectorstores import Chroma
from ..config.models import BackendConfig


def create_vectorstore(cfg: BackendConfig, embedding_fn: Any, dim: int):
	collection_name = f"{cfg.collection_name}_{dim}"
	
	# Ensure the chroma directory exists
	chroma_dir = Path(cfg.chroma_dir)
	chroma_dir.mkdir(parents=True, exist_ok=True)
	
	return Chroma(
		collection_name=collection_name,
		embedding_function=embedding_fn,
		persist_directory=cfg.chroma_dir,
	)
