import logging
import json
import subprocess
import re
from pathlib import Path
from typing import List, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from ..config.models import AppConfig
from ..core.embeddings import create_embedding_fn
from ..db.vectorstores import create_vectorstore
from ..repos.github import clone_or_pull

CODE_EXTS = set(".py .js .ts .tsx .jsx .go .java .kt .rs .c .cpp .h .hpp .rb .php .scala .sql .sh .md .yml .yaml .toml .ini".split())
logger = logging.getLogger("app.indexing.indexer")


def _should_index(path: Path, includes: List[str], excludes: List[str], max_mb: float) -> bool:
	if path.is_dir():
		return False
	if path.suffix.lower() not in CODE_EXTS:
		return False
	s = str(path)
	if includes and not any(Path(".").joinpath(s).match(g.strip()) for g in includes):
		return False
	if excludes and any(Path(".").joinpath(s).match(g.strip()) for g in excludes):
		return False
	try:
		if path.stat().st_size > max_mb * 1024 * 1024:
			return False
	except Exception:
		return False
	return True


class RepoIndexer:
	def __init__(self, cfg: AppConfig):
		self.cfg = cfg
		self.splitter = RecursiveCharacterTextSplitter(
			chunk_size=cfg.indexing.chunk_size,
			chunk_overlap=cfg.indexing.chunk_overlap,
		)
		self.embeddings, self.dim = create_embedding_fn(cfg.embedding)
		self.store = create_vectorstore(cfg.backend, self.embeddings, self.dim)

	def _docs_from_dir(self, repo_dir: Path, repo_name: str) -> List[Document]:
		docs: List[Document] = []
		for p in repo_dir.rglob("*"):
			if not _should_index(p, self.cfg.indexing.include_globs, self.cfg.indexing.exclude_globs, self.cfg.indexing.max_file_mb):
				continue
			try:
				text = p.read_text(errors="ignore")
			except Exception:
				continue
			md = {
				"repo": repo_name,
				"repo_name": repo_name,
				"path": p.relative_to(repo_dir).as_posix(),
				"full_path": str(p),
				"file_type": p.suffix.lower(),
				"repo_folder": repo_name,
			}
			docs.append(Document(page_content=text, metadata=md))
		return self.splitter.split_documents(docs)

	def index_local_root(self, local_root: Path):
		repo_dirs = [d for d in local_root.iterdir() if d.is_dir() and not d.name.startswith('.')]
		all_chunks: List[Document] = []
		repo_names: List[str] = []
		for repo_dir in sorted(repo_dirs):
			repo_name = repo_dir.name
			repo_names.append(repo_name)
			logger.info(f"Indexing repository: {repo_name}")
			chunks = self._docs_from_dir(repo_dir, repo_name)
			logger.info(f"{repo_name}: {len(chunks)} chunks")
			all_chunks.extend(chunks)
		if all_chunks:
			bs = self.cfg.indexing.batch_size
			logger.info(f"Processing {len(all_chunks)} chunks in batches of {bs}")
			for i in range(0, len(all_chunks), bs):
				batch = all_chunks[i:i+bs]
				logger.info(f"Adding batch {i//bs + 1}/{(len(all_chunks)+bs-1)//bs} ({len(batch)} chunks)")
				self.store.add_documents(batch)
			logger.info(f"Indexed {len(all_chunks)} chunks across {len(repo_names)} repos: {', '.join(repo_names)}")

	def index_from_urls_file(self, urls_file: Path):
		root = Path(self.cfg.indexing.local_repo_root).resolve()
		root.mkdir(parents=True, exist_ok=True)
		all_chunks: List[Document] = []
		for url in [l.strip() for l in Path(urls_file).read_text().splitlines() if l.strip() and not l.strip().startswith('#')]:
			try:
				repo_dir = clone_or_pull(
					url, 
					root, 
					self.cfg.indexing.github_token_env,
					self.cfg.indexing.github_auth_required
				)
				chunks = self._docs_from_dir(repo_dir, repo_dir.name)
				logger.info(f"{repo_dir.name}: {len(chunks)} chunks")
				all_chunks.extend(chunks)
			except Exception as e:
				logger.error(f"Failed to process repository {url}: {e}")
				continue
		if all_chunks:
			bs = self.cfg.indexing.batch_size
			for i in range(0, len(all_chunks), bs):
				self.store.add_documents(all_chunks[i:i+bs])

	def index_local_paths(self, paths: List[str]):
		all_chunks: List[Document] = []
		repo_names: List[str] = []
		for p in paths:
			repo_dir = Path(p).resolve()
			if not repo_dir.exists():
				logger.warning(f"Path not found: {repo_dir}")
				continue
			repo_name = repo_dir.name
			repo_names.append(repo_name)
			logger.info(f"Indexing directory: {repo_dir}")
			chunks = self._docs_from_dir(repo_dir, repo_name)
			logger.info(f"{repo_name}: {len(chunks)} chunks")
			all_chunks.extend(chunks)
		if all_chunks:
			bs = self.cfg.indexing.batch_size
			for i in range(0, len(all_chunks), bs):
				self.store.add_documents(all_chunks[i:i+bs])

	def index_repo_urls(self, urls: List[str]):
		root = Path(self.cfg.indexing.local_repo_root).resolve()
		root.mkdir(parents=True, exist_ok=True)
		
		for url in urls:
			try:
				repo_dir = clone_or_pull(
					url, 
					root, 
					self.cfg.indexing.github_token_env,
					self.cfg.indexing.github_auth_required
				)
				repo_name = repo_dir.name
				state_path = repo_dir / ".index_state.json"
				prev_head = None
				if state_path.exists():
					try:
						prev = json.loads(state_path.read_text())
						prev_head = prev.get("last_indexed_commit")
					except Exception:
						prev_head = None
				# get current HEAD commit
				try:
					curr_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_dir).decode().strip()
				except Exception:
					curr_head = None
				# If first time or unable to get previous, do full index
				if not prev_head or not curr_head or prev_head == "":
					logger.info(f"Full indexing for {repo_name}")
					chunks = self._docs_from_dir(repo_dir, repo_name)
					bs = self.cfg.indexing.batch_size
					for i in range(0, len(chunks), bs):
						self.store.add_documents(chunks[i:i+bs])
				else:
					if prev_head == curr_head:
						logger.info(f"No changes in {repo_name}; skipping re-embedding")
					else:
						# compute changed files between commits
						try:
							diff_output = subprocess.check_output(["git", "diff", "--name-status", prev_head, curr_head], cwd=repo_dir).decode()
							changed = []
							for line in diff_output.splitlines():
								parts = line.split("\t")
								if not parts:
									continue
								status = parts[0].strip()
								# handle rename (R100\told\tnew): take new path
								if status.startswith("R") and len(parts) == 3:
									changed.append(parts[2].strip())
								elif len(parts) >= 2:
									changed.append(parts[1].strip())
							logger.info(f"{repo_name}: {len(changed)} changed files since last index")
							self.upsert_changed_files(repo_dir, repo_name, changed)
						except Exception as e:
							logger.exception(f"Failed incremental update for {repo_name}, falling back to full index: {e}")
							chunks = self._docs_from_dir(repo_dir, repo_name)
							bs = self.cfg.indexing.batch_size
							for i in range(0, len(chunks), bs):
								self.store.add_documents(chunks[i:i+bs])
				# persist state
				try:
					(state_path).write_text(json.dumps({"last_indexed_commit": curr_head or ""}))
				except Exception:
					pass
			except Exception as e:
				logger.error(f"Failed to process repository {url}: {e}")
				continue

	def upsert_changed_files(self, repo_dir: Path, repo_name: str, changed_files: List[str]):
		new_docs: List[Document] = []
		ids_to_delete: List[str] = []
		for rel in changed_files:
			f = Path(repo_dir) / rel
			doc_id = f"{repo_name}:{rel}"
			if not f.exists():
				ids_to_delete.append(doc_id)
				continue
			if not _should_index(f, self.cfg.indexing.include_globs, self.cfg.indexing.exclude_globs, self.cfg.indexing.max_file_mb):
				continue
			try:
				text = f.read_text(errors="ignore")
			except Exception:
				continue
			new_docs.append(Document(page_content=text, metadata={"repo": repo_name, "path": rel}, id=doc_id))
		if new_docs:
			chunks = RecursiveCharacterTextSplitter(
				chunk_size=self.cfg.indexing.chunk_size,
				chunk_overlap=self.cfg.indexing.chunk_overlap,
			).split_documents(new_docs)
			self.store.add_documents(chunks)
