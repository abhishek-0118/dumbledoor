import logging
import json
import subprocess
import re
import shutil
from pathlib import Path
from typing import List, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from ..config.models import AppConfig
from ..core.embeddings import create_optimized_embedding_fn
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
		# Enhanced text splitter for better code understanding
		self.splitter = RecursiveCharacterTextSplitter(
			chunk_size=cfg.indexing.chunk_size,
			chunk_overlap=cfg.indexing.chunk_overlap,
			separators=[
				"\n\n",  # Paragraph breaks
				"\nclass ",  # Class definitions
				"\ndef ",   # Function definitions
				"\n@",      # Decorators
				"\nif ",    # Control structures
				"\nfor ",
				"\nwhile ",
				"\ntry:",
				"\nwith ",
				"\n",       # Line breaks
				" ",        # Word breaks
				""
			],
			keep_separator=True,
		)
		self.embeddings, self.dim = create_optimized_embedding_fn(cfg.embedding, enable_optimizations=True)
		self.store = create_vectorstore(cfg.backend, self.embeddings, self.dim)

	def cleanup_all(self):
		"""Remove all embeddings and indexes"""
		logger.info("Starting cleanup of all embeddings and indexes")
		
		# Clean up the Chroma directory
		chroma_dir = Path(self.cfg.backend.chroma_dir)
		if chroma_dir.exists():
			logger.info(f"Removing Chroma directory: {chroma_dir}")
			try:
				shutil.rmtree(chroma_dir)
				logger.info("Chroma directory removed successfully")
			except Exception as e:
				logger.error(f"Failed to remove Chroma directory: {e}")
				raise
		else:
			logger.info(f"Chroma directory does not exist: {chroma_dir}")
		
		# Clean up index state files in repository directories
		if self.cfg.app_env != "local":
			repo_root = Path(self.cfg.indexing.local_repo_root).resolve()
			if repo_root.exists():
				logger.info(f"Cleaning up index state files in: {repo_root}")
				state_files_removed = 0
				for state_file in repo_root.rglob(".index_state.json"):
					try:
						state_file.unlink()
						state_files_removed += 1
						logger.debug(f"Removed state file: {state_file}")
					except Exception as e:
						logger.warning(f"Failed to remove state file {state_file}: {e}")
				logger.info(f"Removed {state_files_removed} index state files")
		
		# Recreate the vectorstore for future operations
		self.store = create_vectorstore(self.cfg.backend, self.embeddings, self.dim)
		
		logger.info("Cleanup completed")

	def _docs_from_dir(self, repo_dir: Path, repo_name: str) -> List[Document]:
		docs: List[Document] = []
		for p in repo_dir.rglob("*"):
			if not _should_index(p, self.cfg.indexing.include_globs, self.cfg.indexing.exclude_globs, self.cfg.indexing.max_file_mb):
				continue
			try:
				text = p.read_text(errors="ignore")
			except Exception:
				continue
			
			# Enhanced metadata with code understanding
			rel_path = p.relative_to(repo_dir).as_posix()
			file_type = p.suffix.lower()
			
			# Extract additional context for code files
			language = self._detect_language(file_type)
			file_size = len(text)
			line_count = text.count('\n') + 1
			
			# Add directory context
			dir_parts = str(p.parent.relative_to(repo_dir)).split('/')
			dir_context = ' / '.join(dir_parts) if dir_parts != ['.'] else 'root'
			
			md = {
				"repo": repo_name,
				"repo_name": repo_name,
				"path": rel_path,
				"full_path": str(p),
				"file_type": file_type,
				"language": language,
				"repo_folder": repo_name,
				"directory": dir_context,
				"file_size": file_size,
				"line_count": line_count,
				"is_test": self._is_test_file(rel_path),
				"is_config": self._is_config_file(rel_path),
				"module_name": self._extract_module_name(rel_path, file_type),
			}
			
			# Create enhanced content with file context
			enhanced_content = self._create_enhanced_content(text, p, repo_name, rel_path)
			docs.append(Document(page_content=enhanced_content, metadata=md))
		
		return self.splitter.split_documents(docs)

	def _detect_language(self, file_type: str) -> str:
		"""Detect programming language from file extension"""
		language_map = {
			'.py': 'python',
			'.js': 'javascript', 
			'.ts': 'typescript',
			'.tsx': 'typescript',
			'.jsx': 'javascript',
			'.go': 'go',
			'.java': 'java',
			'.kt': 'kotlin',
			'.rs': 'rust',
			'.c': 'c',
			'.cpp': 'cpp',
			'.h': 'c',
			'.hpp': 'cpp',
			'.rb': 'ruby',
			'.php': 'php',
			'.scala': 'scala',
			'.sql': 'sql',
			'.sh': 'bash',
			'.md': 'markdown',
			'.yml': 'yaml',
			'.yaml': 'yaml',
			'.toml': 'toml',
			'.ini': 'ini',
			'.json': 'json',
			'.xml': 'xml',
			'.html': 'html',
			'.css': 'css',
		}
		return language_map.get(file_type.lower(), 'text')

	def _is_test_file(self, path: str) -> bool:
		"""Check if file is a test file"""
		path_lower = path.lower()
		return any(indicator in path_lower for indicator in [
			'test', 'spec', '__test__', '__spec__', 'tests/', 'spec/',
			'.test.', '.spec.', '_test.', '_spec.'
		])

	def _is_config_file(self, path: str) -> bool:
		"""Check if file is a configuration file"""
		path_lower = path.lower()
		config_indicators = [
			'config', 'conf', 'settings', 'setup', 'makefile', 'dockerfile',
			'.env', '.ini', '.toml', '.yaml', '.yml', '.json', 'package.json',
			'requirements.txt', 'go.mod', 'cargo.toml'
		]
		return any(indicator in path_lower for indicator in config_indicators)

	def _extract_module_name(self, path: str, file_type: str) -> str:
		"""Extract module/class name from file path"""
		if file_type == '.py':
			# Convert path to Python module notation
			module_path = path.replace('/', '.').replace('\\', '.')
			if module_path.endswith('.py'):
				module_path = module_path[:-3]
			if module_path.startswith('.'):
				module_path = module_path[1:]
			return module_path
		elif file_type in ['.js', '.ts', '.jsx', '.tsx']:
			# Extract component/module name
			parts = path.split('/')
			filename = parts[-1]
			if '.' in filename:
				return filename.split('.')[0]
			return filename
		else:
			# Generic file name without extension
			parts = path.split('/')
			filename = parts[-1]
			if '.' in filename:
				return filename.split('.')[0]
			return filename

	def _create_enhanced_content(self, text: str, file_path: Path, repo_name: str, rel_path: str) -> str:
		"""Create enhanced content with additional context for better understanding"""
		
		# Add file header with context
		header = f"""FILE: {rel_path}
REPOSITORY: {repo_name}
LANGUAGE: {self._detect_language(file_path.suffix)}
PATH: {rel_path}

"""
		
		# For code files, try to extract key structural information
		file_type = file_path.suffix.lower()
		if file_type == '.py':
			structure_info = self._extract_python_structure(text)
		elif file_type in ['.js', '.ts', '.jsx', '.tsx']:
			structure_info = self._extract_js_structure(text)
		else:
			structure_info = ""
		
		if structure_info:
			header += f"STRUCTURE:\n{structure_info}\n\n"
		
		return header + text

	def _extract_python_structure(self, text: str) -> str:
		"""Extract Python file structure (classes, functions, imports)"""
		import ast
		try:
			tree = ast.parse(text)
			structure = []
			
			# Extract imports
			imports = []
			for node in ast.walk(tree):
				if isinstance(node, ast.Import):
					for alias in node.names:
						imports.append(f"import {alias.name}")
				elif isinstance(node, ast.ImportFrom):
					module = node.module or ""
					for alias in node.names:
						imports.append(f"from {module} import {alias.name}")
			
			if imports:
				structure.append("IMPORTS: " + ", ".join(imports[:5]))  # Limit to first 5
			
			# Extract classes and functions
			classes = []
			functions = []
			for node in ast.walk(tree):
				if isinstance(node, ast.ClassDef):
					classes.append(node.name)
				elif isinstance(node, ast.FunctionDef):
					functions.append(node.name)
			
			if classes:
				structure.append("CLASSES: " + ", ".join(classes))
			if functions:
				structure.append("FUNCTIONS: " + ", ".join(functions[:10]))  # Limit to first 10
			
			return "\n".join(structure)
		except:
			return ""

	def _extract_js_structure(self, text: str) -> str:
		"""Extract JavaScript/TypeScript file structure"""
		structure = []
		
		# Simple regex-based extraction for imports, exports, functions, classes
		import re
		
		# Extract imports
		import_pattern = r'import\s+.*?from\s+[\'"]([^\'"]+)[\'"]'
		imports = re.findall(import_pattern, text)
		if imports:
			structure.append("IMPORTS: " + ", ".join(imports[:5]))
		
		# Extract exports
		export_pattern = r'export\s+(?:default\s+)?(?:class|function|const|let|var)\s+(\w+)'
		exports = re.findall(export_pattern, text)
		if exports:
			structure.append("EXPORTS: " + ", ".join(exports))
		
		# Extract function declarations
		func_pattern = r'(?:function\s+(\w+)|const\s+(\w+)\s*=\s*(?:\([^)]*\)\s*=>|\([^)]*\)\s*:\s*[^=]*=>))'
		functions = [match[0] or match[1] for match in re.findall(func_pattern, text)]
		if functions:
			structure.append("FUNCTIONS: " + ", ".join(functions[:10]))
		
		# Extract class declarations
		class_pattern = r'class\s+(\w+)'
		classes = re.findall(class_pattern, text)
		if classes:
			structure.append("CLASSES: " + ", ".join(classes))
		
		return "\n".join(structure)

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
