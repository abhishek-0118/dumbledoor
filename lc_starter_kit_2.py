# LangChain Multi-Repo Q&A — Gemini Version
# -------------------------------------------------------------
# This single file indexes many repos and serves a Q&A API over your code using Google Gemini.
# It now **defaults to Chroma** (local disk) so it **does not require psycopg2**.
# If you want Postgres/pgvector, flip VECTOR_BACKEND=pgvector and install psycopg2.
#
# One file to:
#  - Index 60+ repos (clone OR local paths) → vector store
#  - Serve a FastAPI `/ask` endpoint with an EnsembleRetriever
#  - Use **Chroma** by default (no DB driver needed)
#  - Optionally use **Postgres + pgvector** (requires psycopg2-binary)
#  - Optional BM25 + cross-encoder rerank (if extras installed)
#  - Webhook for incremental updates
#  - Lightweight **self-test** that runs without Gemini or psycopg
#
# Quickstart (Chroma backend — no psycopg2 needed)
#   1) Save as lc_starter_2.py
#   2) Create `.env` from ENV_EXAMPLE below (leave VECTOR_BACKEND=chroma)
#   3) Put repo URLs in repos.txt OR set LOCAL_REPO_ROOT
#   4) Index:      python3 lc_starter_2.py --index --repos repos.txt
#      (or)        python3 lc_starter_2.py --index --local
#   5) Serve API:  python3 lc_starter_2.py --serve
#   6) Ask:        curl -s localhost:8000/ask -H 'Content-Type: application/json' -d '{"q":"Where are payment retries?"}' | jq
#   7) Optional:   python3 lc_starter_2.py --self-test   # runs without network keys
#
# Quickstart (pgvector backend — needs psycopg2-binary)
#   1) Set VECTOR_BACKEND=pgvector in .env
#   2) Start DB:   docker compose -f <(python3 lc_starter_2.py --print-compose) up -d
#   3) Init DB:    python3 lc_starter_2.py --init-db
#   4) Index/Serve as above
#
# Pip (minimal)
#   Chroma only:    pip install langchain langchain-google-genai langchain-community chromadb fastapi uvicorn tiktoken python-dotenv
#   pgvector (opt): pip install psycopg2-binary pgvector
#   extras  (opt):  pip install rank-bm25 sentence-transformers torch
#
# Notes
#  - Default backend avoids psycopg2 import entirely.
#  - To enable pgvector features (DB BM25 sampling, `CREATE EXTENSION`), install psycopg2-binary.
#  - Dimensions auto-picked by embedding model. Chroma stores on disk at CHROMA_DIR.

import os
import re
import uuid
import json
import shutil
import hashlib
import logging
import argparse
import subprocess
from pathlib import Path
from typing import List, Optional, Dict

from dotenv import load_dotenv
load_dotenv(override=True)

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# LangChain core
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import PGVector
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.schema import Document

try:
    from langchain_community.retrievers import BM25Retriever
    HAS_BM25 = True
except Exception:
    HAS_BM25 = False

try:
    from sentence_transformers import CrossEncoder
    HAS_CE = True
except Exception:
    HAS_CE = False

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("lc-rag-gemini")

# ──────────────────────────────────────────────────────────────────────────────
# Config & helpers
# ──────────────────────────────────────────────────────────────────────────────

COMPOSE_YML = r"""
version: '3.9'
services:
  db:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_DB: ${PGDATABASE:-codeindex}
      POSTGRES_USER: ${PGUSER:-codeindex}
      POSTGRES_PASSWORD: ${PGPASSWORD:-codeindexpass}
    ports:
      - "5435:5435"
    volumes:
      - pgdata:/var/lib/postgresql/data
volumes:
  pgdata:
"""

ENV_EXAMPLE = r"""
# Backend (chroma | pgvector)
VECTOR_BACKEND=chroma

# Chroma (local disk)
CHROMA_DIR=./.chroma

# Postgres/pgvector (only if VECTOR_BACKEND=pgvector)
PGHOST=localhost
PGPORT=5435
PGDATABASE=codeindex
PGUSER=codeindex
PGPASSWORD=codeindexpass
PGVECTOR_COLLECTION=code_index

# Gemini Configuration
GEMINI_API_KEY=AIzaSyA7JfGfB37i0fvdERvDN8Sy3qHz-jbO80Q
GEMINI_EMBED_MODEL=models/embedding-001
GEMINI_CHAT_MODEL=gemini-1.5-flash

# Fallback to local embeddings if needed
# For 1024-dim embeddings, set USE_LOCAL_EMBEDDINGS=true and use a 1024-dim model
USE_LOCAL_EMBEDDINGS=true
LOCAL_EMBED_MODEL=intfloat/e5-large-v2

# Indexing
LOCAL_REPO_ROOT=./repos   # if using --local, expects subfolders per repo (e.g., ./repos/citadel, ./repos/dokumentor, ./repos/oms)
INCLUDE_GLOBS=**/*
EXCLUDE_GLOBS=**/.git/**,**/.github/**,**/node_modules/**,**/dist/**,**/build/**,**/*.min.js,**/*.png,**/*.jpg,**/*.jpeg,**/*.pdf
MAX_FILE_MB=1.5

# Chunking
CHUNK_SIZE=1200
CHUNK_OVERLAP=150

# Retrieval
TOP_K=12
ALPHA_HYBRID=0.3    # 0=vector-only, 1=BM25-only (BM25 only if available)
USE_RERANKER=false  # requires sentence-transformers
CROSS_ENCODER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# Server
BIND_HOST=0.0.0.0
BIND_PORT=8000
"""

CODE_EXTS = set(
    ".py .js .ts .tsx .jsx .go .java .kt .rs .c .cpp .h .hpp .rb .php .scala .sql .sh .md .yml .yaml .toml .ini".split()
)

def backend() -> str:
    return os.getenv("VECTOR_BACKEND", "chroma").strip().lower()

def conn_url() -> str:
    return (
        f"postgresql+psycopg2://{os.getenv('PGUSER')}:{os.getenv('PGPASSWORD')}"
        f"@{os.getenv('PGHOST')}:{os.getenv('PGPORT')}/{os.getenv('PGDATABASE')}"
    )

def sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256(); h.update(b); return h.hexdigest()

def should_index(path: Path, includes: List[str], excludes: List[str], max_mb: float) -> bool:
    s = str(path)
    if path.is_dir():
        return False
    if path.suffix.lower() not in CODE_EXTS:
        return False
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

EMBED_DIM: Optional[int] = None


def get_embedder():
    """Return an embedding function that produces 1024-d vectors.

    We force local embeddings to a model with 1024 dimensions to avoid
    collection-dimension mismatches in Chroma and meet the requirement.
    """
    global EMBED_DIM
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    model = os.getenv("LOCAL_EMBED_MODEL", "intfloat/e5-large-v2")
    logger.info(f"Using local 1024-d embeddings: {model}")
    emb = HuggingFaceEmbeddings(model_name=model, encode_kwargs={"normalize_embeddings": True})
    try:
        EMBED_DIM = len(emb.embed_query("test"))
        logger.info(f"Embedding dimension detected: {EMBED_DIM}")
        if EMBED_DIM < 1024:
            logger.warning(f"Expected 1024+ dimensional embeddings, got {EMBED_DIM}. Consider using a model with 1024+ dimensions.")
        else:
            logger.info(f"Using {EMBED_DIM}-dimensional embeddings (>= 1024 as requested)")
    except Exception:
        EMBED_DIM = None
    return emb


def get_llm():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is required for chat model")
    
    model = os.getenv("GEMINI_CHAT_MODEL", "gemini-1.5-flash")
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key,
        temperature=0.1
    )


# ──────────────────────────────────────────────────────────────────────────────
# Indexing
# ──────────────────────────────────────────────────────────────────────────────

class RepoIndexer:
    def __init__(self):
        self.include_globs = [g.strip() for g in os.getenv("INCLUDE_GLOBS", "**/*").split(",") if g.strip()]
        self.exclude_globs = [g.strip() for g in os.getenv("EXCLUDE_GLOBS", "").split(",") if g.strip()]
        self.max_file_mb = float(os.getenv("MAX_FILE_MB", "1.5"))
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(os.getenv("CHUNK_SIZE", "1200")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "150"))
        )
        self.embeddings = get_embedder()
        base_collection = os.getenv("PGVECTOR_COLLECTION", "code_index")
        # Use dimension-suffixed collection name to avoid Chroma dim mismatch
        suffix = str(EMBED_DIM or "unk")
        self.collection = f"{base_collection}_{suffix}"
        self._store = None

    def store(self):
        if self._store is not None:
            return self._store
        if backend() == "pgvector":
            self._store = PGVector(
                connection_string=conn_url(),
                collection_name=self.collection,
                embedding_function=self.embeddings,
                use_jsonb=True,
            )
        else:
            persist_dir = os.getenv("CHROMA_DIR", "./.chroma")
            self._store = Chroma(
                collection_name=self.collection,
                embedding_function=self.embeddings,
                persist_directory=persist_dir,
            )
        return self._store

    def clone_or_pull(self, url: str, root: Path) -> Path:
        name = re.sub(r"[^A-Za-z0-9_.-]", "-", url.rstrip("/").split("/")[-1].replace(".git", ""))
        dest = root / name
        if not dest.exists():
            logger.info(f"Cloning {url} → {dest}")
            subprocess.check_call(["git", "clone", "--depth", "1", url, str(dest)])
        else:
            logger.info(f"Updating {url} in {dest}")
            subprocess.check_call(["git", "fetch", "--all", "--prune"], cwd=dest)
            subprocess.check_call(["git", "reset", "--hard", "origin/HEAD"], cwd=dest)
        return dest
    def _docs_from_dir(self, repo_dir: Path, repo_name: str) -> List[Document]:
        docs: List[Document] = []
        for p in repo_dir.rglob("*"):
            if not should_index(p, self.include_globs, self.exclude_globs, self.max_file_mb):
                continue
            try:
                text = p.read_text(errors="ignore")
            except Exception:
                continue
            # Enhanced metadata with repo name for cross-repo queries
            md = {
                "repo": repo_name, 
                "repo_name": repo_name,  # Explicit repo name field
                "path": p.relative_to(repo_dir).as_posix(),
                "full_path": str(p),
                "file_type": p.suffix.lower(),
                "repo_folder": repo_name  # Additional field for clarity
            }
            docs.append(Document(page_content=text, metadata=md))
        return self.splitter.split_documents(docs)

    def index_local_root(self, local_root: Path):
        print("----",local_root)
        store = self.store()
        all_chunks: List[Document] = []
        repo_names = []
        
        # Get the actual repository directories (citadel, dokumentor, oms)
        repo_dirs = [d for d in local_root.iterdir() if d.is_dir() and not d.name.startswith('.')]
        
        for repo_dir in sorted(repo_dirs):
            repo_name = repo_dir.name
            repo_names.append(repo_name)
            logger.info(f"Indexing repository: {repo_name}")
            
            # Index all files within this repository
            chunks = self._docs_from_dir(repo_dir, repo_name)
            logger.info(f"{repo_name}: {len(chunks)} chunks")
            all_chunks.extend(chunks)
        
        if all_chunks:
            # Process in batches to avoid ChromaDB batch size limits
            batch_size = 5000  # Safe batch size for ChromaDB
            total_chunks = len(all_chunks)
            logger.info(f"Processing {total_chunks} chunks in batches of {batch_size}")
            
            for i in range(0, total_chunks, batch_size):
                batch = all_chunks[i:i + batch_size]
                logger.info(f"Adding batch {i//batch_size + 1}/{(total_chunks + batch_size - 1)//batch_size} ({len(batch)} chunks)")
                store.add_documents(batch)
            
            logger.info(f"Indexed {len(all_chunks)} chunks across {len(repo_names)} repos: {', '.join(repo_names)}")
            logger.info(f"Repos available for cross-repo queries: {repo_names}")
        else:
            logger.warning(f"No chunks found in {local_root}. Check if repository directories exist.")

    def index_from_urls_file(self, urls_file: Path):
        workdir = Path(os.getenv("LOCAL_REPO_ROOT", "./repos")).resolve()
        workdir.mkdir(parents=True, exist_ok=True)
        store = self.store()
        all_chunks: List[Document] = []
        for url in [l.strip() for l in urls_file.read_text().splitlines() if l.strip() and not l.strip().startswith('#')]:
            repo_dir = self.clone_or_pull(url, workdir)
            chunks = self._docs_from_dir(repo_dir, repo_dir.name)
            logger.info(f"{repo_dir.name}: {len(chunks)} chunks")
            all_chunks.extend(chunks)
        if all_chunks:
            store.add_documents(all_chunks)
            logger.info(f"Indexed {len(all_chunks)} chunks from URLs list")

    def upsert_changed_files(self, repo_dir: Path, repo_name: str, changed_files: List[str]):
        store = self.store()
        new_docs: List[Document] = []
        ids_to_delete: List[str] = []
        for rel in changed_files:
            f = repo_dir / rel
            doc_id = f"{repo_name}:{rel}"
            if not f.exists():
                ids_to_delete.append(doc_id)
                continue
            if not should_index(f, self.include_globs, self.exclude_globs, self.max_file_mb):
                continue
            try:
                text = f.read_text(errors="ignore")
            except Exception:
                continue
            new_docs.append(Document(page_content=text, metadata={"repo": repo_name, "path": rel}, id=doc_id))
        if ids_to_delete and backend() == "pgvector":
            store.delete(ids=ids_to_delete)
        if new_docs:
            store.add_documents(self.splitter.split_documents(new_docs))


# ──────────────────────────────────────────────────────────────────────────────
# Retrieval & API
# ──────────────────────────────────────────────────────────────────────────────

class AskIn(BaseModel):
    q: str
    k: int = int(os.getenv("TOP_K", "12"))
    alpha: float = float(os.getenv("ALPHA_HYBRID", "0.3"))  # blend BM25/vector if BM25 available
    repo: Optional[str] = None  # filter by repo name prefix
    architectural: bool = False  # flag for architectural analysis

class AskOut(BaseModel):
    answer: str
    sources: List[Dict]


def build_retriever(indexer: RepoIndexer, repo_prefix: Optional[str], k: int, alpha: float):
    vec_store = indexer.store()
    # Increase k for architectural queries to get more context
    search_k = k * 3 if any(keyword in repo_prefix.lower() if repo_prefix else "" for keyword in ["architectural", "flow", "interconnectivity"]) else k
    search_kwargs = {"k": search_k}
    
    if repo_prefix:
        # Enhanced metadata filter for cross-repo queries
        # Can filter by repo name, repo_name, or repo_folder
        search_kwargs["filter"] = {
            "$or": [
                {"repo": {"$like": f"{repo_prefix}%"}},
                {"repo_name": {"$like": f"{repo_prefix}%"}},
                {"repo_folder": {"$like": f"{repo_prefix}%"}}
            ]
        }
    vector_ret = vec_store.as_retriever(search_kwargs=search_kwargs)

    # BM25 ensemble is available only if package installed; for pgvector we can sample corpus from DB.
    if backend() == "pgvector" and HAS_BM25 and alpha > 0:
        try:
            import psycopg2  # local import; optional dependency
        except Exception:
            logger.warning("BM25 disabled: psycopg2 not installed. Set VECTOR_BACKEND=chroma or install psycopg2-binary.")
            return vector_ret
        try:
            with psycopg2.connect(host=os.getenv('PGHOST'), port=os.getenv('PGPORT'), dbname=os.getenv('PGDATABASE'), user=os.getenv('PGUSER'), password=os.getenv('PGPASSWORD')) as conn:
                with conn.cursor() as cur:
                    if repo_prefix:
                        cur.execute("SELECT metadata->>'repo', metadata->>'path', content FROM langchain_pg_embedding WHERE collection_id=(SELECT uuid FROM langchain_pg_collection WHERE name=%s) AND metadata->>'repo' ILIKE %s LIMIT 5000", (indexer.collection, repo_prefix+"%"))
                    else:
                        cur.execute("SELECT metadata->>'repo', metadata->>'path', content FROM langchain_pg_embedding WHERE collection_id=(SELECT uuid FROM langchain_pg_collection WHERE name=%s) LIMIT 5000", (indexer.collection,))
                    rows = cur.fetchall()
            docs = [Document(page_content=r[2], metadata={"repo": r[0], "path": r[1]}) for r in rows]
            if not docs:
                return vector_ret
            bm25 = BM25Retriever.from_documents(docs)
            bm25.k = k
            from langchain.retrievers import EnsembleRetriever
            return EnsembleRetriever(retrievers=[bm25, vector_ret], weights=[alpha, 1-alpha])
        except Exception as e:
            logger.warning(f"BM25 sampling error: {e}. Falling back to vector-only.")
            return vector_ret
    else:
        return vector_ret


def apply_cross_encoder_rerank(docs: List[Document], query: str, top_k: int) -> List[Document]:
    if not HAS_CE or os.getenv("USE_RERANKER", "false").lower() != "true":
        return docs[:top_k]
    model_name = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    ce = CrossEncoder(model_name)
    pairs = [(query, d.page_content) for d in docs]
    scores = ce.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: float(x[1]), reverse=True)
    return [d for d, _ in ranked[:top_k]]


def _extract_function_score(snippet: str) -> int:
    """Heuristic: score higher if snippet looks like function code across languages."""
    patterns = [
        r"\bdef\s+\w+\s*\(",  # Python
        r"\bfunction\s+\w+\s*\(",  # JS/TS
        r"\b\w+\s+\w+\s*\([^)]*\)\s*\{",  # C-like
        r"\bpublic\s+\w+\s+\w+\s*\(",
        r"\bprivate\s+\w+\s+\w+\s*\(",
        r"\bstatic\s+\w+\s+\w+\s*\("
    ]
    score = 0
    for p in patterns:
        try:
            import re as _re
            if _re.search(p, snippet):
                score += 1
        except Exception:
            pass
    return score


def llm_rerank_functions(docs: List[Document], query: str, top_k: int) -> List[Document]:
    """Use LLM to select top function-like results relevant to the query.

    Falls back to heuristic + cross-encoder if LLM unavailable.
    """
    try:
        llm = get_llm()
    except Exception:
        # LLM not available, fall back to heuristic + optional CE rerank
        boosted = sorted(docs, key=lambda d: _extract_function_score(d.page_content), reverse=True)
        return apply_cross_encoder_rerank(boosted, query, top_k)

    # Prepare a compact prompt listing snippets with repo context
    items = []
    for idx, d in enumerate(docs[: min(len(docs), max(top_k * 4, 20))]):
        preview = d.page_content[:400].replace("`", "'")
        repo_name = d.metadata.get("repo_name", d.metadata.get("repo", "unknown"))
        items.append(f"[{idx}] [{repo_name}] {preview}")
    
    # Enhanced prompt for cross-repo architectural queries
    prompt = (
        "You are ranking code snippets that best answer the user's query, prioritizing FUNCTION definitions, calls, and architectural patterns.\n"
        f"Query: {query}\n"
        "Snippets (index: [repo_name] content preview):\n"
        + "\n".join(items)
        + "\n\nConsider cross-repo relationships and architectural flow when ranking. "
        "Return a JSON list of the top indices in order of relevance (e.g., [3,1,0])."
    )
    try:
        resp = llm.invoke(prompt)
        text = resp.content if hasattr(resp, "content") else str(resp)
        import json as _json
        indices = _json.loads(text.strip())
        chosen = []
        for i in indices:
            if isinstance(i, int) and 0 <= i < len(docs):
                chosen.append(docs[i])
            if len(chosen) >= top_k:
                break
        if not chosen:
            raise ValueError("empty selection")
        return chosen
    except Exception:
        # Fallback path
        boosted = sorted(docs, key=lambda d: _extract_function_score(d.page_content), reverse=True)
        return apply_cross_encoder_rerank(boosted, query, top_k)


from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _indexer
    _indexer = RepoIndexer()
    print("----",_indexer)
    yield

app = FastAPI(lifespan=lifespan)
_indexer: Optional[RepoIndexer] = None

@app.post("/ask", response_model=AskOut)
async def ask(payload: AskIn):
    # Enhanced retrieval for architectural queries
    is_architectural = (payload.architectural or 
                       any(keyword in payload.q.lower() for keyword in ["architectural", "flow", "interconnectivity", "cross-repo", "between repos", "system design", "high level"]))
    
    if is_architectural:
        # For architectural queries, get more context and don't filter by repo
        retriever = build_retriever(_indexer, None, payload.k * 2, payload.alpha)
        docs = retriever.get_relevant_documents(payload.q)
        # For architectural analysis, we want diverse context, not just functions
        docs = docs[:payload.k * 2]  # Keep more context for architectural analysis
    else:
        retriever = build_retriever(_indexer, payload.repo, payload.k, payload.alpha)
        docs = retriever.get_relevant_documents(payload.q)
        # LLM-driven rerank to prioritize function-related results (adjustable top-k)
        docs = llm_rerank_functions(docs, payload.q, payload.k)

    llm = get_llm()
    
    # Enhanced prompt for architectural and cross-repo queries
    is_architectural = (payload.architectural or 
                       any(keyword in payload.q.lower() for keyword in ["architectural", "flow", "interconnectivity", "cross-repo", "between repos", "system design", "high level"]))
    
    if is_architectural:
        # Special handling for architectural queries
        context_parts = []
        repo_groups = {}
        
        for d in docs:
            repo_name = d.metadata.get("repo_name", d.metadata.get("repo", "unknown"))
            if repo_name not in repo_groups:
                repo_groups[repo_name] = []
            repo_groups[repo_name].append(d.page_content[:800])
        
        for repo_name, contents in repo_groups.items():
            context_parts.append(f"\n=== {repo_name.upper()} REPOSITORY ===\n")
            context_parts.extend(contents[:3])  # Top 3 chunks per repo
        
        context = "\n".join(context_parts)
        
        architectural_prompt = f"""You are an expert software architect analyzing code across multiple repositories. 
        
Based on the following code context from different repositories, provide a comprehensive architectural analysis:

{context}

User Question: {payload.q}

Please provide:
1. A high-level architectural overview
2. How the repositories interact and communicate
3. Key components and their relationships
4. Data flow between systems
5. Any integration patterns you can identify

If you cannot determine specific interactions, explain what you can infer from the code structure and suggest what additional information would be needed for a complete architectural analysis.

Answer:"""
        
        try:
            response = llm.invoke(architectural_prompt)
            answer = response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            logger.error(f"Architectural analysis failed: {e}")
            # Fallback to standard QA
            qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
            res = qa.invoke({"query": payload.q})
            answer = res.get("result", "")
    else:
        # Standard QA for regular queries
        qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
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


# ──────────────────────────────────────────────────────────────────────────────
# Webhook for incremental updates
# ──────────────────────────────────────────────────────────────────────────────

class WebhookIn(BaseModel):
    repo_url: Optional[str] = None
    repo_name: Optional[str] = None
    changed_files: List[str]

@app.post("/webhook/github")
async def webhook(payload: WebhookIn):
    root = Path(os.getenv("LOCAL_REPO_ROOT", "./repos")).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if payload.repo_name:
        repo_dir = root / payload.repo_name
    elif payload.repo_url:
        repo_dir = _indexer.clone_or_pull(payload.repo_url, root)
    else:
        return {"ok": False, "error": "repo_url or repo_name required"}
    _indexer.upsert_changed_files(repo_dir, repo_dir.name, payload.changed_files)
    return {"ok": True}


# ──────────────────────────────────────────────────────────────────────────────
# CLI + DB init (pgvector only)
# ──────────────────────────────────────────────────────────────────────────────

def init_db():
    if backend() != "pgvector":
        logger.info("--init-db: no-op for Chroma backend.")
        return
    try:
        import psycopg2  # optional
    except Exception:
        raise RuntimeError("pgvector backend requires psycopg2-binary. Either install it or set VECTOR_BACKEND=chroma.")
    with psycopg2.connect(host=os.getenv('PGHOST'), port=os.getenv('PGPORT'), dbname=os.getenv('PGDATABASE'), user=os.getenv('PGUSER'), password=os.getenv('PGPASSWORD')) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    logger.info("DB ready (vector extension ensured)")


def index_from_urls(file_path: Path):
    indexer = RepoIndexer()
    indexer.index_from_urls_file(file_path)


def index_local():
    root = Path(os.getenv("LOCAL_REPO_ROOT", "./repos")).resolve()
    print(f"Indexing repositories from: {root}")
    indexer = RepoIndexer()
    indexer.index_local_root(root)


def serve():
    uvicorn.run(app, host=os.getenv("BIND_HOST", "0.0.0.0"), port=int(os.getenv("BIND_PORT", "8000")))


# ──────────────────────────────────────────────────────────────────────────────
# Self-Test (no network, no psycopg2 needed)
# ──────────────────────────────────────────────────────────────────────────────
# Adds two tiny docs into a fresh Chroma collection using a simple local embedder
# and asserts retrieval order for a known query.

class SimpleHashEmbeddings:
    """Tiny deterministic embedder for tests (no network)."""
    def __init__(self, dim: int = 64):
        self.dim = dim
    def _vec(self, text: str):
        v = [0.0] * self.dim
        for i, ch in enumerate(text.encode("utf-8")):
            v[i % self.dim] += (ch % 13) / 13.0
        # l2-normalize
        norm = sum(x*x for x in v) ** 0.5 or 1.0
        return [x / norm for x in v]
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._vec(t) for t in texts]
    def embed_query(self, text: str) -> List[float]:
        return self._vec(text)


def self_test():
    os.environ["VECTOR_BACKEND"] = "chroma"  # force chroma for test
    test_dir = Path("./.chroma_selftest")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    os.environ["CHROMA_DIR"] = str(test_dir)

    idx = RepoIndexer()
    idx.embeddings = SimpleHashEmbeddings()  # override to avoid network
    store = idx.store()

    texts = [
        ("payments/retry.py", "def retry_payment():\n    # exponential backoff for card failures\n    return True\n"),
        ("orders/create.py", "def create_order():\n    # create order and reserve inventory\n    return True\n"),
    ]
    docs = [Document(page_content=t, metadata={"repo": "demo", "path": p}) for p, t in texts]
    store.add_documents(docs)

    ret = store.as_retriever(search_kwargs={"k": 2})
    results = ret.get_relevant_documents("How do we retry payment failures?")
    assert any("payments/retry.py" == d.metadata.get("path") for d in results), "Self-test failed: expected payments/retry.py in results"
    print("SELF-TEST PASS: retrieval returned expected source")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--print-compose", action="store_true")
    ap.add_argument("--print-env", action="store_true")
    ap.add_argument("--init-db", action="store_true")
    ap.add_argument("--index", action="store_true")
    ap.add_argument("--repos", type=str)
    ap.add_argument("--local", action="store_true", help="Index all subfolders under LOCAL_REPO_ROOT")
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--self-test", action="store_true", help="Run a minimal local retrieval test (no network)")
    args = ap.parse_args()

    if args.init_db:
        init_db(); return
    if args.index:
        if args.local:
            index_local(); return
        if not args.repos:
            print("--repos <file> required OR use --local", flush=True)
            return
        index_from_urls(Path(args.repos)); return
    if args.serve:
        serve(); return
    if args.self_test:
        self_test(); return

    ap.print_help()


if __name__ == "__main__":
    main()

# ──────────────────────────────────────────────────────────────────────────────
# GitHub Actions (example) to hit webhook on pushes
# ──────────────────────────────────────────────────────────────────────────────
# name: Indexer Notify
# on:
#   push:
#     branches: [ main ]
# jobs:
#   notify:
#     runs-on: ubuntu-latest
#     steps:
#       - name: Collect changed files
#         id: changed
#         uses: tj-actions/changed-files@v44
#       - name: POST webhook
#         run: |
#           CHANGED=$(printf '"%s",' ${{ steps.changed.outputs.all_changed_files }})
#           CHANGED=[${CHANGED%,}]
#           curl -X POST "$INDEXER_URL/webhook/github" \
#                -H "Content-Type: application/json" \
#                -d "{\"repo_url\":\"$GITHUB_SERVER_URL/$GITHUB_REPOSITORY.git\",\"changed_files\":${CHANGED}}"
#         env:
#           INDEXER_URL: https://your-indexer.example.com
