from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Iterable

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings

# Core model for reasoning/tool-calling.
# qwen2.5:3b is more reliable for tool-calling; 1.5b is faster but may
# occasionally fail to invoke tools correctly.
LLM_MODEL = "qwen2.5:3b"

# Maximum tokens the model may generate per response.
# WARNING: setting this too low (e.g. < 256) will cause responses to be
# cut off mid-sentence for any complex answer, code block, or list.
# Raise this if you see truncated replies; lower it only for simple Q&A.
LLM_NUM_PREDICT = 512

# Context window size (input tokens the model evaluates each request).
# This covers: system prompt + retrieved chunks + chat history + question.
# WARNING: setting this too low will silently drop earlier history/context.
# 4096 comfortably fits most conversations; lower only if memory is tight.
LLM_NUM_CTX = 4096

# CPU threads used for inference. Half of logical cores is a safe starting
# point. Increase toward your physical core count for faster generation;
# going above physical cores usually hurts more than it helps.
LLM_NUM_THREAD = max(1, (os.cpu_count() or 4) // 2)

# Shared Ollama generation options used by both CLI and API runtime.
LLM_OPTIONS = {
    "num_predict": LLM_NUM_PREDICT,
    "num_ctx": LLM_NUM_CTX,
    "num_thread": LLM_NUM_THREAD,
}
# Local embedding model for vector search. Pull once: ollama pull nomic-embed-text
EMBED_MODEL = "nomic-embed-text"

BASE_DIR = Path(__file__).resolve().parent
STORAGE_DIR = BASE_DIR / ".agent_store"
CHROMA_DIR = STORAGE_DIR / "chroma"
SQLITE_PATH = STORAGE_DIR / "state.db"

# Add folders you want indexed here.
DEFAULT_SOURCE_DIRS = [BASE_DIR]
INCLUDE_EXTENSIONS = {
    ".py",
    ".md",
    ".txt",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".pdf",
}
IGNORE_DIRS = {".git", ".venv", "__pycache__", ".agent_store", ".vscode"}

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
# Number of vector chunks retrieved per question.
# Lower values reduce prompt size and improve speed; higher values give
# the model more context but increase token cost and latency.
RETRIEVAL_K = 3
# Candidate pool size before reranking. Keep this larger than RETRIEVAL_K.
RETRIEVAL_CANDIDATES = 10
# Toggle lightweight local reranking on/off.
ENABLE_LOCAL_RERANK = True
MAX_HISTORY_TURNS = 6
PROGRESS_EVERY = 25
DEFAULT_THREAD_ID = "legacy-default"
DEFAULT_THREAD_NAME = "Legacy Conversation"


def parse_source_dirs() -> list[Path]:
    raw = os.getenv("AGENT_SOURCE_DIRS", "").strip()
    if not raw:
        return DEFAULT_SOURCE_DIRS

    dirs: list[Path] = []
    for item in raw.split(os.pathsep):
        candidate = item.strip().strip('"')
        if not candidate:
            continue

        path = Path(candidate).expanduser()
        if not path.is_absolute():
            path = (BASE_DIR / path).resolve()

        if path.exists() and path.is_dir():
            dirs.append(path)
        else:
            print(f"Skipping non-existent or invalid source directory: {candidate}")

    return dirs or DEFAULT_SOURCE_DIRS


def ensure_dirs() -> None:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def ensure_chat_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS threads (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )

    thread_columns = {
        row[1] for row in conn.execute("PRAGMA table_info(chat_log)").fetchall()
    }
    if "thread_id" not in thread_columns:
        conn.execute("ALTER TABLE chat_log ADD COLUMN thread_id TEXT")

    conn.execute(
        """
        INSERT OR IGNORE INTO threads(id, name, created_at)
        VALUES (?, ?, ?)
        """,
        (DEFAULT_THREAD_ID, DEFAULT_THREAD_NAME, now_iso()),
    )
    conn.execute(
        """
        UPDATE chat_log
        SET thread_id = ?
        WHERE thread_id IS NULL OR thread_id = ''
        """,
        (DEFAULT_THREAD_ID,),
    )


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(SQLITE_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS indexed_files (
            path TEXT PRIMARY KEY,
            sha256 TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    ensure_chat_schema(conn)
    conn.commit()
    return conn


def safe_file_id(path: Path, chunk_idx: int) -> str:
    return hashlib.sha256(f"{path.as_posix()}::{chunk_idx}".encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def collect_files(source_dirs: list[Path]) -> list[Path]:
    files: list[Path] = []
    for root in source_dirs:
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if any(part in IGNORE_DIRS for part in path.parts):
                continue
            if path.suffix.lower() in INCLUDE_EXTENSIONS:
                files.append(path)
    return files


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> Iterable[str]:
    clean = text.strip()
    if not clean:
        return []
    chunks: list[str] = []
    start = 0
    text_len = len(clean)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = clean[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= text_len:
            break
        start = max(0, end - overlap)
    return chunks


def load_indexed_hashes(conn: sqlite3.Connection) -> dict[str, str]:
    rows = conn.execute("SELECT path, sha256 FROM indexed_files").fetchall()
    return {row[0]: row[1] for row in rows}


def upsert_file_hash(conn: sqlite3.Connection, path: str, sha256: str) -> None:
    conn.execute(
        """
        INSERT INTO indexed_files(path, sha256, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(path) DO UPDATE SET
            sha256 = excluded.sha256,
            updated_at = excluded.updated_at
        """,
        (path, sha256, now_iso()),
    )


def append_chat(
    conn: sqlite3.Connection,
    role: str,
    content: str,
    thread_id: str = DEFAULT_THREAD_ID,
) -> None:
    conn.execute(
        "INSERT INTO chat_log(role, content, created_at, thread_id) VALUES (?, ?, ?, ?)",
        (role, content, now_iso(), thread_id),
    )
    conn.commit()


def recent_chat_messages(
    conn: sqlite3.Connection,
    max_turns: int = MAX_HISTORY_TURNS,
    thread_id: str = DEFAULT_THREAD_ID,
) -> list[dict[str, str]]:
    limit = max_turns * 2
    rows = conn.execute(
        "SELECT role, content FROM chat_log WHERE thread_id = ? ORDER BY id DESC LIMIT ?",
        (thread_id, limit),
    ).fetchall()
    rows.reverse()
    return [{"role": role, "content": content} for role, content in rows]


def list_threads(conn: sqlite3.Connection) -> list[dict[str, str]]:
    rows = conn.execute(
        "SELECT id, name, created_at FROM threads ORDER BY created_at ASC"
    ).fetchall()
    return [{"id": row[0], "name": row[1], "created_at": row[2]} for row in rows]


def create_thread(conn: sqlite3.Connection, name: str) -> dict[str, str]:
    thread_name = name.strip() or "New Thread"
    thread_id = hashlib.sha256(f"{thread_name}-{now_iso()}".encode("utf-8")).hexdigest()[:16]
    created_at = now_iso()
    conn.execute(
        "INSERT INTO threads(id, name, created_at) VALUES (?, ?, ?)",
        (thread_id, thread_name, created_at),
    )
    conn.commit()
    return {"id": thread_id, "name": thread_name, "created_at": created_at}


def recent_chat_records(
    conn: sqlite3.Connection,
    thread_id: str,
    limit: int = 100,
) -> list[dict[str, str]]:
    rows = conn.execute(
        """
        SELECT role, content, created_at
        FROM chat_log
        WHERE thread_id = ?
        ORDER BY id DESC
        LIMIT ?
        """,
        (thread_id, limit),
    ).fetchall()
    rows.reverse()
    return [
        {"role": row[0], "content": row[1], "created_at": row[2]}
        for row in rows
    ]


def init_vectorstore() -> Chroma:
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    return Chroma(
        collection_name="local_docs",
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )


def index_changed_files(
    conn: sqlite3.Connection,
    vectorstore: Chroma,
    source_dirs: list[Path],
) -> dict[str, int]:
    existing = load_indexed_hashes(conn)
    files = collect_files(source_dirs)
    changed_count = 0
    chunk_count = 0
    total_files = len(files)
    started_at = perf_counter()

    if total_files == 0:
        print("No files found to index for the current source directories.")
        return {"changed_files": 0, "chunks_indexed": 0}

    print(f"Starting index scan: {total_files} files")

    for idx, path in enumerate(files, start=1):
        path_str = str(path.resolve())
        try:
            new_hash = file_sha256(path)
        except OSError:
            continue

        if existing.get(path_str) == new_hash:
            continue

        try:
            if path.suffix.lower() == ".pdf":
                loader = PyPDFLoader(str(path))
                pdf_docs = loader.load()
                text = "\n".join(d.page_content for d in pdf_docs)
            else:
                text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        chunks = list(chunk_text(text))
        # Remove prior chunks for this source before adding fresh ones.
        try:
            vectorstore.delete(where={"source": path_str})
        except Exception:
            pass

        docs: list[Document] = []
        ids: list[str] = []
        for chunk_idx, chunk in enumerate(chunks):
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": path_str,
                        "chunk": chunk_idx,
                        "updated_at": now_iso(),
                    },
                )
            )
            ids.append(safe_file_id(path, chunk_idx))

        if docs:
            vectorstore.add_documents(documents=docs, ids=ids)
            chunk_count += len(docs)

        upsert_file_hash(conn, path_str, new_hash)
        changed_count += 1

        if idx % PROGRESS_EVERY == 0 or idx == total_files:
            elapsed = perf_counter() - started_at
            rate = idx / elapsed if elapsed > 0 else 0.0
            print(
                f"Progress: {idx}/{total_files} files scanned | "
                f"changed: {changed_count} | chunks: {chunk_count} | "
                f"elapsed: {elapsed:.1f}s | rate: {rate:.1f} files/s"
            )

    conn.commit()
    total_elapsed = perf_counter() - started_at
    print(
        f"Index scan complete in {total_elapsed:.1f}s. "
        f"Changed files: {changed_count}, chunks indexed: {chunk_count}"
    )
    return {"changed_files": changed_count, "chunks_indexed": chunk_count}


def format_docs_for_prompt(docs: list[Document]) -> str:
    if not docs:
        return "No relevant context found."

    lines: list[str] = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        chunk = doc.metadata.get("chunk", "?")
        snippet = doc.page_content.replace("\n", " ").strip()
        if len(snippet) > 400:
            snippet = snippet[:400] + "..."
        lines.append(f"[{i}] {source} (chunk {chunk})\n{snippet}")
    return "\n\n".join(lines)


def tokenize_for_rerank(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]{2,}", text.lower()))


def rerank_documents(query: str, docs: list[Document], top_k: int = RETRIEVAL_K) -> list[Document]:
    # Lightweight lexical rerank: favors chunks that share more query terms.
    if not docs:
        return []

    query_terms = tokenize_for_rerank(query)
    if not query_terms:
        return docs[:top_k]

    scored: list[tuple[float, int, Document]] = []
    for idx, doc in enumerate(docs):
        content = doc.page_content or ""
        doc_terms = tokenize_for_rerank(content)
        overlap = len(query_terms & doc_terms)
        if overlap == 0:
            score = 0.0
        else:
            coverage = overlap / max(1, len(query_terms))
            precision = overlap / max(1, len(doc_terms))
            score = (coverage * 0.8) + (precision * 0.2)
        scored.append((score, idx, doc))

    scored.sort(key=lambda item: (-item[0], item[1]))
    ranked_docs = [item[2] for item in scored]
    return ranked_docs[:top_k]


def inspect_source_chunk(source: str, chunk: int, radius: int = 1) -> str:
    path = Path(source).expanduser()
    if not path.is_file():
        return f"Error: source file not found: {source}"

    try:
        if path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(path))
            pdf_docs = loader.load()
            text = "\n".join(d.page_content for d in pdf_docs)
        else:
            text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        return f"Error: unable to read source: {e}"

    chunks = list(chunk_text(text))
    if not chunks:
        return "No readable content found in source."

    target = max(0, min(int(chunk), len(chunks) - 1))
    window = max(0, int(radius))
    start = max(0, target - window)
    end = min(len(chunks), target + window + 1)

    lines = [f"Source: {path.resolve()}", f"Target chunk: {target} of {len(chunks) - 1}"]
    for idx in range(start, end):
        marker = "*" if idx == target else "-"
        lines.append(f"{marker} chunk {idx}\n{chunks[idx]}")
    return "\n\n".join(lines)


def safe_calculate(expression: str) -> str:
    allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Constant,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.USub,
        ast.UAdd,
        ast.Load,
        ast.Tuple,
    )
    try:
        tree = ast.parse(expression, mode="eval")
        for node in ast.walk(tree):
            if not isinstance(node, allowed_nodes):
                return "Error: unsupported expression."
        return str(eval(compile(tree, "<expr>", "eval"), {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Error: {e}"


def main() -> None:
    ensure_dirs()
    conn = get_db()
    source_dirs = parse_source_dirs()

    print("Index source directories:")
    for source_dir in source_dirs:
        print(f" - {source_dir}")

    try:
        vectorstore = init_vectorstore()
    except Exception as e:
        print("Failed to initialize vector store.")
        print(str(e))
        print("Tip: make sure Ollama is running and pull embeddings model: ollama pull nomic-embed-text")
        return

    stats = index_changed_files(conn, vectorstore, source_dirs)
    print(
        f"Index ready. Changed files: {stats['changed_files']}, chunks indexed: {stats['chunks_indexed']}"
    )

    llm = ChatOllama(model=LLM_MODEL, **LLM_OPTIONS)

    @tool
    def calculator(expression: str) -> str:
        """Calculate a mathematical expression with basic arithmetic operators."""
        return safe_calculate(expression)

    @tool
    def retrieve_context(query: str) -> str:
        """Search local indexed files for context relevant to the user's question."""
        candidate_docs = vectorstore.similarity_search(query, k=RETRIEVAL_CANDIDATES)
        docs = rerank_documents(query, candidate_docs, top_k=RETRIEVAL_K) if ENABLE_LOCAL_RERANK else candidate_docs[:RETRIEVAL_K]
        return format_docs_for_prompt(docs)

    @tool
    def reindex_knowledge_base(_: str = "") -> str:
        """Reindex changed local files into persistent vector store."""
        run_stats = index_changed_files(conn, vectorstore, source_dirs)
        return json.dumps(run_stats)

    @tool
    def source_inspector(spec_json: str) -> str:
        """Inspect a source/chunk with expanded context.

        Input JSON: {"source": "<absolute_path>", "chunk": <int>, "radius": <int optional>}.
        """
        try:
            spec = json.loads(spec_json or "{}")
            source = str(spec.get("source", "")).strip()
            chunk = int(spec.get("chunk", 0))
            radius = int(spec.get("radius", 1))
            if not source:
                return "Error: source is required."
            return inspect_source_chunk(source=source, chunk=chunk, radius=radius)
        except Exception as e:
            return f"Error: invalid source_inspector input: {e}"

    tools = [calculator, retrieve_context, source_inspector, reindex_knowledge_base]

    system_prompt = (
        "You are a local assistant with access to tools. "
        "Use retrieve_context when questions might depend on local files. "
        "Use source_inspector to fetch expanded chunk context when retrieval snippets are not enough. "
        "When using retrieved text, cite the [n] source labels you were given."
    )
    agent = create_agent(model=llm, tools=tools, system_prompt=system_prompt)

    print("\nLocal agent ready.")
    print("Commands: /quit, /reindex")

    while True:
        user_input = input("\nYou > ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"/quit", "exit", "quit"}:
            print("Bye.")
            break
        if user_input.lower() == "/reindex":
            reindex_result = reindex_knowledge_base.invoke("")
            print(f"Agent > Reindex complete: {reindex_result}")
            continue

        history = recent_chat_messages(conn, thread_id=DEFAULT_THREAD_ID)
        messages = history + [{"role": "user", "content": user_input}]

        try:
            response = agent.invoke({"messages": messages})
            final_message = response["messages"][-1]
            assistant_text = final_message.content if hasattr(final_message, "content") else str(final_message)
            print(f"Agent > {assistant_text}")
            append_chat(conn, "user", user_input, thread_id=DEFAULT_THREAD_ID)
            append_chat(conn, "assistant", assistant_text, thread_id=DEFAULT_THREAD_ID)
        except Exception as e:
            print("\n--- Agent Error ---")
            print(str(e))
            print("Tips: ensure Ollama is running and your selected models are available.")

    conn.close()


if __name__ == "__main__":
    main()
