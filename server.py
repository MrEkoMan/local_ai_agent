from __future__ import annotations

import json
import math
from threading import Lock

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from agent import (
    DEFAULT_THREAD_ID,
    ENABLE_LOCAL_RERANK,
    inspect_source_chunk,
    LLM_MODEL,
    LLM_NUM_CTX,
    LLM_NUM_PREDICT,
    LLM_OPTIONS,
    RETRIEVAL_CANDIDATES,
    RETRIEVAL_K,
    append_chat,
    create_thread,
    format_docs_for_prompt,
    get_db,
    index_changed_files,
    init_vectorstore,
    list_threads,
    parse_source_dirs,
    recent_chat_records,
    recent_chat_messages,
    rerank_documents,
    safe_calculate,
)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    thread_id: str = Field(default=DEFAULT_THREAD_ID)


class ChatResponse(BaseModel):
    answer: str
    thread_id: str
    grounded: bool
    retrieval_count: int
    used_sources: list[dict[str, str | int]]
    token_budget: dict[str, str | int | float]


class RuntimeConfigResponse(BaseModel):
    model: str
    context_limit: int
    max_output_tokens: int
    retrieval_k: int
    retrieval_candidates: int
    rerank_enabled: bool
    token_estimator: str


class ThreadCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)


class ThreadResponse(BaseModel):
    id: str
    name: str
    created_at: str


class ThreadMessageResponse(BaseModel):
    role: str
    content: str
    created_at: str


class ReindexResponse(BaseModel):
    changed_files: int
    chunks_indexed: int


def estimate_text_tokens(text: str) -> int:
    # Lightweight approximation for local UI telemetry without extra tokenizer deps.
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def estimate_message_tokens(messages: list[dict[str, str]]) -> int:
    # Add a small per-message overhead to account for role/format framing.
    per_message_overhead = 4
    trailing_overhead = 2
    return sum(estimate_text_tokens(m.get("content", "")) + per_message_overhead for m in messages) + trailing_overhead


def classify_token_pressure(projected_total: int, context_limit: int) -> tuple[str, float]:
    if context_limit <= 0:
        return "unknown", 0.0
    ratio = projected_total / context_limit
    if ratio >= 0.9:
        return "critical", ratio
    if ratio >= 0.75:
        return "warning", ratio
    return "ok", ratio


def serialize_sources(docs: list) -> list[dict[str, str | int]]:
    sources: list[dict[str, str | int]] = []
    for doc in docs:
        snippet = doc.page_content.replace("\n", " ").strip()
        if len(snippet) > 220:
            snippet = snippet[:220] + "..."
        sources.append(
            {
                "source": str(doc.metadata.get("source", "unknown")),
                "chunk": int(doc.metadata.get("chunk", -1)),
                "snippet": snippet,
            }
        )
    return sources


def build_runtime() -> tuple:
    source_dirs = parse_source_dirs()
    vectorstore = init_vectorstore()
    startup_conn = get_db()
    try:
        index_changed_files(startup_conn, vectorstore, source_dirs)
    finally:
        startup_conn.close()

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
        tool_conn = get_db()
        try:
            run_stats = index_changed_files(tool_conn, vectorstore, source_dirs)
            return json.dumps(run_stats)
        finally:
            tool_conn.close()

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

    system_prompt = (
        "You are a local assistant with access to tools. "
        "Use retrieve_context when questions might depend on local files. "
        "Use source_inspector to fetch expanded chunk context when retrieval snippets are not enough. "
        "When using retrieved text, cite the [n] source labels you were given."
    )
    agent = create_agent(
        model=llm,
        tools=[calculator, retrieve_context, source_inspector, reindex_knowledge_base],
        system_prompt=system_prompt,
    )
    return agent, reindex_knowledge_base, vectorstore


app = FastAPI(title="Local AI Agent API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_chat_lock = Lock()
_reindex_lock = Lock()
agent = None
reindex_tool = None
chat_vectorstore = None


@app.on_event("startup")
def on_startup() -> None:
    global agent, reindex_tool, chat_vectorstore
    agent, reindex_tool, chat_vectorstore = build_runtime()


@app.on_event("shutdown")
def on_shutdown() -> None:
    return


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/runtime-config", response_model=RuntimeConfigResponse)
def runtime_config() -> RuntimeConfigResponse:
    return RuntimeConfigResponse(
        model=LLM_MODEL,
        context_limit=LLM_NUM_CTX,
        max_output_tokens=LLM_NUM_PREDICT,
        retrieval_k=RETRIEVAL_K,
        retrieval_candidates=RETRIEVAL_CANDIDATES,
        rerank_enabled=ENABLE_LOCAL_RERANK,
        token_estimator="approx_chars_div_4",
    )


@app.get("/api/threads", response_model=list[ThreadResponse])
def get_threads() -> list[ThreadResponse]:
    conn = get_db()
    try:
        return [ThreadResponse(**row) for row in list_threads(conn)]
    finally:
        conn.close()


@app.post("/api/threads", response_model=ThreadResponse)
def post_thread(req: ThreadCreateRequest) -> ThreadResponse:
    conn = get_db()
    try:
        created = create_thread(conn, req.name)
        return ThreadResponse(**created)
    finally:
        conn.close()


@app.get("/api/threads/{thread_id}/messages", response_model=list[ThreadMessageResponse])
def get_thread_messages(thread_id: str, limit: int = 100) -> list[ThreadMessageResponse]:
    conn = get_db()
    try:
        records = recent_chat_records(conn, thread_id=thread_id, limit=max(1, min(limit, 500)))
        return [ThreadMessageResponse(**row) for row in records]
    finally:
        conn.close()


@app.post("/api/reindex", response_model=ReindexResponse)
def reindex() -> ReindexResponse:
    if reindex_tool is None:
        raise HTTPException(status_code=503, detail="Runtime not ready")

    with _reindex_lock:
        try:
            result = reindex_tool.invoke("")
            data = json.loads(result)
            return ReindexResponse(**data)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    if agent is None:
        raise HTTPException(status_code=503, detail="Runtime not ready")

    with _chat_lock:
        chat_conn = get_db()
        try:
            history = recent_chat_messages(chat_conn, thread_id=req.thread_id)
            messages = history + [{"role": "user", "content": req.message}]
            docs = []
            if chat_vectorstore is not None:
                candidate_docs = chat_vectorstore.similarity_search(req.message, k=RETRIEVAL_CANDIDATES)
                docs = rerank_documents(req.message, candidate_docs, top_k=RETRIEVAL_K) if ENABLE_LOCAL_RERANK else candidate_docs[:RETRIEVAL_K]
                context = format_docs_for_prompt(docs)
                if context != "No relevant context found.":
                    messages = [
                        {
                            "role": "system",
                            "content": (
                                "Relevant local context for this request is below. "
                                "Use it when answering if it is applicable, and cite the [n] labels when you rely on it.\n\n"
                                f"{context}"
                            ),
                        },
                        *messages,
                    ]

            prompt_tokens = estimate_message_tokens(messages)
            response = agent.invoke({"messages": messages})
            final_message = response["messages"][-1]
            answer = final_message.content if hasattr(final_message, "content") else str(final_message)
            completion_tokens = estimate_text_tokens(answer)
            total_tokens = prompt_tokens + completion_tokens
            projected_with_max_output = prompt_tokens + LLM_NUM_PREDICT
            warning_level, utilization_ratio = classify_token_pressure(projected_with_max_output, LLM_NUM_CTX)
            used_sources = serialize_sources(docs)
            append_chat(chat_conn, "user", req.message, thread_id=req.thread_id)
            append_chat(chat_conn, "assistant", answer, thread_id=req.thread_id)
            return ChatResponse(
                answer=answer,
                thread_id=req.thread_id,
                grounded=bool(used_sources),
                retrieval_count=len(used_sources),
                used_sources=used_sources,
                token_budget={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "context_limit": LLM_NUM_CTX,
                    "max_output_tokens": LLM_NUM_PREDICT,
                    "projected_with_max_output": projected_with_max_output,
                    "utilization_ratio": round(utilization_ratio, 4),
                    "warning_level": warning_level,
                    "estimator": "approx_chars_div_4",
                },
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e
        finally:
            chat_conn.close()
