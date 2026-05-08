from pathlib import Path

from langchain_core.documents import Document

import agent


class FakeVectorStore:
    def __init__(self):
        self.deleted_sources = []
        self.added_docs = []
        self.added_ids = []

    def delete(self, where):
        self.deleted_sources.append(where.get("source"))

    def add_documents(self, documents, ids):
        self.added_docs.extend(documents)
        self.added_ids.extend(ids)


def test_parse_source_dirs_defaults_when_env_missing(monkeypatch):
    monkeypatch.delenv("AGENT_SOURCE_DIRS", raising=False)
    dirs = agent.parse_source_dirs()
    assert dirs == agent.DEFAULT_SOURCE_DIRS


def test_parse_source_dirs_filters_invalid_and_keeps_valid(monkeypatch, tmp_path):
    good = tmp_path / "good"
    good.mkdir()
    bad = tmp_path / "missing"
    monkeypatch.setenv("AGENT_SOURCE_DIRS", f"{good}{agent.os.pathsep}{bad}")

    dirs = agent.parse_source_dirs()

    assert dirs == [good]


def test_chunk_text_with_overlap():
    chunks = list(agent.chunk_text("abcdefghij", chunk_size=4, overlap=1))
    assert chunks == ["abcd", "defg", "ghij"]


def test_chunk_text_empty_returns_empty_list():
    assert list(agent.chunk_text("   ")) == []


def test_safe_calculate_success_and_rejects_unsafe():
    assert agent.safe_calculate("(2 + 3) * 4") == "20"
    assert "unsupported" in agent.safe_calculate("__import__('os').system('whoami')")


def test_safe_calculate_syntax_error_path():
    assert agent.safe_calculate("2 +") .startswith("Error:")


def test_format_docs_for_prompt_handles_empty_and_truncates():
    assert agent.format_docs_for_prompt([]) == "No relevant context found."
    docs = [
        Document(page_content="x" * 500, metadata={"source": "a.txt", "chunk": 2}),
    ]
    out = agent.format_docs_for_prompt(docs)
    assert "[1] a.txt (chunk 2)" in out
    assert out.endswith("...")


def test_rerank_documents_prefers_query_overlap():
    docs = [
        Document(page_content="zebra alpha", metadata={"source": "a"}),
        Document(page_content="alpha beta gamma", metadata={"source": "b"}),
        Document(page_content="nothing useful", metadata={"source": "c"}),
    ]
    ranked = agent.rerank_documents("alpha beta", docs, top_k=2)
    assert len(ranked) == 2
    assert ranked[0].metadata["source"] == "b"


def test_rerank_documents_handles_empty_query_terms():
    docs = [Document(page_content="first", metadata={"source": "a"}), Document(page_content="second", metadata={"source": "b"})]
    ranked = agent.rerank_documents("!", docs, top_k=1)
    assert ranked[0].metadata["source"] == "a"


def test_db_chat_and_thread_flow(monkeypatch, tmp_path):
    monkeypatch.setattr(agent, "SQLITE_PATH", tmp_path / "state.db")
    conn = agent.get_db()
    try:
        created = agent.create_thread(conn, "  Thread A  ")
        assert created["name"] == "Thread A"

        agent.append_chat(conn, "user", "hello", thread_id=created["id"])
        agent.append_chat(conn, "assistant", "world", thread_id=created["id"])

        messages = agent.recent_chat_messages(conn, thread_id=created["id"])
        assert [m["role"] for m in messages] == ["user", "assistant"]

        records = agent.recent_chat_records(conn, thread_id=created["id"], limit=10)
        assert len(records) == 2

        threads = agent.list_threads(conn)
        assert any(t["id"] == created["id"] for t in threads)
    finally:
        conn.close()


def test_index_changed_files_indexes_and_skips_unchanged(monkeypatch, tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    file_a = src / "doc.txt"
    file_a.write_text("hello world " * 50, encoding="utf-8")

    monkeypatch.setattr(agent, "SQLITE_PATH", tmp_path / "state.db")
    conn = agent.get_db()
    try:
        vs = FakeVectorStore()
        first = agent.index_changed_files(conn, vs, [src])
        assert first["changed_files"] == 1
        assert first["chunks_indexed"] > 0
        assert vs.added_docs

        second = agent.index_changed_files(conn, vs, [src])
        assert second == {"changed_files": 0, "chunks_indexed": 0}
    finally:
        conn.close()


def test_collect_files_respects_extension_and_ignored_dirs(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "ok.md").write_text("ok", encoding="utf-8")
    (src / "no.bin").write_text("x", encoding="utf-8")
    ig = src / "__pycache__"
    ig.mkdir()
    (ig / "skip.md").write_text("skip", encoding="utf-8")

    files = agent.collect_files([src])
    paths = {p.name for p in files}
    assert "ok.md" in paths
    assert "no.bin" not in paths
    assert "skip.md" not in paths


def test_inspect_source_chunk_returns_windowed_excerpt(tmp_path):
    src = tmp_path / "doc.txt"
    src.write_text("A" * 1300 + "B" * 1300 + "C" * 1300, encoding="utf-8")

    out = agent.inspect_source_chunk(str(src), chunk=1, radius=1)

    assert "Target chunk: 1" in out
    assert "chunk 0" in out
    assert "chunk 1" in out
    assert "chunk 2" in out


def test_inspect_source_chunk_handles_missing_file(tmp_path):
    missing = tmp_path / "missing.txt"
    out = agent.inspect_source_chunk(str(missing), chunk=0)
    assert out.startswith("Error: source file not found")


def test_index_changed_files_returns_zero_when_no_files(monkeypatch, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setattr(agent, "SQLITE_PATH", tmp_path / "state.db")
    conn = agent.get_db()
    try:
        stats = agent.index_changed_files(conn, FakeVectorStore(), [empty])
        assert stats == {"changed_files": 0, "chunks_indexed": 0}
    finally:
        conn.close()


def test_main_exits_when_vectorstore_init_fails(monkeypatch):
    class FakeConn:
        def close(self):
            return None

    monkeypatch.setattr(agent, "ensure_dirs", lambda: None)
    monkeypatch.setattr(agent, "get_db", lambda: FakeConn())
    monkeypatch.setattr(agent, "parse_source_dirs", lambda: [Path(".")])

    def boom():
        raise RuntimeError("no vectorstore")

    monkeypatch.setattr(agent, "init_vectorstore", boom)
    agent.main()


def test_main_basic_chat_and_quit(monkeypatch):
    class FakeConn:
        def close(self):
            return None

    class FakeVS:
        def similarity_search(self, query, k=3):
            return []

    class FakeModel:
        def __init__(self, content):
            self.content = content

    class FakeAgentRuntime:
        def invoke(self, payload):
            class Msg:
                def __init__(self, content):
                    self.content = content

            return {"messages": [Msg("ok")]} 

    inputs = iter(["hello", "/quit"])
    captured = []

    monkeypatch.setattr(agent, "ensure_dirs", lambda: None)
    monkeypatch.setattr(agent, "get_db", lambda: FakeConn())
    monkeypatch.setattr(agent, "parse_source_dirs", lambda: [Path(".")])
    monkeypatch.setattr(agent, "init_vectorstore", lambda: FakeVS())
    monkeypatch.setattr(agent, "index_changed_files", lambda conn, vs, dirs: {"changed_files": 0, "chunks_indexed": 0})
    monkeypatch.setattr(agent, "ChatOllama", lambda **kwargs: FakeModel("ignored"))
    monkeypatch.setattr(agent, "create_agent", lambda **kwargs: FakeAgentRuntime())
    monkeypatch.setattr(agent, "recent_chat_messages", lambda conn, thread_id: [])
    monkeypatch.setattr(agent, "append_chat", lambda conn, role, content, thread_id=None: captured.append((role, content)))
    monkeypatch.setattr(agent, "tool", lambda f: f)
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    agent.main()
    assert captured == [("user", "hello"), ("assistant", "ok")]
