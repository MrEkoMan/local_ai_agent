from langchain_core.documents import Document
from fastapi.testclient import TestClient

import agent
import server


class FakeAgent:
    def __init__(self, content="answer"):
        self.content = content
        self.calls = []

    def invoke(self, payload):
        self.calls.append(payload)

        class Msg:
            def __init__(self, content):
                self.content = content

        return {"messages": [Msg(self.content)]}


class RaisingAgent:
    def invoke(self, payload):
        raise RuntimeError("invoke failed")


class FakeVectorStore:
    def __init__(self, docs):
        self.docs = docs

    def similarity_search(self, query, k=3):
        return self.docs[:k]


class FakeReindexTool:
    def __init__(self, result='{"changed_files": 1, "chunks_indexed": 2}'):
        self.result = result

    def invoke(self, _):
        return self.result


def _configure_temp_db(monkeypatch, tmp_path):
    monkeypatch.setattr(agent, "SQLITE_PATH", tmp_path / "state.db")


def test_token_estimators_and_pressure_classification():
    assert server.estimate_text_tokens("") == 0
    assert server.estimate_text_tokens("abcd") == 1
    assert server.estimate_message_tokens([{"content": "hello"}]) > 0

    level, ratio = server.classify_token_pressure(95, 100)
    assert level == "critical"
    assert ratio == 0.95
    assert server.classify_token_pressure(1, 0)[0] == "unknown"


def test_serialize_sources_truncates_long_snippet():
    docs = [Document(page_content="x" * 300, metadata={"source": "f", "chunk": 1})]
    out = server.serialize_sources(docs)
    assert out[0]["source"] == "f"
    assert out[0]["snippet"].endswith("...")


def test_runtime_config_endpoint():
    client = TestClient(server.app)
    resp = client.get("/api/runtime-config")
    assert resp.status_code == 200
    data = resp.json()
    assert "context_limit" in data
    assert "retrieval_candidates" in data
    assert "rerank_enabled" in data


def test_build_runtime_returns_expected_tuple(monkeypatch, tmp_path):
    _configure_temp_db(monkeypatch, tmp_path)

    class FakeConn:
        def close(self):
            return None

    class FakeVS:
        def similarity_search(self, query, k=3):
            return []

    fake_runtime_agent = object()

    monkeypatch.setattr(server, "parse_source_dirs", lambda: [tmp_path])
    monkeypatch.setattr(server, "init_vectorstore", lambda: FakeVS())
    monkeypatch.setattr(server, "get_db", lambda: FakeConn())
    monkeypatch.setattr(server, "index_changed_files", lambda conn, vs, dirs: {"changed_files": 0, "chunks_indexed": 0})
    monkeypatch.setattr(server, "ChatOllama", lambda **kwargs: object())
    monkeypatch.setattr(server, "create_agent", lambda **kwargs: fake_runtime_agent)
    monkeypatch.setattr(server, "tool", lambda f: f)

    runtime_agent, reindex_tool, vectorstore = server.build_runtime()
    assert runtime_agent is fake_runtime_agent
    assert callable(reindex_tool)
    assert vectorstore is not None


def test_on_startup_wires_globals(monkeypatch):
    fake_agent = object()
    fake_tool = object()
    fake_vs = object()
    monkeypatch.setattr(server, "build_runtime", lambda: (fake_agent, fake_tool, fake_vs))
    server.on_startup()
    assert server.agent is fake_agent
    assert server.reindex_tool is fake_tool
    assert server.chat_vectorstore is fake_vs


def test_threads_endpoints(monkeypatch, tmp_path):
    _configure_temp_db(monkeypatch, tmp_path)
    client = TestClient(server.app)

    create = client.post("/api/threads", json={"name": "My Thread"})
    assert create.status_code == 200
    thread_id = create.json()["id"]

    listed = client.get("/api/threads")
    assert listed.status_code == 200
    assert any(t["id"] == thread_id for t in listed.json())

    messages = client.get(f"/api/threads/{thread_id}/messages?limit=10")
    assert messages.status_code == 200
    assert messages.json() == []


def test_reindex_endpoint_success_and_not_ready(monkeypatch):
    client = TestClient(server.app)

    monkeypatch.setattr(server, "reindex_tool", None)
    not_ready = client.post("/api/reindex")
    assert not_ready.status_code == 503

    monkeypatch.setattr(server, "reindex_tool", FakeReindexTool())
    ok = client.post("/api/reindex")
    assert ok.status_code == 200
    assert ok.json()["changed_files"] == 1


def test_reindex_endpoint_error(monkeypatch):
    class BadTool:
        def invoke(self, _):
            raise RuntimeError("boom")

    client = TestClient(server.app)
    monkeypatch.setattr(server, "reindex_tool", BadTool())
    resp = client.post("/api/reindex")
    assert resp.status_code == 500


def test_chat_endpoint_with_rag_and_token_budget(monkeypatch, tmp_path):
    _configure_temp_db(monkeypatch, tmp_path)
    fake_agent = FakeAgent("assistant response")
    fake_docs = [
        Document(page_content="alpha beta context", metadata={"source": "a.txt", "chunk": 0}),
        Document(page_content="other", metadata={"source": "b.txt", "chunk": 1}),
    ]

    monkeypatch.setattr(server, "agent", fake_agent)
    monkeypatch.setattr(server, "chat_vectorstore", FakeVectorStore(fake_docs))

    client = TestClient(server.app)

    # Create thread first so messages can be persisted.
    thread = client.post("/api/threads", json={"name": "ChatThread"}).json()
    resp = client.post("/api/chat", json={"message": "alpha question", "thread_id": thread["id"]})

    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "assistant response"
    assert data["grounded"] is True
    assert data["retrieval_count"] >= 1
    assert data["token_budget"]["prompt_tokens"] > 0
    assert data["token_budget"]["warning_level"] in {"ok", "warning", "critical"}

    # Ensure system context was prepended for the model invocation.
    sent_messages = fake_agent.calls[0]["messages"]
    assert sent_messages[0]["role"] == "system"


def test_chat_endpoint_handles_runtime_not_ready(monkeypatch):
    client = TestClient(server.app)
    monkeypatch.setattr(server, "agent", None)
    resp = client.post("/api/chat", json={"message": "hi", "thread_id": "x"})
    assert resp.status_code == 503


def test_chat_endpoint_returns_500_on_agent_error(monkeypatch, tmp_path):
    _configure_temp_db(monkeypatch, tmp_path)
    client = TestClient(server.app)
    monkeypatch.setattr(server, "agent", RaisingAgent())
    monkeypatch.setattr(server, "chat_vectorstore", None)

    thread = client.post("/api/threads", json={"name": "ErrThread"}).json()
    resp = client.post("/api/chat", json={"message": "hi", "thread_id": thread["id"]})
    assert resp.status_code == 500


def test_health_endpoint():
    client = TestClient(server.app)
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
