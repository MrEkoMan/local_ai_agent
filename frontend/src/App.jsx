import { useEffect, useMemo, useState } from "react";

const initialMessages = [
  {
    role: "assistant",
    text: "Loading threads...",
    grounded: false,
    sources: [],
  },
];

const STATUS_TEXT = {
  idle: "online",
  loading: "loading thread",
  creating: "creating thread",
  thinking: "searching local docs",
  generating: "generating answer",
  reindexing: "reindexing knowledge base",
  error: "error",
};

export default function App() {
  const [messages, setMessages] = useState(initialMessages);
  const [threads, setThreads] = useState([]);
  const [selectedThreadId, setSelectedThreadId] = useState("");
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("idle");
  const [activity, setActivity] = useState("");

  const canSend = useMemo(() => input.trim().length > 0 && !busy, [input, busy]);

  useEffect(() => {
    initializeThreads();
  }, []);

  async function initializeThreads() {
    setStatus("loading");
    try {
      const resp = await fetch("/api/threads");
      const data = await resp.json();
      if (!resp.ok) {
        throw new Error(data.detail || "Failed to load threads");
      }

      setThreads(data);
      const firstThreadId = data[0]?.id || "";
      setSelectedThreadId(firstThreadId);
      if (firstThreadId) {
        await loadThreadMessages(firstThreadId);
      }
      setStatus("idle");
      setActivity("");
    } catch (err) {
      setMessages([{ role: "assistant", text: `Error: ${err.message}`, grounded: false, sources: [] }]);
      setStatus("error");
      setActivity("Failed to load thread list.");
    }
  }

  async function loadThreadMessages(threadId) {
    const resp = await fetch(`/api/threads/${threadId}/messages?limit=100`);
    const data = await resp.json();
    if (!resp.ok) {
      throw new Error(data.detail || "Failed to load thread messages");
    }

    if (!data.length) {
      setMessages([
        {
          role: "assistant",
          text: "Thread is empty. Start by asking a question.",
          grounded: false,
          sources: [],
        },
      ]);
      return;
    }

    setMessages(data.map((m) => ({ role: m.role, text: m.content, grounded: false, sources: [] })));
  }

  async function onThreadChange(e) {
    const threadId = e.target.value;
    setSelectedThreadId(threadId);
    setBusy(true);
    setStatus("loading");
    try {
      await loadThreadMessages(threadId);
      setStatus("idle");
      setActivity("");
    } catch (err) {
      setMessages([{ role: "assistant", text: `Error: ${err.message}`, grounded: false, sources: [] }]);
      setStatus("error");
      setActivity("Failed to switch thread.");
    } finally {
      setBusy(false);
    }
  }

  async function createNewThread() {
    const name = window.prompt("Thread name:", `Theme ${threads.length + 1}`);
    if (!name || !name.trim()) {
      return;
    }

    setBusy(true);
    setStatus("creating");
    try {
      const resp = await fetch("/api/threads", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: name.trim() }),
      });
      const data = await resp.json();
      if (!resp.ok) {
        throw new Error(data.detail || "Failed to create thread");
      }
      const nextThreads = [...threads, data];
      setThreads(nextThreads);
      setSelectedThreadId(data.id);
      setMessages([
        {
          role: "assistant",
          text: `Thread '${data.name}' created. Ask your first question.`,
          grounded: false,
          sources: [],
        },
      ]);
      setStatus("idle");
      setActivity("");
    } catch (err) {
      setMessages([{ role: "assistant", text: `Error: ${err.message}`, grounded: false, sources: [] }]);
      setStatus("error");
      setActivity("Failed to create thread.");
    } finally {
      setBusy(false);
    }
  }

  async function sendMessage(e) {
    e.preventDefault();
    const question = input.trim();
    if (!question) return;

    setBusy(true);
    setStatus("thinking");
    setActivity("Searching indexed local content...");
    setMessages((prev) => [...prev, { role: "user", text: question, grounded: false, sources: [] }]);
    setInput("");

    try {
      const phaseTimer = window.setTimeout(() => {
        setStatus("generating");
        setActivity("Generating answer from retrieved context...");
      }, 700);

      const resp = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: question, thread_id: selectedThreadId }),
      });
      const data = await resp.json();
      window.clearTimeout(phaseTimer);
      if (!resp.ok) {
        throw new Error(data.detail || "Request failed");
      }
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          text: data.answer,
          grounded: Boolean(data.grounded),
          sources: data.used_sources || [],
          retrievalCount: data.retrieval_count || 0,
        },
      ]);
      setStatus("idle");
      setActivity(
        data.grounded
          ? `Used ${data.retrieval_count} local source${data.retrieval_count === 1 ? "" : "s"}.`
          : "No relevant local context found for this answer."
      );
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", text: `Error: ${err.message}`, grounded: false, sources: [] },
      ]);
      setStatus("error");
      setActivity("Chat request failed.");
    } finally {
      setBusy(false);
    }
  }

  async function reindex() {
    setBusy(true);
    setStatus("reindexing");
    try {
      const resp = await fetch("/api/reindex", { method: "POST" });
      const data = await resp.json();
      if (!resp.ok) {
        throw new Error(data.detail || "Reindex failed");
      }
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          text: `Reindex complete. Changed files: ${data.changed_files}, chunks indexed: ${data.chunks_indexed}`,
          grounded: false,
          sources: [],
        },
      ]);
      setStatus("idle");
      setActivity(`Reindex finished: ${data.changed_files} changed files, ${data.chunks_indexed} chunks.`);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", text: `Reindex error: ${err.message}`, grounded: false, sources: [] },
      ]);
      setStatus("error");
      setActivity("Reindex failed.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="app-shell">
      <div className="ambient a1" />
      <div className="ambient a2" />

      <header className="topbar">
        <h1>Local Agent Console</h1>
        <div className={`status ${status}`}>{STATUS_TEXT[busy ? status : "idle"]}</div>
      </header>

      <section className="activity-strip">
        <div className={`activity-indicator ${busy ? "active" : ""}`} />
        <div>
          <strong>Activity</strong>
          <span>{activity || "Waiting for the next request."}</span>
        </div>
      </section>

      <section className="composer-wrap">
        <button className="ghost" disabled={busy} onClick={createNewThread}>
          New Thread
        </button>
        <div className="composer">
          <select value={selectedThreadId} onChange={onThreadChange} disabled={busy || !threads.length}>
            {threads.map((thread) => (
              <option key={thread.id} value={thread.id}>
                {thread.name}
              </option>
            ))}
          </select>
        </div>
      </section>

      <main className="chat-panel">
        {messages.map((m, i) => (
          <article key={`${m.role}-${i}`} className={`bubble ${m.role}`}>
            <div className="label">{m.role}</div>
            <p>{m.text}</p>
            {m.role === "assistant" && Array.isArray(m.sources) && m.sources.length > 0 ? (
              <details className="sources-panel">
                <summary>
                  Used {m.sources.length} local source{m.sources.length === 1 ? "" : "s"}
                </summary>
                <div className="sources-list">
                  {m.sources.map((source, sourceIndex) => (
                    <article key={`${source.source}-${source.chunk}-${sourceIndex}`} className="source-card">
                      <div className="source-path">{source.source}</div>
                      <div className="source-meta">chunk {source.chunk}</div>
                      <p>{source.snippet}</p>
                    </article>
                  ))}
                </div>
              </details>
            ) : null}
            {m.role === "assistant" && m.grounded === false && !m.text.startsWith("Error:") ? (
              <div className="ungrounded-note">No matching local source was attached to this answer.</div>
            ) : null}
          </article>
        ))}
      </main>

      <footer className="composer-wrap">
        <button className="ghost" disabled={busy} onClick={reindex}>
          Reindex
        </button>
        <form className="composer" onSubmit={sendMessage}>
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask about your docs, code, or calculations"
          />
          <button type="submit" disabled={!canSend}>
            Send
          </button>
        </form>
      </footer>
    </div>
  );
}