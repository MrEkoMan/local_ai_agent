# local_ai_agent

Local-first AI agent using Ollama + LangChain with:

- Tool-calling chat model
- Persistent local RAG index (Chroma)
- Incremental indexing (only changed files are re-embedded)
- Persistent chat history in SQLite

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install langchain langchain-ollama langchain-chroma chromadb
ollama pull qwen2.5:3b; ollama pull nomic-embed-text
python agent.py
```

## What This Project Does

This agent can:

- Answer questions with a local model
- Use tools for calculations and document retrieval
- Index your local folders (including network shares)
- Keep indexed knowledge and chat history across restarts

Main script: agent.py

## Requirements

- Windows PowerShell
- Python virtual environment
- Ollama running locally
- Node.js 18+ and npm (for React frontend)
- Ollama models:
	- qwen2.5:3b (chat/tool model, good balance of speed and quality)
	- nomic-embed-text (embeddings)

## Install And Setup

1. Create and activate venv

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install Python dependencies

```powershell
pip install --upgrade pip
pip install langchain langchain-ollama langchain-chroma chromadb fastapi uvicorn
```

3. Pull required Ollama models

```powershell
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
```

## Response Speed Settings (Code-Driven)

Performance-related chat settings are defined near the top of [agent.py](agent.py).

- LLM_MODEL
  - Default: qwen2.5:3b
  - What it does: selects the chat model. Smaller models are faster but less
    capable; qwen2.5:1.5b can work for simple Q&A but may fail tool-calling.
- LLM_NUM_PREDICT
  - Default: 512
  - What it does: hard cap on tokens generated per response.
  - WARNING: setting this below ~256 will cut off complex answers mid-sentence.
    Raise to 1024+ if you ask for long explanations or code; lower (e.g. 256)
    only for short factual Q&A where speed matters most.
- LLM_NUM_CTX
  - Default: 4096
  - What it does: context window size — covers system prompt, retrieved chunks,
    chat history, and the current message combined.
  - WARNING: setting this too low silently drops older history and context. The
    default 4096 comfortably fits most conversations; only lower it if memory is
    constrained and you understand the tradeoff.
- LLM_NUM_THREAD
  - Default: half your logical CPU cores.
  - What it does: number of CPU threads used during inference.
  - Tuning: increase toward your physical core count for more throughput;
    going above physical cores usually causes contention and slows things down.
- RETRIEVAL_K
  - Default: 3
  - What it does: number of vector chunks retrieved and passed to the model per
    question. Lower means faster and smaller prompts; higher means richer context.
- RETRIEVAL_CANDIDATES
  - Default: 10
  - What it does: number of initial vector candidates collected before reranking.
  - Tradeoff: higher values can improve hit quality but add retrieval/rerank cost.
- ENABLE_LOCAL_RERANK
  - Default: true
  - What it does: reranks retrieved candidates using query-term overlap before
    selecting top RETRIEVAL_K chunks for prompting.
  - Tradeoff: usually improves local answer relevance, with a small latency cost.

Recommended tuning order:

1. Adjust LLM_NUM_THREAD toward your physical core count (biggest safe win).
2. Raise or lower LLM_NUM_PREDICT based on expected response length.
3. Lower LLM_NUM_CTX only if memory is limited — watch for coherence issues.
4. Lower RETRIEVAL_K if prompt payload feels large and answers are still good.
5. Adjust RETRIEVAL_CANDIDATES (8-12 is a practical range with reranking on).
6. Disable ENABLE_LOCAL_RERANK only if you need the absolute lowest latency.
7. Switch to qwen2.5:1.5b only if speed is still insufficient and answers
   remain acceptable for your use case.
After changing settings, restart the backend process:

```powershell
python -m uvicorn server:app --host 127.0.0.1 --port 8000
```

## Configure Source Folders (Local Or Network)

Set AGENT_SOURCE_DIRS with one or more folders separated by semicolons.

Example (current PowerShell session):

```powershell
$env:AGENT_SOURCE_DIRS="C:\code\eyesky\wiki;\\NAS01\Shared\EngineeringDocs"
```

Make it persistent for new terminals:

```powershell
setx AGENT_SOURCE_DIRS "C:\code\eyesky\wiki;\\NAS01\Shared\EngineeringDocs"
```

Notes:

- Supports absolute local paths and UNC network paths
- Invalid or unreachable paths are skipped with a warning
- If not set, the agent indexes the project folder by default

## Run CLI

```powershell
python agent.py
```

CLI startup behavior:

- Prints configured index source directories
- Performs incremental indexing with progress output
- Loads persistent chat history context

## Run Web API + React Frontend

1. Start backend API (terminal 1)

```powershell
python -m uvicorn server:app --host 127.0.0.1 --port 8000
```

2. Start frontend (terminal 2)

```powershell
cd frontend
npm install
npm run dev
```

3. Open the app

- http://localhost:5173

Frontend notes:

- The React app proxies /api calls to http://localhost:8000
- Reindex button triggers the same persistent indexing process
- Chat uses the same SQLite history and vector index as CLI mode

## In-App Commands (CLI)

- /reindex: force reindex of changed files
- /quit: exit the app

## Persistence

Data is stored in .agent_store:

- .agent_store/chroma: vector store
- .agent_store/state.db: SQLite metadata and chat log

This allows the agent to preserve knowledge and conversation memory after shutdown.

## Performance Tips

- First index can take time for large folders
- Reindex is faster afterward because unchanged files are skipped
- Network locations can be slower due to file I/O latency
- Narrow source folders to improve ingest speed and relevance
- Use a smaller chat model for faster local response times
- Keep retrieval/context settings conservative for lower latency

## Troubleshooting

- If embeddings fail: pull nomic-embed-text and verify Ollama is running
- If tool-calling fails: use a tool-capable model such as qwen2.5 variants
- If indexing is slow: reduce folder scope and file types