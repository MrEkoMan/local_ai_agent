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
	- qwen2.5:3b (chat/tool model)
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

## Troubleshooting

- If embeddings fail: pull nomic-embed-text and verify Ollama is running
- If tool-calling fails: use a tool-capable model such as qwen2.5 variants
- If indexing is slow: reduce folder scope and file types