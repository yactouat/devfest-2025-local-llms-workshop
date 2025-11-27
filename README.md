# DevFest 2025: Local LLMs, RAG, and Multi-Agent Architectures

A hands-on workshop providing a comprehensive beginner's guide to using local Large Language Models (LLMs), Retrieval-Augmented Generation (RAG) systems, and multi-agent architectures to build powerful and efficient applications.

Participants will gain practical insights into running LLMs locally, integrating vectorized knowledge bases for RAG systems, and implementing multi-agent systems for various use cases. By the end of this session, you will have a clear understanding of what is achievable in your applications using open LLMs on personal machines or on-premise infrastructure.

## Prerequisites

### Hardware
- Laptop with at least 8GB RAM (16GB recommended for local LLMs) OR 8GB VRAM

### Software
1. **Python 3.10+** installed
2. **Code Editor:** VS Code (recommended) || PyCharm || Cursor
3. **Ollama:** Download and install from [ollama.com](https://ollama.com)
4. **Pull Base Models** (Run in terminal):
   - `ollama pull llama3.1:latest` || `ollama pull qwen3:8b`
   - `ollama pull nomic-embed-text`
5. **Git:** Installed

## Setting Up a Virtual Environment

To create and activate a Python virtual environment for this project:

### Create the virtual environment
```bash
python -m venv venv
```

### Activate the virtual environment

**On Linux/macOS:**
```bash
source venv/bin/activate
```

**On Windows:**
```bash
venv\Scripts\activate
```

Once activated, you'll see `(venv)` in your terminal prompt. You can then install the project dependencies using:
```bash
pip install -r requirements.txt
```

To deactivate the virtual environment when you're done:
```bash
deactivate
```

## Vector Database Options

This workshop provides **two implementations** for vector storage, allowing you to choose based on your needs:

### Option 1: SQLiteVSS (Classic - Default)
Uses SQLite with the Vector Similarity Search (VSS) extension via `pysqlite3-binary`.

**Folders:** `02_rag_lcel/`, `03_langgraph_react/`, `04_supervisor/`, `05_network/`

**Pros:**
- Lightweight single-file database (`devfest.db`)
- Good for learning and development
- Minimal dependencies

**Cons:**
- Requires `pysqlite3-binary` (not standard Python sqlite3)
- Platform-specific binary dependencies

### Option 2: ChromaDB (Alternative)
Uses ChromaDB, a modern AI-native vector database.

**Folders:** `02_rag_lcel_chromadb/`, `03_langgraph_react_chromadb/`, `04_supervisor_chromadb/`, `05_network_chromadb/`

**Pros:**
- Pure Python implementation (no binary dependencies)
- Production-ready with better scalability
- Modern API designed for AI applications
- Better cross-platform compatibility

**Cons:**
- Directory-based storage (`chroma_db/`)
- Slightly larger footprint

### How to Choose?
- **Use SQLiteVSS** if you want a simple, single-file database for learning
- **Use ChromaDB** if you want better compatibility and production-ready features

Both implementations work identically from a usage perspective - just use the corresponding folder's scripts!

## Runnable Scripts

### Classic SQLiteVSS Implementation

### 01_local_llm/hello_world.py
```bash
python3 01_local_llm/hello_world.py [--thinking]
```
- `--thinking`: Use qwen3:8b thinking model to show reasoning process

### 02_rag_lcel/ingest.py
```bash
python3 02_rag_lcel/ingest.py
```
No arguments.

### 02_rag_lcel/query.py
```bash
python3 02_rag_lcel/query.py [--interactive] [--question "YOUR_QUESTION"] [--thinking]
```
- `--interactive`: Run in interactive mode (ask multiple questions)
- `--question "YOUR_QUESTION"`: Question to ask (default: "Who is the CEO of DevFest Corp?")
- `--thinking`: Use qwen3:8b thinking model to show reasoning process

### 03_langgraph_react/agent.py
```bash
python3 03_langgraph_react/agent.py [--interactive] [--question "YOUR_QUESTION"] [--thinking]
```
- `--interactive`: Run in interactive mode (ask multiple questions)
- `--question "YOUR_QUESTION"`: Question to ask (default: "Who is the CEO of DevFest Corp?")
- `--thinking`: Use qwen3:8b thinking model to show reasoning process

**Note**: Run `python3 02_rag_lcel/ingest.py` first to create the knowledge base used by the agent's `lookup_policy` tool.

### 04_supervisor/supervisor.py
```bash
python3 04_supervisor/supervisor.py [--interactive] [--question "YOUR_QUESTION"] [--thinking]
```
- `--interactive`: Run in interactive mode (ask multiple questions)
- `--question "YOUR_QUESTION"`: Question to ask (default: "Who is the CEO of DevFest Corp?")
- `--thinking`: Use qwen3:8b thinking model for supervisor decisions

**Note**: Run `python3 02_rag_lcel/ingest.py` first to create the knowledge base used by the Researcher agent.

### 05_network/network.py
```bash
python3 05_network/network.py [--interactive] [--question "YOUR_QUESTION"] [--thinking]
```
- `--interactive`: Run in interactive mode (ask multiple questions)
- `--question "YOUR_QUESTION"`: Question to ask (default: "Who is the CEO of DevFest Corp?")
- `--thinking`: Use qwen3:8b thinking model for agent decisions

**Note**: Run `python3 02_rag_lcel/ingest.py` first to create the knowledge base used by the Researcher agent.

---

### ChromaDB Alternative Implementation

All scripts have ChromaDB equivalents in the `*_chromadb` folders. Simply replace the folder name in the commands:

#### 02_rag_lcel_chromadb/ingest.py
```bash
python3 02_rag_lcel_chromadb/ingest.py
```

#### 02_rag_lcel_chromadb/query.py
```bash
python3 02_rag_lcel_chromadb/query.py [--interactive] [--question "YOUR_QUESTION"] [--thinking]
```
- Same options as the classic version

#### 03_langgraph_react_chromadb/agent.py
```bash
python3 03_langgraph_react_chromadb/agent.py [--interactive] [--question "YOUR_QUESTION"] [--thinking]
```
- Same options as the classic version

**Note**: Run `python3 02_rag_lcel_chromadb/ingest.py` first to create the ChromaDB knowledge base.

#### 04_supervisor_chromadb/supervisor.py
```bash
python3 04_supervisor_chromadb/supervisor.py [--interactive] [--question "YOUR_QUESTION"] [--thinking]
```
- Same options as the classic version

**Note**: Run `python3 02_rag_lcel_chromadb/ingest.py` first.

#### 05_network_chromadb/network.py
```bash
python3 05_network_chromadb/network.py [--interactive] [--question "YOUR_QUESTION"] [--thinking]
```
- Same options as the classic version

**Note**: Run `python3 02_rag_lcel_chromadb/ingest.py` first.

### Database Storage Locations

- **SQLiteVSS**: Creates `devfest.db` (single file) in the project root
- **ChromaDB**: Creates `chroma_db/` (directory) in the project root

Both are automatically ignored by git (see `.gitignore`).
