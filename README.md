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
