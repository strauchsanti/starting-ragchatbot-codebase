# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A RAG chatbot over course transcripts: FastAPI backend, ChromaDB vector store, Claude for generation, and a static HTML/JS frontend served by the same FastAPI app. Python 3.13+, managed with `uv`.

## Commands

```bash
uv sync                                   # install dependencies
./run.sh                                  # run the app (http://localhost:8000, API docs at /docs)
cd backend && uv run uvicorn app:app --reload --port 8000   # equivalent manual start
```

- Always use `uv` to run the server and manage dependencies (`uv sync`, `uv run ...`). Never use `pip` directly.
- Requires `ANTHROPIC_API_KEY` in a root-level `.env` (copy `.env.example`). The key must be workspace-scoped, otherwise the API returns a 400 about `anthropic-workspace-id`. `.env` is only read at startup, so restart after changing it.
- There are no tests, linter, or build step configured. `main.py` at the root is an unused stub.
- **Always run the server from `backend/`.** `config.py` uses relative paths (`./chroma_db`) and `app.py` uses `../docs` and `../frontend`, so launching from another directory creates a second empty ChromaDB or breaks static serving.

## Architecture

`RAGSystem` (`backend/rag_system.py`) wires together `DocumentProcessor`, `VectorStore`, `AIGenerator`, `SessionManager` and `ToolManager`. `app.py` instantiates it once at import time.

**Retrieval is tool-based, not automatic.** Each query makes up to two Claude calls (`ai_generator.py`). The first offers the `search_course_content` tool, and Claude decides whether to search. If it does, `ToolManager` → `CourseSearchTool` → `VectorStore.search` runs and the results go into a second call that has *no tools*, so at most one search happens per question. General-knowledge questions are answered without any retrieval. The system prompt in `AIGenerator.SYSTEM_PROMPT` enforces this behavior.

**Two Chroma collections** (`vector_store.py`): `course_catalog` (one entry per course, used to resolve fuzzy course names such as "MCP" to a full title via embedding search) and `course_content` (text chunks, filtered by `course_title` and `lesson_number` metadata). Course title is the unique ID, and chunk IDs are `{title_with_underscores}_{chunk_index}`.

**Ingestion** runs on FastAPI startup (`app.py` startup event) over every `.pdf/.docx/.txt` in `docs/`, skipping courses whose title already exists in Chroma. This means editing a transcript does not update already-indexed content. To re-index, delete `backend/chroma_db/` (or call `add_course_folder(..., clear_existing=True)`).

**Transcript format** (parsed by regex in `document_processor.py`): line 1 `Course Title:`, line 2 `Course Link:`, line 3 `Course Instructor:`, then `Lesson N: <title>` markers, each optionally followed by a `Lesson Link:` line. Files without lesson markers are indexed as one unstructured document.

**Sources shown in the UI** are collected as side state on the `CourseSearchTool` instance (`last_sources`), read and then reset by `RAGSystem.query`. This state is shared across requests.

**Sessions** are in-memory only (`SessionManager`) and are lost on restart. History is injected into the Claude *system prompt* as text, not sent as chat messages, and is limited to `MAX_HISTORY` exchanges.

All tunables (model, chunk size/overlap, result count, history length, embedding model) are in `backend/config.py`. The model is `claude-sonnet-5-5`, which rejects the `temperature` parameter, so don't add it to the request params in `AIGenerator`.
