"""Shared fixtures. Tests build a throwaway ChromaDB from docs/ so the real backend/chroma_db is never touched."""
import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DOCS_DIR = os.path.abspath(os.path.join(BACKEND_DIR, "..", "docs"))
sys.path.insert(0, BACKEND_DIR)

from config import Config  # noqa: E402
from document_processor import DocumentProcessor  # noqa: E402
from vector_store import VectorStore  # noqa: E402


# ---- fake Anthropic response helpers ------------------------------------------------

def text_block(text):
    return SimpleNamespace(type="text", text=text)


def thinking_block(text="hmm"):
    return SimpleNamespace(type="thinking", thinking=text)


def tool_use_block(name, input, id="toolu_1"):
    return SimpleNamespace(type="tool_use", name=name, input=input, id=id)


def response(*blocks, stop_reason="end_turn"):
    return SimpleNamespace(content=list(blocks), stop_reason=stop_reason)


# ---- real vector store over the real transcripts, in a temp dir ---------------------

@pytest.fixture(scope="session")
def real_store(tmp_path_factory):
    path = tmp_path_factory.mktemp("chroma")
    cfg = Config()
    store = VectorStore(str(path), cfg.EMBEDDING_MODEL, cfg.MAX_RESULTS)
    processor = DocumentProcessor(cfg.CHUNK_SIZE, cfg.CHUNK_OVERLAP)
    for name in sorted(os.listdir(DOCS_DIR)):
        if name.endswith(".txt"):
            course, chunks = processor.process_course_document(os.path.join(DOCS_DIR, name))
            store.add_course_metadata(course)
            store.add_course_content(chunks)
    return store


@pytest.fixture
def mock_client(monkeypatch):
    """Patch anthropic.Anthropic so AIGenerator gets a scriptable client."""
    client = MagicMock()
    monkeypatch.setattr("ai_generator.anthropic.Anthropic", lambda api_key=None: client)
    return client


# ---- API test app -------------------------------------------------------------------
# backend/app.py builds a real RAGSystem and mounts ../frontend at import time, so it can't
# be imported in tests. build_test_app mirrors its endpoints over an injected rag_system.

SAMPLE_COURSES = {"total_courses": 2, "course_titles": ["MCP Course", "Chroma Course"]}
SAMPLE_SOURCES = [{"label": "MCP Course - Lesson 1", "url": "https://example.com/l1"}, {"label": "Chroma Course"}]


def build_test_app(rag_system):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel
    from typing import List, Optional

    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None

    class Source(BaseModel):
        label: str
        url: Optional[str] = None

    class QueryResponse(BaseModel):
        answer: str
        sources: List[Source]
        session_id: str

    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]

    app = FastAPI()

    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id or rag_system.session_manager.create_session()
            answer, sources = rag_system.query(request.query, session_id)
            return QueryResponse(answer=answer, sources=sources, session_id=session_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"],
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/api/session/{session_id}")
    async def delete_session(session_id: str):
        return {"deleted": rag_system.session_manager.delete_session(session_id)}

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return "<html><body>Course Materials RAG</body></html>"

    return app


@pytest.fixture
def mock_rag_system():
    rag = MagicMock()
    rag.session_manager.create_session.return_value = "session_1"
    rag.session_manager.delete_session.return_value = True
    rag.query.return_value = ("Test answer", list(SAMPLE_SOURCES))
    rag.get_course_analytics.return_value = dict(SAMPLE_COURSES)
    return rag


@pytest.fixture
def client(mock_rag_system):
    from fastapi.testclient import TestClient

    return TestClient(build_test_app(mock_rag_system))
