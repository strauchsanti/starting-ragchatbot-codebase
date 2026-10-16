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
