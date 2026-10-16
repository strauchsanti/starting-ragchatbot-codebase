"""End-to-end tests of RAGSystem.query with a real vector store and a scripted Claude."""
import os

import pytest

from ai_generator import AIGenerator
from config import Config
from document_processor import DocumentProcessor
from rag_system import RAGSystem
from search_tools import ToolManager, CourseSearchTool, CourseOutlineTool
from session_manager import SessionManager
from conftest import response, text_block, tool_use_block


@pytest.fixture
def rag(real_store, mock_client):
    cfg = Config()
    cfg.ANTHROPIC_API_KEY = "test"
    # Build RAGSystem without opening a second Chroma client: reuse the session store.
    system = RAGSystem.__new__(RAGSystem)
    system.config = cfg
    system.document_processor = DocumentProcessor(cfg.CHUNK_SIZE, cfg.CHUNK_OVERLAP)
    system.vector_store = real_store
    system.ai_generator = AIGenerator("test", cfg.ANTHROPIC_MODEL)
    system.session_manager = SessionManager(cfg.MAX_HISTORY)
    system.tool_manager = ToolManager()
    system.search_tool = CourseSearchTool(real_store)
    system.outline_tool = CourseOutlineTool(real_store)
    system.tool_manager.register_tool(system.search_tool)
    system.tool_manager.register_tool(system.outline_tool)
    system.mock_client = mock_client
    return system


def script_search(rag, tool_input, final="Final answer"):
    rag.mock_client.messages.create.side_effect = [
        response(tool_use_block("search_course_content", tool_input), stop_reason="tool_use"),
        response(text_block(final)),
    ]


def tool_result_sent_to_claude(rag):
    msgs = rag.mock_client.messages.create.call_args_list[1].kwargs["messages"]
    return msgs[-1]["content"][0]["content"]


class TestContentQueries:
    def test_both_tools_are_offered(self, rag):
        rag.mock_client.messages.create.return_value = response(text_block("hi"))
        rag.query("What is MCP?")
        names = {t["name"] for t in rag.mock_client.messages.create.call_args.kwargs["tools"]}
        assert names == {"search_course_content", "get_course_outline"}

    def test_query_is_wrapped_in_prompt(self, rag):
        rag.mock_client.messages.create.return_value = response(text_block("hi"))
        rag.query("What is MCP?")
        content = rag.mock_client.messages.create.call_args.kwargs["messages"][0]["content"]
        assert "What is MCP?" in content

    def test_content_query_returns_answer_and_sources(self, rag):
        script_search(rag, {"query": "computer use", "course_name": "Computer Use"})
        answer, sources = rag.query("How does computer use work?")
        assert answer == "Final answer"
        assert sources and all({"label", "url"} <= s.keys() for s in sources)

    def test_real_chunks_reach_claude_in_second_call(self, rag):
        script_search(rag, {"query": "computer use screenshots"})
        rag.query("q")
        result = tool_result_sent_to_claude(rag)
        assert result.startswith("[") and len(result) > 200
        assert not result.startswith(("Search error", "No relevant", "No course found"))

    def test_no_search_for_general_question(self, rag):
        rag.mock_client.messages.create.return_value = response(text_block("Paris"))
        answer, sources = rag.query("Capital of France?")
        assert answer == "Paris" and sources == []
        assert rag.mock_client.messages.create.call_count == 1

    def test_sources_reset_between_queries(self, rag):
        script_search(rag, {"query": "computer use"})
        _, first = rag.query("q1")
        assert first
        rag.mock_client.messages.create.side_effect = None
        rag.mock_client.messages.create.return_value = response(text_block("no search"))
        _, second = rag.query("q2")
        assert second == []

    def test_lesson_and_course_filters_flow_through(self, rag):
        script_search(rag, {"query": "overview", "course_name": "Computer Use", "lesson_number": 1})
        _, sources = rag.query("q")
        assert sources and all(s["label"].endswith("Lesson 1") for s in sources)

    def test_outline_query(self, rag):
        rag.mock_client.messages.create.side_effect = [
            response(tool_use_block("get_course_outline", {"course_title": "Computer Use"}), stop_reason="tool_use"),
            response(text_block("outline")),
        ]
        rag.query("Outline of the computer use course")
        result = tool_result_sent_to_claude(rag)
        assert "Lessons (" in result and "Lesson 0:" in result

    def test_no_match_message_reaches_claude(self, rag):
        script_search(rag, {"query": "x", "course_name": "Computer Use", "lesson_number": 999})
        rag.query("q")
        assert tool_result_sent_to_claude(rag).startswith("No relevant content found")


class TestSessions:
    def test_history_recorded_and_injected(self, rag):
        rag.mock_client.messages.create.return_value = response(text_block("A1"))
        sid = rag.session_manager.create_session()
        rag.query("Q1", sid)
        rag.query("Q2", sid)
        system = rag.mock_client.messages.create.call_args.kwargs["system"]
        assert "Q1" in system and "A1" in system

    def test_failed_query_does_not_pollute_history(self, rag):
        rag.mock_client.messages.create.side_effect = RuntimeError("api down")
        sid = rag.session_manager.create_session()
        with pytest.raises(RuntimeError):
            rag.query("Q1", sid)
        assert not rag.session_manager.get_conversation_history(sid)


class TestErrorHandling:
    def test_search_backend_error_still_produces_answer(self, rag, monkeypatch):
        class Broken:
            def query(self, **kw):
                raise RuntimeError("db")
        monkeypatch.setattr(rag.vector_store, "course_content", Broken())
        script_search(rag, {"query": "x"})
        answer, _ = rag.query("q")
        assert answer == "Final answer"
        assert tool_result_sent_to_claude(rag).startswith("Search error")

    def test_tool_crash_does_not_fail_whole_query(self, rag):
        # A bad tool call from Claude (unknown kwarg) currently raises TypeError -> HTTP 500 -> "query failed"
        script_search(rag, {"query": "x", "bogus_param": 1})
        answer, _ = rag.query("q")
        assert answer == "Final answer"


@pytest.mark.skipif(not os.environ.get("RUN_LIVE"), reason="set RUN_LIVE=1 to call the real Anthropic API")
class TestLive:
    def test_real_claude_searches_and_answers(self, real_store):
        cfg = Config()
        tm = ToolManager()
        tm.register_tool(CourseSearchTool(real_store))
        calls = []
        orig = tm.execute_tool

        def spy(name, **kw):
            calls.append((name, kw))
            return orig(name, **kw)
        tm.execute_tool = spy
        out = AIGenerator(cfg.ANTHROPIC_API_KEY, cfg.ANTHROPIC_MODEL).generate_response(
            "Answer this question about course materials: What does the computer use course teach about screenshots?",
            tools=tm.get_tool_definitions(), tool_manager=tm)
        assert calls and calls[0][0] == "search_course_content"
        assert isinstance(out, str) and out.strip()
