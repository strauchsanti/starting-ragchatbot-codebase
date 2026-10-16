"""Tests that AIGenerator correctly offers and invokes the CourseSearchTool (ai_generator.py)."""
from unittest.mock import MagicMock

import pytest

from ai_generator import AIGenerator
from search_tools import CourseSearchTool
from conftest import response, text_block, thinking_block, tool_use_block

SEARCH_INPUT = {"query": "what is MCP", "course_name": "MCP", "lesson_number": 1}
TOOLS = [CourseSearchTool(MagicMock()).get_tool_definition()]


@pytest.fixture
def gen(mock_client):
    return AIGenerator("key", "claude-sonnet-5-5")


@pytest.fixture
def tool_manager():
    tm = MagicMock()
    tm.execute_tool.return_value = "[Course - Lesson 1]\nMCP is a protocol"
    return tm


class TestToolOffering:
    def test_tools_and_auto_choice_sent_on_first_call(self, gen, mock_client):
        mock_client.messages.create.return_value = response(text_block("hi"))
        gen.generate_response("q", tools=TOOLS, tool_manager=MagicMock())
        kw = mock_client.messages.create.call_args.kwargs
        assert kw["tools"] == TOOLS
        assert kw["tool_choice"] == {"type": "auto"}

    def test_no_tools_key_when_none_given(self, gen, mock_client):
        mock_client.messages.create.return_value = response(text_block("hi"))
        gen.generate_response("q")
        assert "tools" not in mock_client.messages.create.call_args.kwargs

    def test_no_temperature_param(self, gen, mock_client):
        # claude-sonnet-5-5 rejects `temperature`
        mock_client.messages.create.return_value = response(text_block("hi"))
        gen.generate_response("q", tools=TOOLS)
        assert "temperature" not in mock_client.messages.create.call_args.kwargs

    def test_history_goes_into_system_prompt(self, gen, mock_client):
        mock_client.messages.create.return_value = response(text_block("hi"))
        gen.generate_response("q", conversation_history="User: a\nAssistant: b")
        assert "User: a" in mock_client.messages.create.call_args.kwargs["system"]

    def test_system_prompt_directs_search_for_course_questions(self):
        p = AIGenerator.SYSTEM_PROMPT
        assert "search_course_content" in p and "get_course_outline" in p


class TestToolExecutionFlow:
    def test_direct_answer_makes_one_call_and_no_tool_exec(self, gen, mock_client, tool_manager):
        mock_client.messages.create.return_value = response(text_block("4"), stop_reason="end_turn")
        assert gen.generate_response("2+2?", tools=TOOLS, tool_manager=tool_manager) == "4"
        assert mock_client.messages.create.call_count == 1
        tool_manager.execute_tool.assert_not_called()

    def test_tool_use_executes_tool_with_claudes_arguments(self, gen, mock_client, tool_manager):
        mock_client.messages.create.side_effect = [
            response(tool_use_block("search_course_content", SEARCH_INPUT), stop_reason="tool_use"),
            response(text_block("answer")),
        ]
        out = gen.generate_response("q", tools=TOOLS, tool_manager=tool_manager)
        tool_manager.execute_tool.assert_called_once_with("search_course_content", **SEARCH_INPUT)
        assert out == "answer"

    def test_second_call_contains_tool_result_and_no_tools(self, gen, mock_client, tool_manager):
        first = response(tool_use_block("search_course_content", SEARCH_INPUT, id="toolu_X"), stop_reason="tool_use")
        mock_client.messages.create.side_effect = [first, response(text_block("answer"))]
        gen.generate_response("q", tools=TOOLS, tool_manager=tool_manager)
        assert mock_client.messages.create.call_count == 2
        second = mock_client.messages.create.call_args_list[1].kwargs
        assert "tools" not in second and "tool_choice" not in second
        msgs = second["messages"]
        assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
        assert msgs[1]["content"] == first.content
        assert msgs[2]["content"] == [{"type": "tool_result", "tool_use_id": "toolu_X",
                                        "content": "[Course - Lesson 1]\nMCP is a protocol"}]
        assert second["system"] == mock_client.messages.create.call_args_list[0].kwargs["system"]

    def test_thinking_block_before_text_is_skipped(self, gen, mock_client):
        mock_client.messages.create.return_value = response(thinking_block(), text_block("real"))
        assert gen.generate_response("q") == "real"

    def test_thinking_block_before_tool_use(self, gen, mock_client, tool_manager):
        mock_client.messages.create.side_effect = [
            response(thinking_block(), text_block("let me search"),
                     tool_use_block("search_course_content", SEARCH_INPUT), stop_reason="tool_use"),
            response(thinking_block(), text_block("final")),
        ]
        assert gen.generate_response("q", tools=TOOLS, tool_manager=tool_manager) == "final"

    def test_parallel_tool_calls_all_executed(self, gen, mock_client, tool_manager):
        mock_client.messages.create.side_effect = [
            response(tool_use_block("search_course_content", {"query": "a"}, id="1"),
                     tool_use_block("get_course_outline", {"course_title": "b"}, id="2"), stop_reason="tool_use"),
            response(text_block("done")),
        ]
        gen.generate_response("q", tools=TOOLS, tool_manager=tool_manager)
        assert tool_manager.execute_tool.call_count == 2
        results = mock_client.messages.create.call_args_list[1].kwargs["messages"][-1]["content"]
        assert [r["tool_use_id"] for r in results] == ["1", "2"]

    def test_tool_use_without_tool_manager_returns_text_not_crash(self, gen, mock_client):
        mock_client.messages.create.return_value = response(
            text_block("searching"), tool_use_block("search_course_content", SEARCH_INPUT), stop_reason="tool_use")
        assert gen.generate_response("q", tools=TOOLS, tool_manager=None) == "searching"


class TestFailureModes:
    """These are the paths that surface to the user as 'query failed' (app.py turns any exception into a 500)."""

    def test_tool_exception_is_handled_gracefully(self, gen, mock_client, tool_manager):
        tool_manager.execute_tool.side_effect = RuntimeError("chroma exploded")
        mock_client.messages.create.side_effect = [
            response(tool_use_block("search_course_content", SEARCH_INPUT), stop_reason="tool_use"),
            response(text_block("Sorry, search failed")),
        ]
        # Desired: the error is fed back to Claude as an is_error tool_result and a reply is produced.
        out = gen.generate_response("q", tools=TOOLS, tool_manager=tool_manager)
        assert out == "Sorry, search failed"

    def test_response_without_text_block_is_handled(self, gen, mock_client):
        # e.g. stop_reason=max_tokens while still "thinking"
        mock_client.messages.create.return_value = response(thinking_block(), stop_reason="max_tokens")
        assert isinstance(gen.generate_response("q"), str)

    def test_final_response_without_text_block_is_handled(self, gen, mock_client, tool_manager):
        mock_client.messages.create.side_effect = [
            response(tool_use_block("search_course_content", SEARCH_INPUT), stop_reason="tool_use"),
            response(thinking_block(), stop_reason="max_tokens"),
        ]
        out = gen.generate_response("q", tools=TOOLS, tool_manager=tool_manager)
        assert isinstance(out, str)

    def test_max_tokens_is_large_enough_for_thinking_plus_answer(self, gen):
        assert gen.base_params["max_tokens"] >= 1500
