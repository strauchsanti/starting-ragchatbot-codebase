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
    def test_direct_answer_makes_one_call_and_no_tool_exec(
        self, gen, mock_client, tool_manager
    ):
        mock_client.messages.create.return_value = response(
            text_block("4"), stop_reason="end_turn"
        )
        assert (
            gen.generate_response("2+2?", tools=TOOLS, tool_manager=tool_manager) == "4"
        )
        assert mock_client.messages.create.call_count == 1
        tool_manager.execute_tool.assert_not_called()

    def test_tool_use_executes_tool_with_claudes_arguments(
        self, gen, mock_client, tool_manager
    ):
        mock_client.messages.create.side_effect = [
            response(
                tool_use_block("search_course_content", SEARCH_INPUT),
                stop_reason="tool_use",
            ),
            response(text_block("answer")),
        ]
        out = gen.generate_response("q", tools=TOOLS, tool_manager=tool_manager)
        tool_manager.execute_tool.assert_called_once_with(
            "search_course_content", **SEARCH_INPUT
        )
        assert out == "answer"

    def test_second_call_contains_tool_result_and_keeps_tools(
        self, gen, mock_client, tool_manager
    ):
        first = response(
            tool_use_block("search_course_content", SEARCH_INPUT, id="toolu_X"),
            stop_reason="tool_use",
        )
        mock_client.messages.create.side_effect = [
            first,
            response(text_block("answer")),
        ]
        gen.generate_response("q", tools=TOOLS, tool_manager=tool_manager)
        assert mock_client.messages.create.call_count == 2
        second = mock_client.messages.create.call_args_list[1].kwargs
        # Claude may still chain another tool call after seeing the first result
        assert second["tools"] == TOOLS and second["tool_choice"] == {"type": "auto"}
        msgs = second["messages"]
        assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
        assert msgs[1]["content"] == first.content
        assert msgs[2]["content"] == [
            {
                "type": "tool_result",
                "tool_use_id": "toolu_X",
                "content": "[Course - Lesson 1]\nMCP is a protocol",
            }
        ]
        assert (
            second["system"]
            == mock_client.messages.create.call_args_list[0].kwargs["system"]
        )

    def test_thinking_block_before_text_is_skipped(self, gen, mock_client):
        mock_client.messages.create.return_value = response(
            thinking_block(), text_block("real")
        )
        assert gen.generate_response("q") == "real"

    def test_thinking_block_before_tool_use(self, gen, mock_client, tool_manager):
        mock_client.messages.create.side_effect = [
            response(
                thinking_block(),
                text_block("let me search"),
                tool_use_block("search_course_content", SEARCH_INPUT),
                stop_reason="tool_use",
            ),
            response(thinking_block(), text_block("final")),
        ]
        assert (
            gen.generate_response("q", tools=TOOLS, tool_manager=tool_manager)
            == "final"
        )

    def test_parallel_tool_calls_all_executed(self, gen, mock_client, tool_manager):
        mock_client.messages.create.side_effect = [
            response(
                tool_use_block("search_course_content", {"query": "a"}, id="1"),
                tool_use_block("get_course_outline", {"course_title": "b"}, id="2"),
                stop_reason="tool_use",
            ),
            response(text_block("done")),
        ]
        gen.generate_response("q", tools=TOOLS, tool_manager=tool_manager)
        assert tool_manager.execute_tool.call_count == 2
        results = mock_client.messages.create.call_args_list[1].kwargs["messages"][-1][
            "content"
        ]
        assert [r["tool_use_id"] for r in results] == ["1", "2"]

    def test_tool_use_without_tool_manager_returns_text_not_crash(
        self, gen, mock_client
    ):
        mock_client.messages.create.return_value = response(
            text_block("searching"),
            tool_use_block("search_course_content", SEARCH_INPUT),
            stop_reason="tool_use",
        )
        assert gen.generate_response("q", tools=TOOLS, tool_manager=None) == "searching"


OUTLINE_INPUT = {"course_title": "MCP"}


def outline_use(id="o1"):
    return response(
        tool_use_block("get_course_outline", OUTLINE_INPUT, id=id),
        stop_reason="tool_use",
    )


def search_use(id="s1", input=None):
    return response(
        tool_use_block("search_course_content", input or SEARCH_INPUT, id=id),
        stop_reason="tool_use",
    )


def calls(mock_client):
    return [c.kwargs for c in mock_client.messages.create.call_args_list]


class TestSequentialToolCalls:
    def test_two_rounds_make_three_requests_and_return_final_text(
        self, gen, mock_client, tool_manager
    ):
        tool_manager.execute_tool.side_effect = [
            "Lesson 4: Prompt caching",
            "[Other course] caching intro",
        ]
        mock_client.messages.create.side_effect = [
            outline_use(),
            search_use(input={"query": "Prompt caching"}),
            response(text_block("Course Y covers it")),
        ]
        out = gen.generate_response(
            "same topic as lesson 4 of MCP?", tools=TOOLS, tool_manager=tool_manager
        )
        assert out == "Course Y covers it"
        assert mock_client.messages.create.call_count == 3
        assert [c.args[0] for c in tool_manager.execute_tool.call_args_list] == [
            "get_course_outline",
            "search_course_content",
        ]
        assert tool_manager.execute_tool.call_args_list[1].kwargs == {
            "query": "Prompt caching"
        }

    def test_stops_after_one_round_when_no_further_tool_use(
        self, gen, mock_client, tool_manager
    ):
        mock_client.messages.create.side_effect = [
            search_use(),
            response(text_block("done")),
        ]
        assert (
            gen.generate_response("q", tools=TOOLS, tool_manager=tool_manager) == "done"
        )
        assert mock_client.messages.create.call_count == 2
        assert tool_manager.execute_tool.call_count == 1

    def test_capped_at_two_rounds_even_if_claude_keeps_asking(
        self, gen, mock_client, tool_manager
    ):
        mock_client.messages.create.side_effect = [
            search_use("1"),
            search_use("2"),
            search_use("3"),
            search_use("4"),
        ]
        out = gen.generate_response("q", tools=TOOLS, tool_manager=tool_manager)
        assert isinstance(out, str)
        assert tool_manager.execute_tool.call_count == 2
        assert mock_client.messages.create.call_count == 3

    def test_tools_offered_on_first_two_requests_but_not_after_the_cap(
        self, gen, mock_client, tool_manager
    ):
        mock_client.messages.create.side_effect = [
            search_use("1"),
            search_use("2"),
            response(text_block("x")),
        ]
        gen.generate_response("q", tools=TOOLS, tool_manager=tool_manager)
        first, second, third = calls(mock_client)
        assert first["tools"] == TOOLS and second["tools"] == TOOLS
        assert "tools" not in third and "tool_choice" not in third

    def test_context_accumulates_across_rounds(self, gen, mock_client, tool_manager):
        tool_manager.execute_tool.side_effect = ["outline text", "search text"]
        r1, r2 = outline_use("o1"), search_use("s1")
        mock_client.messages.create.side_effect = [r1, r2, response(text_block("x"))]
        gen.generate_response(
            "the question",
            tools=TOOLS,
            tool_manager=tool_manager,
            conversation_history="User: a\nAssistant: b",
        )
        third = calls(mock_client)[2]
        msgs = third["messages"]
        assert [m["role"] for m in msgs] == [
            "user",
            "assistant",
            "user",
            "assistant",
            "user",
        ]
        assert msgs[0]["content"] == "the question"
        assert msgs[1]["content"] == r1.content and msgs[3]["content"] == r2.content
        assert msgs[2]["content"][0]["content"] == "outline text"
        assert msgs[4]["content"][0]["content"] == "search text"
        assert len({c["system"] for c in calls(mock_client)}) == 1
        assert "User: a" in third["system"]

    def test_failure_in_first_round_ends_with_toolless_request(
        self, gen, mock_client, tool_manager
    ):
        tool_manager.execute_tool.side_effect = RuntimeError("boom")
        mock_client.messages.create.side_effect = [
            search_use(),
            response(text_block("sorry")),
        ]
        assert (
            gen.generate_response("q", tools=TOOLS, tool_manager=tool_manager)
            == "sorry"
        )
        assert mock_client.messages.create.call_count == 2
        assert "tools" not in calls(mock_client)[1]

    def test_failure_in_second_round_ends_the_loop(
        self, gen, mock_client, tool_manager
    ):
        tool_manager.execute_tool.side_effect = ["ok", RuntimeError("boom")]
        mock_client.messages.create.side_effect = [
            outline_use(),
            search_use(),
            response(text_block("partial")),
        ]
        assert (
            gen.generate_response("q", tools=TOOLS, tool_manager=tool_manager)
            == "partial"
        )
        last = calls(mock_client)[2]
        assert "tools" not in last
        assert last["messages"][-1]["content"][0]["is_error"] is True

    def test_parallel_failure_still_returns_a_result_for_every_tool_use(
        self, gen, mock_client, tool_manager
    ):
        tool_manager.execute_tool.side_effect = [RuntimeError("boom"), "fine"]
        mock_client.messages.create.side_effect = [
            response(
                tool_use_block("search_course_content", SEARCH_INPUT, id="1"),
                tool_use_block("get_course_outline", OUTLINE_INPUT, id="2"),
                stop_reason="tool_use",
            ),
            response(text_block("done")),
        ]
        gen.generate_response("q", tools=TOOLS, tool_manager=tool_manager)
        results = calls(mock_client)[1]["messages"][-1]["content"]
        assert [r["tool_use_id"] for r in results] == ["1", "2"]
        assert results[0].get("is_error") is True and "is_error" not in results[1]

    def test_no_temperature_on_any_request(self, gen, mock_client, tool_manager):
        mock_client.messages.create.side_effect = [
            search_use("1"),
            search_use("2"),
            response(text_block("x")),
        ]
        gen.generate_response("q", tools=TOOLS, tool_manager=tool_manager)
        assert all("temperature" not in c for c in calls(mock_client))

    def test_system_prompt_allows_two_calls_and_keeps_outline_format(self):
        p = AIGenerator.SYSTEM_PROMPT
        assert "One tool call per query maximum" not in p
        assert "Up to 2 sequential tool calls" in p
        assert "Lessons (N):" in p and "- Lesson <number>: <title>" in p


class TestFailureModes:
    """These are the paths that surface to the user as 'query failed' (app.py turns any exception into a 500)."""

    def test_tool_exception_is_handled_gracefully(self, gen, mock_client, tool_manager):
        tool_manager.execute_tool.side_effect = RuntimeError("chroma exploded")
        mock_client.messages.create.side_effect = [
            response(
                tool_use_block("search_course_content", SEARCH_INPUT),
                stop_reason="tool_use",
            ),
            response(text_block("Sorry, search failed")),
        ]
        # Desired: the error is fed back to Claude as an is_error tool_result and a reply is produced.
        out = gen.generate_response("q", tools=TOOLS, tool_manager=tool_manager)
        assert out == "Sorry, search failed"
        second = mock_client.messages.create.call_args_list[1].kwargs
        assert "tools" not in second
        result = second["messages"][-1]["content"][0]
        assert result["is_error"] is True and "chroma exploded" in result["content"]

    def test_response_without_text_block_is_handled(self, gen, mock_client):
        # e.g. stop_reason=max_tokens while still "thinking"
        mock_client.messages.create.return_value = response(
            thinking_block(), stop_reason="max_tokens"
        )
        assert isinstance(gen.generate_response("q"), str)

    def test_final_response_without_text_block_is_handled(
        self, gen, mock_client, tool_manager
    ):
        mock_client.messages.create.side_effect = [
            response(
                tool_use_block("search_course_content", SEARCH_INPUT),
                stop_reason="tool_use",
            ),
            response(thinking_block(), stop_reason="max_tokens"),
        ]
        out = gen.generate_response("q", tools=TOOLS, tool_manager=tool_manager)
        assert isinstance(out, str)

    def test_max_tokens_is_large_enough_for_thinking_plus_answer(self, gen):
        assert gen.base_params["max_tokens"] >= 1500
