"""Tests for CourseSearchTool.execute (search_tools.py)."""
from unittest.mock import MagicMock

import pytest

from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager
from vector_store import SearchResults


def make_results(docs, metas):
    return SearchResults(documents=docs, metadata=metas, distances=[0.1] * len(docs))


# ---------- unit tests with a mocked VectorStore ----------

class TestExecuteWithMockedStore:
    def setup_method(self):
        self.store = MagicMock()
        self.store.get_lesson_link.return_value = "http://lesson"
        self.tool = CourseSearchTool(self.store)

    def test_passes_all_params_to_store(self):
        self.store.search.return_value = make_results(["x"], [{"course_title": "C", "lesson_number": 1}])
        self.tool.execute(query="q", course_name="MCP", lesson_number=2)
        self.store.search.assert_called_once_with(query="q", course_name="MCP", lesson_number=2)

    def test_formats_results_with_header(self):
        self.store.search.return_value = make_results(
            ["content A"], [{"course_title": "Course X", "lesson_number": 3}])
        out = self.tool.execute(query="q")
        assert out == "[Course X - Lesson 3]\ncontent A"

    def test_header_without_lesson(self):
        self.store.search.return_value = make_results(["c"], [{"course_title": "Course X"}])
        assert self.tool.execute(query="q").startswith("[Course X]\n")

    def test_lesson_zero_is_shown_in_header(self):
        self.store.search.return_value = make_results(["c"], [{"course_title": "C", "lesson_number": 0}])
        assert self.tool.execute(query="q").startswith("[C - Lesson 0]")

    def test_error_is_returned_verbatim(self):
        self.store.search.return_value = SearchResults.empty("Search error: boom")
        assert self.tool.execute(query="q") == "Search error: boom"

    def test_empty_results_message(self):
        self.store.search.return_value = make_results([], [])
        assert self.tool.execute(query="q") == "No relevant content found."

    def test_empty_results_message_mentions_filters(self):
        self.store.search.return_value = make_results([], [])
        out = self.tool.execute(query="q", course_name="MCP", lesson_number=4)
        assert "course 'MCP'" in out and "lesson 4" in out

    def test_empty_results_message_mentions_lesson_zero(self):
        # Lesson 0 is a valid lesson (every transcript has "Lesson 0: Introduction")
        self.store.search.return_value = make_results([], [])
        assert "lesson 0" in self.tool.execute(query="q", lesson_number=0)

    def test_sources_deduplicated_per_lesson_with_links(self):
        metas = [{"course_title": "C", "lesson_number": 1},
                 {"course_title": "C", "lesson_number": 1},
                 {"course_title": "C", "lesson_number": 2}]
        self.store.search.return_value = make_results(["a", "b", "c"], metas)
        self.tool.execute(query="q")
        assert self.tool.last_sources == [
            {"label": "C - Lesson 1", "url": "http://lesson"},
            {"label": "C - Lesson 2", "url": "http://lesson"},
        ]

    def test_sources_not_set_on_empty_or_error(self):
        self.store.search.return_value = make_results([], [])
        self.tool.execute(query="q")
        assert self.tool.last_sources == []

    def test_tool_definition_shape(self):
        d = self.tool.get_tool_definition()
        assert d["name"] == "search_course_content"
        assert d["input_schema"]["required"] == ["query"]
        assert set(d["input_schema"]["properties"]) == {"query", "course_name", "lesson_number"}


class TestToolManager:
    def test_execute_unknown_tool(self):
        assert "not found" in ToolManager().execute_tool("nope")

    def test_register_and_execute(self):
        store = MagicMock()
        store.search.return_value = make_results(["c"], [{"course_title": "C", "lesson_number": 1}])
        store.get_lesson_link.return_value = None
        tm = ToolManager()
        tm.register_tool(CourseSearchTool(store))
        assert "c" in tm.execute_tool("search_course_content", query="q")
        assert tm.get_last_sources() == [{"label": "C - Lesson 1", "url": None}]
        tm.reset_sources()
        assert tm.get_last_sources() == []

    def test_unexpected_kwargs_raise(self):
        # If Claude hallucinates a parameter the tool raises TypeError rather than returning a string.
        store = MagicMock()
        tm = ToolManager()
        tm.register_tool(CourseSearchTool(store))
        with pytest.raises(TypeError):
            tm.execute_tool("search_course_content", query="q", bogus=1)


# ---------- integration tests against a real ChromaDB built from docs/ ----------

class TestExecuteAgainstRealStore:
    def test_content_query_returns_content(self, real_store):
        tool = CourseSearchTool(real_store)
        out = tool.execute(query="computer use screenshots")
        assert out and not out.startswith(("Search error", "No relevant", "No course found"))
        assert out.startswith("[")

    def test_returns_at_most_max_results_chunks(self, real_store):
        out = CourseSearchTool(real_store).execute(query="model")
        assert out.count("\n\n[") + 1 <= real_store.max_results

    def test_course_name_filter_restricts_results(self, real_store):
        tool = CourseSearchTool(real_store)
        out = tool.execute(query="what is covered", course_name="MCP")
        assert not out.startswith(("Search error", "No relevant", "No course found")), out
        titles = {s["label"].split(" - Lesson")[0] for s in tool.last_sources}
        assert len(titles) == 1

    def test_lesson_filter_restricts_results(self, real_store):
        tool = CourseSearchTool(real_store)
        out = tool.execute(query="introduction", lesson_number=0)
        assert not out.startswith(("Search error", "No relevant")), out
        assert all(s["label"].endswith("Lesson 0") for s in tool.last_sources)

    def test_course_and_lesson_filter_combined(self, real_store):
        tool = CourseSearchTool(real_store)
        out = tool.execute(query="overview", course_name="Computer Use", lesson_number=1)
        assert not out.startswith(("Search error", "No relevant")), out
        assert all(s["label"].endswith("Lesson 1") for s in tool.last_sources)

    def test_sources_have_clickable_urls(self, real_store):
        tool = CourseSearchTool(real_store)
        tool.execute(query="introduction", lesson_number=0)
        assert tool.last_sources and all(s["url"] and s["url"].startswith("http") for s in tool.last_sources)

    @pytest.mark.xfail(reason="Known limitation: _resolve_course_name always returns the nearest course. "
                              "A distance cutoff can't separate bogus names (Python=1.54) from valid short "
                              "ones (MCP=1.55), so no threshold is applied.", strict=True)
    def test_unknown_course_name_should_not_silently_match_another_course(self, real_store):
        # Desired behaviour: a bogus name yields "No course found". Currently returns another course's content.
        out = CourseSearchTool(real_store).execute(query="anything", course_name="Quantum Basket Weaving 101")
        assert out.startswith(("No course found", "No relevant")), out[:120]

    def test_outline_tool(self, real_store):
        out = CourseOutlineTool(real_store).execute("Computer Use")
        assert out.startswith("Course: ") and "Lesson 0:" in out and "Link: http" in out
