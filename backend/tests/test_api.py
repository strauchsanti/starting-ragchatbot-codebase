"""API endpoint tests against the test app from conftest (real app.py can't be imported without ../frontend)."""

import pytest

from conftest import SAMPLE_COURSES, SAMPLE_SOURCES

pytestmark = pytest.mark.api


class TestQueryEndpoint:
    def test_new_session_created_when_missing(self, client, mock_rag_system):
        r = client.post("/api/query", json={"query": "What is MCP?"})
        assert r.status_code == 200
        body = r.json()
        assert body["answer"] == "Test answer"
        assert body["session_id"] == "session_1"
        assert body["sources"] == [
            {"label": "MCP Course - Lesson 1", "url": "https://example.com/l1"},
            {"label": "Chroma Course", "url": None},
        ]
        mock_rag_system.session_manager.create_session.assert_called_once()
        mock_rag_system.query.assert_called_once_with("What is MCP?", "session_1")

    def test_existing_session_reused(self, client, mock_rag_system):
        r = client.post("/api/query", json={"query": "q", "session_id": "abc"})
        assert r.json()["session_id"] == "abc"
        mock_rag_system.session_manager.create_session.assert_not_called()
        mock_rag_system.query.assert_called_once_with("q", "abc")

    def test_empty_sources(self, client, mock_rag_system):
        mock_rag_system.query.return_value = ("General answer", [])
        assert client.post("/api/query", json={"query": "hi"}).json()["sources"] == []

    @pytest.mark.parametrize(
        "payload", [{}, {"session_id": "x"}, {"query": 123}, {"query": None}]
    )
    def test_invalid_body_rejected(self, client, mock_rag_system, payload):
        assert client.post("/api/query", json=payload).status_code == 422
        mock_rag_system.query.assert_not_called()

    def test_non_json_body_rejected(self, client):
        assert client.post("/api/query", content="not json").status_code == 422

    def test_rag_failure_returns_500(self, client, mock_rag_system):
        mock_rag_system.query.side_effect = RuntimeError("boom")
        r = client.post("/api/query", json={"query": "q"})
        assert r.status_code == 500
        assert r.json()["detail"] == "boom"

    def test_get_not_allowed(self, client):
        assert client.get("/api/query").status_code == 405


class TestCoursesEndpoint:
    def test_returns_stats(self, client):
        r = client.get("/api/courses")
        assert r.status_code == 200
        assert r.json() == SAMPLE_COURSES

    def test_no_courses(self, client, mock_rag_system):
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": [],
        }
        assert client.get("/api/courses").json() == {
            "total_courses": 0,
            "course_titles": [],
        }

    def test_failure_returns_500(self, client, mock_rag_system):
        mock_rag_system.get_course_analytics.side_effect = RuntimeError("chroma down")
        r = client.get("/api/courses")
        assert r.status_code == 500
        assert r.json()["detail"] == "chroma down"

    def test_post_not_allowed(self, client):
        assert client.post("/api/courses").status_code == 405


class TestSessionEndpoint:
    @pytest.mark.parametrize("existed", [True, False])
    def test_delete_is_idempotent(self, client, mock_rag_system, existed):
        mock_rag_system.session_manager.delete_session.return_value = existed
        r = client.delete("/api/session/abc")
        assert r.status_code == 200
        assert r.json() == {"deleted": existed}
        mock_rag_system.session_manager.delete_session.assert_called_once_with("abc")


class TestRoot:
    def test_root_serves_html(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/html")

    def test_unknown_route_404(self, client):
        assert client.get("/api/nope").status_code == 404
