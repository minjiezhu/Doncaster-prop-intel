"""Integration tests for POST /agent route (agent runner is stubbed)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.api.agent import router as agent_router
from backend.app.agent.runner import get_agent_runner


STUB_RESULT = {
    "answer": "The school zone for 3109 is Templestowe College.",
    "tools_used": ["rag_search"],
    "tool_latencies_ms": {"rag_search": 120},
    "total_time_ms": 350,
    "fallback_triggered": False,
}


@pytest.fixture
def client():
    stub_runner = MagicMock()
    stub_runner.run.return_value = STUB_RESULT
    app.dependency_overrides[get_agent_runner] = lambda: stub_runner
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


class TestAgentRoute:
    def test_post_agent_returns_200(self, client):
        resp = client.post("/agent", json={"question": "What is the school zone for 3109?"})
        assert resp.status_code == 200

    def test_response_shape(self, client):
        resp = client.post("/agent", json={"question": "school zone 3109"})
        data = resp.json()
        assert "answer" in data
        assert "tools_used" in data
        assert "tool_latencies_ms" in data
        assert "total_time_ms" in data
        assert "fallback_triggered" in data

    def test_answer_content(self, client):
        resp = client.post("/agent", json={"question": "school zone 3109"})
        assert "Templestowe College" in resp.json()["answer"]

    def test_tools_used_field(self, client):
        resp = client.post("/agent", json={"question": "school zone 3109"})
        assert resp.json()["tools_used"] == ["rag_search"]

    def test_short_question_rejected(self, client):
        resp = client.post("/agent", json={"question": "hi"})
        assert resp.status_code == 422
