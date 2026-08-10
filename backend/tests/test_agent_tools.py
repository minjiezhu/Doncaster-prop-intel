"""Unit tests for Phase 3 agent tools (no real HTTP calls, no Ollama)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from backend.app.agent.tools import build_tools, _make_web_search_tool, _make_domain_tool
from backend.app.config import Settings


@pytest.fixture
def settings():
    return Settings(
        domain_api_key="",          # stub mode
        web_search_max_results=3,
        domain_listings_limit=3,
        agent_tool_timeout_seconds=5.0,
    )


@pytest.fixture
def mock_query_service():
    svc = MagicMock()
    svc.answer_question.return_value = {
        "answer": "Doncaster East is in the Donvale zone.",
        "sources": [{"source": "zones.pdf", "rerank_score": 7.2}],
        "retrieval_debug": {},
    }
    return svc


class TestBuildTools:
    def test_returns_three_tools(self, settings, mock_query_service):
        tools = build_tools(mock_query_service, settings)
        assert len(tools) == 3

    def test_tool_names(self, settings, mock_query_service):
        tools = build_tools(mock_query_service, settings)
        names = [t.name for t in tools]
        assert names == ["rag_search", "domain_listings", "web_search"]


class TestRagTool:
    def test_calls_query_service(self, settings, mock_query_service):
        tools = build_tools(mock_query_service, settings)
        rag_tool = next(t for t in tools if t.name == "rag_search")
        result = rag_tool.func("what is the school zone for 3108?")
        mock_query_service.answer_question.assert_called_once()
        assert "Doncaster East" in result

    def test_result_contains_sources(self, settings, mock_query_service):
        tools = build_tools(mock_query_service, settings)
        rag_tool = next(t for t in tools if t.name == "rag_search")
        result = rag_tool.func("school zone?")
        assert "zones.pdf" in result


class TestDomainTool:
    def test_stub_mode_returns_mock_listings(self, settings):
        tool = _make_domain_tool(settings)
        result = tool.func("Doncaster")
        assert "stub mode" in result.lower() or "mock" in result.lower()
        assert "Domain API" in result

    def test_stub_includes_query_text(self, settings):
        tool = _make_domain_tool(settings)
        result = tool.func("Templestowe")
        assert "Templestowe" in result

    def test_live_mode_parses_nested_listing_shape(self):
        settings = Settings(
            domain_api_key="live-key",
            domain_api_url="https://api.domain.com.au/v1",
            domain_listings_limit=2,
            agent_tool_timeout_seconds=5.0,
        )
        tool = _make_domain_tool(settings)

        response_payload = {
            "listings": [
                {
                    "listing": {
                        "propertyDetails": {"displayableAddress": "12 Sample St Doncaster"},
                        "priceDetails": {"displayPrice": "$1,200,000"},
                    }
                }
            ]
        }

        mock_response = MagicMock()
        mock_response.json.return_value = response_payload
        mock_response.raise_for_status.return_value = None

        with patch("httpx.post", return_value=mock_response) as mock_post:
            result = tool.func("Doncaster")

        mock_post.assert_called_once()
        assert "12 Sample St Doncaster" in result
        assert "$1,200,000" in result

    def test_live_mode_parses_flat_listing_shape(self):
        settings = Settings(
            domain_api_key="live-key",
            domain_api_url="https://api.domain.com.au/v1",
            domain_listings_limit=2,
            agent_tool_timeout_seconds=5.0,
        )
        tool = _make_domain_tool(settings)

        response_payload = {
            "listings": [
                {
                    "propertyDetails": {"address": {"display": "8 Test Ave Templestowe"}},
                    "price": {"display": "$1,450,000"},
                }
            ]
        }

        mock_response = MagicMock()
        mock_response.json.return_value = response_payload
        mock_response.raise_for_status.return_value = None

        with patch("httpx.post", return_value=mock_response):
            result = tool.func("Templestowe")

        assert "8 Test Ave Templestowe" in result
        assert "$1,450,000" in result


class TestWebSearchTool:
    def test_ddg_not_installed_returns_graceful_message(self, settings):
        tool = _make_web_search_tool(settings)
        with patch.dict("sys.modules", {"duckduckgo_search": None}):
            # simulate ImportError by patching the import inside the func
            with patch("builtins.__import__", side_effect=ImportError("no module")):
                result = tool.func("council rezoning Doncaster")
        # Should not raise; returns a string
        assert isinstance(result, str)

    def test_ddg_result_formatted(self, settings):
        mock_results = [
            {"title": "Rezoning Update", "href": "http://example.com", "body": "Council approved new zones."},
        ]
        tool = _make_web_search_tool(settings)

        class MockDDGS:
            def __enter__(self): return self
            def __exit__(self, *a): pass
            def text(self, *a, **kw): return mock_results

        with patch("backend.app.agent.tools.DDGS", MockDDGS, create=True):
            with patch("duckduckgo_search.DDGS", MockDDGS, create=True):
                with patch.dict("sys.modules", {"duckduckgo_search": MagicMock(DDGS=MockDDGS)}):
                    result = tool.func("rezoning Doncaster")

        assert isinstance(result, str)
