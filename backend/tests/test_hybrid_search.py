from unittest.mock import MagicMock, patch

import httpx
import pytest

from backend.app.retrieval.weaviate_store import WeaviateStore


def _make_store() -> WeaviateStore:
    store = WeaviateStore.__new__(WeaviateStore)
    store.settings = MagicMock()
    store.settings.weaviate_url = "http://localhost:8080"
    store.settings.weaviate_class_name = "PropertyDocChunk"
    store.settings.retrieval_timeout_seconds = 5.0
    store.base_url = "http://localhost:8080"
    store.class_name = "PropertyDocChunk"
    store.timeout = 5.0
    return store


def test_hybrid_search_builds_correct_graphql() -> None:
    store = _make_store()
    captured = {}

    def fake_run_graphql(query: str) -> list[dict]:
        captured["query"] = query
        return []

    store._run_graphql = fake_run_graphql  # type: ignore[method-assign]
    store.ensure_schema = MagicMock()  # type: ignore[method-assign]

    store.hybrid_search(query="school doncaster", vector=[0.1, 0.2], top_k=5, alpha=0.5)

    assert "hybrid" in captured["query"]
    assert "school doncaster" in captured["query"]
    assert "alpha: 0.5" in captured["query"]
    assert "limit: 5" in captured["query"]


def test_hybrid_search_escapes_double_quotes() -> None:
    store = _make_store()
    captured = {}

    def fake_run_graphql(query: str) -> list[dict]:
        captured["query"] = query
        return []

    store._run_graphql = fake_run_graphql  # type: ignore[method-assign]
    store.ensure_schema = MagicMock()  # type: ignore[method-assign]

    store.hybrid_search(query='3-bed "under $1M"', vector=[0.1], top_k=3, alpha=0.5)
    # Double-quotes in query text must be escaped so the GraphQL stays valid
    assert '\\"' in captured["query"]
