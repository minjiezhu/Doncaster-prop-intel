from unittest.mock import MagicMock, patch

import pytest

from backend.app.retrieval.reranker import LocalCrossEncoderReranker, RankedChunk


def _make_chunks(n: int = 4) -> list[dict]:
    return [{"text": f"chunk {i} about Doncaster property prices", "source": f"doc{i}.pdf"} for i in range(n)]


def test_local_reranker_returns_top_n() -> None:
    reranker = LocalCrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    fake_scores = [0.9, 0.1, 0.7, 0.4]

    mock_model = MagicMock()
    mock_model.predict.return_value = fake_scores
    reranker._model = mock_model  # inject without loading real weights

    chunks = _make_chunks(4)
    ranked = reranker.rerank(query="median price doncaster", chunks=chunks, top_n=2)

    assert len(ranked) == 2
    assert ranked[0].score == 0.9  # highest score first
    assert ranked[1].score == 0.7
    assert isinstance(ranked[0], RankedChunk)


def test_local_reranker_empty_input() -> None:
    reranker = LocalCrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    result = reranker.rerank(query="anything", chunks=[], top_n=5)
    assert result == []


def test_local_reranker_top_n_larger_than_input() -> None:
    reranker = LocalCrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    fake_scores = [0.5, 0.3]
    mock_model = MagicMock()
    mock_model.predict.return_value = fake_scores
    reranker._model = mock_model

    ranked = reranker.rerank(query="school zones", chunks=_make_chunks(2), top_n=10)
    assert len(ranked) == 2  # capped at input size
