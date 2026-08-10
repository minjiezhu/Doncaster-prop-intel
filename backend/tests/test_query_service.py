from unittest.mock import MagicMock

from backend.app.config import Settings
from backend.app.retrieval.query_service import QueryService
from backend.app.retrieval.reranker import RankedChunk


def _make_service(reranker_min_score: float | None) -> QueryService:
    settings = Settings(reranker_min_score=reranker_min_score)
    service = QueryService(settings)

    service.embeddings = MagicMock()
    service.embeddings.embed_query.return_value = [0.1, 0.2, 0.3]

    service.store = MagicMock()
    service.store.hybrid_search.return_value = [
        {"text": "chunk A", "source": "a.pdf", "suburb": "doncaster", "strategy": "fixed", "chunk_index": 0},
        {"text": "chunk B", "source": "b.pdf", "suburb": "doncaster", "strategy": "fixed", "chunk_index": 1},
    ]

    service.llm = MagicMock()
    service.llm.invoke.return_value.content = "answer text"

    return service


class TestRerankerMinScore:
    def test_disabled_by_default_keeps_all_reranked_results(self) -> None:
        service = _make_service(reranker_min_score=None)
        service.reranker = MagicMock()
        service.reranker.rerank.return_value = [
            RankedChunk(text="chunk A", score=-8.0, original_index=0, metadata={"source": "a.pdf"}),
            RankedChunk(text="chunk B", score=-9.0, original_index=1, metadata={"source": "b.pdf"}),
        ]

        result = service.answer_question("what is the school zone?", mode="hybrid_rerank")

        assert result["answer"] == "answer text"
        assert len(result["sources"]) == 2

    def test_filters_out_low_scoring_candidates(self) -> None:
        service = _make_service(reranker_min_score=0.0)
        service.reranker = MagicMock()
        service.reranker.rerank.return_value = [
            RankedChunk(text="chunk A", score=-8.0, original_index=0, metadata={"source": "a.pdf"}),
            RankedChunk(text="chunk B", score=2.5, original_index=1, metadata={"source": "b.pdf"}),
        ]

        result = service.answer_question("what is the school zone?", mode="hybrid_rerank")

        assert len(result["sources"]) == 1
        assert result["sources"][0]["source"] == "b.pdf"

    def test_all_candidates_below_threshold_returns_no_results_response(self) -> None:
        service = _make_service(reranker_min_score=5.0)
        service.reranker = MagicMock()
        service.reranker.rerank.return_value = [
            RankedChunk(text="chunk A", score=-8.0, original_index=0, metadata={"source": "a.pdf"}),
            RankedChunk(text="chunk B", score=-9.0, original_index=1, metadata={"source": "b.pdf"}),
        ]

        result = service.answer_question("what is the school zone?", mode="hybrid_rerank")

        assert result["sources"] == []
        assert "未检索到相关资料" in result["answer"]
        assert result["retrieval_debug"]["hits"] == 0
        # This is a quality filter, not an infra failure — shouldn't be reported as a fallback.
        assert result["retrieval_debug"]["fallback_triggered"] is False
        service.llm.invoke.assert_not_called()
