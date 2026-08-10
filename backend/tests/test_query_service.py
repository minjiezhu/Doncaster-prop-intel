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


class TestHybridModeScoring:
    def test_truncates_overfetched_candidates_to_top_k(self) -> None:
        # _recall() over-fetches candidate_k = top_k * multiplier for hybrid mode so the
        # reranker has a richer pool in hybrid_rerank mode; plain "hybrid" mode must still
        # truncate back down to what was actually requested.
        service = _make_service(reranker_min_score=None)
        service.store.hybrid_search.return_value = [
            {
                "text": f"chunk {i}",
                "source": f"{i}.pdf",
                "suburb": "doncaster",
                "strategy": "fixed",
                "chunk_index": i,
                "_additional": {"distance": None, "score": "0.9", "id": str(i)},
            }
            for i in range(15)
        ]

        result = service.answer_question("median income?", top_k=5, mode="hybrid")

        assert len(result["sources"]) == 5

    def test_extracts_score_not_distance_for_hybrid_mode(self) -> None:
        # Weaviate's hybrid endpoint returns null for `distance` (only meaningful for
        # nearVector) and a string-typed `score` instead; naively doing
        # float(additional.get("distance", 0.0)) crashes with TypeError since the key is
        # present with value None, so .get()'s default never kicks in.
        service = _make_service(reranker_min_score=None)
        service.store.hybrid_search.return_value = [
            {
                "text": "chunk A",
                "source": "a.pdf",
                "suburb": "doncaster",
                "strategy": "fixed",
                "chunk_index": 0,
                "_additional": {"distance": None, "score": "0.87", "id": "1"},
            },
        ]

        result = service.answer_question("median income?", top_k=5, mode="hybrid")

        assert result["sources"][0]["score"] == 0.87
        assert "distance" not in result["sources"][0]


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
