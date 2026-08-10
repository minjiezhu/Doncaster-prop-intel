import json
from pathlib import Path
from unittest.mock import MagicMock

from backend.app.config import Settings
from backend.app.eval.cli import load_golden_set, retrieve_sources, run_eval, write_json_report
from backend.app.eval.metrics import ModeMetrics
from backend.app.retrieval.reranker import RankedChunk


def test_load_golden_set_parses_questions_and_sources(tmp_path: Path) -> None:
    golden_path = tmp_path / "golden.yaml"
    golden_path.write_text(
        "- question: what is the median income?\n"
        "  expected_sources:\n"
        "    - census.csv\n",
        encoding="utf-8",
    )

    golden_set = load_golden_set(str(golden_path))

    assert golden_set == [{"question": "what is the median income?", "expected_sources": {"census.csv"}}]


class TestRetrieveSources:
    def _settings(self) -> Settings:
        return Settings(hybrid_candidate_multiplier=3, reranker_top_n=2)

    def test_vector_mode_calls_vector_search(self) -> None:
        store = MagicMock()
        store.vector_search.return_value = [{"source": "a.pdf"}, {"source": "b.pdf"}]

        sources = retrieve_sources("q", [0.1], "vector", 5, store, self._settings(), reranker=MagicMock())

        store.vector_search.assert_called_once_with(vector=[0.1], top_k=5)
        assert sources == ["a.pdf", "b.pdf"]

    def test_hybrid_mode_truncates_to_top_k(self) -> None:
        store = MagicMock()
        store.hybrid_search.return_value = [{"source": f"{i}.pdf"} for i in range(15)]

        sources = retrieve_sources("q", [0.1], "hybrid", 5, store, self._settings(), reranker=MagicMock())

        assert len(sources) == 5

    def test_hybrid_rerank_mode_uses_reranker_output(self) -> None:
        store = MagicMock()
        store.hybrid_search.return_value = [{"source": "a.pdf", "text": "x"}, {"source": "b.pdf", "text": "y"}]
        reranker = MagicMock()
        reranker.rerank.return_value = [
            RankedChunk(text="y", score=5.0, original_index=1, metadata={"source": "b.pdf"}),
        ]

        sources = retrieve_sources("q", [0.1], "hybrid_rerank", 5, store, self._settings(), reranker)

        assert sources == ["b.pdf"]


def test_run_eval_computes_recall_and_mrr_per_mode() -> None:
    golden_set = [
        {"question": "q1", "expected_sources": {"a.pdf"}},
        {"question": "q2", "expected_sources": {"z.pdf"}},
    ]
    store = MagicMock()
    store.vector_search.return_value = [{"source": "a.pdf"}]
    embeddings = MagicMock()
    embeddings.embed_query.return_value = [0.1]

    results = run_eval(golden_set, ["vector"], 5, store, embeddings, Settings(), reranker=MagicMock())

    assert results == [ModeMetrics(mode="vector", n_questions=2, recall_at_k=0.5, mrr=0.5)]


def test_write_json_report_creates_parent_dirs_and_valid_json(tmp_path: Path) -> None:
    report_path = tmp_path / "nested" / "report.json"
    results = [ModeMetrics(mode="hybrid", n_questions=3, recall_at_k=0.667, mrr=0.5)]

    write_json_report(str(report_path), "backend/eval/golden_set.yaml", 5, results)

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["top_k"] == 5
    assert payload["modes"][0]["mode"] == "hybrid"
