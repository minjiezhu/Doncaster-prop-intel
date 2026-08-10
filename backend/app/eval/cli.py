"""Retrieval quality evaluation CLI.

Runs every question in a golden set (question -> expected source file(s)) through each
search mode and reports recall@k / MRR per mode. Calls WeaviateStore directly and skips
LLM answer generation entirely — this measures retrieval quality, not answer quality, so
there's no reason to pay for a qwen3:14b call per question.

Usage:
    python -m backend.app.eval.cli
    python -m backend.app.eval.cli --golden-set backend/eval/golden_set.yaml --top-k 5
    python -m backend.app.eval.cli --modes hybrid hybrid_rerank --report backend/logs/eval_report.json
"""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import yaml
from langchain_ollama import OllamaEmbeddings

from backend.app.config import Settings, get_settings
from backend.app.eval.metrics import ModeMetrics, hit, reciprocal_rank
from backend.app.retrieval.reranker import build_reranker
from backend.app.retrieval.weaviate_store import WeaviateStore

_ALL_MODES = ["vector", "hybrid", "hybrid_rerank"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality against a golden question set")
    parser.add_argument("--golden-set", default="backend/eval/golden_set.yaml")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--modes", nargs="+", default=_ALL_MODES, choices=_ALL_MODES)
    parser.add_argument("--report", default=None, help="Optional path to write a JSON report")
    return parser.parse_args()


def load_golden_set(path: str) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or []
    return [{"question": item["question"], "expected_sources": set(item["expected_sources"])} for item in raw]


def retrieve_sources(
    question: str,
    vector: list[float],
    mode: str,
    top_k: int,
    store: WeaviateStore,
    settings: Settings,
    reranker,
) -> list[str]:
    """Mirrors QueryService's recall/rerank logic, minus LLM generation."""
    candidate_k = top_k * settings.hybrid_candidate_multiplier if mode in {"hybrid", "hybrid_rerank"} else top_k

    if mode == "vector":
        rows = store.vector_search(vector=vector, top_k=top_k)
    else:
        rows = store.hybrid_search(query=question, vector=vector, top_k=candidate_k, alpha=settings.hybrid_alpha)

    if mode == "hybrid_rerank":
        ranked = reranker.rerank(query=question, chunks=rows, top_n=settings.reranker_top_n)
        return [r.metadata.get("source", "unknown") for r in ranked]

    if mode == "hybrid":
        rows = rows[:top_k]
    return [row.get("source", "unknown") for row in rows]


def run_eval(
    golden_set: list[dict],
    modes: list[str],
    top_k: int,
    store: WeaviateStore,
    embeddings: OllamaEmbeddings,
    settings: Settings,
    reranker,
) -> list[ModeMetrics]:
    hits: dict[str, list[bool]] = {mode: [] for mode in modes}
    rrs: dict[str, list[float]] = {mode: [] for mode in modes}

    for item in golden_set:
        vector = embeddings.embed_query(item["question"])
        for mode in modes:
            sources = retrieve_sources(item["question"], vector, mode, top_k, store, settings, reranker)
            hits[mode].append(hit(sources, item["expected_sources"]))
            rrs[mode].append(reciprocal_rank(sources, item["expected_sources"]))

    return [
        ModeMetrics(
            mode=mode,
            n_questions=len(golden_set),
            recall_at_k=sum(hits[mode]) / len(hits[mode]),
            mrr=sum(rrs[mode]) / len(rrs[mode]),
        )
        for mode in modes
    ]


def print_report(golden_set_path: str, top_k: int, n_questions: int, results: list[ModeMetrics]) -> None:
    print(f"Golden set: {golden_set_path} ({n_questions} questions), top_k={top_k}\n")
    print(f"{'mode':<16}{'recall@k':<12}{'MRR':<10}")
    for m in results:
        print(f"{m.mode:<16}{m.recall_at_k:<12.3f}{m.mrr:<10.3f}")


def write_json_report(path: str, golden_set_path: str, top_k: int, results: list[ModeMetrics]) -> None:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "evaluated_at": datetime.now(UTC).isoformat(),
        "golden_set": golden_set_path,
        "top_k": top_k,
        "modes": [
            {"mode": m.mode, "n_questions": m.n_questions, "recall_at_k": m.recall_at_k, "mrr": m.mrr}
            for m in results
        ],
    }
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    settings = get_settings()
    golden_set = load_golden_set(args.golden_set)

    store = WeaviateStore()
    embeddings = OllamaEmbeddings(model=settings.ollama_embed_model, base_url=settings.ollama_base_url)
    reranker = build_reranker(
        provider=settings.reranker_provider,
        model_name=settings.reranker_model,
        cohere_api_key=settings.cohere_api_key,
    )

    results = run_eval(golden_set, args.modes, args.top_k, store, embeddings, settings, reranker)
    print_report(args.golden_set, args.top_k, len(golden_set), results)

    if args.report:
        write_json_report(args.report, args.golden_set, args.top_k, results)
        print(f"\nReport written to {args.report}")


if __name__ == "__main__":
    main()
