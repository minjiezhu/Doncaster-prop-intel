"""Retrieval comparison logger for Phase 2.

Writes a JSONL entry per query that records which search mode was used,
candidate counts, scores, and latencies.  These logs feed the side-by-side
comparison of vector vs hybrid vs hybrid+rerank in Phase 2 analysis.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def log_retrieval_comparison(
    path: str,
    *,
    question: str,
    mode: str,
    top_k: int,
    candidates_before_rerank: int,
    final_hits: int,
    top_scores: list[float],
    retrieval_time_ms: int,
    rerank_time_ms: int = 0,
    fallback_triggered: bool = False,
    fallback_reason: str = "",
) -> None:
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "mode": mode,
        "top_k": top_k,
        "candidates_before_rerank": candidates_before_rerank,
        "final_hits": final_hits,
        "top_scores": [round(s, 6) for s in top_scores[:5]],
        "retrieval_time_ms": retrieval_time_ms,
        "rerank_time_ms": rerank_time_ms,
        "fallback_triggered": fallback_triggered,
        "fallback_reason": fallback_reason,
    }
    with log_path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(entry, ensure_ascii=True) + "\n")
