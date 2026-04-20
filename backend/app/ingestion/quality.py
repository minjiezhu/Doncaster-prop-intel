from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from backend.app.ingestion.chunking import estimate_tokens


def summarize_chunks(chunks: list[str], source_tokens: int) -> dict[str, float | int]:
    if not chunks:
        return {
            "chunk_count": 0,
            "avg_tokens": 0,
            "min_tokens": 0,
            "max_tokens": 0,
            "duplicate_ratio": 0.0,
            "coverage_ratio": 0.0,
        }

    sizes = [estimate_tokens(chunk) for chunk in chunks]
    unique_count = len(set(chunks))
    duplicate_ratio = 1 - (unique_count / len(chunks))
    coverage_ratio = sum(sizes) / max(1, source_tokens)

    return {
        "chunk_count": len(chunks),
        "avg_tokens": round(sum(sizes) / len(sizes), 2),
        "min_tokens": min(sizes),
        "max_tokens": max(sizes),
        "duplicate_ratio": round(duplicate_ratio, 4),
        "coverage_ratio": round(coverage_ratio, 4),
    }


def append_quality_log(path: str, payload: dict) -> None:
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    enriched = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **payload,
    }
    with log_path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(enriched, ensure_ascii=True) + "\n")
