"""Retrieval quality metrics for the golden-set evaluation CLI.

Both metrics work off an ordered list of retrieved `source` filenames compared against
the set of filenames a golden question expects to see — they don't care about chunk-level
detail, so they stay valid across chunking-strategy changes and re-ingestion.
"""
from __future__ import annotations

from dataclasses import dataclass


def hit(retrieved_sources: list[str], expected_sources: set[str]) -> bool:
    """Did at least one expected source appear anywhere in the retrieved results?"""
    return any(source in expected_sources for source in retrieved_sources)


def reciprocal_rank(retrieved_sources: list[str], expected_sources: set[str]) -> float:
    """1 / (rank of the first expected source), or 0.0 if none appeared."""
    for rank, source in enumerate(retrieved_sources, start=1):
        if source in expected_sources:
            return 1.0 / rank
    return 0.0


@dataclass
class ModeMetrics:
    mode: str
    n_questions: int
    recall_at_k: float
    mrr: float
