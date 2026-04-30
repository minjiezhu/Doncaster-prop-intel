"""Structured JSONL logging for Phase 3 agent tool calls."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Sequence


def log_agent_call(
    path: str,
    *,
    question: str,
    tools_selected: Sequence[str],
    tool_latencies_ms: dict[str, int],
    tool_errors: dict[str, str],
    final_answer_length: int,
    total_time_ms: int,
    fallback_triggered: bool,
    fallback_reason: str = "",
) -> None:
    """Append one JSON line describing a complete agent run."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    record = {
        "ts": datetime.now(tz=timezone.utc).isoformat(),
        "question": question,
        "tools_selected": list(tools_selected),
        "tool_latencies_ms": tool_latencies_ms,
        "tool_errors": tool_errors,
        "final_answer_length": final_answer_length,
        "total_time_ms": total_time_ms,
        "fallback_triggered": fallback_triggered,
        "fallback_reason": fallback_reason,
    }
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
