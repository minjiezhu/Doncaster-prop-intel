"""Tool contracts reserved for Phase 3 agentic workflows."""

from dataclasses import dataclass


@dataclass
class ToolCallLog:
    tool_name: str
    latency_ms: int
    confidence: float
