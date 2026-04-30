"""Phase 3 agent runner: LangChain ReAct agent over three property-research tools."""

from __future__ import annotations

import time
from functools import lru_cache
from typing import Any

from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_ollama import ChatOllama

from backend.app.agent.agent_log import log_agent_call
from backend.app.agent.tools import build_tools
from backend.app.config import Settings, get_settings
from backend.app.retrieval.query_service import QueryService


class AgentRunner:
    """
    Wraps a LangChain ReAct agent that autonomously selects among:
      1. rag_search        — local vector knowledge base
      2. domain_listings   — Domain.com.au (stub or live)
      3. web_search        — DuckDuckGo fallback

    The agent logs every run to agent_calls.jsonl for later analysis.
    """

    # ReAct prompt pulled from LangChain Hub — no custom prompt needed for demo
    _REACT_PROMPT_REPO = "hwchase17/react"

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._query_service = QueryService(self._settings)
        self._executor = self._build_executor()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build_executor(self) -> AgentExecutor:
        llm = ChatOllama(
            base_url=self._settings.ollama_base_url,
            model=self._settings.ollama_chat_model,
        )
        tools = build_tools(self._query_service, self._settings)
        prompt = hub.pull(self._REACT_PROMPT_REPO)
        agent = create_react_agent(llm, tools, prompt)
        return AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=False,          # set True locally for ReAct trace debugging
            handle_parsing_errors=True,
            max_iterations=6,       # guard against runaway loops
            return_intermediate_steps=True,
        )

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self, question: str) -> dict[str, Any]:
        """
        Execute the agent and return:
          {
            "answer": str,
            "tools_used": list[str],
            "tool_latencies_ms": dict[str, int],
            "total_time_ms": int,
            "fallback_triggered": bool,
            "intermediate_steps": list  (raw LangChain steps for inspection)
          }
        """
        t0 = time.perf_counter()
        tools_used: list[str] = []
        tool_latencies: dict[str, int] = {}
        tool_errors: dict[str, str] = {}
        fallback_triggered = False
        fallback_reason = ""
        answer = ""

        try:
            result = self._executor.invoke({"input": question})
            answer = result.get("output", "")

            # Parse intermediate steps to extract tool telemetry
            for action, observation in result.get("intermediate_steps", []):
                tool_name = getattr(action, "tool", "unknown")
                tools_used.append(tool_name)
                # LangChain doesn't natively expose per-tool latency;
                # we record presence and any error signals from the observation string.
                if str(observation).startswith(f"[{tool_name.replace('_', ' ').title()} Error]"):
                    tool_errors[tool_name] = str(observation)[:200]

        except Exception as exc:  # noqa: BLE001
            fallback_triggered = True
            fallback_reason = str(exc)[:200]
            answer = (
                "I encountered an error while researching your question. "
                f"Details: {fallback_reason}"
            )

        total_ms = round((time.perf_counter() - t0) * 1000)

        # Best-effort latency: divide total by number of tool calls
        if tools_used:
            approx_per_tool = total_ms // len(tools_used)
            tool_latencies = {t: approx_per_tool for t in tools_used}

        log_agent_call(
            self._settings.agent_log_path,
            question=question,
            tools_selected=tools_used,
            tool_latencies_ms=tool_latencies,
            tool_errors=tool_errors,
            final_answer_length=len(answer),
            total_time_ms=total_ms,
            fallback_triggered=fallback_triggered,
            fallback_reason=fallback_reason,
        )

        return {
            "answer": answer,
            "tools_used": tools_used,
            "tool_latencies_ms": tool_latencies,
            "total_time_ms": total_ms,
            "fallback_triggered": fallback_triggered,
        }


@lru_cache(maxsize=1)
def get_agent_runner() -> AgentRunner:
    return AgentRunner()
