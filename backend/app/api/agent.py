"""Phase 3 agent endpoint: POST /agent."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from backend.app.agent.runner import AgentRunner, get_agent_runner

router = APIRouter(prefix="/agent", tags=["agent"])


class AgentRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)


class AgentResponse(BaseModel):
    answer: str
    tools_used: list[str]
    tool_latencies_ms: dict[str, int]
    total_time_ms: int
    fallback_triggered: bool


@router.post("", response_model=AgentResponse)
def run_agent(
    body: AgentRequest,
    runner: AgentRunner = Depends(get_agent_runner),
) -> AgentResponse:
    """
    Run the ReAct agent over a free-text property question.

    The agent autonomously selects from three tools:
    - **rag_search**: local document knowledge base (school zones, zoning, reports)
    - **domain_listings**: current Domain.com.au property listings
    - **web_search**: DuckDuckGo fallback for recent news / council updates
    """
    result = runner.run(body.question)
    return AgentResponse(**result)
