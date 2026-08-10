"""Phase 3 agent tools: RAG search, Domain API listings, web search fallback."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from langchain.tools import Tool

if TYPE_CHECKING:
    from backend.app.config import Settings


@dataclass
class ToolCallLog:
    tool_name: str
    latency_ms: int
    confidence: float
    result_count: int = 0
    error: str = ""


# ---------------------------------------------------------------------------
# Tool 1 — RAG document search
# ---------------------------------------------------------------------------

def _make_rag_tool(query_service, settings: "Settings") -> Tool:
    """Return a LangChain Tool that queries the local RAG pipeline."""

    def rag_search(question: str) -> str:
        result = query_service.answer_question(
            question,
            top_k=settings.retrieval_top_k,
            mode="hybrid_rerank",
        )
        # Expose answer + source metadata for the agent to reason over
        sources = "; ".join(
            f"{s.get('source', 'unknown')} (score={s.get('rerank_score', s.get('distance', '?'))})"
            for s in result.get("sources", [])[:3]
        )
        return f"[RAG]\n{result['answer']}\nSources: {sources}"

    return Tool(
        name="rag_search",
        func=rag_search,
        description=(
            "Search the local property document knowledge base (PDFs, CSVs, reports). "
            "Use this first for questions about school zones, council zoning, "
            "suburb profiles, or any document-level facts about Manningham/Doncaster."
        ),
    )


# ---------------------------------------------------------------------------
# Tool 2 — Domain.com.au listings stub
# ---------------------------------------------------------------------------

def _make_domain_tool(settings: "Settings") -> Tool:
    """Return a LangChain Tool that calls the Domain API (stub implementation)."""

    import httpx

    def _pick_listing_node(item: dict) -> dict:
        if isinstance(item.get("listing"), dict):
            return item["listing"]
        return item

    def _extract_address(listing: dict) -> str:
        details = listing.get("propertyDetails", {}) if isinstance(listing, dict) else {}
        address = (
            details.get("displayableAddress")
            or details.get("displayAddress")
            or details.get("address")
        )
        if isinstance(address, dict):
            return address.get("display") or address.get("street") or "?"
        return address or "?"

    def _extract_price(listing: dict) -> str:
        candidates = []
        if isinstance(listing, dict):
            candidates.extend(
                [
                    listing.get("priceDetails"),
                    listing.get("price"),
                    listing.get("listingPrice"),
                ]
            )

        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            value = (
                candidate.get("displayPrice")
                or candidate.get("price")
                or candidate.get("display")
                or candidate.get("value")
            )
            if value:
                return str(value)

        return "Price unavailable"

    def _format_listings(items: list[dict]) -> str:
        lines = []
        for item in items[: settings.domain_listings_limit]:
            listing = _pick_listing_node(item)
            lines.append(f"• {_extract_address(listing)} — {_extract_price(listing)}")
        return "\n".join(lines)

    def domain_listings(query: str) -> str:
        """
        Production implementation would call:
          GET {domain_api_url}/listings/residential/_search
        with suburb + price filters parsed from *query*.

        Stub: return a fixed mock response so the agent can practise tool selection
        without a live API key.
        """
        if not settings.domain_api_key:
            # Return a plausible stub rather than failing hard
            return (
                "[Domain API — stub mode]\n"
                "3 mock listings found for query: " + query + "\n"
                "• 12 Sample St Doncaster — $1,200,000 — 4br/2ba\n"
                "• 8 Test Ave Templestowe — $1,450,000 — 4br/2ba\n"
                "• 22 Demo Rd Warrandyte — $980,000 — 3br/1ba\n"
                "(Set DOMAIN_API_KEY in .env for live results)"
            )

        # Live path — Domain API v1 residential search uses POST with a JSON body.
        # Response is a JSON array; each element has a "listing" key with property details.
        body = {
            "listingType": "Sale",
            "locations": [
                {
                    "state": "VIC",
                    "suburb": query,
                    "includeSurroundingSuburbs": False,
                }
            ],
            "pageSize": settings.domain_listings_limit,
        }
        try:
            resp = httpx.post(
                f"{settings.domain_api_url}/listings/residential/_search",
                headers={
                    "X-Api-Key": settings.domain_api_key,
                    "Content-Type": "application/json",
                },
                json=body,
                timeout=settings.agent_tool_timeout_seconds,
            )
            resp.raise_for_status()
            data = resp.json()
            # Response may be a list directly or a dict with a "listings" key
            if isinstance(data, list):
                listings = data[: settings.domain_listings_limit]
            else:
                listings = data.get("listings", [])[: settings.domain_listings_limit]
            if not listings:
                return "[Domain API] No listings found for: " + query
            return "[Domain API]\n" + _format_listings(listings)
        except (TypeError, ValueError, KeyError) as exc:
            return f"[Domain API] Error parsing response: {exc}"
        except httpx.HTTPError as exc:
            return f"[Domain API] HTTP error: {exc}"

    return Tool(
        name="domain_listings",
        func=domain_listings,
        description=(
            "Fetch current real-estate listings from Domain.com.au for a suburb or address. "
            "Use this for questions about current asking prices, available properties, "
            "or 'what is on the market now' style queries."
        ),
    )


# ---------------------------------------------------------------------------
# Tool 3 — DuckDuckGo web search fallback
# ---------------------------------------------------------------------------

def _make_web_search_tool(settings: "Settings") -> Tool:
    """Return a LangChain Tool backed by DuckDuckGo (no API key required)."""

    def web_search(query: str) -> str:
        try:
            from ddgs import DDGS
        except ImportError:
            return "[Web Search] ddgs package not installed."

        results = []
        try:
            with DDGS() as ddgs:
                for r in ddgs.text(
                    query + " Manningham Doncaster Victoria",
                    max_results=settings.web_search_max_results,
                ):
                    results.append(f"• {r['title']}: {r['href']}\n  {r['body'][:200]}")
        except Exception as exc:  # noqa: BLE001
            return f"[Web Search] Error: {exc}"

        if not results:
            return "[Web Search] No results found for: " + query
        return "[Web Search]\n" + "\n".join(results)

    return Tool(
        name="web_search",
        func=web_search,
        description=(
            "Search the web for recent news, council announcements, "
            "planning permits, or any question not answered by local documents or "
            "Domain listings. Use as a fallback when the other tools return insufficient results."
        ),
    )


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def build_tools(query_service, settings: "Settings") -> list[Tool]:
    """Return all three agent tools in priority order."""
    return [
        _make_rag_tool(query_service, settings),
        _make_domain_tool(settings),
        _make_web_search_tool(settings),
    ]
