"""Reranker module — Phase 2.

Architecture decision:
  Reranking sits *after* the recall stage (vector or hybrid) and *before* prompt
  assembly.  This two-stage pattern lets recall cast a wide net (high recall) while
  the reranker enforces precision, without over-constraining the candidate pool early.

  Provider toggle (RERANKER_PROVIDER in .env):
  - "local"  → sentence-transformers cross-encoder, fully offline, CPU-friendly.
  - "cohere" → Cohere Rerank API (free tier), requires COHERE_API_KEY.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RankedChunk:
    text: str
    score: float
    original_index: int
    metadata: dict


class LocalCrossEncoderReranker:
    """Lazy-loads the cross-encoder on first call to avoid startup cost."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None  # deferred load

    def _load(self) -> None:
        if self._model is None:
            from sentence_transformers import CrossEncoder  # noqa: PLC0415

            self._model = CrossEncoder(self.model_name)

    def rerank(self, query: str, chunks: list[dict], top_n: int) -> list[RankedChunk]:
        if not chunks:
            return []
        self._load()
        texts = [c.get("text", "") for c in chunks]
        pairs = [[query, t] for t in texts]
        scores: list[float] = self._model.predict(pairs).tolist()  # type: ignore[union-attr]
        ranked = sorted(
            (
                RankedChunk(
                    text=texts[i],
                    score=float(scores[i]),
                    original_index=i,
                    metadata={k: v for k, v in chunks[i].items() if k != "text"},
                )
                for i in range(len(chunks))
            ),
            key=lambda r: r.score,
            reverse=True,
        )
        return ranked[:top_n]


class CohereReranker:
    """Thin wrapper around the Cohere Rerank API (free tier: 1k req/month)."""

    def __init__(self, api_key: str, model_name: str = "rerank-english-v3.0") -> None:
        self.api_key = api_key
        self.model_name = model_name

    def rerank(self, query: str, chunks: list[dict], top_n: int) -> list[RankedChunk]:
        if not chunks:
            return []
        import httpx  # noqa: PLC0415

        texts = [c.get("text", "") for c in chunks]
        payload = {
            "query": query,
            "documents": texts,
            "top_n": top_n,
            "model": self.model_name,
        }
        resp = httpx.post(
            "https://api.cohere.ai/v1/rerank",
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=15.0,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        return [
            RankedChunk(
                text=texts[r["index"]],
                score=float(r["relevance_score"]),
                original_index=r["index"],
                metadata={k: v for k, v in chunks[r["index"]].items() if k != "text"},
            )
            for r in results
        ]


def build_reranker(provider: str, model_name: str, cohere_api_key: str) -> LocalCrossEncoderReranker | CohereReranker:
    if provider == "cohere":
        if not cohere_api_key:
            raise ValueError("COHERE_API_KEY must be set when RERANKER_PROVIDER=cohere")
        return CohereReranker(api_key=cohere_api_key, model_name=model_name)
    # Default: local cross-encoder
    return LocalCrossEncoderReranker(model_name=model_name)
