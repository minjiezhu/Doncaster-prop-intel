import uuid
from collections.abc import Iterable

import httpx

from backend.app.config import get_settings


class WeaviateStore:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.base_url = self.settings.weaviate_url.rstrip("/")
        self.class_name = self.settings.weaviate_class_name
        self.timeout = self.settings.retrieval_timeout_seconds

    # ------------------------------------------------------------------
    # Schema management
    # ------------------------------------------------------------------

    def ensure_schema(self) -> None:
        with httpx.Client(timeout=self.timeout) as client:
            check = client.get(f"{self.base_url}/v1/schema/{self.class_name}")
            if check.status_code == 200:
                return

            schema = {
                "class": self.class_name,
                "vectorizer": "none",
                "properties": [
                    {"name": "text", "dataType": ["text"]},
                    {"name": "source", "dataType": ["text"]},
                    {"name": "suburb", "dataType": ["text"]},
                    {"name": "doc_type", "dataType": ["text"]},
                    {"name": "strategy", "dataType": ["text"]},
                    {"name": "chunk_index", "dataType": ["int"]},
                ],
            }
            create = client.post(f"{self.base_url}/v1/schema", json=schema)
            create.raise_for_status()

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def upsert_chunks(self, chunks: Iterable[dict]) -> int:
        self.ensure_schema()
        inserted = 0
        with httpx.Client(timeout=self.timeout) as client:
            for chunk in chunks:
                payload = {
                    "class": self.class_name,
                    "id": str(uuid.uuid4()),
                    "properties": {
                        "text": chunk["text"],
                        "source": chunk["source"],
                        "suburb": chunk.get("suburb", "unknown"),
                        "doc_type": chunk.get("doc_type", "unknown"),
                        "strategy": chunk["strategy"],
                        "chunk_index": chunk["chunk_index"],
                    },
                    "vector": chunk["embedding"],
                }
                resp = client.post(f"{self.base_url}/v1/objects", json=payload)
                resp.raise_for_status()
                inserted += 1
        return inserted

    # ------------------------------------------------------------------
    # Retrieval helpers
    # ------------------------------------------------------------------

    # `distance` is only populated for nearVector (vector_search); Weaviate's hybrid queries
    # return null for it and use `score` instead (returned as a string, not a number).
    _RETURN_FIELDS = "{ text source suburb doc_type strategy chunk_index _additional { distance score id } }"

    def vector_search(self, vector: list[float], top_k: int = 5) -> list[dict]:
        """Pure vector (cosine) recall — Phase 1 baseline."""
        self.ensure_schema()
        vector_str = ", ".join(f"{v:.8f}" for v in vector)
        gql = (
            "{"
            f"Get{{{self.class_name}("
            f"nearVector: {{vector: [{vector_str}]}} "
            f"limit: {top_k}"
            ")"
            f"{self._RETURN_FIELDS}"
            "}}"
        )
        return self._run_graphql(gql)

    def hybrid_search(self, query: str, vector: list[float], top_k: int = 10, alpha: float = 0.5) -> list[dict]:
        """Hybrid BM25 + vector recall.

        Architecture note:
          alpha=0   → pure BM25 keyword recall
          alpha=1   → pure vector semantic recall
          alpha=0.5 → equal weight (default; tunable via HYBRID_ALPHA in .env)

        Why hybrid over pure vector?
          Property queries often contain specific terms (suburb names, price ranges,
          postcode numbers) that benefit from exact-match BM25 recall.  Pure vector
          search may miss these if the embedding space does not separate them cleanly.
          Hybrid fuses both signals, improving coverage without sacrificing semantic
          relevance.
        """
        self.ensure_schema()
        vector_str = ", ".join(f"{v:.8f}" for v in vector)
        # Escape double-quotes inside query string to avoid breaking GraphQL
        safe_query = query.replace('"', '\\"')
        gql = (
            "{"
            f"Get{{{self.class_name}("
            f'hybrid: {{query: "{safe_query}", vector: [{vector_str}], alpha: {alpha}}} '
            f"limit: {top_k}"
            ")"
            f"{self._RETURN_FIELDS}"
            "}}"
        )
        return self._run_graphql(gql)

    def _run_graphql(self, query: str) -> list[dict]:
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(f"{self.base_url}/v1/graphql", json={"query": query})
            resp.raise_for_status()
            data = resp.json()
        return data.get("data", {}).get("Get", {}).get(self.class_name, [])

