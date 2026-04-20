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

    def vector_search(self, vector: list[float], top_k: int = 5) -> list[dict]:
        self.ensure_schema()
        vector_str = ", ".join(f"{v:.8f}" for v in vector)
        query = {
            "query": (
                "{"
                f"Get{{{self.class_name}("
                f"nearVector: {{vector: [{vector_str}]}} "
                f"limit: {top_k}"
                ")"
                "{ text source suburb doc_type strategy chunk_index _additional { distance id } }"
                "}}"
            )
        }

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(f"{self.base_url}/v1/graphql", json=query)
            resp.raise_for_status()
            data = resp.json()

        rows = data.get("data", {}).get("Get", {}).get(self.class_name, [])
        return rows
