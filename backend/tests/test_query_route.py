from fastapi.testclient import TestClient

from backend.app.api.query import get_query_service
from backend.app.main import app


class StubService:
    def answer_question(self, question: str, top_k: int | None = None, mode: str = "hybrid_rerank") -> dict:
        return {
            "answer": f"echo: {question}",
            "sources": [{"source": "stub.txt", "distance": 0.1, "chunk_index": 0, "mode": mode}],
            "retrieval_debug": {
                "mode": mode,
                "top_k": top_k or 5,
                "retrieval_time_ms": 1,
                "hits": 1,
                "fallback_triggered": False,
            },
        }


def test_query_endpoint_happy_path() -> None:
    app.dependency_overrides[get_query_service] = lambda: StubService()
    client = TestClient(app)

    resp = client.get("/query", params={"question": "Doncaster median price?", "top_k": 3, "mode": "hybrid"})
    data = resp.json()

    assert resp.status_code == 200
    assert "answer" in data
    assert data["retrieval_debug"]["top_k"] == 3
    assert data["retrieval_debug"]["mode"] == "hybrid"

    app.dependency_overrides.clear()
