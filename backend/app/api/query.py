from functools import lru_cache

from fastapi import APIRouter, Depends, Query

from backend.app.api.models import QueryResponse
from backend.app.retrieval.query_service import QueryService

router = APIRouter(tags=["query"])


@lru_cache(maxsize=1)
def get_query_service() -> QueryService:
    return QueryService()


@router.get("/query", response_model=QueryResponse)
def query(
    question: str = Query(..., min_length=3, description="Question to answer"),
    top_k: int = Query(default=5, ge=1, le=20),
    mode: str = Query(
        default="hybrid_rerank",
        description="Search mode: 'vector' | 'hybrid' | 'hybrid_rerank'",
    ),
    service: QueryService = Depends(get_query_service),
) -> QueryResponse:
    result = service.answer_question(question=question, top_k=top_k, mode=mode)
    return QueryResponse(**result)
