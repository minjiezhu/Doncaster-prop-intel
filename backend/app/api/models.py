from pydantic import BaseModel, Field


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict[str, str | int | float]] = Field(default_factory=list)
    retrieval_debug: dict[str, int | float | str] = Field(default_factory=dict)
