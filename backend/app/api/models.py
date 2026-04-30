from pydantic import BaseModel, Field


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict[str, str | int | float | bool]] = Field(default_factory=list)
    retrieval_debug: dict[str, int | float | str | bool] = Field(default_factory=dict)
