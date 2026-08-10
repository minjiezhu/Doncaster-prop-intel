from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "doncaster-property-intel"
    app_env: str = "development"
    log_level: str = "INFO"

    weaviate_url: str = Field(default="http://localhost:8080")
    weaviate_class_name: str = Field(default="PropertyDocChunk")

    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_chat_model: str = Field(default="qwen3:14b")
    ollama_embed_model: str = Field(default="nomic-embed-text")

    fixed_chunk_size: int = Field(default=512)
    fixed_chunk_overlap: int = Field(default=50)
    semantic_min_tokens: int = Field(default=180)
    semantic_max_tokens: int = Field(default=520)
    semantic_breakpoint_threshold: float = Field(default=0.72)

    retrieval_top_k: int = Field(default=5)
    retrieval_timeout_seconds: float = Field(default=30.0)

    # Hybrid search: alpha=0 → pure BM25, alpha=1 → pure vector, 0.5 → balanced.
    # We default to 0.5 so keyword recall and semantic recall contribute equally.
    hybrid_alpha: float = Field(default=0.5)
    # Fetch more candidates before reranking so the reranker has a richer pool to choose from.
    hybrid_candidate_multiplier: int = Field(default=3)

    # Reranker: local cross-encoder model (runs on CPU, no external API key required).
    # Swap to a Cohere model name and set reranker_provider=cohere to use Cohere free tier.
    reranker_provider: str = Field(default="local")   # "local" | "cohere"
    reranker_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    reranker_top_n: int = Field(default=5)
    cohere_api_key: str = Field(default="")
    # Minimum reranker score to keep a result; below this, treated as "no relevant results".
    # None (default) disables filtering. Score scale differs by provider (local cross-encoder
    # logits are unbounded, Cohere returns 0-1), so this needs calibrating against real data
    # before enabling — leave unset until then.
    reranker_min_score: float | None = Field(default=None)

    metrics_log_path: str = Field(default="backend/logs/chunk_quality.jsonl")
    retrieval_comparison_log_path: str = Field(default="backend/logs/retrieval_comparison.jsonl")
    ingestion_manifest_path: str = Field(default="backend/logs/ingestion_manifest.jsonl")

    # Phase 3 agent settings
    # Domain API endpoint (stub: returns mock listings; swap for real API in prod)
    domain_api_url: str = Field(default="https://api.domain.com.au/v1")
    domain_api_key: str = Field(default="")
    # Max listings returned by the domain tool per query
    domain_listings_limit: int = Field(default=5)
    # DuckDuckGo web search: max results to return to the agent
    web_search_max_results: int = Field(default=5)
    # Agent tool-call log file path
    agent_log_path: str = Field(default="backend/logs/agent_calls.jsonl")
    # Timeout (seconds) applied to each agent tool invocation
    agent_tool_timeout_seconds: float = Field(default=20.0)

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
