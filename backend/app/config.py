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

    metrics_log_path: str = Field(default="backend/logs/chunk_quality.jsonl")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
