from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from langchain_ollama import OllamaEmbeddings

from backend.app.config import get_settings


def estimate_tokens(text: str) -> int:
    words = text.split()
    return max(1, len(words))


def split_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


@dataclass
class ChunkedDocument:
    text: str
    chunk_index: int
    strategy: str
    metadata: dict[str, str]


def fixed_size_chunk(text: str, chunk_size: int, overlap: int) -> list[str]:
    tokens = text.split()
    if not tokens:
        return []

    chunks: list[str] = []
    step = max(1, chunk_size - overlap)
    for start in range(0, len(tokens), step):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        if not chunk_tokens:
            break
        chunks.append(" ".join(chunk_tokens))
        if end >= len(tokens):
            break
    return chunks


class SemanticChunker:
    def __init__(self) -> None:
        settings = get_settings()
        self.min_tokens = settings.semantic_min_tokens
        self.max_tokens = settings.semantic_max_tokens
        self.threshold = settings.semantic_breakpoint_threshold
        self.embedder = OllamaEmbeddings(
            model=settings.ollama_embed_model,
            base_url=settings.ollama_base_url,
        )

    @staticmethod
    def _cosine(v1: list[float], v2: list[float]) -> float:
        arr1 = np.array(v1)
        arr2 = np.array(v2)
        denom = np.linalg.norm(arr1) * np.linalg.norm(arr2)
        if denom == 0:
            return 0.0
        return float(np.dot(arr1, arr2) / denom)

    def chunk(self, text: str) -> list[str]:
        sentences = split_sentences(text)
        if not sentences:
            return []
        if len(sentences) == 1:
            return [sentences[0]]

        vectors = self.embedder.embed_documents(sentences)
        chunks: list[str] = []
        current: list[str] = [sentences[0]]
        current_tokens = estimate_tokens(sentences[0])

        for idx in range(1, len(sentences)):
            sentence = sentences[idx]
            sim = self._cosine(vectors[idx - 1], vectors[idx])
            sentence_tokens = estimate_tokens(sentence)

            should_break = sim < self.threshold and current_tokens >= self.min_tokens
            would_exceed_max = (current_tokens + sentence_tokens) > self.max_tokens

            if should_break or would_exceed_max:
                chunks.append(" ".join(current).strip())
                current = [sentence]
                current_tokens = sentence_tokens
            else:
                current.append(sentence)
                current_tokens += sentence_tokens

        if current:
            chunks.append(" ".join(current).strip())
        return chunks
