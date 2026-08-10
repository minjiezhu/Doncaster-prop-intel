"""Query service — Phase 2.

Search modes:
  "vector"        — Phase 1 baseline: pure cosine vector recall.
  "hybrid"        — BM25 + vector recall (Weaviate native hybrid).
  "hybrid_rerank" — hybrid recall followed by cross-encoder reranking.

Failure handling:
  - Weaviate unreachable → return graceful empty-result response, log fallback.
  - Ollama timeout       → return retrieved sources with a generation-failure notice.
  - Empty recall         → return informative no-results message.

Every query writes a structured entry to retrieval_comparison.jsonl so the three
modes can be compared side-by-side in Phase 2 analysis.
"""
from __future__ import annotations

from time import perf_counter

import httpx
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings

from backend.app.config import Settings, get_settings
from backend.app.retrieval.comparison_log import log_retrieval_comparison
from backend.app.retrieval.reranker import RankedChunk, build_reranker
from backend.app.retrieval.weaviate_store import WeaviateStore

_SEARCH_MODES = {"vector", "hybrid", "hybrid_rerank"}


class QueryService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.store = WeaviateStore()
        self.embeddings = OllamaEmbeddings(
            model=self.settings.ollama_embed_model,
            base_url=self.settings.ollama_base_url,
        )
        self.llm = ChatOllama(
            model=self.settings.ollama_chat_model,
            base_url=self.settings.ollama_base_url,
            temperature=0.1,
        )
        # Reranker is built once and lazy-loads the model on first use.
        self.reranker = build_reranker(
            provider=self.settings.reranker_provider,
            model_name=self.settings.reranker_model,
            cohere_api_key=self.settings.cohere_api_key,
        )
        self.prompt = ChatPromptTemplate.from_template(
            """
你是 Manningham 区域房产研究助手。仅根据提供上下文回答。
如果上下文不足，请明确说明不确定。

问题:
{question}

上下文:
{context}

请给出简洁、可追溯来源的回答。
""".strip()
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def answer_question(
        self,
        question: str,
        top_k: int | None = None,
        mode: str = "hybrid_rerank",
    ) -> dict:
        if mode not in _SEARCH_MODES:
            mode = "hybrid_rerank"

        k = top_k or self.settings.retrieval_top_k
        start = perf_counter()

        # --- Step 1: embed query ---
        try:
            query_embedding = self.embeddings.embed_query(question)
        except Exception as exc:
            return self._fallback_response(
                top_k=k,
                elapsed_start=start,
                mode=mode,
                reason=f"Embedding failed: {type(exc).__name__}",
            )

        # --- Step 2: recall ---
        recall_start = perf_counter()
        rows, fallback_triggered, fallback_reason = self._recall(
            question=question,
            vector=query_embedding,
            mode=mode,
            k=k,
        )
        recall_ms = int((perf_counter() - recall_start) * 1000)

        if not rows:
            return self._no_results_response(
                question=question,
                mode=mode,
                k=k,
                elapsed_start=start,
                candidates_before_rerank=0,
                fallback_triggered=fallback_triggered,
                fallback_reason=fallback_reason,
            )

        candidates_count = len(rows)

        # --- Step 3: rerank (only for hybrid_rerank mode) ---
        rerank_ms = 0
        if mode == "hybrid_rerank":
            rerank_start = perf_counter()
            ranked: list[RankedChunk] = self.reranker.rerank(
                query=question,
                chunks=rows,
                top_n=self.settings.reranker_top_n,
            )
            rerank_ms = int((perf_counter() - rerank_start) * 1000)

            min_score = self.settings.reranker_min_score
            if min_score is not None:
                ranked = [r for r in ranked if r.score >= min_score]
                if not ranked:
                    return self._no_results_response(
                        question=question,
                        mode=mode,
                        k=k,
                        elapsed_start=start,
                        candidates_before_rerank=candidates_count,
                        fallback_triggered=False,
                        fallback_reason=(
                            f"All reranked candidates scored below reranker_min_score={min_score}"
                        ),
                    )

            # Rebuild rows from ranked output so downstream code stays uniform
            rows = [
                {
                    "text": r.text,
                    **r.metadata,
                    "_rerank_score": r.score,
                }
                for r in ranked
            ]
            top_scores = [r.score for r in ranked]
        else:
            top_scores = [
                float(r.get("_additional", {}).get("distance", 0.0)) for r in rows
            ]

        # --- Step 4: assemble context and generate answer ---
        context_chunks = [row.get("text", "") for row in rows]
        sources = self._build_sources(rows, mode=mode)

        try:
            messages = self.prompt.format_messages(
                question=question,
                context="\n\n".join(context_chunks),
            )
            answer = str(self.llm.invoke(messages).content)
        except (httpx.TimeoutException, Exception) as exc:
            answer = f"[Answer generation failed: {type(exc).__name__}. Retrieved sources are shown below.]"

        total_ms = int((perf_counter() - start) * 1000)

        # --- Step 5: log comparison entry ---
        log_retrieval_comparison(
            self.settings.retrieval_comparison_log_path,
            question=question,
            mode=mode,
            top_k=k,
            candidates_before_rerank=candidates_count,
            final_hits=len(rows),
            top_scores=top_scores,
            retrieval_time_ms=recall_ms,
            rerank_time_ms=rerank_ms,
            fallback_triggered=fallback_triggered,
            fallback_reason=fallback_reason,
        )

        return {
            "answer": answer,
            "sources": sources,
            "retrieval_debug": {
                "mode": mode,
                "top_k": k,
                "candidates_before_rerank": candidates_count,
                "final_hits": len(rows),
                "retrieval_time_ms": recall_ms,
                "rerank_time_ms": rerank_ms,
                "total_time_ms": total_ms,
                "fallback_triggered": fallback_triggered,
            },
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _recall(
        self,
        question: str,
        vector: list[float],
        mode: str,
        k: int,
    ) -> tuple[list[dict], bool, str]:
        """Run recall for the selected mode.  Returns (rows, fallback_triggered, reason)."""
        # For hybrid_rerank, fetch extra candidates before reranking narrows the pool.
        candidate_k = (
            k * self.settings.hybrid_candidate_multiplier
            if mode in {"hybrid", "hybrid_rerank"}
            else k
        )
        try:
            if mode == "vector":
                rows = self.store.vector_search(vector=vector, top_k=k)
            else:
                rows = self.store.hybrid_search(
                    query=question,
                    vector=vector,
                    top_k=candidate_k,
                    alpha=self.settings.hybrid_alpha,
                )
            return rows, False, ""
        except httpx.ConnectError:
            return [], True, "Weaviate unreachable"
        except httpx.TimeoutException:
            return [], True, "Weaviate timeout"
        except Exception as exc:
            return [], True, f"Retrieval error: {type(exc).__name__}"

    def _build_sources(self, rows: list[dict], mode: str) -> list[dict]:
        sources = []
        for row in rows:
            additional = row.get("_additional", {})
            entry: dict = {
                "id": additional.get("id", ""),
                "source": row.get("source", "unknown"),
                "suburb": row.get("suburb", "unknown"),
                "strategy": row.get("strategy", "unknown"),
                "chunk_index": int(row.get("chunk_index", 0)),
                "mode": mode,
            }
            if mode == "hybrid_rerank":
                entry["rerank_score"] = float(row.get("_rerank_score", 0.0))
            else:
                entry["distance"] = float(additional.get("distance", 0.0))
            sources.append(entry)
        return sources

    def _no_results_response(
        self,
        question: str,
        mode: str,
        k: int,
        elapsed_start: float,
        candidates_before_rerank: int,
        fallback_triggered: bool,
        fallback_reason: str,
    ) -> dict:
        elapsed_ms = int((perf_counter() - elapsed_start) * 1000)
        log_retrieval_comparison(
            self.settings.retrieval_comparison_log_path,
            question=question,
            mode=mode,
            top_k=k,
            candidates_before_rerank=candidates_before_rerank,
            final_hits=0,
            top_scores=[],
            retrieval_time_ms=elapsed_ms,
            fallback_triggered=fallback_triggered,
            fallback_reason=fallback_reason,
        )
        return {
            "answer": "未检索到相关资料，请先摄入文档或调整问题范围。",
            "sources": [],
            "retrieval_debug": {
                "mode": mode,
                "top_k": k,
                "retrieval_time_ms": elapsed_ms,
                "hits": 0,
                "fallback_triggered": fallback_triggered,
                "fallback_reason": fallback_reason,
            },
        }

    def _fallback_response(
        self,
        top_k: int,
        elapsed_start: float,
        mode: str,
        reason: str,
    ) -> dict:
        elapsed_ms = int((perf_counter() - elapsed_start) * 1000)
        return {
            "answer": f"検索に失敗しました。原因: {reason}",
            "sources": [],
            "retrieval_debug": {
                "mode": mode,
                "top_k": top_k,
                "retrieval_time_ms": elapsed_ms,
                "hits": 0,
                "fallback_triggered": True,
                "fallback_reason": reason,
            },
        }

