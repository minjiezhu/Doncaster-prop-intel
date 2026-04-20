from time import perf_counter

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings

from backend.app.config import get_settings
from backend.app.retrieval.weaviate_store import WeaviateStore


class QueryService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.store = WeaviateStore()
        # Phase 1 keeps retrieval pure-vector to validate ingestion/chunk quality first.
        # Phase 2 will insert hybrid recall before answer generation.
        self.embeddings = OllamaEmbeddings(
            model=self.settings.ollama_embed_model,
            base_url=self.settings.ollama_base_url,
        )
        self.llm = ChatOllama(
            model=self.settings.ollama_chat_model,
            base_url=self.settings.ollama_base_url,
            temperature=0.1,
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

    def answer_question(self, question: str, top_k: int | None = None) -> dict:
        start = perf_counter()
        k = top_k or self.settings.retrieval_top_k
        query_embedding = self.embeddings.embed_query(question)
        rows = self.store.vector_search(vector=query_embedding, top_k=k)

        if not rows:
            elapsed_ms = int((perf_counter() - start) * 1000)
            return {
                "answer": "未检索到相关资料，请先摄入文档或调整问题范围。",
                "sources": [],
                "retrieval_debug": {"top_k": k, "retrieval_time_ms": elapsed_ms, "hits": 0},
            }

        context_chunks = []
        sources = []
        for row in rows:
            context_chunks.append(row.get("text", ""))
            sources.append(
                {
                    "id": row.get("_additional", {}).get("id", ""),
                    "source": row.get("source", "unknown"),
                    "suburb": row.get("suburb", "unknown"),
                    "strategy": row.get("strategy", "unknown"),
                    "distance": float(row.get("_additional", {}).get("distance", 0.0)),
                    "chunk_index": int(row.get("chunk_index", 0)),
                }
            )

        # Future reranker should be inserted here: after recall, before prompt assembly.
        messages = self.prompt.format_messages(question=question, context="\n\n".join(context_chunks))
        answer = self.llm.invoke(messages).content

        elapsed_ms = int((perf_counter() - start) * 1000)
        return {
            "answer": str(answer),
            "sources": sources,
            "retrieval_debug": {"top_k": k, "retrieval_time_ms": elapsed_ms, "hits": len(rows)},
        }
