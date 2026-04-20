from __future__ import annotations

from pathlib import Path

from langchain_ollama import OllamaEmbeddings

from backend.app.config import get_settings
from backend.app.ingestion.chunking import SemanticChunker, estimate_tokens, fixed_size_chunk
from backend.app.ingestion.loaders import RawDocument, load_documents
from backend.app.ingestion.quality import append_quality_log, summarize_chunks
from backend.app.retrieval.weaviate_store import WeaviateStore


class IngestionPipeline:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.store = WeaviateStore()
        self.embedder = OllamaEmbeddings(
            model=self.settings.ollama_embed_model,
            base_url=self.settings.ollama_base_url,
        )
        self.semantic_chunker = SemanticChunker()

    def ingest_file(self, file_path: str, strategy: str = "both", suburb: str | None = None) -> dict:
        path = Path(file_path)
        docs = load_documents(path, suburb=suburb)

        all_inserted = 0
        by_strategy: dict[str, dict] = {}

        for doc in docs:
            result = self._ingest_document(doc=doc, strategy=strategy)
            all_inserted += result["inserted"]
            for key, value in result["metrics"].items():
                by_strategy.setdefault(key, {"docs": 0, "chunk_count": 0})
                by_strategy[key]["docs"] += 1
                by_strategy[key]["chunk_count"] += int(value["chunk_count"])

        return {
            "file": str(path),
            "documents": len(docs),
            "inserted_chunks": all_inserted,
            "strategy_rollup": by_strategy,
        }

    def _ingest_document(self, doc: RawDocument, strategy: str) -> dict:
        source_tokens = estimate_tokens(doc.text)
        selected = ["fixed", "semantic"] if strategy == "both" else [strategy]

        inserted = 0
        metric_rows: dict[str, dict] = {}

        for one_strategy in selected:
            if one_strategy == "fixed":
                # Fixed windows are deterministic and easier to tune for latency and recall stability.
                chunks = fixed_size_chunk(
                    doc.text,
                    chunk_size=self.settings.fixed_chunk_size,
                    overlap=self.settings.fixed_chunk_overlap,
                )
            elif one_strategy == "semantic":
                # Semantic boundaries usually improve coherence but can vary chunk sizes considerably.
                chunks = self.semantic_chunker.chunk(doc.text)
            else:
                raise ValueError(f"Unknown chunking strategy: {one_strategy}")

            metric = summarize_chunks(chunks, source_tokens=source_tokens)
            metric_rows[one_strategy] = metric

            append_quality_log(
                self.settings.metrics_log_path,
                {
                    "source": doc.metadata.get("source", "unknown"),
                    "doc_type": doc.metadata.get("doc_type", "unknown"),
                    "strategy": one_strategy,
                    "source_tokens": source_tokens,
                    "metrics": metric,
                    "sample_chunk_preview": (chunks[0][:300] if chunks else ""),
                },
            )

            if chunks:
                embeddings = self.embedder.embed_documents(chunks)
                records = []
                for idx, (chunk_text, embedding) in enumerate(zip(chunks, embeddings, strict=False)):
                    records.append(
                        {
                            "text": chunk_text,
                            "source": doc.metadata.get("source", "unknown"),
                            "suburb": doc.metadata.get("suburb", "manningham"),
                            "doc_type": doc.metadata.get("doc_type", "unknown"),
                            "strategy": one_strategy,
                            "chunk_index": idx,
                            "embedding": embedding,
                        }
                    )
                inserted += self.store.upsert_chunks(records)

        if set(selected) == {"fixed", "semantic"}:
            append_quality_log(
                self.settings.metrics_log_path,
                {
                    "source": doc.metadata.get("source", "unknown"),
                    "comparison": {
                        "fixed_chunk_count": metric_rows["fixed"]["chunk_count"],
                        "semantic_chunk_count": metric_rows["semantic"]["chunk_count"],
                        "fixed_avg_tokens": metric_rows["fixed"]["avg_tokens"],
                        "semantic_avg_tokens": metric_rows["semantic"]["avg_tokens"],
                        "note": (
                            "Fixed chunking gives predictable window sizes and stable recall; "
                            "semantic chunking preserves meaning boundaries and often improves context cohesion."
                        ),
                    },
                },
            )

        return {"inserted": inserted, "metrics": metric_rows}
