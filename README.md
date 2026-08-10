# doncaster-property-intel

> [中文文档](README.zh.md)

A RAG-powered property research assistant focused on Manningham, VIC.

Primary suburb: Doncaster (3108)
Extended coverage: Doncaster East (3109), Templestowe and Templestowe Lower (3106), Bulleen (3107)

## Tech Stack

- Backend: Python + FastAPI
- LLM orchestration: LangChain
- Local LLM: Ollama (default qwen3:14b)
- Vector store: Weaviate (Docker)
- Frontend: React (scaffold only in Phase 1)

## Phase 1 Deliverables

- Docker Compose with Weaviate + FastAPI backend
- Ingestion pipeline for PDF, CSV, TXT
- Two chunking strategies:
	- Fixed-size chunking (default: 512, overlap 50)
	- Semantic chunking (sentence-embedding breakpoint)
- Structured quality logging for chunking comparison
- Query endpoint: GET /query for vector retrieval + LangChain + Ollama answer generation

## Phase 2 Deliverables

- Hybrid search: Weaviate BM25 + vector with configurable alpha weighting
- Local cross-encoder reranker (sentence-transformers, CPU-friendly) with optional Cohere API swap
- Three search modes via `?mode=` param: `vector`, `hybrid`, `hybrid_rerank`
- Retrieval failure handling: Weaviate unreachable, Ollama timeout, empty recall
- Structured comparison log (retrieval_comparison.jsonl) for side-by-side mode analysis

## Phase 3 Deliverables

- LangChain ReAct agent that autonomously selects tools per query
- **Tool 1 — `rag_search`**: local knowledge base (PDFs, CSVs, reports) via hybrid_rerank pipeline
- **Tool 2 — `domain_listings`**: Domain.com.au property listings (stub mode without key; live with `DOMAIN_API_KEY`).
  Live mode requires a Domain Developer Portal API key, which requires completing a business
  profile — in practice this gates live mode behind being a registered real estate business, so
  stub mode is expected to be the permanent state for non-commercial use of this project.
- **Tool 3 — `web_search`**: DuckDuckGo fallback for recent news, council updates, planning permits
- Structured agent call log (`agent_calls.jsonl`): tools selected, per-tool latency, fallback flag
- New endpoint: `POST /agent` — accepts free-text question, returns answer + tool telemetry

## Project Structure

doncaster-property-intel/
- backend/
	- app/
		- api/
		- ingestion/
		- retrieval/
		- agent/
		- config.py
	- tests/
	- Dockerfile
	- requirements.txt
- frontend/
- data/
- docker-compose.yml
- .env.example
- .gitignore
- README.md

## Architecture Notes

### Why compare fixed and semantic chunking?

- Fixed chunking gives stable window sizes and deterministic retrieval behavior.
- Semantic chunking uses sentence-level meaning boundaries to reduce context breakage.
- The ingestion pipeline writes JSONL quality logs to make this trade-off measurable, not subjective.

### Why Phase 1 uses pure vector search first

- Pure vector search is the minimal end-to-end baseline.
- It simplifies validation of ingestion quality before introducing hybrid BM25 weighting.
- Phase 2 will extend this baseline with hybrid search and reranking for higher precision.

### Why hybrid over pure vector?

- Property queries often contain specific terms (suburb names, postcodes, price thresholds) that benefit
  from exact keyword (BM25) recall.
- Pure vector search may miss these if the embedding space does not cleanly separate them.
- Hybrid fuses both signals; `alpha` is configurable so the balance can be tuned per query type.

### Reranker placement

- Retrieval flow: recall (vector or hybrid) → rerank → prompt assembly → answer generation.
- Recall casts a wide net (`top_k × HYBRID_CANDIDATE_MULTIPLIER`); the reranker then narrows to `RERANKER_TOP_N`.
- Keeping reranker after recall avoids over-constraining the candidate pool too early.
- Local cross-encoder (`ms-marco-MiniLM-L-6-v2`) runs on CPU with no external dependency; swap to
  Cohere by setting `RERANKER_PROVIDER=cohere` and `COHERE_API_KEY` in `.env`.
- Reranking only narrows/reorders candidates — it does not reject irrelevant ones by default.
  Set `RERANKER_MIN_SCORE` in `.env` to drop candidates scoring below a threshold (treated the
  same as an empty recall). Left unset by default: score scale differs by provider (local
  cross-encoder logits are unbounded, Cohere returns 0-1), so a sane value has to be calibrated
  against real ingested data, not guessed.

### Why a ReAct agent over a fixed pipeline?

- Property research spans multiple information sources: local documents, current listings, and real-time web.
- A single pipeline cannot reliably serve all query types; an agent selects the right tool per query.
- ReAct (Reason + Act) forces the LLM to articulate its tool choice before calling it — this trace is
  visible in `agent_calls.jsonl` for explainability and debugging.
- `max_iterations=6` caps runaway loops; `handle_parsing_errors=True` keeps the API stable even when
  the LLM produces malformed tool-call JSON.

## Setup

> Running the backend outside Docker (tests, the ingestion CLI, etc.) requires **Python 3.11–3.12** — the pinned `numpy`/`langchain` versions have no wheels for 3.13+. The Docker image itself is unaffected (pinned to `python:3.11-slim`).

1. Copy environment file.

```bash
cp .env.example .env
```

2. Ensure Ollama is running locally and model is available.

```bash
ollama pull qwen3:14b
ollama pull nomic-embed-text
```

3. Start services.

```bash
docker compose up --build
```

4. Open API docs.

- http://localhost:8000/docs

## Ingestion CLI

Ingest a single file using both chunking strategies:

```bash
python -m backend.app.ingestion.cli --path data/sample.txt --strategy both --suburb doncaster
```

Ingest an entire folder:

```bash
python -m backend.app.ingestion.cli --path data/ --strategy fixed
```

Write an ingestion manifest while processing files:

```bash
python -m backend.app.ingestion.cli --path data/doncaster --strategy both --suburb Doncaster --manifest backend/logs/ingestion_manifest.jsonl
```

Quality logs are written to the path in METRICS_LOG_PATH (default backend/logs/chunk_quality.jsonl).
Ingestion manifests default to `INGESTION_MANIFEST_PATH` (default `backend/logs/ingestion_manifest.jsonl`).

## Query Endpoint

The `mode` parameter selects the retrieval pipeline:

| mode | behaviour |
|------|-----------|
| `vector` | Phase 1 baseline — pure cosine vector recall |
| `hybrid` | BM25 + vector recall (Weaviate native hybrid) |
| `hybrid_rerank` | hybrid recall + cross-encoder reranking (default) |

Example request:

```bash
curl "http://localhost:8000/query?question=What%20are%20school%20zones%20in%203108%3F&top_k=5&mode=hybrid_rerank"
```

Example response shape:

```json
{
	"answer": "...",
	"sources": [
		{
			"id": "...",
			"source": "school-zones.pdf",
			"suburb": "doncaster",
			"strategy": "semantic",
			"rerank_score": 8.43,
			"chunk_index": 3,
			"mode": "hybrid_rerank"
		}
	],
	"retrieval_debug": {
		"mode": "hybrid_rerank",
		"top_k": 5,
		"candidates_before_rerank": 15,
		"final_hits": 5,
		"retrieval_time_ms": 120,
		"rerank_time_ms": 38,
		"total_time_ms": 310,
		"fallback_triggered": false
	}
}
```

## Phase 1 Done Criteria

- Weaviate and backend start successfully via Docker Compose
- At least one PDF or CSV can be ingested into Weaviate
- chunk_quality.jsonl contains both fixed and semantic comparison entries
- GET /query returns an answer and source chunks

## Agent Endpoint

`POST /agent` — runs the ReAct agent over a free-text question.

Example request:

```bash
curl -X POST http://localhost:8000/agent \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the median price in Doncaster East and are there any listings under $1.2M?"}'
```

Example response:

```json
{
  "answer": "The median price in Doncaster East (3109) is ...",
  "tools_used": ["rag_search", "domain_listings"],
  "tool_latencies_ms": {"rag_search": 280, "domain_listings": 95},
  "total_time_ms": 1840,
  "fallback_triggered": false
}
```

## Phase 2 Done Criteria

- `?mode=hybrid` returns results with BM25 + vector fusion
- `?mode=hybrid_rerank` returns results reordered by cross-encoder score
- retrieval_comparison.jsonl captures entries for all three modes
- Weaviate unreachable and Ollama timeout both return graceful responses
- All Phase 2 tests pass

## Phase 3 Done Criteria

- `POST /agent` routes questions to the appropriate tool without hard-coded routing logic
- All three tools (rag_search, domain_listings, web_search) are callable and return valid string output
- agent_calls.jsonl records tools_selected, tool_latencies_ms, fallback_triggered for every request
- Domain stub mode works without a DOMAIN_API_KEY; live mode activates when key is set
- All Phase 3 tests pass

## Roadmap

| Phase | Goal | Status |
|-------|------|--------|
| 1 | Basic RAG pipeline (ingestion + vector search + Q&A) | Done |
| 2 | Retrieval quality (hybrid search, reranking, failure handling) | Done |
| 3 | Agentic workflow (multi-tool agent, Domain API, structured logging) | Done |
