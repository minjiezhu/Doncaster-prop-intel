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

### Reranker placement in later phases

- Retrieval flow will evolve to: recall (vector or hybrid) -> rerank -> answer generation.
- Keeping reranker after recall avoids over-constraining candidate set too early.

## Setup

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

Quality logs are written to the path in METRICS_LOG_PATH (default backend/logs/chunk_quality.jsonl).

## Query Endpoint

Example request:

```bash
curl "http://localhost:8000/query?question=What%20are%20school%20zones%20in%203108%3F&top_k=5"
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
			"distance": 0.12,
			"chunk_index": 3
		}
	],
	"retrieval_debug": {
		"top_k": 5,
		"retrieval_time_ms": 143,
		"hits": 5
	}
}
```

## Phase 1 Done Criteria

- Weaviate and backend start successfully via Docker Compose
- At least one PDF or CSV can be ingested into Weaviate
- chunk_quality.jsonl contains both fixed and semantic comparison entries
- GET /query returns an answer and source chunks
