from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from backend.app.config import get_settings
from backend.app.ingestion.pipeline import IngestionPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest documents into Weaviate")
    parser.add_argument("--path", required=True, help="File path or directory path")
    parser.add_argument(
        "--strategy",
        default="both",
        choices=["fixed", "semantic", "both"],
        help="Chunking strategy",
    )
    parser.add_argument("--suburb", default=None, help="Optional suburb metadata")
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional manifest output path; defaults to INGESTION_MANIFEST_PATH",
    )
    return parser.parse_args()


def iter_files(path_arg: str) -> list[Path]:
    path = Path(path_arg)
    if path.is_file():
        return [path]
    return [
        item
        for item in path.rglob("*")
        if item.is_file() and item.suffix.lower() in {".pdf", ".csv", ".txt", ".md"}
    ]


def append_manifest(manifest_path: Path, payload: dict) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def build_manifest_row(summary: dict, strategy: str, suburb: str | None) -> dict:
    return {
        "ingested_at": datetime.now(UTC).isoformat(),
        "file": summary["file"],
        "documents": summary["documents"],
        "inserted_chunks": summary["inserted_chunks"],
        "strategy": strategy,
        "suburb": suburb,
        "strategy_rollup": summary["strategy_rollup"],
    }


def main() -> None:
    args = parse_args()
    settings = get_settings()
    pipeline = IngestionPipeline()
    summaries = []
    manifest_path = Path(args.manifest or settings.ingestion_manifest_path)

    for file_path in iter_files(args.path):
        summary = pipeline.ingest_file(
            file_path=str(file_path),
            strategy=args.strategy,
            suburb=args.suburb,
        )
        summaries.append(summary)
        append_manifest(
            manifest_path,
            build_manifest_row(summary, strategy=args.strategy, suburb=args.suburb),
        )

    print(json.dumps({"items": summaries}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
