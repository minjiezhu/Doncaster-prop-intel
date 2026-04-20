from __future__ import annotations

import argparse
import json
from pathlib import Path

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


def main() -> None:
    args = parse_args()
    pipeline = IngestionPipeline()
    summaries = []

    for file_path in iter_files(args.path):
        summaries.append(
            pipeline.ingest_file(
                file_path=str(file_path),
                strategy=args.strategy,
                suburb=args.suburb,
            )
        )

    print(json.dumps({"items": summaries}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
