from __future__ import annotations

import json
from pathlib import Path

from backend.app.ingestion.cli import append_manifest, build_manifest_row, iter_files


def test_iter_files_filters_supported_extensions(tmp_path: Path):
    supported = tmp_path / "market-report.pdf"
    supported.write_text("pdf", encoding="utf-8")
    ignored = tmp_path / "notes.docx"
    ignored.write_text("docx", encoding="utf-8")

    result = iter_files(str(tmp_path))

    assert result == [supported]


def test_build_manifest_row_uses_summary_fields():
    summary = {
        "file": "data/doncaster/report.pdf",
        "documents": 2,
        "inserted_chunks": 17,
        "strategy_rollup": {"fixed": {"docs": 2, "chunk_count": 17}},
    }

    row = build_manifest_row(summary, strategy="fixed", suburb="Doncaster")

    assert row["file"] == summary["file"]
    assert row["strategy"] == "fixed"
    assert row["suburb"] == "Doncaster"
    assert row["inserted_chunks"] == 17
    assert "ingested_at" in row


def test_append_manifest_writes_jsonl(tmp_path: Path):
    manifest_path = tmp_path / "logs" / "ingestion_manifest.jsonl"
    payload = {"file": "data/doncaster/report.pdf", "inserted_chunks": 4}

    append_manifest(manifest_path, payload)

    lines = manifest_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == payload