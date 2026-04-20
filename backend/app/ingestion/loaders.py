from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass
class RawDocument:
    text: str
    metadata: dict[str, str]


def load_documents(file_path: Path, suburb: str | None = None) -> list[RawDocument]:
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(file_path, suburb=suburb)
    if suffix == ".csv":
        return _load_csv(file_path, suburb=suburb)
    if suffix in {".txt", ".md"}:
        return _load_text(file_path, suburb=suburb)
    raise ValueError(f"Unsupported file type: {suffix}")


def _base_meta(file_path: Path, doc_type: str, suburb: str | None) -> dict[str, str]:
    return {
        "source": str(file_path.name),
        "doc_type": doc_type,
        "suburb": suburb or "manningham",
    }


def _load_text(file_path: Path, suburb: str | None) -> list[RawDocument]:
    text = file_path.read_text(encoding="utf-8", errors="ignore")
    return [RawDocument(text=text, metadata=_base_meta(file_path, doc_type="text", suburb=suburb))]


def _load_pdf(file_path: Path, suburb: str | None) -> list[RawDocument]:
    reader = PdfReader(str(file_path))
    pages = []
    for page_num, page in enumerate(reader.pages):
        content = page.extract_text() or ""
        if content.strip():
            meta = _base_meta(file_path, doc_type="pdf", suburb=suburb)
            meta["page"] = str(page_num + 1)
            pages.append(RawDocument(text=content, metadata=meta))
    return pages


def _load_csv(file_path: Path, suburb: str | None) -> list[RawDocument]:
    docs: list[RawDocument] = []
    with file_path.open("r", encoding="utf-8", errors="ignore", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row_index, row in enumerate(reader):
            pairs = [f"{key}: {value}" for key, value in row.items()]
            text = "\n".join(pairs)
            meta = _base_meta(file_path, doc_type="csv", suburb=suburb)
            meta["row"] = str(row_index)
            docs.append(RawDocument(text=text, metadata=meta))
    return docs
