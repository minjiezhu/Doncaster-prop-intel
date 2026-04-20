from backend.app.ingestion.chunking import fixed_size_chunk


def test_fixed_size_chunk_generates_multiple_chunks() -> None:
    text = " ".join(["token"] * 1200)
    chunks = fixed_size_chunk(text=text, chunk_size=512, overlap=50)
    assert len(chunks) >= 2
    assert all(len(chunk) > 0 for chunk in chunks)
