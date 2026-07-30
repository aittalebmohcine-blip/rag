import bm25s

from pathlib import Path

from .searcher import load_chunks

def load_index_and_chunks(processed_data_path: Path):
    try:
        retriever = bm25s.BM25.load(processed_data_path / "bm25_index")
        chunks = load_chunks(processed_data_path / "chunks.json")
    except Exception:
        raise FileNotFoundError(
                "Index is invalid or corrupted. Please run:\nuv run python -m src index"
                )
    return retriever, chunks
