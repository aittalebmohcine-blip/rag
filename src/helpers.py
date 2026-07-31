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


def validate_strict_pos_int(param_name: str, param_val: any) -> None:
    if isinstance(param_val, bool):
        raise ValueError(
            f"{param_name} requires an integer value."
        )
    if not isinstance(param_val, int) or param_val <= 0:
        raise ValueError(f"{param_name} must be a positive (non-zero) integer.")


def validate_str_arg(param_name: str, param_val: any):
    if isinstance(param_val, bool):
        raise ValueError(
                f"{param_name} should be a string.\n"
                "Note: a True/False value is NOT treated as a string. "
                "if you dont want that, then for example use '\"True\"'"
                )
