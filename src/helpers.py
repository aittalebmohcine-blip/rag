import bm25s

from pathlib import Path
from typing import Any

from .searcher import load_chunks
from .models import Chunk


def load_index_and_chunks(
        processed_data_path: Path
) -> tuple[bm25s.BM25, list[Chunk]]:
    """Load a BM25 retriever and its serialized chunk corpus from disk.

    Args:
        processed_data_path: Directory containing the saved BM25 index and
            chunk JSON file.

    Returns:
        tuple[bm25s.BM25, list[Chunk]]: The loaded retriever and chunk corpus.

    Raises:
        FileNotFoundError:
            If the processed data directory is invalid or missing.
    """
    try:
        retriever = bm25s.BM25.load(processed_data_path / "bm25_index")
        chunks = load_chunks(processed_data_path / "chunks.json")
    except Exception:
        raise FileNotFoundError(
            "Index is invalid or corrupted. "
            "Please run:\nuv run python -m src index"
        )
    return retriever, chunks


def validate_strict_pos_int(param_name: str, param_val: Any) -> None:
    """Validate that a parameter is a positive integer and not a boolean.

    Args:
        param_name: Name of the argument being validated.
        param_val: Candidate value to inspect.

    Raises:
        ValueError: If the input is not a strictly positive integer.
    """
    if isinstance(param_val, bool):
        raise ValueError(
            f"{param_name} requires an integer value."
        )
    if not isinstance(param_val, int) or param_val <= 0:
        raise ValueError(
            f"{param_name} must be a positive (non-zero) integer.")


def validate_str_arg(param_name: str, param_val: Any) -> None:
    """Validate that a parameter is a non-empty string-like argument.

    Args:
        param_name: Name of the argument being validated.
        param_val: Candidate value to inspect.

    Raises:
        ValueError: If the input is a boolean or an empty string.
    """
    if isinstance(param_val, bool):
        raise ValueError(
            f"{param_name} should be a string.\n"
            "Note: a True/False value is NOT treated as a string. "
            "if you dont want that, then for example use '\"True\"'"
        )

    if isinstance(param_val, str) and not param_val.strip():
        raise ValueError(f"{param_name} cannot be empty.")
