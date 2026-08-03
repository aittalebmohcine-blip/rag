import bm25s
from pydantic import ValidationError

from pathlib import Path
import json

from .models import (
    MinimalSource,
    Chunk,
    RagDataset
)


def print_search_results(sources: list[MinimalSource]) -> None:
    for source in sources:
        print(
            f"{source.file_path} "
            f"[{source.first_character_index}:{source.last_character_index}]"
        )


def retrieve_ids(
    query: str,
    retriever: bm25s.BM25,
    k: int = 5,
) -> list[int]:
    query_tokens = bm25s.tokenize(query)
    docs, scores = retriever.retrieve(query_tokens, k=k)
    doc_ids = docs[0]
    return doc_ids


def single_question_searcher(
    query: str,
    retriever: bm25s.BM25,
    k: int,
    chunks: list[Chunk]
) -> list[MinimalSource]:
    try:
        doc_ids = retrieve_ids(query, retriever, k)

        found_chunks = [chunks[i] for i in doc_ids]

        sources = [chunk.to_minimal_source() for chunk in found_chunks]

        return sources

    except Exception:
        raise FileNotFoundError(
            "An error accured due to invalid or corrupted index. "
            "Please run:\nuv run python -m src index"
        )


def load_chunks(chunks_path: Path) -> list[Chunk]:
    with chunks_path.open() as f:
        return [Chunk(**obj) for obj in json.load(f)]


def load_questions(path: Path) -> RagDataset:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    try:
        return RagDataset.model_validate(data)
    except ValidationError as e:
        raise ValueError(
            f"'{path}' does not conform to the RagDataset schema.\n{e}"
        ) from e
