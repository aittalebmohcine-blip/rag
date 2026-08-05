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
    """Print a compact representation of retrieved source locations.

    Args:
        sources: Source spans returned by the retriever.

    Returns:
        None: Emits each source line to standard output.
    """
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
    """Return the top-k document IDs for a query using the BM25 retriever.

    Args:
        query: Search string to rank against the index.
        retriever: BM25 index used for retrieval.
        k: Number of document IDs to return.

    Returns:
        list[int]: Ranked document IDs retrieved for the query.
    """
    query_tokens = bm25s.tokenize(query)
    docs, scores = retriever.retrieve(query_tokens, k=k)
    doc_ids = docs[0]
    return list(doc_ids)


def single_question_searcher(
    query: str,
    retriever: bm25s.BM25,
    k: int,
    chunks: list[Chunk]
) -> list[MinimalSource]:
    """Retrieve the top-k minimal source snippets for a single question.

    Args:
        query: Natural-language question to search for.
        retriever: Indexed BM25 retriever.
        k: Number of source documents to retrieve.
        chunks: Corpus chunks used to convert document IDs into source spans.

    Returns:
        list[MinimalSource]: Ranked source references for the query.

    Raises:
        FileNotFoundError: If the index is invalid or cannot be used for
            retrieval.
    """
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
    """Load the chunk JSON payload into a list of `Chunk` models.

    Args:
        chunks_path: Path to the serialized chunk JSON file.

    Returns:
        list[Chunk]: Deserialized chunk objects.
    """
    with chunks_path.open() as f:
        return [Chunk(**obj) for obj in json.load(f)]


def load_questions(path: Path) -> RagDataset:
    """Load and validate a dataset JSON payload into a `RagDataset` model.

    Args:
        path: Path to the dataset JSON file.

    Returns:
        RagDataset: Deserialized and validated dataset model.

    Raises:
        ValueError: If the payload does not conform to the expected schema.
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    try:
        return RagDataset.model_validate(data)
    except ValidationError as e:
        raise ValueError(
            f"'{path}' does not conform to the RagDataset schema.\n{e}"
        ) from e
