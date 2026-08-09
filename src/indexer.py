import bm25s

from pathlib import Path

from .models import Chunk


def save_index(retriever: bm25s.BM25, output_dir: Path) -> None:
    """Persist a BM25 retriever to disk.

    Args:
        retriever: BM25 retriever object to serialize.
        output_dir: Directory where the index files should be written.

    Returns:
        None: Saves the BM25 index to the provided directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    retriever.save(str(output_dir / "bm25_index"))


def build_bm25(chunks: list[Chunk]) -> bm25s.BM25:
    """Build a BM25 retriever from a list of chunk objects.

    Args:
        chunks: Chunk corpus used to populate the index.

    Returns:
        bm25s.BM25: An indexed BM25 retriever ready for retrieval.
    """
    corpus_texts = [chunk.text for chunk in chunks]

    tokenized_corpus: bm25s.tokenization.Tokenized = bm25s.tokenize(
        corpus_texts
    )

    retriever = bm25s.BM25()
    retriever.index(tokenized_corpus)

    return retriever
