import bm25s

from pathlib import Path

from .models import Chunk


def save_index(retriever: bm25s.BM25, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    retriever.save(str(output_dir / "bm25_index"))


def build_bm25(chunks: list[Chunk]) -> bm25s.BM25:
    corpus_texts = [chunk.text for chunk in chunks]

    tokenized_corpus = bm25s.tokenize(corpus_texts)

    retriever = bm25s.BM25()
    retriever.index(tokenized_corpus)

    return retriever
