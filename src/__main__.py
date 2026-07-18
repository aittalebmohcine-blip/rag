from typing import Any
import json
from pathlib import Path
import argparse

import bm25s
from tqdm import tqdm

from .Chunkers import PythonChunker, TextChunker
from .Chunk import Chunk
from .MinimalSource import MinimalSource

TEXT_EXTENSIONS = {".md", ".txt"}
CODE_EXTENSIONS = {".py"}


def collect_files(
    root: Path,
    extensions: set[str],
) -> list[Path]:
    return [
        file_path
        for file_path in root.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() in extensions
    ]


def chunk_repository(
    files: list[Path],
    chunk_size: int,
    overlap: int,
) -> list[Chunk]:

    text_chunker: TextChunker = TextChunker(chunk_size, overlap)
    python_chunker = PythonChunker(chunk_size, overlap)  # temporary
    all_chunks: list[Chunk] = []

    for file in tqdm(files, desc="Chunking"):
        try:
            chunker = (
                python_chunker
                if file.suffix.lower() in CODE_EXTENSIONS
                else text_chunker
            )

            all_chunks.extend(chunker.chunk_file(file))

        except (OSError, UnicodeDecodeError) as e:
            print(f"Skipping {file}: {e}")

    if not all_chunks:
        raise ValueError("No supported files found.")

    return all_chunks


def save_chunks(chunks: list[Chunk], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_ready_list = [chunk.model_dump(mode="json") for chunk in chunks]
    chunks_path = output_dir / "chunks.json"
    with chunks_path.open("w", encoding="utf-8") as f:
        json.dump(json_ready_list, f, indent=6)


def build_bm25(chunks: list[Chunk]) -> bm25s.BM25:
    corpus_texts = [chunk.text for chunk in chunks]

    tokenized_corpus = bm25s.tokenize(corpus_texts)

    retriever = bm25s.BM25(corpus=chunks)
    retriever.index(tokenized_corpus)

    return retriever


def save_index(retriever: bm25s.BM25, output_dir: Path) -> None:
    retriever.save(str(output_dir / "bm25_index"))


def search(
    query: str,
    retriever: bm25s.BM25,
    k: int = 5,
) -> Any:
    query_tokens = bm25s.tokenize(query)
    return retriever.retrieve(query_tokens, k=k)


def load_chunks(chunks_path: Path) -> list[Chunk]:
    with chunks_path.open() as f:
        return [Chunk(**obj) for obj in json.load(f)]


def index_repository(
    repository: Path,
    output_dir: Path,
    chunk_size: int,
    overlap: int,
) -> None:
    files = collect_files(repository, TEXT_EXTENSIONS | CODE_EXTENSIONS)

    chunks = chunk_repository(files, chunk_size, overlap)

    save_chunks(chunks, output_dir)

    retriever = build_bm25(chunks)

    save_index(retriever, output_dir)


def print_search_results(
    sources: list[MinimalSource],
    retrieval_scores: list[float],
) -> None:
    for c, s in zip(sources, retrieval_scores):
        print(f"Score: {s}\n")
        print(f"File:\n {c.file_path}\n")
        print(
            f"Characters:\n {c.first_character_index}-{c.last_character_index}"
        )
        print("-" * 20)
        print(f"{c.text}")
        print("-" * 20)
        print()


def main() -> None:
    # ----- cli ------#
    parser = argparse.ArgumentParser(
        description="Retrieval Augmented Generation"
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    subparsers.add_parser("index")

    search_parser = subparsers.add_parser("search")
    search_parser.add_argument("query")
    search_parser.add_argument("--k", type=int, default=5)

    subparsers.add_parser("search_dataset")

    args = parser.parse_args()
    # -----------#

    target_dir = Path("/home/mait-tal/Documents/rag/vllm-0.10.1")
    output_dir = Path("data/processed/")

    # ----- indexing ------#
    if args.command == "index":
        index_repository(target_dir, output_dir, 2000, 20)
        return
    # -----------#

    retriever = bm25s.BM25.load(output_dir / "bm25_index")
    chunks = load_chunks(output_dir / "chunks.json")
    # ----- one question retrieving ------#
    if args.command == "search":
        # TODO:
        # - put the retrieving logic in a separet funtion
        #   (maybe edit search return sources directly)
        docs, scores = search(args.query, retriever, args.k)

        doc_ids = docs[0]
        doc_scores = scores[0]

        found_chunks = [chunks[i] for i in doc_ids]
        retrieval_scores = [round(float(s), 2) for s in doc_scores]

        # ----- convert and print ------#
        sources = [chunk.to_minimal_source() for chunk in found_chunks]

        print_search_results(sources, retrieval_scores)
        # -----------#
    # -----------#

    # ----- search dataset ------#
    if args.command == "search_dataset":
        dataset_docs_file = ("/home/mait-tal/Documents/rag/datasets_public/"
                             "public/UnansweredQuestions/"
                             "dataset_docs_public.json")
        dataset_docs_path = Path(dataset_docs_file)
        with dataset_docs_path.open("r") as dp:
            data = json.load(dp)
        questions = [element["question"] for element in data["rag_questions"]]


if __name__ == "__main__":
    main()
