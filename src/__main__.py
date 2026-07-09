import json

import bm25s
from pathlib import Path
from tqdm import tqdm

from .Chunkers import TextChunker
from .Chunk import Chunk

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
    python_chunker = text_chunker  # temporary
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
        json.dump(json_ready_list, f, indent=4)


def build_bm25(chunks: list[Chunk]) -> bm25s.BM25:
    corpus_texts = [chunk.text for chunk in chunks]

    tokenized_corpus = bm25s.tokenize(corpus_texts)

    retriever = bm25s.BM25(corpus=chunks)
    retriever.index(tokenized_corpus)

    return retriever


def save_index(retriever: bm25s.BM25, output_dir: Path) -> None:
    retriever.save(str(output_dir / "bm25_index"))


def main() -> None:
    target_dir = Path("/home/mait-tal/Documents/rag1/vllm-0.10.1")
    output_dir = Path("data/processed/")

    files = collect_files(target_dir, TEXT_EXTENSIONS | CODE_EXTENSIONS)

    chunks = chunk_repository(files, 2000, 20)

    save_chunks(chunks, output_dir)

    retriever = build_bm25(chunks)

    save_index(retriever, output_dir)


if __name__ == "__main__":
    main()
