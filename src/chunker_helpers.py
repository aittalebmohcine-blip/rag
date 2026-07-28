from pathlib import Path
from tqdm import tqdm

import json

from .models import Chunk
from .Chunkers import TextChunker, PythonChunker

TEXT_EXTENSIONS = {".md", ".txt"}
CODE_EXTENSIONS = {".py"}

def save_chunks(chunks: list[Chunk], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_ready_list = [chunk.model_dump(mode="json") for chunk in chunks]
    chunks_path = output_dir / "chunks.json"
    with chunks_path.open("w", encoding="utf-8") as f:
        json.dump(json_ready_list, f, indent=6)

def chunk_repository(
    files: list[Path],
    chunk_size: int,
    overlap: int,
) -> list[Chunk]:

    text_chunker = TextChunker(chunk_size, overlap)
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

def collect_files(
    root: Path,
    extensions: set[str],
) -> list[Path]:

    return [
        file_path
        for file_path in root.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() in extensions
    ]
