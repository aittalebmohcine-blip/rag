from pathlib import Path
from tqdm import tqdm

import json

from .models import Chunk
from .Chunkers import TextChunker, PythonChunker

TEXT_EXTENSIONS = {".md", ".txt"}
CODE_EXTENSIONS = {".py"}


def save_chunks(chunks: list[Chunk], output_dir: Path) -> None:
    """Persist a list of chunk objects to a JSON file.

    Args:
        chunks: Chunk objects to serialize.
        output_dir: Directory where the serialized chunk data should be saved.

    Returns:
        None:
            Writes the chunk payload to `chunks.json` in the output directory.
    """
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
    """Chunk every supported file in a repository
    and return all resulting chunks.

    Args:
        files: Source files to process.
        chunk_size: Maximum size of a single chunk in characters.
        overlap: Number of overlapping characters between adjacent chunks.

    Returns:
        list[Chunk]:
            A flattened list of all chunks produced from the repository.

    Raises:
        ValueError: If no chunks were produced.
    """

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
            print(f"\nSkipping {file}: {e}")

    if not all_chunks:
        raise ValueError("No data to be processed!")

    return all_chunks


def collect_files(
    root: Path,
    extensions: set[str],
) -> list[Path]:
    """Collect all files under a root path
    whose suffix is in the supported set.

    Args:
        root: Directory tree to scan.
        extensions: Allowed file suffixes to include.

    Returns:
        list[Path]: Matching file paths discovered recursively.
    """

    return [
        file_path
        for file_path in root.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() in extensions
    ]
