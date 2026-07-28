from abc import ABC, abstractmethod
from pathlib import Path

from .models import Chunk


class Chunker(ABC):
    def __init__(self, chunk_size: int, overlap: int) -> None:
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")

        self.chunk_size: int = chunk_size
        self.overlap: int = overlap
        self.step: int = chunk_size - overlap

    @abstractmethod
    def chunk(self, text: str) -> list[Chunk]:
        """Chunk a string into smaller pieces."""
        ...

    def chunk_file(self, path: str | Path) -> list[Chunk]:
        """Read a file, chunk it, and attach the file path to every chunk."""

        path = Path(path)

        with path.open("r", encoding="utf-8") as f:
            text = f.read()

        chunks = self.chunk(text)

        for chunk in chunks:
            chunk.file_path = str(path)

        return chunks
