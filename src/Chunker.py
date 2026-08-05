from abc import ABC, abstractmethod
from pathlib import Path

from .models import Chunk


class Chunker(ABC):
    """Abstract base class for chunking text into structured `Chunk` objects.

    Args:
        chunk_size: Maximum number of characters in a single chunk.
        overlap: Number of overlapping characters shared between adjacent
            chunks.
    """

    def __init__(self, chunk_size: int, overlap: int) -> None:
        """Initialize chunking parameters and derived step size.

        Args:
            chunk_size: Maximum size of each chunk in characters.
            overlap: Number of overlapping characters between chunks.

        Raises:
            ValueError: If overlap is not smaller than chunk_size.
        """
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")

        self.chunk_size: int = chunk_size
        self.overlap: int = overlap
        self.step: int = chunk_size - overlap

    @abstractmethod
    def chunk(self, text: str) -> list[Chunk]:
        """Split a text string into a list of chunk objects.

        Args:
            text: Raw string content to chunk.

        Returns:
            list[Chunk]: Chunk objects describing the segmented text.
        """
        ...

    def chunk_file(self, path: str | Path) -> list[Chunk]:
        """Read a file, chunk it, and attach the file path to every chunk.

        Args:
            path: Location of the file to read and segment.

        Returns:
            list[Chunk]: Chunks derived from the file text, each annotated with
                the file path.
        """

        path = Path(path)

        with path.open("r", encoding="utf-8") as f:
            text = f.read()

        chunks = self.chunk(text)

        for chunk in chunks:
            chunk.file_path = str(path)

        return chunks
