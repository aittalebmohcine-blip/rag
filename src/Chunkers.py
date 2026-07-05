from .Chunker import Chunker
from .Chunk import Chunk


class TextChunker(Chunker):
    def chunk(self, text: str) -> list[Chunk]:

        chunks: list[Chunk] = []
        for i in range(0, len(text), self.step):
            end = min(i + self.chunk_size, len(text))
            chunk = Chunk(
                text=text[i:end],
                start=i,
                end=end - 1,
            )
            chunks.append(chunk)
            if i + self.chunk_size >= len(text):
                break

        return chunks


# class PythonChunker(Chunker): ...
