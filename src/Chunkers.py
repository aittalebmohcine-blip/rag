import ast
from tracemalloc import start

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


class PythonChunker(Chunker):
    def chunk(self, text: str) -> list[Chunk]:
        code_lines = text.splitlines()
        tree = ast.parse(text)
        chunks = []

        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start_line = node.lineno + 1
                end_line = node.end_lineno
                chunk_text = "/n".join(code_lines[start_line:end_line])
                if end_line is not None:
                    chunk = Chunk(text=chunk_text, start=start_line, end=end_line)
                    chunks.append(chunk)
        return chunks
