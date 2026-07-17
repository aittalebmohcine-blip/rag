import ast

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


# TODO:
# - Split oversized functions/classes (> chunk_size)
# - Merge small module-level statements (imports, constants)
# - Preserve node metadata (FunctionDef/ClassDef)
# - Consider docstrings separately
class PythonChunker(Chunker):
    def chunk(self, text: str) -> list[Chunk]:
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return TextChunker(self.chunk_size, self.overlap).chunk(text)

        # Pre-calculate the starting character index for every line
        code_lines = text.splitlines(keepends=True)
        line_offsets = [0]
        for line in code_lines:
            line_offsets.append(line_offsets[-1] + len(line))

        chunks: list[Chunk] = []

        for node in tree.body:
            start_char, end_char = self.get_absolute_char_positions(
                node, line_offsets
            )
            chunk = Chunk(
                text=text[start_char:end_char], start=start_char, end=end_char
            )
            chunks.append(chunk)
        return chunks

    @staticmethod
    def get_absolute_char_positions(
        node: ast.stmt, line_offsets: list[int]
    ) -> tuple[int, int]:
        # ast lines are 1-indexed; convert to 0-indexed index
        start_line_idx = node.lineno - 1
        start_col = node.col_offset
        start_char = line_offsets[start_line_idx] + start_col

        # Fallback to the end of the text if end coordinates are missing
        end_line_idx = getattr(node, "end_lineno", len(line_offsets)) - 1
        end_col = getattr(node, "end_col_offset", 0)
        end_char = line_offsets[end_line_idx] + end_col

        return start_char, end_char
