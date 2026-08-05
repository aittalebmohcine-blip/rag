import ast

from .Chunker import Chunker
from .models import Chunk


class TextChunker(Chunker):
    """Chunker for plain text content using a fixed-size sliding window."""

    def chunk(self, text: str) -> list[Chunk]:
        """Split plain text into overlapping chunks.

        Args:
            text: Raw text content to segment.

        Returns:
            list[Chunk]: Chunk objects produced from the text.
        """
        return self.chunk_with_offset(text, 0)

    def chunk_with_offset(
        self,
        text: str,
        offset: int,
    ) -> list[Chunk]:
        """Chunk text while preserving absolute character offsets.

        Args:
            text: The string to split.
            offset: Starting character offset for the chunk positions.

        Returns:
            list[Chunk]: Chunks aligned to the provided global offset.
        """
        chunks: list[Chunk] = []
        for i in range(0, len(text), self.step):
            end = min(i + self.chunk_size, len(text))
            chunk = Chunk(
                text=text[i:end],
                start=offset + i,
                end=offset + end - 1,
            )
            chunks.append(chunk)
            if i + self.chunk_size >= len(text):
                break

        return chunks


class PythonChunker(Chunker):
    """Chunker that prefers code-structure boundaries
    for Python source files."""

    def chunk(self, text: str) -> list[Chunk]:
        """Chunk Python source code using AST function and class boundaries.

        Args:
            text: Python source text to analyze.

        Returns:
            list[Chunk]:
                Structured chunks derived from function and class nodes,
                or a fallback text chunking result if parsing fails.
        """
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

        fallback_chunker = TextChunker(
            self.chunk_size,
            self.overlap,
        )
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                continue
            start_char, end_char = self.get_absolute_char_positions(
                node, line_offsets
            )
            chunk_text = text[start_char:end_char]
            if len(chunk_text) <= self.chunk_size:
                chunk = Chunk(
                    text=chunk_text, start=start_char, end=end_char - 1
                )
                chunks.append(chunk)
            else:
                chunks.extend(
                    fallback_chunker.chunk_with_offset(
                        chunk_text, offset=start_char)
                )
        return chunks

    @staticmethod
    def get_absolute_char_positions(
        node: ast.stmt, line_offsets: list[int]
    ) -> tuple[int, int]:
        """Convert AST node line/column positions into
        absolute character offsets.

        Args:
            node: AST statement node to convert.
            line_offsets: Prefix sums of each line length in the source text.

        Returns:
            tuple[int, int]: The inclusive start and exclusive end character
                offsets for the node.
        """
        # ast lines are 1-indexed; convert to 0-indexed index
        start_line_idx = node.lineno - 1
        start_col = node.col_offset
        start_char = line_offsets[start_line_idx] + start_col

        # Fallback to the end of the text if end coordinates are missing
        end_line_idx = getattr(node, "end_lineno", len(line_offsets)) - 1
        end_col = getattr(node, "end_col_offset", 0)
        end_char = line_offsets[end_line_idx] + end_col

        return start_char, end_char
