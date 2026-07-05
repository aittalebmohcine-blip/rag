from .Chunkers import TextChunker
from .Chunk import Chunk


def main() -> None:
    file = "datasets_public/public/AnsweredQuestions/dataset_code_public.json"
    text_chunker: TextChunker = TextChunker(2000, 20)
    result: list[Chunk] = text_chunker.chunk_file(file)
    print(result[:10], end="\n")


if __name__ == "__main__":
    main()
