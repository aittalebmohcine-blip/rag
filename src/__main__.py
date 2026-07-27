from fire import Fire

from .CLI import CLI
# ---

from typing import Any
import json
from pathlib import Path
import argparse

# import pydantic
import bm25s
from tqdm import tqdm

from .Chunkers import PythonChunker, TextChunker
from .Chunk import Chunk
from .MinimalSource import (
    MinimalSource,
    MinimalSearchResults,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
    MinimalAnswer,
    )
from .Question import AnsweredQuestion, RagDataset, UnansweredQuestion
from .Generator import Generator

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


def save_chunks(chunks: list[Chunk], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_ready_list = [chunk.model_dump(mode="json") for chunk in chunks]
    chunks_path = output_dir / "chunks.json"
    with chunks_path.open("w", encoding="utf-8") as f:
        json.dump(json_ready_list, f, indent=6)


def build_bm25(chunks: list[Chunk]) -> bm25s.BM25:
    corpus_texts = [chunk.text for chunk in chunks]

    tokenized_corpus = bm25s.tokenize(corpus_texts)

    retriever = bm25s.BM25(corpus=chunks)
    retriever.index(tokenized_corpus)

    return retriever


def save_index(retriever: bm25s.BM25, output_dir: Path) -> None:
    retriever.save(str(output_dir / "bm25_index"))


def search(
    query: str,
    retriever: bm25s.BM25,
    k: int = 5,
) -> Any:
    query_tokens = bm25s.tokenize(query)
    docs, scores = retriever.retrieve(query_tokens, k=k)
    doc_ids = docs[0]
    doc_scores = scores[0]
    return doc_ids, doc_scores


def load_chunks(chunks_path: Path) -> list[Chunk]:
    with chunks_path.open() as f:
        return [Chunk(**obj) for obj in json.load(f)]


def index_repository(
    repository: Path,
    output_dir: Path,
    chunk_size: int,
    overlap: int,
) -> None:
    files = collect_files(repository, TEXT_EXTENSIONS | CODE_EXTENSIONS)

    chunks = chunk_repository(files, chunk_size, overlap)

    save_chunks(chunks, output_dir)

    retriever = build_bm25(chunks)

    save_index(retriever, output_dir)


def print_search_results(
    sources: list[MinimalSource],
    retrieval_scores: list[float],
) -> None:
    for c, s in zip(sources, retrieval_scores):
        print(f"Score: {s}\n")
        print(f"File:\n {c.file_path}\n")
        print(
            f"Characters:\n {c.first_character_index}-{c.last_character_index}"
        )
        print("-" * 20)
        print(f"{c.text}")
        print("-" * 20)
        print()


def searcher(
    query: str,
    retriever: bm25s.BM25,
    k: int,
    chunks: list[Chunk]
) -> tuple[list[MinimalSource], list[int]]:
    doc_ids, doc_scores = search(query, retriever, k)

    found_chunks = [chunks[i] for i in doc_ids]
    retrieval_scores = [round(float(s), 2) for s in doc_scores]

    # ----- convert ------#
    sources = [chunk.to_minimal_source() for chunk in found_chunks]

    return sources, retrieval_scores


def load_questions(path: Path) -> list[dict[str, Any]]:
    with path.open("r") as dp:
        data = json.load(dp)
        return data["rag_questions"]


def overlap(
    retrieved: MinimalSource,
    expected: MinimalSource,
) -> float:
    if retrieved.file_path != expected.file_path:
        return 0
    last_indexe = min(retrieved.last_character_index,
                      expected.last_character_index)
    first_indexe = max(retrieved.first_character_index,
                       expected.first_character_index)

    intersection = max(0, last_indexe - first_indexe)

    expected_length = (
        expected.last_character_index
        - expected.first_character_index
    )
    if expected_length <= 0:
        return 0.0

    return intersection / expected_length


def source_found(
    retrieved: list[MinimalSource],
    expected: MinimalSource,
) -> bool:
    return any(
        overlap(source, expected) >= 0.05
        for source in retrieved
    )


def question_recall(
    retrieved: list[MinimalSource],
    expected: list[MinimalSource],
) -> float:
    if not expected:
        return 0.0

    found = sum(
        source_found(retrieved, correct_source)
        for correct_source in expected
    )
    recall = found / len(expected)

    return recall


def evaluate(
    student_results: StudentSearchResults,
    ground_truth: RagDataset,
) -> float:
    ground_truth_by_id: dict[str, AnsweredQuestion] = {
        question.question_id: question
        for question in ground_truth.rag_questions
    }

    total = 0.0
    for result in student_results.search_results:
        expected = ground_truth_by_id.get(result.question_id)
        if expected is None:
            raise ValueError(
                f"Unknown question_id: {result.question_id}"
            )

        total += question_recall(
            result.retrieved_sources,
            expected.sources,
        )

    return total / len(ground_truth_by_id)


def load_answered_questions(answered_questions_dir: Path) -> RagDataset:
    file = answered_questions_dir/"dataset_docs_public.json"
    with file.open("r", encoding="utf-8") as f:
        return RagDataset(**json.load(f))


def load_student_results(student_results_dir: Path) -> StudentSearchResults:
    file = student_results_dir/"dataset_docs_public.json"
    with file.open("r", encoding="utf-8") as f:
        return StudentSearchResults(**json.load(f))


def answer(
    unasnswered_question: UnansweredQuestion,
    retriever: bm25s.BM25,
    chunks: list[Chunk],
    generator: Generator,
    k: int,
) -> MinimalAnswer:

    # Retrieve the top k chunks.
    chunk_ids, _ = search(
            query=unasnswered_question.question,
            retriever=retriever,
            k=k
            )
    retrieved_chunks = [chunks[i] for i in chunk_ids]

    # Convert them to context.
    context = "\n\n".join(chunk.text for chunk in retrieved_chunks)

    # Build the prompt.
    prompt = f"""If the answer is not present in the context, answer:
"I don't know."

Context:
{context}


Question:
{unasnswered_question.question}


Answer:
"""

    generated_answer = generator.generate(prompt)
    retrieved_sources=[chunk.to_minimal_source() for chunk in retrieved_chunks]
    return MinimalAnswer(
            answer=generated_answer,
            question_id=unasnswered_question.question_id,
            question=unasnswered_question.question,
            retrieved_sources=retrieved_sources,
            )

def answer_dataset(
    rag_dataset: RagDataset,
    retriever: bm25s.BM25,
    chunks: list[Chunk],
    generator: Generator,
    k: int,
) -> StudentSearchResultsAndAnswer:
    answers: list[MinimalAnswer] = []

    for question in tqdm(rag_dataset.rag_questions, desc="Answering"):
        result = answer(
                question,
                retriever,
                chunks,
                generator,
                k
                )
        answers.append(result)
    results_and_answer = StudentSearchResultsAndAnswer(
            search_results=answers,
            k=k
            )
    return results_and_answer

def main() -> None:
    # ----- cli ------#
    parser = argparse.ArgumentParser(
        description="Retrieval Augmented Generation"
    )
    # ---
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )
    # ---
    subparsers.add_parser("index")
    # ---
    search_parser = subparsers.add_parser("search")
    search_parser.add_argument("query")
    search_parser.add_argument("--k", type=int, default=5)
    # ---
    search_dataset_parser = subparsers.add_parser("search_dataset")
    search_dataset_parser.add_argument(
        "--dataset_path",
        type=Path,
        default=Path("datasets_public/public/UnansweredQuestions/"
                     "dataset_docs_public.json"),
        help="Path to the dataset JSON file.",
    )

    search_dataset_parser.add_argument(
        "--k",
        type=int,
        default=2,
        help="Number of chunks to retrieve per question.",
    )

    search_dataset_parser.add_argument(
        "--save_directory",
        type=Path,
        default=Path("data/output/search_results/UnansweredQuestions"),
        help="Target directory for the dataset_docs_public.json file.",
    )
    # ---
    subparsers.add_parser("evaluate")
    # ---
    subparsers.add_parser("answer")
    # ---
    subparsers.add_parser("answer_dataset")
    # ---

    args = parser.parse_args()
    # -----------#

    target_dir = Path("data/raw/vllm-0.10.1")
    output_dir = Path("data/processed/")

    # ----- indexing ------#
    if args.command == "index":
        index_repository(target_dir, output_dir, 2000, 20)
        return
        # -----------#

    retriever = bm25s.BM25.load(output_dir / "bm25_index")
    chunks = load_chunks(output_dir / "chunks.json")
    match args.command:
        # ----- one question retrieving ------#
        case "search":
            sources, retrieval_scores = searcher(
                args.query, retriever, args.k, chunks)

            print_search_results(sources, retrieval_scores)
        # -----------#

        # ----- search dataset ------#
        case "search_dataset":
            # dataset_docs_path: Path = Path(args.dataset_path)

            questions: list[dict[str, Any]] = load_questions(args.dataset_path)

            # save_directory: Path = Path(args.save_directory)
            file_path = args.save_directory / "dataset_docs_public.json"
            results: list[MinimalSearchResults] = []
            for question in questions:
                sources, _ = searcher(
                    question["question"], retriever, args.k, chunks)
                results.append(
                    MinimalSearchResults(
                        question_id=question["question_id"],
                        question=question["question"],
                        retrieved_sources=sources
                    )
                )
            student_search_results = StudentSearchResults(
                search_results=results, k=args.k)

            args.save_directory.mkdir(parents=True, exist_ok=True)
            with file_path.open("w", encoding="utf-8") as file:
                json.dump(student_search_results.model_dump(
                    mode="json"), file, indent=4)

            print(
                f"Saved student_search_results to {file_path}"
            )

        case "evaluate":
            results: StudentSearchResults = load_student_results(
                Path("data/output/search_results"))
            answered: RagDataset = load_answered_questions(
                Path("datasets_public/public/AnsweredQuestions"))
            print(evaluate(results, answered))

            return
        case "answer":
            import time
            k = 2
            generator = Generator("Qwen/Qwen3-0.6B")
            q = "could the vllm one dat be slef conscious"
            question = UnansweredQuestion(
                    question_id="",
                    question=q,
                    )
            start_time = time.time()
            minimal_answer = answer(
                    unasnswered_question=question,
                    retriever=retriever,
                    chunks=chunks,
                    generator=generator,
                    k=k
                    )
            print(minimal_answer.answer)
            print("taux: ", time.time() - start_time)
        case "answer_dataset":
            k = 2
            generator = Generator("Qwen/Qwen3-0.6B")

            print("loading questions\n...")
            with open("datasets_public/public/UnansweredQuestions/dataset_docs_public.json", "r") as f:
                rag_dataset = RagDataset(**json.load(f))
            print("rag_dataset loaded\n")

            results = answer_dataset(
                    rag_dataset,
                    retriever=retriever,
                    chunks=chunks,
                    generator=generator,
                    k=k
                    )

            save_file = "/home/mait-tal/Documents/rag/data/output/search_results/AnsweredQuestions/dataset_docs_public.json"
            with open(save_file, "w", encoding="utf-8") as f:
                json.dump(
                    results.model_dump(mode="json"),
                    f,
                    indent=4,
                    )

def main_1() -> None:
    Fire(CLI)

if __name__ == "__main__":
    main_1()
