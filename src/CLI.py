import bm25s
from tqdm import tqdm

from pathlib import Path
import json

from .models import(
        RagDataset,
        MinimalSearchResults,
        StudentSearchResults,
        MinimalAnswer,
        StudentSearchResultsAndAnswer
        )
from .answerer import build_prompt, source_to_text, load_student_search_results
from .chunker_helpers import collect_files, chunk_repository, save_chunks
from .indexer import build_bm25, save_index
from .searcher import (
        print_search_results,
        retrieve_ids,
        single_question_searcher,
        load_chunks,
        load_questions,
        )
from .evaluator import (
        recall,
        load_dataset,
        load_student_results,
        )
from .Generator import Generator

TEXT_EXTENSIONS = {".md", ".txt"}
CODE_EXTENSIONS = {".py"}

OVERLAP_RATIO = 0.15

class CLI:
    def index(
        self,
        max_chunk_size: int=2000
    ) -> None:
        if not isinstance(max_chunk_size, int) or not 0 < max_chunk_size <= 2000:
            raise ValueError("max_chunk_size must be a positive (non-zero) integer and cannot exceed 2000.")

        overlap = int(max_chunk_size * OVERLAP_RATIO)

        repository = Path("data/raw")
        output_dir = Path("data/processed")

        print("---- Ingesting data/raw to build an index ----\n")

        print("Collecting files...")
        files: list[Path] = collect_files(repository, TEXT_EXTENSIONS | CODE_EXTENSIONS)
        if not files:
            raise ValueError(
                    "No supported files found. Make sure the target repository is under data/raw"
                    )
        print("Files collected.\n")

        chunks: list[Chunk] = chunk_repository(files, max_chunk_size, overlap)

        print(f"Saving chunks under {output_dir}...")
        save_chunks(chunks, output_dir)
        print(f"Chunks saved with max_chunk_size={max_chunk_size}.\n")

        print("Building the index...")
        retriever: bm25s.BM25 = build_bm25(chunks)
        print("Index has ben built.\n")

        print(f"Saving the index under {output_dir}...\n")
        save_index(retriever, output_dir)

        print(f"Ingestion complete! Indices saved under {output_dir}\n")


    def search(
        self,
        query: str,
        k: int = 5,
    ) -> None:
        query = str(query)
        if not isinstance(k, int) or k <= 0:
            raise ValueError("'k' must be a positive (non-zero) integer.\n")

        output_dir = Path("data/processed")

        print(f"Loading the index from {output_dir}...")
        try:
            retriever = bm25s.BM25.load(output_dir / "bm25_index")
            chunks = load_chunks(output_dir / "chunks.json")
        except Exception:
            raise FileNotFoundError(
                    "Index is invalid or corrupted. Please run:\nuv run python -m src index"
                    )
        print("Index loaded.\n")

        print("Looking for relevant sources...\n")
        try:
            sources: list[MinimalSource] = single_question_searcher(
                query, retriever, k, chunks)
        except Exception:
            raise FileNotFoundError(
                    "An error accured due to invalid or corrupted index. Please run:\nuv run python -m src index"
                    )

        print("results:")
        print_search_results(sources)

    def search_dataset(
        self,
        dataset_path: str,
        k: int = 5,
        save_directory: str = "data/output/search_results/UnansweredQuestions",
    ) -> None:
        output_dir = Path("data/processed")
        save_directory = Path(save_directory)
        dataset_path = Path(dataset_path)

        retriever = bm25s.BM25.load(output_dir / "bm25_index")
        chunks = load_chunks(output_dir / "chunks.json")
        rag_dataset: RagDataset = load_questions(dataset_path)

        results: list[MinimalSearchResults] = []
        for unanswered_question in rag_dataset.rag_questions:
            sources = single_question_searcher(
                    unanswered_question.question,
                    retriever,
                    k,
                    chunks
                    )
            results.append(
                MinimalSearchResults(
                    question_id=unanswered_question.question_id,
                    question=unanswered_question.question,
                    retrieved_sources=sources
                )
            )
        student_search_results = StudentSearchResults(
            search_results=results, k=k)

        save_directory.mkdir(parents=True, exist_ok=True)
        save_path = save_directory / dataset_path.name
        with save_path.open("w", encoding="utf-8") as file:
            json.dump(student_search_results.model_dump(
                mode="json"), file, indent=4)
        print(
            f"Saved student_search_results to {save_path}"
        )

    def answer(
        self,
        query: str,
        k: int = 5,
    ) -> None:
        output_dir = Path("data/processed")

        retriever = bm25s.BM25.load(output_dir / "bm25_index")
        chunks = load_chunks(output_dir / "chunks.json")

        generator = Generator("Qwen/Qwen3-0.6B")

        sources: list[MinimalSource] = single_question_searcher(
            query, retriever, k, chunks)

        prompt = build_prompt(query, sources)
        generated_answer = generator.generate(prompt)
        print(generated_answer)


    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str = "data/output/search_results/AnsweredQuestions"
    ):
        student_search_results_path = Path(student_search_results_path)
        save_directory = Path(save_directory)

        generator = Generator("Qwen/Qwen3-0.6B")

        student_search_results: StudentSearchResults = load_student_search_results(
                student_search_results_path
                )

        answers: list[MinimalAnswer] = []
        for result in tqdm(student_search_results.search_results, desc="Answering"):
            prompt = build_prompt(
                    result.question,
                    result.retrieved_sources
                    )
            generated_answer = generator.generate(prompt)
            answers.append(MinimalAnswer(
                **result.model_dump(),
                answer=generated_answer
                ))

        results_and_answer = StudentSearchResultsAndAnswer(
                search_results=answers,
                k=student_search_results.k
                )

        save_directory.mkdir(parents=True, exist_ok=True)
        save_path = save_directory / student_search_results_path.name
        with save_path.open("w", encoding="utf-8") as file:
            json.dump(results_and_answer.model_dump(
                mode="json"), file, indent=4)
        print(
            f"Saved student_search_results to {save_path}"
        )


    def evaluate(
        self,
        student_search_results_path : str,
        dataset_path: str,
    ) -> None:
        student_search_results_path = Path(student_search_results_path)
        dataset_path = Path(dataset_path)

        student_search_results: StudentSearchResults = load_student_results(
                student_search_results_path
                )
        dataset: RagDataset = load_dataset(dataset_path)

        recall_at_k = recall(student_search_results, dataset)
        print(f"recall@{student_search_results.k}: {recall_at_k}")
