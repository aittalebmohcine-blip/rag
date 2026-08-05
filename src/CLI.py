import bm25s
from tqdm import tqdm

from pathlib import Path
import json

from .models import (
    RagDataset,
    MinimalSearchResults,
    StudentSearchResults,
    MinimalAnswer,
    StudentSearchResultsAndAnswer,
    AnsweredQuestion,
    Chunk,
    MinimalSource,
)
from .answerer import build_prompt, load_student_search_results
from .chunker_helpers import collect_files, chunk_repository, save_chunks
from .indexer import build_bm25, save_index
from .searcher import (
    print_search_results,
    single_question_searcher,
    load_questions,
)
from .evaluator import recall
from .Generator import Generator
from .helpers import (
    load_index_and_chunks,
    validate_strict_pos_int, validate_str_arg
)

TEXT_EXTENSIONS = {".md", ".txt"}
CODE_EXTENSIONS = {".py"}

OVERLAP_RATIO = 0.15


class CLI:
    """Command-line interface for
    indexing, searching, answering, and evaluating RAG data.

    The CLI exposes the workflow steps used for the repository ingestion and
    retrieval pipeline.
    """

    def index(
        self,
        max_chunk_size: int = 2000
    ) -> None:
        """Build the BM25 index and chunked corpus for repository data.

        Args:
            max_chunk_size: Maximum character size used for each chunk.

        Returns:
            None: Saves processed chunks and the BM25 index to the configured
                data directory.
        """
        validate_strict_pos_int("max_chunk_size", max_chunk_size)
        if max_chunk_size > 2000:
            raise ValueError("max_chunk_size cannot exceed 2000.")

        overlap = int(max_chunk_size * OVERLAP_RATIO)

        repository = Path("data/raw")
        output_dir = Path("data/processed")

        print("---- Ingesting data/raw to build an index ----\n")

        print("Collecting files...")
        files: list[Path] = collect_files(
            repository, TEXT_EXTENSIONS | CODE_EXTENSIONS)
        if not files:
            raise ValueError(
                "No supported files found. "
                "Make sure the target repository is under data/raw"
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
        """Search the processed index for a single query string.

        Args:
            query: Natural-language question to search for.
            k: Number of retrieved sources to return.

        Returns:
            None: Prints the retrieved sources to the console.
        """
        validate_strict_pos_int("k", k)
        validate_str_arg("query", query)
        query = str(query)

        output_dir = Path("data/processed")

        print(f"Loading the index from {output_dir}...")
        retriever, chunks = load_index_and_chunks(output_dir)
        print("Index loaded.\n")

        print("Looking for relevant sources...\n")
        sources: list[MinimalSource] = single_question_searcher(
            query, retriever, k, chunks)

        print("results:")
        print_search_results(sources)

    def search_dataset(
        self,
        dataset_path: str | Path,
        k: int = 5,
        save_directory: str | Path =
        "data/output/search_results/UnansweredQuestions",
    ) -> None:
        """Run retrieval over an entire dataset and persist the results.

        Args:
            dataset_path: Path to the dataset file containing questions.
            k: Number of sources to retrieve per question.
            save_directory: Directory where the JSON search results should be
                written.

        Returns:
            None: Writes JSON results for the dataset to disk.
        """
        validate_str_arg("dataset_path", dataset_path)
        validate_strict_pos_int("k", k)
        validate_str_arg("save_directory", save_directory)

        dataset_path = Path(str(dataset_path))
        save_directory = Path(str(save_directory))

        index_dir = Path("data/processed")

        print("Loading the dataset...")
        rag_dataset: RagDataset = load_questions(dataset_path)
        print("Dataset loaded.\n")

        print(f"Loading the index from {index_dir}...")
        retriever, chunks = load_index_and_chunks(index_dir)
        print("Index loaded.\n")

        print("Generating search results...")
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
        print("Search results generated.\n")

        try:
            save_directory.mkdir(parents=True, exist_ok=True)
        except FileExistsError:
            raise ValueError(
                f"Unable to created the directory '{save_directory}'."
                " A file exists with the same path!"
            )

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
        """Generate an answer for a single question using the indexed sources.

        Args:
            query: Question to answer.
            k: Number of relevant sources to include in the prompt.

        Returns:
            None: Prints the generated answer to the console.
        """
        validate_str_arg("query", query)
        validate_strict_pos_int("k", k)

        query = str(query)
        index_dir = Path("data/processed")

        print(f"Loading the index from {index_dir}...")
        retriever, chunks = load_index_and_chunks(index_dir)
        print("Index loaded.\n")

        print("Looking for relevant sources...\n")
        sources: list[MinimalSource] = single_question_searcher(
            query, retriever, k, chunks)

        print("Running the LLM...")
        generator = Generator("Qwen/Qwen3-0.6B")

        prompt = build_prompt(query, sources)
        generated_answer = generator.generate(prompt)

        print("\nLLM's Answer:")
        print(generated_answer)

    def answer_dataset(
        self,
        student_search_results_path: str | Path,
        save_directory: str | Path =
        "data/output/search_results/AnsweredQuestions"
    ) -> None:
        """Answer every question in a search-result dataset
        and save the output.

        Args:
            student_search_results_path: Path to a JSON file containing
                retrieved search results.
            save_directory: Directory where the answered dataset should be
                written.

        Returns:
            None: Writes the answered dataset to disk.
        """
        validate_str_arg(
            "student_search_results_path",
            student_search_results_path
        )
        validate_str_arg("save_directory", save_directory)

        student_search_results_path = Path(str(student_search_results_path))
        save_directory = Path(str(save_directory))

        print("Loading student search results...")
        student_search_results: StudentSearchResults = \
            load_student_search_results(student_search_results_path)
        print("Search results loaded.\n")

        print("Running the LLM...")
        generator = Generator("Qwen/Qwen3-0.6B")
        answers: list[MinimalAnswer] = []
        for result in tqdm(
                student_search_results.search_results,
                desc="Answering"
        ):
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

        try:
            save_directory.mkdir(parents=True, exist_ok=True)
        except FileExistsError:
            raise ValueError(
                f"Unable to created the directory '{save_directory}'."
                " A file exists with the same path!"
            )
        print(f"\n{save_directory} created.\n")
        save_path = save_directory / student_search_results_path.name
        with save_path.open("w", encoding="utf-8") as file:
            json.dump(results_and_answer.model_dump(
                mode="json"), file, indent=4)
        print(
            f"Saved student_search_results to {save_path}"
        )

    def evaluate(
        self,
        student_search_results_path: str | Path,
        dataset_path: str | Path,
    ) -> None:
        """Compute retrieval recall for a student search-results file.

        Args:
            student_search_results_path:
                Path to the predicted retrieval output.
            dataset_path: Path to the ground-truth dataset.

        Returns:
            None: Prints the recall score for the supplied results.
        """
        validate_str_arg(
            "student_search_results_path",
            student_search_results_path
        )
        validate_str_arg("dataset_path", dataset_path)

        student_search_results_path = Path(str(student_search_results_path))
        dataset_path = Path(str(dataset_path))

        print("Loading student search results...")
        student_search_results: StudentSearchResults = \
            load_student_search_results(student_search_results_path)
        print("Search results loaded.\n")

        print("Loading the dataset...")
        dataset: RagDataset = load_questions(dataset_path)
        if not all(isinstance(q, AnsweredQuestion)
                   for q in dataset.rag_questions):
            raise ValueError(
                f"the RagDataset.rag_questions in '{dataset_path}' "
                "does not conform to the list[AnsweredQuestion] schema."
            )
        print("Dataset loaded.\n")

        recall_at_k = recall(student_search_results, dataset)
        print(f"recall@{student_search_results.k}: {recall_at_k}")
