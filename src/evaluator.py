import json
from pathlib import Path

from .Question import RagDataset, AnsweredQuestion
from .MinimalSource import StudentSearchResults, MinimalSource


def recall(
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

def load_dataset(dataset_path: Path) -> RagDataset:
    with dataset_path.open("r", encoding="utf-8") as f:
        return RagDataset(**json.load(f))


def load_student_results(
    student_search_results_path: Path
) -> StudentSearchResults:
    with student_search_results_path.open("r", encoding="utf-8") as f:
        return StudentSearchResults(**json.load(f))


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
