from .models import (
    RagDataset,
    AnsweredQuestion,
    StudentSearchResults,
    MinimalSource
)


def recall(
    student_results: StudentSearchResults,
    ground_truth: RagDataset,
) -> float:
    """Compute the average question recall for a set of search results.

    Args:
        student_results: Predicted search results for the dataset.
        ground_truth: Ground-truth dataset containing expected answers and
            source spans.

    Returns:
        float: Mean recall across all questions in the dataset.
    """
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


def overlap(
    retrieved: MinimalSource,
    expected: MinimalSource,
) -> float:
    """Measure overlap between a retrieved source span and an expected one.

    Args:
        retrieved: Predicted source span.
        expected: Ground-truth source span.

    Returns:
        float: Ratio of overlap relative to the expected span length.
    """
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

    return float(intersection / expected_length)


def source_found(
    retrieved: list[MinimalSource],
    expected: MinimalSource,
) -> bool:
    """Check whether a predicted source list contains the expected source.

    Args:
        retrieved: Retrieved source spans.
        expected: Source span that should be found.

    Returns:
        bool: True when the expected source overlaps sufficiently with at least
            one retrieved source.
    """
    return any(
        overlap(source, expected) >= 0.05
        for source in retrieved
    )


def question_recall(
    retrieved: list[MinimalSource],
    expected: list[MinimalSource],
) -> float:
    """Compute recall for one question using retrieved and expected sources.

    Args:
        retrieved: Predicted source spans.
        expected: Ground-truth source spans.

    Returns:
        float: Fraction of expected sources that were successfully retrieved.
    """
    if not expected:
        return 0.0

    found = sum(
        source_found(retrieved, correct_source)
        for correct_source in expected
    )
    recall = found / len(expected)

    return recall
