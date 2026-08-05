from pydantic import ValidationError

from pathlib import Path
import json

from .models import MinimalSource, StudentSearchResults


def build_prompt(query: str, sources: list[MinimalSource]) -> str:
    """Construct a prompt for an LLM answerer using the top-ranked sources.

    Args:
        query: User question to answer.
        sources: Ranked source excerpts to include as context.

    Returns:
        str: Prompt string with the retrieved context and question.
    """
    prompt: str

    context = "\n\n".join(source_to_text(s) for s in sources[:3])

    prompt = f"""If the answer is not present in the context, answer:
"I don't know."

Context:
{context}


Question:
{query}


Answer:
"""

    return prompt


def source_to_text(source: MinimalSource) -> str:
    """Read the text content for a source span and return just that span.

    Args:
        source: Minimal source reference including file path and character range.

    Returns:
        str: The substring covering the requested source span.
    """
    with open(source.file_path, encoding="utf-8") as f:
        text = f.read()

    return text[
        source.first_character_index:
        source.last_character_index + 1
    ]


def load_student_search_results(path: Path) -> StudentSearchResults:
    """Load and validate a student search-results JSON payload.

    Args:
        path: Path to the JSON file describing student retrieval results.

    Returns:
        StudentSearchResults: Deserialized model instance.

    Raises:
        ValueError: If the file does not conform to the expected schema.
    """
    with path.open("r", encoding="utf-8") as f:
        content = json.load(f)
    try:
        return StudentSearchResults.model_validate(content)
    except ValidationError as e:
        raise ValueError(
            f"'{path}' does not conform to the"
            f" StudentSearchResults schema.\n{e}"
        ) from e
