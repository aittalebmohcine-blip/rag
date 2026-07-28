from pathlib import Path
import json

from .models import MinimalSource, StudentSearchResults


def build_prompt(query: str, sources: list[MinimalSource]):
    prompt: str

    context = "\n\n".join(source_to_text(s) for s in sources)

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
    with open(source.file_path, encoding="utf-8") as f:
        text = f.read()

    return text[
        source.first_character_index :
        source.last_character_index + 1
    ]


def load_student_search_results(student_search_results_path: Path):
    with student_search_results_path.open("r", encoding="utf-8") as f:
        content = json.load(f)
    return StudentSearchResults(**content)
