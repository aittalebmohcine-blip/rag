from typing import List
from pydantic import BaseModel


class MinimalSource(BaseModel):
    file_path: str
    first_character_index: int
    last_character_index: int

    # extra field
    text: str


class MinimalSearchResults(BaseModel):
    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class StudentSearchResults(BaseModel):
    search_results: List[MinimalSearchResults]
    k: int
