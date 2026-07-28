from pydantic import BaseModel, Field

import uuid
from typing import List


class MinimalSource(BaseModel):
    file_path: str
    first_character_index: int
    last_character_index: int


class Chunk(BaseModel):
    text: str
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    file_path: str | None = None

    def to_minimal_source(self) -> MinimalSource:
        return MinimalSource(
            file_path=self.file_path,
            first_character_index=self.start,
            last_character_index=self.end,
        )


class MinimalSearchResults(BaseModel):
    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    answer: str


class StudentSearchResults(BaseModel):
    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    search_results: List[MinimalAnswer]
    k: int


class UnansweredQuestion(BaseModel):
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    rag_questions: List[AnsweredQuestion | UnansweredQuestion]
