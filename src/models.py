from pydantic import BaseModel, Field

import uuid
from typing import List


class MinimalSource(BaseModel):
    """Minimal file-and-span reference used in retrieval outputs."""

    file_path: str
    first_character_index: int
    last_character_index: int


class Chunk(BaseModel):
    """Text chunk with its absolute start/end offsets
    and optional file path."""

    text: str
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    file_path: str | None = None

    def to_minimal_source(self) -> MinimalSource:
        """Convert this chunk into a minimal source reference.

        Returns:
            MinimalSource: Source reference containing the file path and span.
        """
        return MinimalSource(
            file_path=self.file_path,
            first_character_index=self.start,
            last_character_index=self.end,
        )


class MinimalSearchResults(BaseModel):
    """Search output containing the question and its retrieved sources."""

    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """A searched question enriched with its generated answer."""

    answer: str


class StudentSearchResults(BaseModel):
    """Collection of search results produced for a batch of questions."""

    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Search results augmented with LLM-generated answers."""

    search_results: List[MinimalAnswer]
    k: int


class UnansweredQuestion(BaseModel):
    """Question entry that has not yet been answered."""

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """Question entry that includes the expected answer and source spans."""

    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """Dataset container holding both answered and unanswered questions."""

    rag_questions: List[AnsweredQuestion | UnansweredQuestion]
