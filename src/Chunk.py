from pydantic import BaseModel, Field

from .MinimalSource import MinimalSource


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
            text=self.text,
        )
