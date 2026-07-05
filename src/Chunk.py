from pydantic import BaseModel, Field


class Chunk(BaseModel):
    text: str
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    file_path: str | None = None
