"""
Pydantic models used across the project.

Keeping these in one place makes the data flowing through the LangGraph
workflow explicit and easy to reason about for a viva/demo.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A single retrieved chunk with its provenance metadata."""

    text: str
    filename: str
    page_number: int
    department: str
    score: float = 0.0


class Citation(BaseModel):
    """A citation shown to the user, derived purely from retrieval metadata."""

    filename: str
    page_number: int

    def display(self) -> str:
        return f"📄 {self.filename} — Page {self.page_number}"


class RouterOutput(BaseModel):
    """Structured output expected from the Manager (routing) Agent."""

    domain: str = Field(description="One of HR, TECHNICAL, PROJECTS, GENERAL")
