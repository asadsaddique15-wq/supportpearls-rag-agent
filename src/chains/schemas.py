# Pydantic schema (Task 07/10): defines the validated shape of every RAG answer.
# Every response -- including refusals -- must conform to this schema.
from __future__ import annotations
from enum import Enum
from pydantic import BaseModel, Field
class Confidence(str, Enum):
    # High = fully grounded, clear match. Partial = some but not all covered. None = refusal.
    HIGH = "high"
    PARTIAL = "partial"
    NONE = "none"

class RAGResponse(BaseModel):
    # The single validated output shape for the whole pipeline.
    answer: str = Field(description="The answer text shown to the customer, or a refusal message.")
    sources: list[str] = Field(default_factory=list, description="Source labels (e.g. S1, S2) actually used by the model.")
    confidence: Confidence = Field(description="high / partial / none")
    answered: bool = Field(description="True if the question was substantively answered, False if refused.")

class Citation(BaseModel):
    # A resolved, display-ready citation -- built AFTER validation, not by the model directly.
    source: str
    location: str