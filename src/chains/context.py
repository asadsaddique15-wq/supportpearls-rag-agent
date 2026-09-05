# Context formatting (Task 07/10): turns retrieved chunks into labeled [S1], [S2]... blocks,
# and tracks source+location so citations can show more than just a filename.

from __future__ import annotations
from dataclasses import dataclass
from langchain_core.documents import Document

MAX_CONTEXT_CHARS = 6000  # context budget -- prevents unbounded prompt growth


@dataclass
class FormattedContext:
    # Holds prompt-ready text plus lookup tables for citation resolution and de-duplication.
    text: str
    label_to_source: dict[str, str]        # e.g. {"S1": "warranty_policy.md"}
    label_to_location: dict[str, str]      # e.g. {"S1": "page 0"} or {"S1": "section unknown"}
    truncated: bool


def format_context(chunks: list[Document]) -> FormattedContext:
    # Build [S1]...[Sk] blocks, each labeled with its source and location, capped at a character budget.
    blocks = []
    label_to_source = {}
    label_to_location = {}
    total_chars = 0
    truncated = False

    for i, chunk in enumerate(chunks, start=1):
        label = f"S{i}"
        source = chunk.metadata.get("source", "unknown")
        location = chunk.metadata.get("location", "")
        label_to_source[label] = source
        label_to_location[label] = location

        block = f"[{label}] source={source} location={location}\n{chunk.page_content}\n"

        if total_chars + len(block) > MAX_CONTEXT_CHARS:
            truncated = True
            break

        blocks.append(block)
        total_chars += len(block)

    return FormattedContext(
        text="\n".join(blocks),
        label_to_source=label_to_source,
        label_to_location=label_to_location,
        truncated=truncated,
    )