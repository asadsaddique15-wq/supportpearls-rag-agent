# Loader dispatch for ingestion (Task 02): walks the KB folder, loads each file by
# extension, normalizes text, attaches metadata, skips duplicates and bad files safely.

from __future__ import annotations
import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Filename keyword -> doc_type, used to auto-tag documents.
DOC_TYPE_HINTS = {
    "manual": "manual",
    "warranty": "policy",
    "refund": "policy",
    "return": "policy",
    "shipping": "policy",
    "privacy": "policy",
    "faq": "faq",
    "pricing": "pricing",
    "price": "pricing",
    "install": "guide",
    "troubleshoot": "guide",
    "service": "guide",
    "handbook": "guide",
    "changelog": "guide",
    "release": "guide",
}

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".md", ".txt", ".csv"}


@dataclass
class IngestionResult:
    # Holds everything needed for the ingestion report.
    documents: list[Document] = field(default_factory=list)
    loaded_files: list[str] = field(default_factory=list)
    skipped_files: list[tuple[str, str]] = field(default_factory=list)  # (filename, reason)
    duplicate_files: list[str] = field(default_factory=list)

    def summary(self) -> str:
        # Human-readable report printed after ingestion runs.
        lines = [
            "=" * 60,
            "INGESTION REPORT",
            "=" * 60,
            f"Files loaded successfully : {len(self.loaded_files)}",
            f"Files skipped (errors)    : {len(self.skipped_files)}",
            f"Duplicate files skipped   : {len(self.duplicate_files)}",
            f"Total chunks/documents    : {len(self.documents)}",
            "-" * 60,
        ]
        if self.loaded_files:
            lines.append("Loaded:")
            lines.extend(f"  ✓ {f}" for f in self.loaded_files)
        if self.skipped_files:
            lines.append("Skipped (with reason):")
            lines.extend(f"  ✗ {f}: {reason}" for f, reason in self.skipped_files)
        if self.duplicate_files:
            lines.append("Duplicates skipped:")
            lines.extend(f"  = {f}" for f in self.duplicate_files)
        lines.append("=" * 60)
        return "\n".join(lines)


def _infer_doc_type(filename: str) -> str:
    # Guess doc_type from filename keywords.
    name = filename.lower()
    for hint, doc_type in DOC_TYPE_HINTS.items():
        if hint in name:
            return doc_type
    return "unknown"


def _normalise_whitespace(text: str) -> str:
    # Collapse repeated spaces/blank lines without destroying paragraph breaks.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _content_hash(text: str) -> str:
    # Used to detect duplicate content across files.
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def _load_pdf(path: Path) -> list[Document]:
    # One Document per page; page number lands in metadata automatically.
    from langchain_community.document_loaders import PyPDFLoader
    return PyPDFLoader(str(path)).load()


def _load_docx(path: Path) -> list[Document]:
    from langchain_community.document_loaders import Docx2txtLoader
    return Docx2txtLoader(str(path)).load()


def _load_text(path: Path) -> list[Document]:
    # Handles both .md and .txt.
    from langchain_community.document_loaders import TextLoader
    return TextLoader(str(path), encoding="utf-8").load()


def _load_csv(path: Path) -> list[Document]:
    # Each row becomes one Document — keeps price/SKU pairs intact (see manual Mistake #13).
    from langchain_community.document_loaders import CSVLoader
    return CSVLoader(str(path), encoding="utf-8").load()


_LOADER_DISPATCH = {
    ".pdf": _load_pdf,
    ".docx": _load_docx,
    ".md": _load_text,
    ".txt": _load_text,
    ".csv": _load_csv,
}


def load_knowledge_base(kb_dir: Path) -> IngestionResult:
    # Main ingestion entry point — never let one bad file abort the whole run.
    result = IngestionResult()
    seen_hashes: dict[str, str] = {}  # content_hash -> first filename that had it

    if not kb_dir.exists():
        raise FileNotFoundError(f"Knowledge base directory not found: {kb_dir}")

    all_files = sorted(p for p in kb_dir.rglob("*") if p.is_file())
    logger.info("Found %d file(s) in %s", len(all_files), kb_dir)

    for path in all_files:
        ext = path.suffix.lower()
        rel_name = path.relative_to(kb_dir).as_posix()

        if ext not in SUPPORTED_EXTENSIONS:
            reason = f"unsupported extension '{ext}'"
            logger.warning("Skipping %s: %s", rel_name, reason)
            result.skipped_files.append((rel_name, reason))
            continue

        try:
            raw_docs = _LOADER_DISPATCH[ext](path)  # per-file try/except is the key error-handling requirement
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            logger.warning("Skipping %s (failed to load): %s", rel_name, reason)
            result.skipped_files.append((rel_name, reason))
            continue

        if not raw_docs:
            reason = "loader returned zero documents"
            logger.warning("Skipping %s: %s", rel_name, reason)
            result.skipped_files.append((rel_name, reason))
            continue

        doc_type = _infer_doc_type(path.name)
        file_had_content = False
        file_was_duplicate = False

        for i, doc in enumerate(raw_docs):
            cleaned = _normalise_whitespace(doc.page_content)
            if not cleaned:
                continue  # drop empty pages/documents

            content_hash = _content_hash(cleaned)
            if content_hash in seen_hashes:
                logger.info("Duplicate content detected in %s (matches %s) — skipping.", rel_name, seen_hashes[content_hash])
                file_was_duplicate = True
                if rel_name not in result.duplicate_files:
                    result.duplicate_files.append(rel_name)
                continue
            seen_hashes[content_hash] = rel_name

            # Preserve loader-supplied metadata (e.g. PDF page number), then layer the metadata contract on top.
            merged_metadata = dict(doc.metadata)
            location = merged_metadata.get("page")
            location = f"page {location}" if location is not None else (f"row {i}" if ext == ".csv" else "section unknown")

            merged_metadata.update({
                "source": path.name,
                "doc_type": doc_type,
                "product_line": "all",       # refine once real KB content exists
                "version": "unknown",        # refine once documents carry a date header
                "location": location,
            })

            result.documents.append(Document(page_content=cleaned, metadata=merged_metadata))
            file_had_content = True

        if file_had_content:
            result.loaded_files.append(rel_name)
        elif not file_was_duplicate:
            reason = "all content was empty after normalisation"
            logger.warning("Skipping %s: %s", rel_name, reason)
            result.skipped_files.append((rel_name, reason))
        # else: entirely duplicate content, already recorded — don't double-report as an error.

    logger.info(
        "Ingestion complete: %d loaded, %d skipped, %d duplicates, %d total documents.",
        len(result.loaded_files), len(result.skipped_files), len(result.duplicate_files), len(result.documents),
    )
    return result