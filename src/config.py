# Central config module — loads env vars ONCE and exposes a single validated `settings` object.
# Every other module imports `settings` from here instead of calling os.getenv() directly.

from __future__ import annotations
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # reads .env from project root


class ConfigError(RuntimeError):
    # Raised when required configuration is missing or invalid.
    pass


def _require(name: str) -> str:
    # Fetch a required env var, raise a clear error if missing.
    value = os.getenv(name)
    if value is None or value.strip() == "":
        raise ConfigError(f"Missing required environment variable: {name}. Copy .env.example to .env and fill it in.")
    return value


def _optional(name: str, default: str) -> str:
    # Fetch an optional env var, falling back to a default.
    value = os.getenv(name)
    return value if value not in (None, "") else default


def _require_int(name: str) -> int:
    raw = _require(name)
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"Environment variable {name}={raw!r} is not a valid int.") from exc


def _require_float(name: str) -> float:
    raw = _require(name)
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"Environment variable {name}={raw!r} is not a valid float.") from exc


@dataclass(frozen=True)
class Settings:
    # All tunable/config values live here — nothing hard-coded elsewhere in the app.
    llm_provider: str
    llm_model: str
    llm_temperature: float
    openai_api_key: str | None
    anthropic_api_key: str | None
    embedding_provider: str
    embedding_model: str
    vector_store_path: Path
    vector_store_collection: str
    chunk_size: int
    chunk_overlap: int
    retrieval_k: int
    retrieval_score_threshold: float
    knowledge_base_dir: Path
    log_level: str
    log_dir: Path

    def masked_dict(self) -> dict:
        # Return settings as a dict with secrets masked, safe to print/log.
        def mask(secret: str | None) -> str:
            if not secret:
                return "<not set>"
            return secret[:4] + "…" + "*" * 6

        return {
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "llm_temperature": self.llm_temperature,
            "openai_api_key": mask(self.openai_api_key),
            "anthropic_api_key": mask(self.anthropic_api_key),
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
            "vector_store_path": str(self.vector_store_path),
            "vector_store_collection": self.vector_store_collection,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "retrieval_k": self.retrieval_k,
            "retrieval_score_threshold": self.retrieval_score_threshold,
            "knowledge_base_dir": str(self.knowledge_base_dir),
            "log_level": self.log_level,
            "log_dir": str(self.log_dir),
        }


def load_settings() -> Settings:
    # Build and validate the Settings object from environment variables.
    llm_provider = _require("LLM_PROVIDER").lower()

    openai_key = os.getenv("OPENAI_API_KEY") or None
    anthropic_key = os.getenv("ANTHROPIC_API_KEY") or None

    # Only demand the API key that's actually needed for the chosen provider.
    if llm_provider == "openai" and not openai_key:
        raise ConfigError("LLM_PROVIDER=openai requires OPENAI_API_KEY to be set.")
    if llm_provider == "anthropic" and not anthropic_key:
        raise ConfigError("LLM_PROVIDER=anthropic requires ANTHROPIC_API_KEY to be set.")

    settings = Settings(
        llm_provider=llm_provider,
        llm_model=_require("LLM_MODEL"),
        llm_temperature=_require_float("LLM_TEMPERATURE"),
        openai_api_key=openai_key,
        anthropic_api_key=anthropic_key,
        embedding_provider=_require("EMBEDDING_PROVIDER"),
        embedding_model=_require("EMBEDDING_MODEL"),
        vector_store_path=Path(_require("VECTOR_STORE_PATH")),
        vector_store_collection=_require("VECTOR_STORE_COLLECTION"),
        chunk_size=_require_int("CHUNK_SIZE"),
        chunk_overlap=_require_int("CHUNK_OVERLAP"),
        retrieval_k=_require_int("RETRIEVAL_K"),
        retrieval_score_threshold=_require_float("RETRIEVAL_SCORE_THRESHOLD"),
        knowledge_base_dir=Path(_require("KNOWLEDGE_BASE_DIR")),
        log_level=_optional("LOG_LEVEL", "INFO"),
        log_dir=Path(_optional("LOG_DIR", "logs")),
    )

    # Sanity check: overlap must be smaller than chunk size or splitting breaks.
    if settings.chunk_overlap >= settings.chunk_size:
        raise ConfigError(f"CHUNK_OVERLAP ({settings.chunk_overlap}) must be smaller than CHUNK_SIZE ({settings.chunk_size}).")

    return settings


# Module-level singleton — import `settings` everywhere else in the app.
try:
    settings = load_settings()
except ConfigError as exc:
    # Print ONE clean error and exit — no traceback noise for a config problem.
    sys.exit(f"[CONFIG ERROR] {exc}")


if __name__ == "__main__":
    # Smoke check: `python -m src.config`
    import json
    print("SupportPearlz — loaded configuration (secrets masked):")
    print(json.dumps(settings.masked_dict(), indent=2))