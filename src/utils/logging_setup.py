# Centralized logging setup — call configure_logging() once at the start of any entry point.

from __future__ import annotations
import logging
import sys
from src.config import settings

_CONFIGURED = False  # guards against double-configuring handlers

def configure_logging() -> None:
    # Idempotent: safe to call multiple times, only configures once.
    global _CONFIGURED
    if _CONFIGURED:
        return

    settings.log_dir.mkdir(parents=True, exist_ok=True)
    log_file = settings.log_dir / "supportpearlz.log"
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)  # logs to terminal
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")  # logs to file
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    _CONFIGURED = True
    logging.getLogger(__name__).info("Logging configured (level=%s, file=%s)", settings.log_level, log_file)