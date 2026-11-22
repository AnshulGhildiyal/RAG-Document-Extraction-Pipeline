"""Centralized logging configuration for the RAG pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


_CONFIGURED = False


def configure_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Configure application logging once.

    Args:
        log_level: Logging level name (e.g. INFO, DEBUG).
        log_file: Optional path to a log file.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    level = getattr(logging, str(log_level).upper(), logging.INFO)
    handlers: list[logging.Handler] = [logging.StreamHandler()]

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
    )

    # Reduce noisy third-party logs by default.
    logging.getLogger("chromadb").setLevel(max(level, logging.WARNING))
    logging.getLogger("sentence_transformers").setLevel(max(level, logging.WARNING))
    logging.getLogger("httpx").setLevel(max(level, logging.WARNING))

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module logger, configuring defaults if needed."""
    if not _CONFIGURED:
        configure_logging()
    return logging.getLogger(name)
