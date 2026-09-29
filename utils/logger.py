"""Standardised logging configuration."""
from __future__ import annotations

import logging
import sys

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_configured = False


def setup_logging(level: str | int = "INFO") -> None:
    """Configure the root logger once (idempotent apart from level updates)."""
    global _configured
    root = logging.getLogger()
    root.setLevel(level)

    if not _configured:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))
        root.addHandler(handler)
        # Quiet noisy third-party loggers.
        for noisy in ("httpx", "httpcore", "urllib3", "sentence_transformers"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
        _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger."""
    return logging.getLogger(name)
