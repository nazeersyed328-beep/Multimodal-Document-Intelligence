"""Reusable console logging setup.

Log messages include a UTC-local timestamp, level, module name, and message.
This module never logs API keys, file contents, or extracted document values;
callers must pass only non-sensitive status information to the logger.
"""

from __future__ import annotations

import logging
import sys

DEFAULT_LOGGER_NAME = "multimodal_document_intelligence"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(
    level: str = "INFO",
    logger_name: str = DEFAULT_LOGGER_NAME,
) -> logging.Logger:
    """Configure and return a console logger without duplicating handlers."""
    logger = logging.getLogger(logger_name)
    level_name = str(level).upper()
    numeric_level = getattr(logging, level_name, None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Unsupported logging level: {level!r}.")

    logger.setLevel(numeric_level)
    logger.propagate = False

    if not any(getattr(handler, "_mdi_console", False) for handler in logger.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
        handler.setLevel(numeric_level)
        setattr(handler, "_mdi_console", True)
        logger.addHandler(handler)
    else:
        for handler in logger.handlers:
            if getattr(handler, "_mdi_console", False):
                handler.setLevel(numeric_level)

    return logger


def get_logger(module_name: str) -> logging.Logger:
    """Return a child logger named after the requesting module."""
    return logging.getLogger(f"{DEFAULT_LOGGER_NAME}.{module_name}")
