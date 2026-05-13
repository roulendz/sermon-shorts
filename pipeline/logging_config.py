"""
pipeline/logging_config.py

Centralized file logging configuration for the entire pipeline.
Call configure_file_logging() once at app startup (main.py).
All modules use standard logging.getLogger(__name__) — this configures
the root logger to write everything to a rotating log file.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_DIRECTORY = Path("logs")
LOG_FILENAME = "sermon_shorts.log"
MAX_LOG_FILE_BYTES = 10 * 1024 * 1024  # 10 MB
BACKUP_COUNT = 3
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_file_logging(log_level: int = logging.DEBUG) -> Path:
    """
    Set up rotating file handler on the root logger.
    Returns the path to the active log file.
    """
    LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)
    log_file_path = LOG_DIRECTORY / LOG_FILENAME

    file_handler = RotatingFileHandler(
        log_file_path,
        maxBytes=MAX_LOG_FILE_BYTES,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(file_handler)

    logging.getLogger(__name__).info(
        "File logging initialized: %s (level=%s)",
        log_file_path,
        logging.getLevelName(log_level),
    )

    return log_file_path
