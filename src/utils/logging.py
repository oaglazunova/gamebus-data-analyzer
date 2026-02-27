"""
Logging utilities for the GameBus-HealthBehaviorMining project.
"""
from __future__ import annotations

import logging
import os
import sys
import warnings

from config.settings import LOG_LEVEL, LOG_FORMAT
from config.paths import PROJECT_ROOT


CONSOLE_FORMAT = "%(message)s"
NOISY_LOGGERS = (
    "urllib3",
    "requests",
    "matplotlib",
    "PIL",
    "fontTools",
    "openpyxl",
)


class ProgressConsoleFilter(logging.Filter):
    """
    Only show explicitly marked user-facing messages on the console.
    Everything else still goes to the file log.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return bool(getattr(record, "console_print", False))


def console_info(logger: logging.Logger, message: str) -> None:
    logger.info(message, extra={"console_print": True})


def console_warning(logger: logging.Logger, message: str) -> None:
    logger.warning(message, extra={"console_print": True})


def console_error(logger: logging.Logger, message: str) -> None:
    logger.error(message, extra={"console_print": True})


def setup_logging(
    log_to_file: bool = True,
    log_level: str | None = None,
    log_type: str = "extraction",
) -> logging.Logger:
    """
        Configure root logging:
        - file handler = full logs
        - console handler = explicit user-facing messages only
    """

    if log_level is None:
        log_level = LOG_LEVEL

    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Avoid duplicated handlers if reconfigured (pipeline -> analysis)
    if root_logger.hasHandlers():
        for handler in root_logger.handlers[:]:
            handler.close()
            root_logger.removeHandler(handler)

    # Route Python warnings into logging so they go to file, not raw stderr
    logging.captureWarnings(True)
    warnings.simplefilter("default")

    # Quiet noisy third-party loggers
    for noisy_name in NOISY_LOGGERS:
        logging.getLogger(noisy_name).setLevel(logging.ERROR)

    # Console: explicit user-facing messages only
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.addFilter(ProgressConsoleFilter())
    console_handler.setFormatter(logging.Formatter(CONSOLE_FORMAT))
    root_logger.addHandler(console_handler)

    # File: full detail
    if log_to_file:
        log_dir = os.path.join(PROJECT_ROOT, "logs")
        os.makedirs(log_dir, exist_ok=True)

        log_file = (
            os.path.join(log_dir, "data_analysis.log")
            if log_type == "analysis"
            else os.path.join(log_dir, "data_extraction.log")
        )

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        root_logger.addHandler(file_handler)

    # Return a named app logger that propagates to root
    app_logger = logging.getLogger(f"gamebus.{log_type}")
    app_logger.setLevel(numeric_level)
    return app_logger