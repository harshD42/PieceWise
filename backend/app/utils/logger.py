# Copyright (c) 2026 Harsh Dwivedi
# Licensed under the Harsh Non-Commercial Attribution License (HNCAL) v1.0
# Commercial use requires written permission. See LICENSE for details.

"""
PieceWise — Structured Logging
JSON-formatted logs via structlog. Every log entry carries a stage label
and optional job_id for full request traceability.
"""

import logging
import sys
from typing import Any

import structlog
from structlog.types import EventDict, Processor

from app.config import get_settings


def _add_app_info(
    logger: Any, method: str, event_dict: EventDict
) -> EventDict:
    """Inject application name into every log entry."""
    event_dict["app"] = "piecewise"
    return event_dict


def _drop_color_message_key(
    logger: Any, method: str, event_dict: EventDict
) -> EventDict:
    """Remove uvicorn's color_message to keep logs clean."""
    event_dict.pop("color_message", None)
    return event_dict


def configure_logging() -> None:
    """
    Configure structlog for JSON output in production and
    human-readable console output in development (DEBUG level).
    Called once at application startup.
    """
    settings = get_settings()
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        _add_app_info,
        _drop_color_message_key,
    ]

    if settings.log_level == "DEBUG":
        # Pretty console output for local development
        processors: list[Processor] = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True)
        ]
    else:
        # JSON output for production / CI
        processors = shared_processors + [
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(sys.stdout),
        cache_logger_on_first_use=True,
    )

    # Configure stdlib logging level (for uvicorn/fastapi passthrough)
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )


def get_logger(name: str = "piecewise") -> structlog.BoundLogger:
    """
    Return a structlog bound logger.

    Usage:
        log = get_logger(__name__)
        log.info("stage_complete", stage="segmentation", piece_count=42)

    To bind job_id for a full request context:
        structlog.contextvars.bind_contextvars(job_id=job_id)
        log.info("pipeline_start")
        structlog.contextvars.clear_contextvars()
    """
    return structlog.get_logger(name)