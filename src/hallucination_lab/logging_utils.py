"""Logging setup.

Library modules only call ``logging.getLogger(__name__)``; entry points (CLI,
UI) call :func:`configure_logging` once.
"""

from __future__ import annotations

import logging
import os

LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"


def configure_logging(level: str | int | None = None) -> None:
    """Configure root logging. ``HALLAB_LOG_LEVEL`` overrides the default."""
    resolved = level or os.environ.get("HALLAB_LOG_LEVEL", "WARNING")
    if isinstance(resolved, str):
        resolved = resolved.upper()
    logging.basicConfig(level=resolved, format=LOG_FORMAT, force=True)
