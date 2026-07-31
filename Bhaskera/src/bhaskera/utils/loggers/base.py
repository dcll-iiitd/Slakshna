"""Base class for experiment trackers."""
from __future__ import annotations

from typing import Any


class BaseLogger:
    """
    Minimal logger interface.

    Subclasses implement ``log()`` (publish a flat metrics dict at the
    given training step) and ``finish()`` (release resources, flush
    buffers).  Both are best-effort: a logger that loses connectivity
    must never raise into the training loop.
    """

    def log(self, metrics: dict[str, Any], step: int) -> None:  # pragma: no cover
        ...

    def finish(self) -> None:  # pragma: no cover
        ...
