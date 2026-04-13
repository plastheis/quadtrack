"""Base fusion algorithm interface and the no-op passthrough implementation."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.frame import Frame
    from trackers.base import TrackResult, BaseTracker


class BaseFusionAlgorithm(ABC):
    """Common interface for all tracker-fusion algorithms."""

    @abstractmethod
    def fuse(self, cfg: dict, trackers: list[BaseTracker], results: list[TrackResult], frame: Frame) -> TrackResult:
        """Combine multiple tracker outputs into one result."""


class PassthroughFusion(BaseFusionAlgorithm):
    """Identity fusion — used when exactly one tracker is active."""

    def fuse(self, cfg: dict, trackers: list[BaseTracker], results: list[TrackResult], frame: Frame) -> TrackResult:
        if len(results) != 1:
            raise ValueError(
                f"PassthroughFusion expects exactly 1 result, got {len(results)}."
            )
        return results[0]
