"""Base tracker interface and standardised result type."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.bbox import BBox
    from core.frame import Frame


@dataclass(frozen=True)
class TrackResult:
    """Standardised output returned by every tracker."""

    bbox:       BBox   # target position in canonical (cx, cy, w, h) format
    confidence: float  # tracker score in [0, 1]; 0.0 means target lost
    latency_s:  float  # wall-clock inference time (seconds)
    source:     str    # tracker identifier, e.g. "kcf", "nanotrack_accel"


class BaseTracker(ABC):
    """Common interface that all tracker implementations must satisfy."""

    @abstractmethod
    def init(self, frame: Frame, bbox: BBox) -> None:
        """Initialise the tracker on the first frame."""

    @abstractmethod
    def update(self, frame: Frame) -> TrackResult:
        """Advance the tracker by one frame and return the result."""
