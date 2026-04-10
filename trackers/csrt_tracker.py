from __future__ import annotations

import time

import cv2

from core.bbox import BBox
from core.frame import Frame
from trackers.base import BaseTracker, TrackResult


class CSRTTracker(BaseTracker):
    """CSRT tracker via OpenCV's legacy contrib module.

    Requires opencv-contrib-python (drop-in replacement for opencv-python).
    CSRT lives under cv2.legacy in OpenCV 4.5+.

    Unlike KCF, CSRT exposes a real confidence score through getTrackingScore(),
    which returns the PSR (Peak-to-Sidelobe Ratio) of the response map.
    The raw PSR is normalised to [0, 1] via a sigmoid so downstream fusion
    receives a consistent scale.
    """

    _PSR_SCALE = 20.0  # sigmoid steepness; PSR ~10–30 maps to ~0.5–0.98

    def __init__(self, cfg: dict) -> None:  # cfg accepted for interface uniformity
        self._tracker: cv2.legacy.TrackerCSRT | None = None

    def init(self, frame: Frame, bbox: BBox) -> None:
        """Initialise the tracker on *frame* with the given bounding box."""
        self._tracker = cv2.legacy.TrackerCSRT_create()
        self._tracker.init(frame.image, bbox.to_xywh())

    def update(self, frame: Frame) -> TrackResult:
        """Run one tracking step."""
        t0 = time.perf_counter()
        if self._tracker is None:
            return TrackResult(
                bbox=BBox(cx=0.0, cy=0.0, w=0.0, h=0.0),
                confidence=0.0,
                latency_s=0.0,
                source="csrt",
            )
        ok, cv2_bbox = self._tracker.update(frame.image)
        if ok:
            import math
            psr = self._tracker.getTrackingScore()
            confidence = 1.0 / (1.0 + math.exp(-psr / self._PSR_SCALE))
        else:
            confidence = 0.0
        return TrackResult(
            bbox=BBox.from_xywh(*cv2_bbox),
            confidence=confidence,
            latency_s=time.perf_counter() - t0,
            source="csrt",
        )
