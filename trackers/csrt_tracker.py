from __future__ import annotations

import math
import time

import cv2

from core.bbox import BBox
from core.frame import Frame
from trackers.base import BaseTracker, TrackResult


class CSRTTracker(BaseTracker):
    """CSRT tracker via OpenCV module.

    CSRT does not produce a confidence score; 1.0 is used as a placeholder
    when tracking is active.
    """

    def __init__(self, cfg: dict) -> None:  # cfg accepted for interface uniformity
        self._tracker: cv2.TrackerCSRT | None = None

    def init(self, frame: Frame, bbox: BBox) -> None:
        """Initialise the tracker on *frame* with the given bounding box."""
        self._tracker = cv2.TrackerCSRT.create()
        self._tracker.init(frame.image, bbox.to_xywh())

    def update(self, frame: Frame) -> TrackResult:
        """Run one tracking step."""
        if self._tracker is None:
            return TrackResult(
                bbox=BBox(cx=0.0, cy=0.0, w=0.0, h=0.0),
                confidence=0.0,
                latency_s=0.0,
                source="csrt",
            )
        t0 = time.perf_counter()
        ok, cv2_bbox = self._tracker.update(frame.image)
        if ok:
            confidence = 1.0 
        else:
            confidence = 0.0
        return TrackResult(
            bbox=BBox.from_xywh(*cv2_bbox),
            confidence=confidence,
            latency_s=time.perf_counter() - t0,
            source="csrt",
        )
    
    def name(self) -> str:
        return "csrt"
