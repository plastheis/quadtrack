from __future__ import annotations

import time

import cv2

from core.bbox import BBox
from core.frame import Frame
from trackers.base import BaseTracker, TrackResult


class NanoTracker(BaseTracker):
    """NanoTrack tracker via OpenCV's built-in cv2.TrackerNano.

    Requires two v2 ONNX model files (v3 has a shape mismatch in OpenCV):
      - models/nanotrack/nanotrack_backbone_sim.onnx
      - models/nanotrack/nanotrack_head_sim.onnx

    Paths are set in config.yaml under tracker.nanotrack_backbone and
    tracker.nanotrack_head.
    cv2.TrackerNano does not expose a confidence score; 1.0 is used as a
    placeholder when tracking is active.
    """

    def __init__(self, cfg: dict) -> None:
        params = cv2.TrackerNano_Params()
        params.backbone = cfg["tracker"]["nanotrack_backbone"]
        params.neckhead  = cfg["tracker"]["nanotrack_head"]
        self._tracker = cv2.TrackerNano_create(params)

    def init(self, frame: Frame, bbox: BBox) -> None:
        """Initialise tracker with the first-frame bounding box."""
        self._tracker.init(frame.image, bbox.to_xywh())

    def update(self, frame: Frame) -> TrackResult:
        """Run one tracking step."""
        t0 = time.perf_counter()
        ok, cv2_bbox = self._tracker.update(frame.image)
        return TrackResult(
            bbox=BBox.from_xywh(*cv2_bbox),
            confidence=1.0 if ok else 0.0,
            latency_s=time.perf_counter() - t0,
            source="nanotrack",
        )
