from __future__ import annotations

import cv2
import numpy as np


class NanoTracker:
    """NanoTrack tracker via OpenCV's built-in cv2.TrackerNano.

    Requires two v2 ONNX model files (v3 has a shape mismatch in OpenCV):
      - models/nanotrack/nanotrack_backbone_sim.onnx
      - models/nanotrack/nanotrack_head_sim.onnx

    Paths are set in config.yaml under tracker.nanotrack_backbone and
    tracker.nanotrack_head.
    """

    def __init__(self, cfg: dict) -> None:
        params = cv2.TrackerNano_Params()
        params.backbone = cfg["tracker"]["nanotrack_backbone"]
        params.neckhead  = cfg["tracker"]["nanotrack_head"]
        self._tracker = cv2.TrackerNano_create(params)

    def init(self, frame: np.ndarray, bbox: tuple) -> None:
        """Initialise tracker with the first-frame bounding box.

        Args:
            frame: BGR image.
            bbox:  (x, y, w, h) in pixels.
        """
        self._tracker.init(frame, bbox)

    def update(self, frame: np.ndarray) -> tuple[tuple, bool]:
        """Run one tracking step.

        Returns:
            (bbox, ok) where bbox is (x, y, w, h) and ok is False if lost.
        """
        ok, bbox = self._tracker.update(frame)
        return tuple(int(v) for v in bbox), bool(ok)
