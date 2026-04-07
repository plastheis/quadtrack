from __future__ import annotations

import cv2
import numpy as np


class KCFTracker:
    """KCF tracker via OpenCV's legacy contrib module.

    Requires opencv-contrib-python (drop-in replacement for opencv-python).
    KCF lives under cv2.legacy in OpenCV 4.5+.
    """

    def __init__(self) -> None:
        self._tracker: cv2.legacy.Tracker | None = None

    def init(self, frame: np.ndarray, bbox: tuple) -> None:
        """Initialise the tracker on *frame* with the given bounding box.

        Args:
            frame: BGR image (np.ndarray).
            bbox:  (x, y, w, h) in pixels.
        """
        self._tracker = cv2.legacy.TrackerKCF_create()
        self._tracker.init(frame, bbox)

    def update(self, frame: np.ndarray) -> tuple[tuple, bool]:
        """Run one tracking step.

        Returns:
            (bbox, ok) where bbox is (x, y, w, h) and ok is False if the
            tracker lost the target.
        """
        if self._tracker is None:
            return (0, 0, 0, 0), False
        ok, bbox = self._tracker.update(frame)
        return tuple(int(v) for v in bbox), bool(ok)
