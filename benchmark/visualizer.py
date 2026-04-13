"""Live visualisation window for benchmark playback."""
from __future__ import annotations

import cv2
import numpy as np

from benchmark.datasets.base import GroundTruthFrame
from trackers.base import TrackResult


# BGR colours
_GREEN  = (0,   200,  0)    # ground-truth box
_ORANGE = (0,   140, 255)   # tracker prediction box
_WHITE  = (255, 255, 255)
_SHADOW = (30,  30,  30)

_FONT = cv2.FONT_HERSHEY_SIMPLEX

# Reference resolution that the base constants were tuned for.
_REF_H = 480


class BenchmarkVisualizer:
    """Manages the OpenCV playback window for benchmark runs.

    The frame is resized to the configured display resolution before any
    drawing occurs, so all sizes (font, thickness, padding) are expressed
    in display pixels and scale automatically with the window resolution.

    Call ``show()`` each frame; it returns ``(skip_sequence, quit)``.

    Key bindings
    ------------
    q / Esc   — quit the entire benchmark
    n / Space — skip to the next sequence
    """

    _WIN = "Benchmark Viewer"

    def __init__(self, width: int, height: int) -> None:
        self._w = width
        self._h = height

        # Scale all UI constants relative to the configured display height.
        s = height / _REF_H
        self._font_scale  = round(0.50 * s, 2)
        self._thickness   = max(1, round(s))
        self._box_thick   = max(1, round(2 * s))
        self._line_h      = max(1, int(22 * s))
        self._pad         = max(4, int(8 * s))

        cv2.namedWindow(self._WIN, cv2.WINDOW_AUTOSIZE)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def show(
        self,
        frame_img:    np.ndarray,
        gt:           GroundTruthFrame,
        result:       TrackResult,
        seq_name:     str,
        frame_idx:    int,
        n_frames:     int,
        running_iou:  float,
        iou:          float,
        fps:          float = 60.0,
    ) -> tuple[bool, bool]:
        """Render the frame and return (skip_sequence, quit)."""
        src_h, src_w = frame_img.shape[:2]

        # Resize to display resolution — all drawing is in display pixels.
        vis = cv2.resize(frame_img, (self._w, self._h), interpolation=cv2.INTER_LINEAR)

        # Coordinate scale factors from source frame → display frame.
        sx = self._w / src_w
        sy = self._h / src_h

        # Draw ground-truth box (green)
        if gt.exists and gt.bbox is not None:
            x1, y1, x2, y2 = gt.bbox.to_xyxy()
            cv2.rectangle(
                vis,
                (int(x1 * sx), int(y1 * sy)),
                (int(x2 * sx), int(y2 * sy)),
                _GREEN, self._box_thick,
            )

        # Draw tracker prediction box (orange)
        if result.bbox is not None:
            x1, y1, x2, y2 = result.bbox.to_xyxy()
            cv2.rectangle(
                vis,
                (int(x1 * sx), int(y1 * sy)),
                (int(x2 * sx), int(y2 * sy)),
                _ORANGE, self._box_thick,
            )

        # Overlay stats — top-left corner
        lines = [
            f"Seq : {seq_name}  ({frame_idx}/{n_frames})",
            f"Tracker : {result.source}",
            f"Conf    : {result.confidence:.3f}",
            f"IoU     : {iou:.3f}",
            f"Avg IoU : {running_iou:.3f}",
            f"Latency : {result.latency_s * 1000:.1f} ms",
            "",
            "n/Space = next seq   q/Esc = quit",
        ]
        for i, line in enumerate(lines):
            x = self._pad
            y = self._pad + self._line_h * (i + 1)
            cv2.putText(vis, line, (x + 1, y + 1), _FONT, self._font_scale,
                        _SHADOW, self._thickness + 1, cv2.LINE_AA)
            cv2.putText(vis, line, (x, y),          _FONT, self._font_scale,
                        _WHITE,  self._thickness,     cv2.LINE_AA)

        cv2.imshow(self._WIN, vis)

        delay_ms = max(1, int(1000 / fps))
        key = cv2.waitKeyEx(delay_ms) & 0xFFFF
        skip  = key in (ord("n"), ord(" "), 0xFF53, 0x27)  # n, space, right-arrow
        quit_ = key in (ord("q"), 27)                       # q, Esc
        return skip, quit_

    def close(self) -> None:
        cv2.destroyWindow(self._WIN)
