"""Canonical bounding-box type and format conversions.

Internally all bounding boxes are stored as (cx, cy, w, h) floats.
Convert to integer pixel coordinates only when calling OpenCV.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BBox:
    """Bounding box in centre-size format (floats)."""

    cx: float   # centre x
    cy: float   # centre y
    w:  float   # width
    h:  float   # height

    # ------------------------------------------------------------------
    # Constructors from other formats
    # ------------------------------------------------------------------

    @classmethod
    def from_xywh(cls, x: float, y: float, w: float, h: float) -> BBox:
        """Create from top-left origin + size."""
        return cls(cx=x + w * 0.5, cy=y + h * 0.5, w=float(w), h=float(h))

    @classmethod
    def from_xyxy(cls, x1: float, y1: float, x2: float, y2: float) -> BBox:
        """Create from corner coordinates."""
        w = x2 - x1
        h = y2 - y1
        return cls(cx=x1 + w * 0.5, cy=y1 + h * 0.5, w=float(w), h=float(h))

    # ------------------------------------------------------------------
    # Conversions to OpenCV formats (integer pixel coordinates)
    # ------------------------------------------------------------------

    def to_xywh(self) -> tuple[int, int, int, int]:
        """Integer (x, y, w, h) — for cv2.rectangle / tracker init."""
        return (
            int(self.cx - self.w * 0.5),
            int(self.cy - self.h * 0.5),
            int(self.w),
            int(self.h),
        )

    def to_xyxy(self) -> tuple[int, int, int, int]:
        """Integer (x1, y1, x2, y2) — for cv2.rectangle (two-point form)."""
        x = int(self.cx - self.w * 0.5)
        y = int(self.cy - self.h * 0.5)
        return (x, y, x + int(self.w), y + int(self.h))
