"""Centroid utilities shared across tracking and flight-interface modules."""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.bbox import BBox


class Centroid:
    """Static centroid helpers.

    Current implementation derives centroid directly from a BBox.
    The flight-interface module will extend this with image-moments-based
    centroid computation once the control loop is implemented.
    """

    @staticmethod
    def from_bbox(bbox: BBox) -> tuple[float, float]:
        """Return the (cx, cy) centre of a bounding box."""
        return bbox.cx, bbox.cy

    @staticmethod
    def distance(c1: tuple[float, float], c2: tuple[float, float]) -> float:
        """Euclidean pixel distance between two (x, y) centroids."""
        return math.hypot(c1[0] - c2[0], c1[1] - c2[1])
