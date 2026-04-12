from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from core.bbox import BBox

def bbox_iou(b1: BBox, b2: BBox) -> float:
    """Intersection-over-Union between two (cx, cy, w, h) bounding boxes."""
    b1x1 = b1.cx - b1.w / 2;  b1x2 = b1.cx + b1.w / 2
    b1y1 = b1.cy - b1.h / 2;  b1y2 = b1.cy + b1.h / 2
    b2x1 = b2.cx   - b2.w   / 2;  b2x2 = b2.cx   + b2.w   / 2
    b2y1 = b2.cy   - b2.h   / 2;  b2y2 = b2.cy   + b2.h   / 2

    inter_w = max(0.0, min(b1x2, b2x2) - max(b1x1, b2x1))
    inter_h = max(0.0, min(b1y2, b2y2) - max(b1y1, b2y1))
    inter   = inter_w * inter_h

    union = b1.w * b1.h + b2.w * b2.h - inter
    return inter / union if union > 0.0 else 0.0
