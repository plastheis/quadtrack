"""Frame container: raw camera image paired with its capture timestamp."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Frame:
    """A single camera frame with an associated capture timestamp."""

    image:     np.ndarray  # BGR uint8, shape (H, W, 3)
    timestamp: float       # seconds, from time.perf_counter()
