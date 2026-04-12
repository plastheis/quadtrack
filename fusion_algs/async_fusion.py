from __future__ import annotations

import math
import time

import cv2

from core.bbox import BBox
from core.frame import Frame
from core.iou import bbox_iou
from trackers.base import BaseTracker, TrackResult
from fusion_algs.base import BaseFusionAlgorithm

class AsyncFusion(BaseFusionAlgorithm):
    def fuse(self, trackers: list[BaseTracker], results: list[TrackResult]) -> TrackResult:
        bbox_fast = BBox
        bbox_slow = BBox

        #for r in results:
            #if r.source

        fused = TrackResult
        return fused
