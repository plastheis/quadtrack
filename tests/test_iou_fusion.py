import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from core.bbox import BBox
from core.frame import Frame
from trackers.base import BaseTracker, TrackResult
from fusion_algs.iou_fusion import IoUFusion

_FRAME = Frame(image=np.zeros((480, 640, 3), dtype=np.uint8), timestamp=0.0)


def _r(cx, cy, w=60.0, h=60.0, conf=0.9, src="fast"):
    return TrackResult(BBox(cx, cy, w, h), conf, 0.0, src)


def _tracker(name, is_async=False, age=0):
    class T(BaseTracker):
        def init(self, f, b): pass
        def update(self, f): pass
        def name(self): return name
    t = T()
    t.is_async = is_async
    t.result_age = age
    return t


CFG = {
    "tracker": {"async_corr_thresh1": 0.7, "async_corr_thresh2": 0.3},
    "fusion_velocity_ema_alpha": 0.5,
}


def test_low_iou_uses_age_corrected_slow():
    """IoU < thresh2 → fuse returns age-corrected slow bbox."""
    fusion = IoUFusion()
    fast_t = _tracker("fast")
    slow_t = _tracker("slow", is_async=True, age=0)
    trackers = [fast_t, slow_t]
    # Fast at (100,100), slow at (400,400) → zero IoU → use slow
    results = [_r(100, 100, src="fast"), _r(400, 400, src="slow")]
    fused = fusion.fuse(CFG, trackers, results, _FRAME)
    assert abs(fused.bbox.cx - 400) < 1.0
    assert abs(fused.bbox.cy - 400) < 1.0


def test_velocity_ema_updates_each_frame():
    """Velocity EMA should shift after repeated fast-tracker movement."""
    fusion = IoUFusion()
    fast_t = _tracker("fast")
    slow_t = _tracker("slow", is_async=True, age=0)
    trackers = [fast_t, slow_t]
    # Move fast tracker 10px right each frame; slow is far away (low IoU path)
    for cx in [100, 110, 120, 130]:
        results = [_r(cx, 100, src="fast"), _r(500, 100, src="slow")]
        fusion.fuse(CFG, trackers, results, _FRAME)
    # After 4 frames at +10px/frame, velocity EMA should be positive
    assert fusion._v_cx > 0.0


def test_age_correct_shifts_slow_by_velocity():
    """With age > 0, slow bbox is shifted by age * velocity before blending."""
    fusion = IoUFusion()
    fast_t = _tracker("fast")
    slow_t = _tracker("slow", is_async=True, age=0)
    trackers = [fast_t, slow_t]
    # Prime velocity EMA: fast moves +10px/frame
    for cx in [100, 110, 120, 130]:
        results = [_r(cx, 100, src="fast"), _r(500, 100, src="slow")]
        fusion.fuse(CFG, trackers, results, _FRAME)
    # Now slow result is 3 frames stale, fast at (140,100), slow at (50,100)
    slow_t.result_age = 3
    results = [_r(140, 100, src="fast"), _r(50, 100, src="slow")]
    fused_stale = fusion.fuse(CFG, trackers, results, _FRAME)
    # With age=0 (no correction)
    fusion2 = IoUFusion()
    fusion2._v_cx = fusion._v_cx
    fusion2._prev_cx = 130.0
    fusion2._prev_cy = 100.0
    slow_t.result_age = 0
    results2 = [_r(140, 100, src="fast"), _r(50, 100, src="slow")]
    fused_fresh = fusion2.fuse(CFG, trackers, results2, _FRAME)
    # Age-corrected slow is shifted right → fused cx should be further right
    # (both are low-IoU paths so they return the slow bbox, corrected)
    assert fused_stale.bbox.cx > fused_fresh.bbox.cx
