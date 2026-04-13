import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from core.bbox import BBox
from trackers.base import BaseTracker, TrackResult
from fusion_algs.ioukf_fusion import IoUKFFusion


def _r(cx, cy, w=60.0, h=60.0, conf=0.9, src="fast"):
    return TrackResult(BBox(cx, cy, w, h), conf, 0.0, src)


def _tracker(name, is_async=False, age=0):
    class T(BaseTracker):
        def init(self, f, b): pass
        def update(self, f): pass
        def name(self): return name
    t = T()
    t.is_async   = is_async
    t.result_age = age
    return t


CFG = {
    "tracker": {"async_corr_thresh2": 0.3},
    "kf_process_noise": 0.01,
    "kf_fast_meas_noise": 5.0,
    "kf_slow_meas_noise": 1.0,
    "kf_age_noise_scale": 0.5,
}


def test_predict_increases_uncertainty():
    fusion = IoUKFFusion()
    fusion._x = np.array([100.0, 100.0, 5.0, 0.0])
    fusion._P = np.eye(4) * 1.0
    p_before = fusion._P[0, 0]
    fusion._predict(np.eye(4) * 0.01)
    assert fusion._P[0, 0] > p_before


def test_update_reduces_uncertainty():
    fusion = IoUKFFusion()
    fusion._x = np.array([100.0, 100.0, 0.0, 0.0])
    fusion._P = np.eye(4) * 100.0
    p_before = fusion._P[0, 0]
    fusion._update(np.array([100.0, 100.0]), np.eye(2) * 5.0)
    assert fusion._P[0, 0] < p_before


def test_update_moves_state_toward_measurement():
    fusion = IoUKFFusion()
    fusion._x = np.array([100.0, 100.0, 0.0, 0.0])
    fusion._P = np.eye(4) * 100.0
    fusion._update(np.array([150.0, 100.0]), np.eye(2) * 5.0)
    assert fusion._x[0] > 100.0


def test_fuse_initialises_from_fast_tracker():
    fusion = IoUKFFusion()
    fast_t = _tracker("fast")
    slow_t = _tracker("slow", is_async=True, age=1)
    results = [_r(200, 300, src="fast"), _r(0, 0, src="slow")]
    fused = fusion.fuse(CFG, [fast_t, slow_t], results)
    assert abs(fused.bbox.cx - 200) < 5.0
    assert abs(fused.bbox.cy - 300) < 5.0


def test_fuse_applies_slow_update_when_age_zero():
    """When slow result_age==0 and IoU passes gate, slow measurement should pull state."""
    fusion = IoUKFFusion()
    fast_t = _tracker("fast")
    slow_t = _tracker("slow", is_async=True, age=1)
    for _ in range(5):
        fusion.fuse(CFG, [fast_t, slow_t], [_r(100, 100, src="fast"), _r(100, 100, src="slow")])
    slow_t.result_age = 0
    results = [_r(100, 100, src="fast"), _r(130, 100, src="slow")]
    fused = fusion.fuse(CFG, [fast_t, slow_t], results)
    assert fused.bbox.cx > 100.0


def test_fast_tracker_confidence_propagated():
    fusion = IoUKFFusion()
    fast_t = _tracker("fast")
    slow_t = _tracker("slow", is_async=True, age=1)
    results = [_r(100, 100, conf=0.0, src="fast"), _r(100, 100, src="slow")]
    fused = fusion.fuse(CFG, [fast_t, slow_t], results)
    assert fused.confidence == 0.0
