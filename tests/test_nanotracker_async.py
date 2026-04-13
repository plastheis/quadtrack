import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from concurrent.futures import ThreadPoolExecutor
from core.bbox import BBox
from trackers.base import TrackResult
from trackers.nanotrack_tracker import NanoTracker


def _make_frame():
    class F:
        image = np.zeros((480, 640, 3), dtype=np.uint8)
    return F()


def _make_tracker():
    """NanoTracker with __init__ bypassed; _run_inference mocked."""
    t = object.__new__(NanoTracker)
    t.is_async              = True
    t.result_age            = 0
    t.async_submit_strategy = "on_completion"
    t.async_min_interval    = 1
    t._cx = 100.0; t._cy = 100.0; t._w = 50.0; t._h = 50.0
    t._executor        = ThreadPoolExecutor(max_workers=1)
    t._pending_future  = None
    t._cached_result   = None
    t._frame_count     = 0

    # Backbone/head attributes needed by init() template extraction
    t._use_rknn = False
    t._t_lo     = 4
    t._t_hi     = 12
    t._t_feat   = np.zeros((1, 96, 8, 8), dtype=np.float32)

    def _mock_backbone(tensor):
        return np.zeros((1, 96, 16, 16), dtype=np.float32)

    t._run_backbone = _mock_backbone

    def _mock_inference(frame, cx, cy, w, h):
        time.sleep(0.02)
        return TrackResult(BBox(cx + 1, cy, w, h), 0.9, 0.02, "nanotrack"), cx + 1, cy, w, h

    t._run_inference = _mock_inference
    return t


def test_first_update_returns_result():
    t = _make_tracker()
    r = t.update(_make_frame())
    assert r is not None
    assert r.source == "nanotrack"


def test_second_update_is_non_blocking():
    t = _make_tracker()
    t.update(_make_frame())
    t0 = time.perf_counter()
    t.update(_make_frame())
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.005, f"update() blocked for {elapsed:.3f}s — expected < 5ms"


def test_result_age_increments_while_inference_running():
    t = _make_tracker()
    t.update(_make_frame())
    t.update(_make_frame())
    assert t.result_age >= 1


def test_result_age_resets_when_fresh_result_arrives():
    t = _make_tracker()
    t.update(_make_frame())
    time.sleep(0.05)
    t.update(_make_frame())
    assert t.result_age == 0


def test_init_resets_async_state():
    t = _make_tracker()
    t.update(_make_frame())
    t.result_age = 5
    t.init(_make_frame(), BBox(200, 200, 60, 60))
    assert t.result_age == 0
    assert t._cached_result is None
