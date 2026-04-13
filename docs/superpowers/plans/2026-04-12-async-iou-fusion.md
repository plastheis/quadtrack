# Non-Blocking Async Tracker & Improved IoU Fusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make NanoTracker run inference in a background thread so it never blocks the fast tracker, replace the jittery linear blend in `iou_fusion.py` with velocity-EMA dead-reckoning, and add a new `ioukf_fusion.py` using a constant-velocity Kalman filter.

**Architecture:** `BaseTracker` gains a `result_age` property (0 = fresh this frame, N = cached N frames). `NanoTracker.update()` becomes non-blocking via an internal `ThreadPoolExecutor`; inference runs in a background thread and the latest cached result is returned immediately. The `fuse()` interface drops the external `nframe` arg in favour of reading `t.result_age` directly. `iou_fusion.py` age-corrects the stale slow bbox using a fast-tracker velocity EMA before IoU gating. `ioukf_fusion.py` tracks `[cx, cy, vx, vy]` with a Kalman filter, using fast-tracker measurements every frame and slow-tracker measurements when a fresh result arrives.

**Tech Stack:** Python, NumPy, `concurrent.futures.ThreadPoolExecutor`, ONNX Runtime (existing), pytest

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `trackers/base.py` | Modify | Add `result_age`, `async_submit_strategy`, `async_min_interval` class attrs |
| `fusion_algs/base.py` | Modify | Remove `nframe` from `fuse()` signature |
| `fusion_algs/iou_fusion.py` | Rewrite | Dead-reckoning velocity EMA; replace `smooth_correction` |
| `fusion_algs/ioukf_fusion.py` | Create | Kalman filter fusion (new file) |
| `trackers/nanotrack_tracker.py` | Modify | Non-blocking `update()` via `ThreadPoolExecutor` |
| `trackers/factory.py` | Modify | Set scheduling attrs from spec; add `"ioukf"` to `_FUSION_MAP` |
| `testing/main.py` | Modify | Remove `nframe`/`rslow`/`interval`; call all trackers uniformly |
| `benchmark/runner.py` | Modify | Fix stale `fuse(results)` call → `fuse(cfg, trackers, results)` |
| `config.yaml` | Modify | Remove `async_interval`; add scheduling + KF keys |
| `tests/__init__.py` | Create | Empty |
| `tests/test_base.py` | Create | `result_age` default |
| `tests/test_iou_fusion.py` | Create | Age correction + velocity EMA |
| `tests/test_ioukf_fusion.py` | Create | Kalman predict/update math |
| `tests/test_nanotracker_async.py` | Create | Non-blocking behaviour |

---

### Task 1: Add `result_age` and scheduling attrs to `BaseTracker`; update `factory.py`

**Files:**
- Modify: `trackers/base.py`
- Modify: `trackers/factory.py`
- Create: `tests/__init__.py`, `tests/test_base.py`

- [ ] **Step 1: Write the failing test**

Create `tests/__init__.py` (empty) and `tests/test_base.py`:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.bbox import BBox
from trackers.base import BaseTracker, TrackResult


class _Stub(BaseTracker):
    def init(self, frame, bbox): pass
    def update(self, frame): return TrackResult(BBox(0,0,1,1), 1.0, 0.0, "stub")
    def name(self): return "stub"


def test_result_age_default_zero():
    assert _Stub().result_age == 0


def test_is_async_default_false():
    assert _Stub().is_async is False


def test_async_submit_strategy_default():
    assert _Stub().async_submit_strategy == "on_completion"


def test_async_min_interval_default():
    assert _Stub().async_min_interval == 1
```

- [ ] **Step 2: Run — expect failure**

```
pytest tests/test_base.py -v
```
Expected: `AttributeError` — `result_age` not defined yet.

- [ ] **Step 3: Update `trackers/base.py`**

```python
"""Base tracker interface and standardised result type."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.bbox import BBox
    from core.frame import Frame


@dataclass(frozen=True)
class TrackResult:
    """Standardised output returned by every tracker."""

    bbox:       BBox
    confidence: float
    latency_s:  float
    source:     str


class BaseTracker(ABC):
    """Common interface that all tracker implementations must satisfy."""

    is_async: bool = False
    result_age: int = 0          # frames since last fresh result; 0 = fresh this frame
    async_submit_strategy: str = "on_completion"  # "on_completion" | "fixed_interval"
    async_min_interval: int = 1  # min frames between submissions (fixed_interval only)

    @abstractmethod
    def init(self, frame: Frame, bbox: BBox) -> None:
        """Initialise the tracker on the first frame."""

    @abstractmethod
    def update(self, frame: Frame) -> TrackResult:
        """Advance the tracker by one frame and return the result."""

    @abstractmethod
    def name(self) -> str:
        """Return the tracker name."""
```

- [ ] **Step 4: Update `trackers/factory.py` — set scheduling attrs from spec**

After `t.is_async = spec.get("async", False)` add:

```python
t.async_submit_strategy = spec.get("async_submit_strategy", "on_completion")
t.async_min_interval    = spec.get("async_min_interval", 1)
```

Full updated `build_trackers` loop:

```python
for spec in tracker_cfg["algorithms"]:
    algo = spec["algorithm"].strip().lower()
    if algo not in _ALGO_MAP:
        raise ValueError(
            f"Unknown tracker algorithm: {algo!r}. "
            f"Valid options: {list(_ALGO_MAP)}"
        )
    t = _ALGO_MAP[algo](cfg)
    t.is_async               = spec.get("async", False)
    t.async_submit_strategy  = spec.get("async_submit_strategy", "on_completion")
    t.async_min_interval     = spec.get("async_min_interval", 1)
    trackers.append(t)
```

- [ ] **Step 5: Run — expect pass**

```
pytest tests/test_base.py -v
```
Expected: 4 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add trackers/base.py trackers/factory.py tests/__init__.py tests/test_base.py
git commit -m "feat: add result_age and async scheduling attrs to BaseTracker"
```

---

### Task 2: Remove `nframe` from fusion interface; rewrite `iou_fusion.py`; fix all call sites

**Note:** `benchmark/runner.py:128` currently calls `fuse(results)` — wrong signature even before this task. Both are fixed here atomically.

**Files:**
- Modify: `fusion_algs/base.py`
- Rewrite: `fusion_algs/iou_fusion.py`
- Modify: `testing/main.py`
- Modify: `benchmark/runner.py`
- Create: `tests/test_iou_fusion.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_iou_fusion.py`:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.bbox import BBox
from trackers.base import BaseTracker, TrackResult
from fusion_algs.iou_fusion import IoUFusion


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
    fused = fusion.fuse(CFG, trackers, results)
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
        fusion.fuse(CFG, trackers, results)
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
        fusion.fuse(CFG, trackers, results)
    # Now slow result is 3 frames stale, fast at (140,100), slow at (50,100)
    slow_t.result_age = 3
    results = [_r(140, 100, src="fast"), _r(50, 100, src="slow")]
    fused_stale = fusion.fuse(CFG, trackers, results)
    # With age=0 (no correction)
    fusion2 = IoUFusion()
    fusion2._v_cx = fusion._v_cx
    fusion2._prev_cx = 130.0
    fusion2._prev_cy = 100.0
    slow_t.result_age = 0
    results2 = [_r(140, 100, src="fast"), _r(50, 100, src="slow")]
    fused_fresh = fusion2.fuse(CFG, trackers, results2)
    # Age-corrected slow is shifted right → fused cx should be further right
    # (both are low-IoU paths so they return the slow bbox, corrected)
    assert fused_stale.bbox.cx > fused_fresh.bbox.cx
```

- [ ] **Step 2: Run — expect failure**

```
pytest tests/test_iou_fusion.py -v
```
Expected: failures because `fuse()` still takes `nframe`.

- [ ] **Step 3: Update `fusion_algs/base.py`**

```python
"""Base fusion algorithm interface and the no-op passthrough implementation."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trackers.base import TrackResult, BaseTracker


class BaseFusionAlgorithm(ABC):
    """Common interface for all tracker-fusion algorithms."""

    @abstractmethod
    def fuse(self, cfg: dict, trackers: list[BaseTracker], results: list[TrackResult]) -> TrackResult:
        """Combine multiple tracker outputs into one result."""


class PassthroughFusion(BaseFusionAlgorithm):
    """Identity fusion — used when exactly one tracker is active."""

    def fuse(self, cfg: dict, trackers: list[BaseTracker], results: list[TrackResult]) -> TrackResult:
        if len(results) != 1:
            raise ValueError(
                f"PassthroughFusion expects exactly 1 result, got {len(results)}."
            )
        return results[0]
```

- [ ] **Step 4: Rewrite `fusion_algs/iou_fusion.py`**

```python
"""IoU-gated fusion with dead-reckoning correction for stale async results."""
from __future__ import annotations

from core.bbox import BBox
from core.iou import bbox_iou
from trackers.base import BaseTracker, TrackResult
from fusion_algs.base import BaseFusionAlgorithm


class IoUFusion(BaseFusionAlgorithm):
    """Fuse a fast sync tracker with a slow async tracker.

    Before IoU gating, the slow bbox is age-corrected: the fast tracker's
    recent velocity (EMA) projects it forward by result_age frames so both
    positions are temporally aligned.
    """

    def __init__(self) -> None:
        self._v_cx: float = 0.0
        self._v_cy: float = 0.0
        self._prev_cx: float | None = None
        self._prev_cy: float | None = None

    def _age_correct(self, bslow: BBox, age: int) -> BBox:
        return BBox(
            cx=bslow.cx + age * self._v_cx,
            cy=bslow.cy + age * self._v_cy,
            w=bslow.w,
            h=bslow.h,
        )

    def fuse(self, cfg: dict, trackers: list[BaseTracker], results: list[TrackResult]) -> TrackResult:
        alpha  = cfg.get("fusion_velocity_ema_alpha", 0.3)
        thrsh1 = cfg["tracker"]["async_corr_thresh1"]
        thrsh2 = cfg["tracker"]["async_corr_thresh2"]
        a1 = 1.0 - thrsh1
        a2 = 1.0 - thrsh2

        bfast     = BBox(0.0, 0.0, 0.0, 0.0)
        bslow     = BBox(0.0, 0.0, 0.0, 0.0)
        namefast  = ""
        nameslow  = ""
        conffast  = 0.0
        age       = 0

        for t in trackers:
            for r in results:
                if r.source == t.name():
                    if t.is_async:
                        bslow    = r.bbox
                        nameslow = t.name()
                        age      = t.result_age
                    else:
                        bfast    = r.bbox
                        conffast = r.confidence
                        namefast = t.name()

        # Update fast-tracker velocity EMA
        if self._prev_cx is not None:
            dcx = bfast.cx - self._prev_cx
            dcy = bfast.cy - self._prev_cy
            self._v_cx = alpha * dcx + (1.0 - alpha) * self._v_cx
            self._v_cy = alpha * dcy + (1.0 - alpha) * self._v_cy
        self._prev_cx = bfast.cx
        self._prev_cy = bfast.cy

        bslow_c = self._age_correct(bslow, age)
        iou     = bbox_iou(bfast, bslow_c)

        def _blend(b1: BBox, b2: BBox, w: float) -> BBox:
            """Return b1 + w*(b2-b1)."""
            return BBox(
                cx=b1.cx + w * (b2.cx - b1.cx),
                cy=b1.cy + w * (b2.cy - b1.cy),
                w =b1.w  + w * (b2.w  - b1.w),
                h =b1.h  + w * (b2.h  - b1.h),
            )

        if iou > thrsh1:
            fusedbbox = _blend(bfast, bslow_c, a1)
            namefused = namefast + " + " + nameslow
            conf = iou
        elif iou > thrsh2:
            fusedbbox = _blend(bfast, bslow_c, a2)
            namefused = namefast + " + " + nameslow
            conf = iou
        else:
            fusedbbox = bslow_c
            namefused = nameslow
            conf = iou

        return TrackResult(fusedbbox, conf, 0.0, namefused)
```

- [ ] **Step 5: Simplify `testing/main.py`**

Replace the `main()` function body (keep imports and helpers unchanged). Remove `nframe`, `interval`, `rslow`. Also remove the unused `from numpy import integer` import at the top.

New `main()`:

```python
def main(config_path: str = "config.yaml") -> None:
    cfg = _load_config(config_path)
    trackers, fusion = build_trackers(cfg)
    cam = Camera(config_path)
    frame_q = _start_capture_thread(cam)

    tracking   = False
    bbox: BBox | None = None
    roi_half   = _DEFAULT_ROI_HALF
    prev_time  = time.perf_counter()
    fps        = 0.

    algo_names = " + ".join(
        s["algorithm"] for s in cfg["tracker"]["algorithms"]
    )
    print(f"QuadTrack started [{algo_names}]. "
          "Press SPACE to lock on, R to release, Q to quit.")

    try:
        while True:
            frame = frame_q.get()

            now       = time.perf_counter()
            fps       = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            if tracking:
                t0      = time.perf_counter()
                results = [t.update(frame) for t in trackers]
                result  = fusion.fuse(cfg, trackers, results)
                bbox    = result.bbox
                print("latency: " + str(time.perf_counter() - t0))

                if not result.confidence:
                    tracking = False
                    bbox = None

            display = draw_overlay(frame.image, bbox, tracking, roi_half, fps)
            cv2.imshow("QuadTrack", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("r"):
                tracking = False
                bbox = None
            elif key == ord("["):
                roi_half = max(_ROI_MIN, roi_half - _ROI_STEP)
            elif key == ord("]"):
                roi_half = min(_ROI_MAX, roi_half + _ROI_STEP)
            elif key == ord(" ") and not tracking:
                h, w = frame.image.shape[:2]
                cx, cy = w // 2, h // 2
                bbox = BBox(cx=float(cx), cy=float(cy),
                            w=float(roi_half * 2), h=float(roi_half * 2))
                for t in trackers:
                    t.init(frame, bbox)
                tracking = True

    finally:
        cam.release()
        cv2.destroyAllWindows()
```

- [ ] **Step 6: Fix `benchmark/runner.py` line 128**

```python
# Before:
fused = self._fusion.fuse(results)

# After:
fused = self._fusion.fuse(self._cfg, self._trackers, results)
```

- [ ] **Step 7: Run tests — expect pass**

```
pytest tests/ -v
```
Expected: all tests in `test_base.py` and `test_iou_fusion.py` PASS.

- [ ] **Step 8: Commit**

```bash
git add fusion_algs/base.py fusion_algs/iou_fusion.py testing/main.py benchmark/runner.py tests/test_iou_fusion.py
git commit -m "feat: dead-reckoning IoU fusion; remove nframe from fuse() interface"
```

---

### Task 3: Make `NanoTracker` non-blocking

**Files:**
- Modify: `trackers/nanotrack_tracker.py`
- Create: `tests/test_nanotracker_async.py`

**Design:** `_run_inference(frame, cx, cy, w, h)` contains all ONNX logic, takes state as args (thread-safe snapshot), returns `(TrackResult, new_cx, new_cy, new_w, new_h)`. `update()` always returns immediately.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_nanotracker_async.py`:

```python
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
from core.bbox import BBox
from trackers.base import TrackResult
from trackers.nanotrack_tracker import NanoTracker


def _make_frame():
    """Minimal frame-like object."""
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

    def _mock_inference(frame, cx, cy, w, h):
        time.sleep(0.02)  # simulate 20 ms inference
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
    t.update(_make_frame())          # first call: seeds cached result
    t0 = time.perf_counter()
    t.update(_make_frame())          # second call: must return cached immediately
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.005, f"update() blocked for {elapsed:.3f}s — expected < 5ms"


def test_result_age_increments_while_inference_running():
    t = _make_tracker()
    t.update(_make_frame())          # seed
    t.update(_make_frame())          # inference still running → cached
    assert t.result_age >= 1


def test_result_age_resets_when_fresh_result_arrives():
    t = _make_tracker()
    t.update(_make_frame())          # seed; kicks off first background job
    time.sleep(0.05)                 # wait for 20ms inference to complete
    t.update(_make_frame())          # harvests result → result_age resets to 0
    assert t.result_age == 0


def test_init_resets_async_state():
    t = _make_tracker()
    t.update(_make_frame())
    t.result_age = 5                 # simulate stale state
    t.init(_make_frame(), BBox(200, 200, 60, 60))
    assert t.result_age == 0
    assert t._cached_result is None
```

- [ ] **Step 2: Run — expect failure**

```
pytest tests/test_nanotracker_async.py -v
```
Expected: `AttributeError` — `_run_inference`, `_pending_future`, `_cached_result` not yet on the class.

- [ ] **Step 3: Refactor `trackers/nanotrack_tracker.py`**

Add to `__init__` (at the top, before existing device setup code):

```python
from concurrent.futures import ThreadPoolExecutor

# Async inference state (populated here; used by update())
self._executor       = ThreadPoolExecutor(max_workers=1)
self._pending_future = None
self._cached_result: TrackResult | None = None
self._frame_count    = 0
```

Add `_run_inference` method — move the entire body of the current `update()` into it, changing all `self._cx/cy/w/h` reads to use the passed-in args, and returning state alongside the result:

```python
def _run_inference(
    self,
    frame: Frame,
    cx: float, cy: float, w: float, h: float,
) -> tuple[TrackResult, float, float, float, float]:
    """Run backbone + head inference. Thread-safe: all mutable state passed as args."""
    t0 = time.perf_counter()
    h_img, w_img = frame.image.shape[:2]

    s_sum  = w + h
    wc     = w + CONTEXT_AMOUNT * s_sum
    hc     = h + CONTEXT_AMOUNT * s_sum
    sz     = np.sqrt(wc * hc)
    scale_z = EXEMPLAR_SIZE / sz
    sx      = sz * (INSTANCE_SIZE / EXEMPLAR_SIZE)
    tw_s = w * scale_z
    th_s = h * scale_z

    crop   = self._get_subwindow(frame.image, cx, cy, int(sx), INSTANCE_SIZE)
    s_feat = self._run_backbone(self._preprocess(crop))
    cls, loc = self._run_head(self._t_feat, s_feat)

    cls0 = cls[0]
    mx   = cls0.max(axis=0, keepdims=True)
    exp  = np.exp(cls0 - mx)
    score = (exp / exp.sum(axis=0, keepdims=True))[1]

    l, t, r, b = loc[0, 0], loc[0, 1], loc[0, 2], loc[0, 3]
    pred_x1 = self._grid_x - l
    pred_y1 = self._grid_y - t
    pred_x2 = self._grid_x + r
    pred_y2 = self._grid_y + b
    pred_w  = pred_x2 - pred_x1
    pred_h  = pred_y2 - pred_y1

    def _rmax(v):
        return np.maximum(v, 1.0 / np.maximum(v, 1e-6))

    sc = _rmax(self._size_cal(pred_w, pred_h) / self._size_cal(tw_s, th_s))
    rc = _rmax((pred_w / np.maximum(pred_h, 1e-6)) / (w / max(h, 1e-6)))
    penalty  = np.exp(-(sc * rc - 1.0) * PENALTY_K)
    pscore   = score * penalty * (1.0 - WINDOW_INFLUENCE) + self._window * WINDOW_INFLUENCE
    best     = np.unravel_index(np.argmax(pscore), pscore.shape)
    bi, bj   = best

    best_score   = float(score[bi, bj])
    best_penalty = float(penalty[bi, bj])

    pred_xs = (pred_x1[bi, bj] + pred_x2[bi, bj]) / 2.0
    pred_ys = (pred_y1[bi, bj] + pred_y2[bi, bj]) / 2.0
    pred_bw = float(pred_w[bi, bj])
    pred_bh = float(pred_h[bi, bj])

    diff_x = (pred_xs - (INSTANCE_SIZE // 2)) / scale_z
    diff_y = (pred_ys - (INSTANCE_SIZE // 2)) / scale_z
    out_w  = pred_bw / scale_z
    out_h  = pred_bh / scale_z

    lr     = best_penalty * best_score * LR
    new_cx = float(np.clip(cx + diff_x, 0, w_img))
    new_cy = float(np.clip(cy + diff_y, 0, h_img))
    new_w  = float(np.clip(out_w * lr + w * (1.0 - lr), MIN_SIZE, w_img))
    new_h  = float(np.clip(out_h * lr + h * (1.0 - lr), MIN_SIZE, h_img))

    ok = best_score > SCORE_THRESHOLD
    result = TrackResult(
        bbox=BBox(cx=new_cx, cy=new_cy, w=new_w, h=new_h),
        confidence=best_score if ok else 0.0,
        latency_s=time.perf_counter() - t0,
        source="nanotrack",
    )
    return result, new_cx, new_cy, new_w, new_h
```

Replace the existing `update()` with the non-blocking version:

```python
def update(self, frame: Frame) -> TrackResult:
    """Return latest cached result immediately; inference runs in background."""
    # First call after init: must seed a result synchronously
    if self._cached_result is None:
        result, cx, cy, w, h = self._run_inference(
            frame, self._cx, self._cy, self._w, self._h
        )
        self._cx, self._cy, self._w, self._h = cx, cy, w, h
        self._cached_result = result
        self.result_age = 0
        self._pending_future = self._executor.submit(
            self._run_inference, frame, self._cx, self._cy, self._w, self._h
        )
        return self._cached_result

    # Harvest completed future
    if self._pending_future is not None and self._pending_future.done():
        result, cx, cy, w, h = self._pending_future.result()
        self._cx, self._cy, self._w, self._h = cx, cy, w, h
        self._cached_result = result
        self._pending_future = None
        self.result_age = 0
    else:
        self.result_age += 1

    # Submit new inference if slot is free
    if self._pending_future is None:
        should_submit = True
        if self.async_submit_strategy == "fixed_interval":
            self._frame_count += 1
            should_submit = self._frame_count >= self.async_min_interval
            if should_submit:
                self._frame_count = 0
        if should_submit:
            self._pending_future = self._executor.submit(
                self._run_inference, frame, self._cx, self._cy, self._w, self._h
            )

    return self._cached_result
```

Update `init()` — prepend reset of async state before the existing template-extraction code:

```python
def init(self, frame: Frame, bbox: BBox) -> None:
    # Reset async state
    if self._pending_future is not None and not self._pending_future.done():
        self._pending_future.cancel()
    self._pending_future  = None
    self._cached_result   = None
    self.result_age       = 0
    self._frame_count     = 0

    self._cx = bbox.cx
    self._cy = bbox.cy
    self._w  = bbox.w
    self._h  = bbox.h

    # ... rest of existing init (template crop + backbone) unchanged ...
```

- [ ] **Step 4: Run tests — expect pass**

```
pytest tests/test_nanotracker_async.py -v
```
Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add trackers/nanotrack_tracker.py tests/test_nanotracker_async.py
git commit -m "feat: non-blocking NanoTracker update via ThreadPoolExecutor"
```

---

### Task 4: Create `ioukf_fusion.py` (Kalman filter)

**Files:**
- Create: `fusion_algs/ioukf_fusion.py`
- Create: `tests/test_ioukf_fusion.py`

**State vector:** `[cx, cy, vx, vy]`. Constant-velocity model (dt=1 frame).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ioukf_fusion.py`:

```python
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
    # Warm up filter at (100, 100)
    for _ in range(5):
        fusion.fuse(CFG, [fast_t, slow_t], [_r(100, 100, src="fast"), _r(100, 100, src="slow")])
    # Now deliver a slow update with age=0 (fresh), pulling toward (130, 100)
    slow_t.result_age = 0
    results = [_r(100, 100, src="fast"), _r(130, 100, src="slow")]
    fused = fusion.fuse(CFG, [fast_t, slow_t], results)
    assert fused.bbox.cx > 100.0  # pulled right by slow update


def test_fast_tracker_confidence_propagated():
    fusion = IoUKFFusion()
    fast_t = _tracker("fast")
    slow_t = _tracker("slow", is_async=True, age=1)
    results = [_r(100, 100, conf=0.0, src="fast"), _r(100, 100, src="slow")]
    fused = fusion.fuse(CFG, [fast_t, slow_t], results)
    assert fused.confidence == 0.0
```

- [ ] **Step 2: Run — expect failure**

```
pytest tests/test_ioukf_fusion.py -v
```
Expected: `ModuleNotFoundError: No module named 'fusion_algs.ioukf_fusion'`

- [ ] **Step 3: Create `fusion_algs/ioukf_fusion.py`**

```python
"""IoU-gated constant-velocity Kalman filter fusion for async tracker pairs."""
from __future__ import annotations

import numpy as np

from core.bbox import BBox
from core.iou import bbox_iou
from trackers.base import BaseTracker, TrackResult
from fusion_algs.base import BaseFusionAlgorithm

_F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], dtype=float)
_H = np.array([[1,0,0,0],[0,1,0,0]], dtype=float)
_I4 = np.eye(4)


class IoUKFFusion(BaseFusionAlgorithm):
    """Fuse fast + slow tracker via a 4-state constant-velocity Kalman filter.

    State: [cx, cy, vx, vy].
    - Fast tracker: measurement update every frame (R = kf_fast_meas_noise * I).
    - Slow tracker: measurement update only when result_age == 0 (fresh result)
      and IoU with Kalman prediction >= async_corr_thresh2.
      R is inflated by kf_age_noise_scale * age to reflect past staleness.
    - Size (w, h): tracked separately via EMA; not part of Kalman state.
    """

    def __init__(self) -> None:
        self._initialized = False
        self._x = np.zeros(4)
        self._P = np.eye(4) * 100.0
        self._fused_w = 0.0
        self._fused_h = 0.0

    def _predict(self, Q: np.ndarray) -> None:
        self._x = _F @ self._x
        self._P = _F @ self._P @ _F.T + Q

    def _update(self, z: np.ndarray, R: np.ndarray) -> None:
        y = z - _H @ self._x
        S = _H @ self._P @ _H.T + R
        K = self._P @ _H.T @ np.linalg.inv(S)
        self._x = self._x + K @ y
        self._P = (_I4 - K @ _H) @ self._P

    def fuse(self, cfg: dict, trackers: list[BaseTracker], results: list[TrackResult]) -> TrackResult:
        q          = cfg.get("kf_process_noise",   0.01)
        r_fast     = cfg.get("kf_fast_meas_noise", 5.0)
        r_slow_base = cfg.get("kf_slow_meas_noise", 1.0)
        age_scale  = cfg.get("kf_age_noise_scale", 0.5)
        thrsh2     = cfg["tracker"]["async_corr_thresh2"]
        size_alpha = 0.3  # EMA weight toward slow size when fresh result arrives

        bfast = BBox(0.0, 0.0, 0.0, 0.0)
        bslow = BBox(0.0, 0.0, 0.0, 0.0)
        namefast = ""
        nameslow = ""
        conffast = 0.0
        age = 0

        for t in trackers:
            for r in results:
                if r.source == t.name():
                    if t.is_async:
                        bslow    = r.bbox
                        nameslow = t.name()
                        age      = t.result_age
                    else:
                        bfast    = r.bbox
                        conffast = r.confidence
                        namefast = t.name()

        if not self._initialized:
            self._x = np.array([bfast.cx, bfast.cy, 0.0, 0.0])
            self._P = np.eye(4) * 100.0
            self._fused_w = bfast.w
            self._fused_h = bfast.h
            self._initialized = True

        # Predict
        self._predict(np.eye(4) * q)

        # Update with fast tracker measurement (every frame)
        self._update(np.array([bfast.cx, bfast.cy]), np.eye(2) * r_fast)

        # Update with slow tracker measurement when fresh result arrives
        if age == 0:
            pred_bbox = BBox(float(self._x[0]), float(self._x[1]),
                             self._fused_w, self._fused_h)
            if bbox_iou(pred_bbox, bslow) >= thrsh2:
                r_slow = r_slow_base + age_scale * age
                self._update(np.array([bslow.cx, bslow.cy]), np.eye(2) * r_slow)
                self._fused_w = size_alpha * bslow.w + (1.0 - size_alpha) * self._fused_w
                self._fused_h = size_alpha * bslow.h + (1.0 - size_alpha) * self._fused_h

        fused_bbox = BBox(float(self._x[0]), float(self._x[1]),
                          self._fused_w, self._fused_h)
        namefused  = namefast + " + " + nameslow if age == 0 else namefast
        return TrackResult(fused_bbox, conffast, 0.0, namefused)
```

- [ ] **Step 4: Run tests — expect pass**

```
pytest tests/test_ioukf_fusion.py -v
```
Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add fusion_algs/ioukf_fusion.py tests/test_ioukf_fusion.py
git commit -m "feat: add IoUKFFusion — Kalman filter fusion for async tracker pairs"
```

---

### Task 5: Register `ioukf` in factory; update `config.yaml`

**Files:**
- Modify: `trackers/factory.py`
- Modify: `config.yaml`

- [ ] **Step 1: Update `trackers/factory.py`**

Add import after the existing IoUFusion import:

```python
from fusion_algs.ioukf_fusion import IoUKFFusion
```

Add entry to `_FUSION_MAP`:

```python
_FUSION_MAP: dict[str, type[BaseFusionAlgorithm]] = {
    "iou":   IoUFusion,
    "ioukf": IoUKFFusion,
}
```

- [ ] **Step 2: Update `config.yaml`**

Replace the tracker + fusion section with:

```yaml
tracker:
  algorithms:
    - algorithm: mosse
    - algorithm: nanotrack
      async: true
      async_submit_strategy: "on_completion"   # "on_completion" | "fixed_interval"
      async_min_interval: 1                    # used only with fixed_interval
  fusion: iou              # iou | ioukf
  async_corr_thresh1: 0.7
  async_corr_thresh2: 0.3
  # nanotrack: ONNX Runtime (CUDA/CPU)
  nanotrack_backbone: models/nanotrackv3/nanotrackv3_backbone.onnx
  nanotrack_head:     models/nanotrackv3/nanotrackv3_head.onnx
  # nanotrack: RKNN (NPU)
  nanotrack_backbone_rknn: models/nanotrackv3/nanotrackv3_backbone.rknn
  nanotrack_head_rknn:     models/nanotrackv3/nanotrackv3_head.rknn

# IoU dead-reckoning fusion settings
fusion_velocity_ema_alpha: 0.3    # EMA smoothing for fast-tracker velocity

# Kalman filter fusion settings (used when fusion: ioukf)
kf_process_noise:   0.01   # Q — target acceleration uncertainty
kf_fast_meas_noise: 5.0    # R_fast — fast tracker centroid noise (pixels²)
kf_slow_meas_noise: 1.0    # R_slow — slow tracker centroid noise (pixels²)
kf_age_noise_scale: 0.5    # inflates R_slow per frame of staleness
```

- [ ] **Step 3: Run full test suite**

```
pytest tests/ -v
```
Expected: all tests PASS.

- [ ] **Step 4: Commit**

```bash
git add trackers/factory.py config.yaml
git commit -m "feat: register ioukf fusion; update config with scheduling + KF keys"
```

---

## Verification

1. **Non-blocking fast rate:** Run `python testing/main.py`. Press SPACE to lock on. Observe the FPS counter in the HUD — it should stay near 30 fps. The `latency:` print should reflect fast-tracker latency only (< 5ms for MOSSE), not NanoTrack inference time.

2. **Dead-reckoning smoothness (`fusion: iou`):** Track a moving target. Observe the fused bbox centroid in the HUD — it should not snap when a new NanoTrack result arrives, even during fast lateral motion.

3. **Kalman fusion (`fusion: ioukf`):** Change `config.yaml` to `fusion: ioukf`. Track the same moving target. Centroid should be smoother than `iou` on fast motion; tracking should persist briefly if MOSSE loses the target (Kalman coasts on velocity).

4. **Scheduling modes:** In `config.yaml`, change the nanotrack algorithm entry to `async_submit_strategy: "fixed_interval"` and `async_min_interval: 5`. Confirm NanoTrack submits less frequently (higher `result_age` values printed or observable via HUD lag).
