# Async Non-Blocking Tracker & Improved IoU Fusion Design

**Date:** 2026-04-12  
**Status:** Approved

---

## Problem Statement

Two independent issues degrade real-time tracking performance:

1. **Blocking slow tracker:** `NanoTracker.update()` runs dual ONNX inference synchronously in the main loop. When called (every `async_interval` frames), it stalls the fast tracker and drops effective frame rate to the slow tracker's throughput.

2. **Jittery fused centroid:** `IoUFusion.smooth_correction()` blends the slow result linearly without compensating for the fact that the slow result is stale by several frames. When the slow result arrives, the position offset causes the centroid to snap rather than transition smoothly.

---

## Design

### 1. Non-Blocking Async Tracker

**Location:** Inside each tracker that has `is_async=True` (primarily `NanoTracker`).

**Mechanism:** The tracker owns a `ThreadPoolExecutor(max_workers=1)`. `update()` always returns immediately:

- If no inference is pending: submit a new job to the thread pool; return the last cached `TrackResult`
- If inference is running and not complete: return the last cached `TrackResult` (non-blocking)
- If inference just completed: cache the new result; submit the next job; return the new result

**`result_age` property:** Exposed by `BaseTracker`. Counts how many frames have elapsed since the cached result was computed. Incremented each time `update()` returns a cached (non-fresh) result. Reset to 0 when a fresh result is returned. Fusion algorithms use this instead of the external `nframe` counter.

**Main loop simplification:** `testing/main.py` drops the `if t.is_async and nframe == 0` branching. All trackers are called with `t.update(frame)` every frame. The `nframe` counter is removed.

**Config (per-algorithm in `config.yaml`):**
```yaml
tracker:
  algorithms:
    - algorithm: nanotrack
      async: true
      async_submit_strategy: "on_completion"  # "on_completion" | "fixed_interval"
      async_min_interval: 1                   # minimum frames between job submissions
```

- `async_submit_strategy: "on_completion"` — submit a new job as soon as the previous completes (maximises NanoTrack utilisation)
- `async_submit_strategy: "fixed_interval"` — submit every `async_min_interval` frames regardless

> **Note:** The top-level `async_interval` key (previously used by the main loop) is removed. Scheduling is now owned entirely by the tracker via `async_min_interval`.

---

### 2. `iou_fusion.py` — Dead-Reckoning Correction

**Replaces:** `smooth_correction()` static method.

**New helper: `_age_correct(slow_bbox, fast_velocity, age)`**

Before IoU gating, the slow result is age-corrected to the current frame:

```
slow_corrected_cx = slow_cx + age * v_ema_cx
slow_corrected_cy = slow_cy + age * v_ema_cy
```

Where `v_ema` is an exponential moving average of the fast tracker's frame-to-frame centroid displacement, updated every frame:

```
delta = (fast_cx_now - fast_cx_prev, fast_cy_now - fast_cy_prev)
v_ema = alpha * delta + (1 - alpha) * v_ema
```

**Size (w, h):** Not age-corrected (velocity of bounding box size is unreliable). Existing EMA-blend on width/height dimensions is retained.

**IoU gating and blending:** Unchanged — the same `async_corr_thresh1` / `async_corr_thresh2` thresholds apply, but now operate on the age-corrected slow bbox instead of the raw stale one.

**Config addition:**
```yaml
fusion_velocity_ema_alpha: 0.3    # smoothing factor for fast-tracker velocity EMA
```

---

### 3. `ioukf_fusion.py` — Kalman Filter Correction (new file)

**State vector:** `[cx, cy, vx, vy]` — 2D centroid position and velocity.

**Prediction step (every frame):** Standard constant-velocity Kalman predict using process noise `Q`.

**Measurement updates:**

| Source | When | Covariance |
|---|---|---|
| Fast tracker | Every frame | `R_fast` (higher — less accurate) |
| Slow tracker | When `result_age` resets to 0 | `R_slow * (1 + age_noise_scale * age)` — inflated for staleness |

**Outlier rejection (IoU gate):** Before applying the slow tracker measurement, compute IoU between the Kalman-predicted centroid bbox and the slow result. If `IoU < async_corr_thresh2`, skip the slow update (treats it as an outlier).

**Size (w, h):** Not tracked by Kalman — blended separately using EMA (same as `iou_fusion.py`).

**Initialisation:** On first frame (or tracking re-init), state is seeded from the fast tracker result; velocity initialised to zero; covariance set to identity.

**Config additions:**
```yaml
kf_process_noise: 0.01        # Q — target acceleration uncertainty
kf_fast_meas_noise: 5.0       # R_fast — fast tracker centroid noise (pixels²)
kf_slow_meas_noise: 1.0       # R_slow — slow tracker centroid noise (pixels²)
kf_age_noise_scale: 0.5       # inflates R_slow per frame of staleness
```

**Fusion map entry:** `"ioukf"` added to `_FUSION_MAP` in `trackers/factory.py`.

---

## Files Changed

| File | Change |
|---|---|
| `trackers/base.py` | Add `result_age: int` property to `BaseTracker`; add threading plumbing for async trackers |
| `trackers/nanotrack_tracker.py` | Implement non-blocking `update()` using `ThreadPoolExecutor` |
| `fusion_algs/iou_fusion.py` | Replace `smooth_correction()` with `_age_correct()` + velocity EMA |
| `fusion_algs/ioukf_fusion.py` | New file: Kalman filter fusion |
| `trackers/factory.py` | Add `"ioukf"` to `_FUSION_MAP` |
| `testing/main.py` | Remove `nframe` async branching; call all trackers uniformly |
| `config.yaml` | Add new config keys for all three components |

---

## Verification

1. **Non-blocking:** Run with `mosse + nanotrack (async)` and `fusion: iou`. Confirm fast tracker maintains target FPS (e.g. 30fps) while NanoTrack runs in background — check via the FPS counter in the HUD.
2. **Dead-reckoning:** Track a moving target. Confirm the fused centroid no longer snaps when the slow result arrives — observe smooth centroid trajectory.
3. **Kalman fusion:** Switch config to `fusion: ioukf`. Confirm similar or better smoothness vs dead-reckoning on a moving target. Confirm tracking survives brief fast-tracker failures (Kalman coasts on velocity).
4. **Config switching:** Toggle `async_submit_strategy` between `"on_completion"` and `"fixed_interval"` and confirm different NanoTrack submission rates.
