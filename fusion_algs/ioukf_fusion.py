"""IoU-gated constant-velocity Kalman filter fusion for async tracker pairs."""
from __future__ import annotations

import numpy as np

from core.bbox import BBox
from core.frame import Frame
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
    - Reinit: when a fresh slow result arrives and IoU with KF prediction == 0
      (trackers fully diverged), the fast tracker and KF state are reinitialised
      from the slow bbox so the fast tracker can recover.
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

    def fuse(self, cfg: dict, trackers: list[BaseTracker], results: list[TrackResult], frame: Frame) -> TrackResult:
        q           = cfg.get("kf_process_noise",   0.01)
        r_fast      = cfg.get("kf_fast_meas_noise", 5.0)
        r_slow_base = cfg.get("kf_slow_meas_noise", 1.0)
        age_scale   = cfg.get("kf_age_noise_scale", 0.5)
        thrsh2      = cfg["tracker"]["async_corr_thresh2"]
        size_alpha  = 0.3

        bfast        = BBox(0.0, 0.0, 0.0, 0.0)
        bslow        = BBox(0.0, 0.0, 0.0, 0.0)
        namefast     = ""
        nameslow     = ""
        conffast     = 0.0
        confslow     = 0.0
        age          = 0
        fast_tracker: BaseTracker | None = None

        for t in trackers:
            for r in results:
                if r.source == t.name():
                    if t.is_async:
                        bslow        = r.bbox
                        nameslow     = t.name()
                        confslow     = r.confidence
                        age          = t.result_age
                    else:
                        bfast        = r.bbox
                        conffast     = r.confidence
                        namefast     = t.name()
                        fast_tracker = t

        if not self._initialized:
            self._x = np.array([bfast.cx, bfast.cy, 0.0, 0.0])
            self._P = np.eye(4) * 100.0
            self._fused_w = bfast.w
            self._fused_h = bfast.h
            self._initialized = True

        self._predict(np.eye(4) * q)
        self._update(np.array([bfast.cx, bfast.cy]), np.eye(2) * r_fast)

        conf = conffast
        if age == 0 and bslow.w > 0:
            pred_bbox = BBox(float(self._x[0]), float(self._x[1]),
                             self._fused_w, self._fused_h)
            iou = bbox_iou(pred_bbox, bslow)
            if iou >= thrsh2:
                r_slow = r_slow_base + age_scale * age
                self._update(np.array([bslow.cx, bslow.cy]), np.eye(2) * r_slow)
                self._fused_w = size_alpha * bslow.w + (1.0 - size_alpha) * self._fused_w
                self._fused_h = size_alpha * bslow.h + (1.0 - size_alpha) * self._fused_h
            elif iou == 0.0 and fast_tracker is not None:
                # Trackers fully diverged — reinitialise fast tracker and KF from slow bbox.
                fast_tracker.init(frame, bslow)
                self._x = np.array([bslow.cx, bslow.cy, 0.0, 0.0])
                self._P = np.eye(4) * 100.0
                self._fused_w = bslow.w
                self._fused_h = bslow.h
                conf = confslow

        fused_bbox = BBox(float(self._x[0]), float(self._x[1]),
                          self._fused_w, self._fused_h)
        namefused  = namefast + " + " + nameslow if age == 0 else namefast
        return TrackResult(fused_bbox, conf, 0.0, namefused)
