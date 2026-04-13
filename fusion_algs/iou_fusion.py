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
