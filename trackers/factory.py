"""Factory functions for building trackers and fusion algorithms from config.

Used by both the live tracking pipeline (testing/main.py) and the offline
benchmark (testing/benchmark.py) so that tracker construction is defined in
exactly one place.
"""
from __future__ import annotations

from trackers.base import BaseTracker
from trackers.kcf_tracker import KCFTracker
from trackers.csrt_tracker import CSRTTracker
from trackers.mosse_tracker import MOSSETracker
from trackers.nanotrack_tracker import NanoTracker
from fusion_algs.base import BaseFusionAlgorithm, PassthroughFusion
from fusion_algs.async_fusion import AsyncFusion


_ALGO_MAP: dict[str, type[BaseTracker]] = {
    "kcf":             KCFTracker,
    "csrt":            CSRTTracker,
    "mosse":           MOSSETracker,
    "nanotrack":       NanoTracker,
}

_FUSION_MAP: dict[str, type[BaseFusionAlgorithm]] = {
    "async": AsyncFusion,
}


def build_trackers(cfg: dict) -> tuple[list[BaseTracker], BaseFusionAlgorithm]:
    """Build trackers and fusion algorithm from cfg['tracker'].

    Args:
        cfg: Full config dict.

    Returns:
        Tuple of (list of initialised BaseTracker instances, BaseFusionAlgorithm).

    Raises:
        ValueError: if an algorithm name or fusion name is not recognised.
    """
    tracker_cfg = cfg["tracker"]
    trackers: list[BaseTracker] = []
    for spec in tracker_cfg["algorithms"]:
        algo = spec["algorithm"].strip().lower()
        if algo not in _ALGO_MAP:
            raise ValueError(
                f"Unknown tracker algorithm: {algo!r}. "
                f"Valid options: {list(_ALGO_MAP)}"
            )
        t = _ALGO_MAP[algo](cfg)
        t.is_async = spec.get("async", False)
        trackers.append(t)

    n = len(trackers)
    fusion_name = tracker_cfg.get("fusion", "passthrough")
    if n == 1:
        fusion: BaseFusionAlgorithm = PassthroughFusion()
    else:
        name = fusion_name.strip().lower()
        if name == "passthrough":
            raise ValueError(
                f"'passthrough' fusion cannot be used with {n} trackers. "
                "Set a real fusion algorithm (e.g. async) in config."
            )
        if name not in _FUSION_MAP:
            raise ValueError(
                f"Unknown fusion algorithm: {name!r}. "
                f"Valid options: ['passthrough', {list(_FUSION_MAP)}]"
            )
        fusion = _FUSION_MAP[name]()

    return trackers, fusion
