"""Factory functions for building trackers and fusion algorithms from config.

Used by both the live tracking pipeline (testing/main.py) and the offline
benchmark (testing/benchmark.py) so that tracker construction is defined in
exactly one place.
"""
from __future__ import annotations

from trackers.base import BaseTracker
from trackers.kcf_tracker import KCFTracker
from trackers.csrt_tracker import CSRTTracker
from trackers.nanotrack_tracker import NanoTracker
from trackers.nanotrack_accel_tracker import NanoTrackAccel
from fusion_algs.base import BaseFusionAlgorithm, PassthroughFusion


_ALGO_MAP: dict[str, type[BaseTracker]] = {
    "kcf":             KCFTracker,
    "csrt":            CSRTTracker,
    "nanotrack":       NanoTracker,
    "nanotrack_accel": NanoTrackAccel,
}

# Registry for future fusion algorithms.
# Add entries here when fusion_algs/fusion.py gains real implementations.
_FUSION_MAP: dict[str, type[BaseFusionAlgorithm]] = {
    # "weighted_mean": WeightedMeanFusion,
}


def build_trackers(algorithm_specs: list[dict], cfg: dict) -> list[BaseTracker]:
    """Instantiate trackers from a list of algorithm spec dicts.

    Args:
        algorithm_specs: List of dicts, each with an "algorithm" key.
                         e.g. [{"algorithm": "kcf"}, {"algorithm": "nanotrack_accel"}]
        cfg: Full config dict (passed to each tracker constructor).

    Returns:
        List of initialised BaseTracker instances.

    Raises:
        ValueError: if an algorithm name is not recognised.
    """
    trackers: list[BaseTracker] = []
    for spec in algorithm_specs:
        algo = spec["algorithm"].strip().lower()
        if algo not in _ALGO_MAP:
            raise ValueError(
                f"Unknown tracker algorithm: {algo!r}. "
                f"Valid options: {list(_ALGO_MAP)}"
            )
        trackers.append(_ALGO_MAP[algo](cfg))
    return trackers


def build_fusion(fusion_name: str, n_trackers: int) -> BaseFusionAlgorithm:
    """Instantiate a fusion algorithm.

    Args:
        fusion_name: Name from config (e.g. "passthrough", "weighted_mean").
        n_trackers:  Number of trackers that will feed into the fusion step.

    Returns:
        A BaseFusionAlgorithm instance.

    Raises:
        ValueError: if fusion_name is not recognised, or "passthrough" is
                    requested with more than one tracker.
    """
    if n_trackers == 1:
        return PassthroughFusion()

    name = fusion_name.strip().lower()
    if name == "passthrough":
        raise ValueError(
            f"'passthrough' fusion cannot be used with {n_trackers} trackers. "
            "Set a real fusion algorithm (e.g. weighted_mean) in config."
        )
    if name not in _FUSION_MAP:
        raise ValueError(
            f"Unknown fusion algorithm: {name!r}. "
            f"Valid options: ['passthrough', {list(_FUSION_MAP)}]"
        )
    return _FUSION_MAP[name]()


def build_from_tracker_section(cfg: dict) -> tuple[list[BaseTracker], BaseFusionAlgorithm]:
    """Convenience: build trackers + fusion from cfg['tracker'].

    Used by the live tracking pipeline (testing/main.py).
    """
    specs = cfg["tracker"]["algorithms"]
    trackers = build_trackers(specs, cfg)
    fusion = build_fusion(cfg["tracker"].get("fusion", "passthrough"), len(trackers))
    return trackers, fusion


def build_from_benchmark_section(cfg: dict) -> tuple[list[BaseTracker], BaseFusionAlgorithm]:
    """Convenience: build trackers + fusion from cfg['benchmark'].

    Used by the offline benchmark (testing/benchmark.py).
    """
    specs = cfg["benchmark"]["trackers"]
    trackers = build_trackers(specs, cfg)
    fusion = build_fusion(cfg["benchmark"].get("fusion", "passthrough"), len(trackers))
    return trackers, fusion
