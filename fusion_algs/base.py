"""Base fusion algorithm interface and the no-op passthrough implementation."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trackers.base import TrackResult


class BaseFusionAlgorithm(ABC):
    """Common interface for all tracker-fusion algorithms.

    A fusion algorithm accepts N TrackResult objects (one per active tracker)
    and returns a single fused TrackResult for downstream use.
    """

    @abstractmethod
    def fuse(self, results: list[TrackResult]) -> TrackResult:
        """Combine multiple tracker outputs into one result.

        Args:
            results: One TrackResult per active tracker, in the same order
                     as the tracker list passed to BenchmarkRunner / main loop.

        Returns:
            A single TrackResult representing the fused estimate.
        """


class PassthroughFusion(BaseFusionAlgorithm):
    """Identity fusion — used when exactly one tracker is active.

    Passes results[0] through unchanged.  build_fusion() always selects this
    when n_trackers == 1, regardless of the config key.
    """

    def fuse(self, results: list[TrackResult]) -> TrackResult:
        if len(results) != 1:
            raise ValueError(
                f"PassthroughFusion expects exactly 1 tracker result, got {len(results)}. "
                "Set a real fusion algorithm (e.g. weighted_mean) in config when using "
                "multiple trackers."
            )
        return results[0]
