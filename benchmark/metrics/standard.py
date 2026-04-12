"""Standard tracking evaluation metrics.

Metrics implemented:
- IoU (Intersection over Union)
- AUC (Success):   area under the success plot (IoU threshold 0–1)
- AUC (Precision): area under the precision plot (centroid distance 0–50 px)
- Failure rate:    fraction of visible frames where the tracker reports lost
- Mean IoU:        average IoU over visible frames
- Per-frame and average latency
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from core.bbox import BBox


# ---------------------------------------------------------------------------
# Per-frame record
# ---------------------------------------------------------------------------

@dataclass
class FrameRecord:
    """One row of data accumulated during a sequence run."""

    gt_exists:   bool   # was the target visible?
    pred_conf:   float  # tracker confidence (0.0 = lost)
    iou:         float  # IoU with GT; 0.0 when GT absent or tracker lost
    center_dist: float  # centroid distance in pixels; inf when GT absent
    latency_s:   float  # wall-clock inference time for this frame


# ---------------------------------------------------------------------------
# Aggregated stats
# ---------------------------------------------------------------------------

@dataclass
class SequenceStats:
    name:              str
    n_frames:          int
    n_visible:         int           # frames where gt_exists=True
    mean_iou:          float
    failure_rate:      float         # visible frames where b1_conf==0.0
    auc_success:       float
    auc_precision:     float
    mean_latency_s:    float
    latency_per_frame: list[float]   # one entry per evaluated frame


@dataclass
class AggregateStats:
    tracker_names:  list[str]
    device:         str
    dataset:        str
    split:          str
    modality:       str
    n_sequences:    int
    mean_iou:       float
    failure_rate:   float
    auc_success:    float
    auc_precision:  float
    mean_latency_s: float
    per_sequence:   list[SequenceStats] = field(default_factory=list)


# ---------------------------------------------------------------------------
# AUC helpers
# ---------------------------------------------------------------------------

def success_auc(iou_values: list[float]) -> float:
    """Area under the success plot.

    Success plot: for each threshold t in [0, 1], the fraction of visible
    frames where IoU >= t.  The curve is integrated and normalised to [0, 1].

    Args:
        iou_values: IoU value per visible frame (gt_exists=True).  Frames
                    where the tracker reported lost contribute 0.0.

    Returns:
        Scalar AUC in [0, 1].  0 = worst, 1 = perfect.
    """
    if not iou_values:
        return 0.0
    arr = np.asarray(iou_values, dtype=float)
    thresholds = np.linspace(0.0, 1.0, 101)
    rates = np.array([(arr >= t).mean() for t in thresholds])
    # span is 1.0, so np.trapz gives the normalised AUC directly
    return float(np.trapezoid(rates, thresholds))


def precision_auc(center_distances: list[float]) -> float:
    """Area under the precision plot.

    Precision plot: for each threshold d in [0, 50] pixels, the fraction of
    visible frames where the centroid distance <= d.  Normalised to [0, 1].

    Args:
        center_distances: Pixel distance per visible frame.  Frames where the
                          tracker is lost contribute math.inf.

    Returns:
        Scalar AUC in [0, 1].  0 = worst, 1 = perfect.
    """
    if not center_distances:
        return 0.0
    arr = np.asarray(center_distances, dtype=float)
    thresholds = np.linspace(0.0, 50.0, 51)
    rates = np.array([(arr <= d).mean() for d in thresholds])
    # span is 50 px, explicit normalisation required
    return float(np.trapezoid(rates, thresholds) / 50.0)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def compute_sequence_stats(name: str, records: list[FrameRecord]) -> SequenceStats:
    """Reduce a list of FrameRecords into per-sequence statistics."""
    if not records:
        return SequenceStats(
            name=name, n_frames=0, n_visible=0,
            mean_iou=0.0, failure_rate=0.0,
            auc_success=0.0, auc_precision=0.0,
            mean_latency_s=0.0, latency_per_frame=[],
        )

    visible = [r for r in records if r.gt_exists]
    n_visible = len(visible)

    iou_vals   = [r.iou for r in visible]
    dist_vals  = [r.center_dist for r in visible]
    fail_count = sum(1 for r in visible if r.pred_conf == 0.0)

    latencies = [r.latency_s for r in records]

    return SequenceStats(
        name=name,
        n_frames=len(records),
        n_visible=n_visible,
        mean_iou=float(np.mean(iou_vals)) if iou_vals else 0.0,
        failure_rate=fail_count / n_visible if n_visible > 0 else 0.0,
        auc_success=success_auc(iou_vals),
        auc_precision=precision_auc(dist_vals),
        mean_latency_s=float(np.mean(latencies)) if latencies else 0.0,
        latency_per_frame=latencies,
    )


def compute_aggregate_stats(
    seq_stats:     list[SequenceStats],
    tracker_names: list[str],
    device:        str,
    dataset:       str,
    split:         str,
    modality:      str,
) -> AggregateStats:
    """Macro-average across sequences (each sequence weighted equally)."""
    n = len(seq_stats)
    if n == 0:
        return AggregateStats(
            tracker_names=tracker_names, device=device,
            dataset=dataset, split=split, modality=modality,
            n_sequences=0, mean_iou=0.0, failure_rate=0.0,
            auc_success=0.0, auc_precision=0.0, mean_latency_s=0.0,
        )

    return AggregateStats(
        tracker_names=tracker_names,
        device=device,
        dataset=dataset,
        split=split,
        modality=modality,
        n_sequences=n,
        mean_iou=sum(s.mean_iou for s in seq_stats) / n,
        failure_rate=sum(s.failure_rate for s in seq_stats) / n,
        auc_success=sum(s.auc_success for s in seq_stats) / n,
        auc_precision=sum(s.auc_precision for s in seq_stats) / n,
        mean_latency_s=sum(s.mean_latency_s for s in seq_stats) / n,
        per_sequence=seq_stats,
    )
