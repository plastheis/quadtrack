"""Console output and JSON file writer for benchmark results."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from benchmark.metrics.standard import AggregateStats, SequenceStats


# ---------------------------------------------------------------------------
# Filename
# ---------------------------------------------------------------------------

def build_output_filename(stats: AggregateStats, timestamp: str) -> str:
    """Build the result filename.

    Format: ``{trackers}_{device}_{dataset}_{split}_{modality}_{timestamp}.json``

    Multiple trackers are joined with '+'.  Spaces are replaced with '-'.
    """
    tracker_str = "+".join(stats.tracker_names).replace(" ", "-")
    parts = [
        tracker_str,
        stats.device,
        stats.dataset,
        stats.split,
        stats.modality,
        timestamp,
    ]
    return "_".join(parts) + ".json"


# ---------------------------------------------------------------------------
# JSON writer
# ---------------------------------------------------------------------------

def write_json(
    stats:      AggregateStats,
    output_dir: Path,
    timestamp:  str,
) -> Path:
    """Serialise AggregateStats to a JSON file and return its path."""
    filename = build_output_filename(stats, timestamp)
    out_path = output_dir / filename

    payload = {
        "metadata": {
            "trackers":  stats.tracker_names,
            "device":    stats.device,
            "dataset":   stats.dataset,
            "split":     stats.split,
            "modality":  stats.modality,
            "timestamp": datetime.strptime(timestamp, "%Y%m%dT%H%M%S").isoformat(),
        },
        "aggregate": {
            "n_sequences":    stats.n_sequences,
            "auc_success":    round(stats.auc_success,    4),
            "auc_precision":  round(stats.auc_precision,  4),
            "mean_iou":       round(stats.mean_iou,       4),
            "failure_rate":   round(stats.failure_rate,   4),
            "mean_latency_s": round(stats.mean_latency_s, 6),
        },
        "sequences": [
            {
                "name":              s.name,
                "n_frames":          s.n_frames,
                "n_visible":         s.n_visible,
                "auc_success":       round(s.auc_success,    4),
                "auc_precision":     round(s.auc_precision,  4),
                "mean_iou":          round(s.mean_iou,       4),
                "failure_rate":      round(s.failure_rate,   4),
                "mean_latency_s":    round(s.mean_latency_s, 6),
                "latency_per_frame": [round(v, 6) for v in s.latency_per_frame],
            }
            for s in stats.per_sequence
        ],
    }

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    return out_path


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

def print_sequence_progress(seq_idx: int, total: int, stats: SequenceStats) -> None:
    """Print one compact per-sequence summary line."""
    name_col = stats.name[:14].ljust(14)
    lat_ms   = stats.mean_latency_s * 1000
    print(
        f"[{seq_idx + 1:>4}/{total}] {name_col}  "
        f"AUC-S={stats.auc_success:.3f}  "
        f"AUC-P={stats.auc_precision:.3f}  "
        f"IoU={stats.mean_iou:.3f}  "
        f"Fail={stats.failure_rate:.3f}  "
        f"Lat={lat_ms:.1f}ms"
    )


def print_aggregate_summary(stats: AggregateStats, output_path: Path) -> None:
    """Print the final aggregate results table."""
    sep  = "=" * 62
    dash = "-" * 62
    lat_ms = stats.mean_latency_s * 1000

    print(f"\n{sep}")
    print(f"  QuadTrack Benchmark — {stats.dataset} / {stats.split} / {stats.modality}")
    print(f"  Trackers : {' + '.join(stats.tracker_names)}")
    print(f"  Device   : {stats.device}")
    print(f"  Sequences: {stats.n_sequences}")
    print(dash)
    print(f"  AUC (Success)  :  {stats.auc_success:.4f}")
    print(f"  AUC (Precision):  {stats.auc_precision:.4f}")
    print(f"  Mean IoU       :  {stats.mean_iou:.4f}")
    print(f"  Failure Rate   :  {stats.failure_rate:.4f}")
    print(f"  Mean Latency   :  {lat_ms:.1f} ms/frame")
    print(dash)
    print(f"  Results → {output_path}")
    print(sep)
