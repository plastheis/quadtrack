"""BenchmarkRunner — sequences a dataset through tracker(s) and collects metrics."""
from __future__ import annotations

import math

from benchmark.datasets.base import BaseDataset, BaseSequence
from benchmark.metrics.standard import (
    AggregateStats,
    FrameRecord,
    SequenceStats,
    bbox_iou,
    compute_aggregate_stats,
    compute_sequence_stats,
)
from benchmark.writer import print_sequence_progress
from core.centroid import Centroid
from fusion_algs.base import BaseFusionAlgorithm
from trackers.base import BaseTracker


class BenchmarkRunner:
    """Runs every sequence in a dataset through a tracker (or fused tracker set).

    Protocol (matching VOT / OTB convention):
    - Initialise all trackers on the first *visible* frame (gt.exists=True).
    - Evaluate every subsequent frame in sequence order.
    - Frames before or at the init frame are skipped (not counted in metrics).
    - Invisible frames (gt.exists=False) are recorded but excluded from all
      metric calculations (they still consume tracker updates).
    """

    def __init__(
        self,
        dataset:  BaseDataset,
        trackers: list[BaseTracker],
        fusion:   BaseFusionAlgorithm,
        cfg:      dict,
    ) -> None:
        self._dataset  = dataset
        self._trackers = trackers
        self._fusion   = fusion
        self._cfg      = cfg

    def run(self) -> AggregateStats:
        """Execute the benchmark and return aggregate statistics."""
        sequences = self._dataset.sequences()
        total = len(sequences)
        tracker_names = [
            spec["algorithm"]
            for spec in self._cfg.get("benchmark", {}).get(
                "trackers",
                self._cfg.get("tracker", {}).get("algorithms", [])
            )
        ]
        device = self._cfg.get("inference", {}).get("device", "cpu")
        all_seq_stats: list[SequenceStats] = []

        print(f"\nRunning benchmark: {self._dataset.name} "
              f"[{self._cfg.get('benchmark', {}).get('split', '?')} / "
              f"{self._cfg.get('benchmark', {}).get('modality', '?')}]")
        print(f"Trackers : {' + '.join(tracker_names)}  |  Device: {device}")
        print(f"Sequences: {total}\n")

        for seq_idx, seq in enumerate(sequences):
            stats = self._run_sequence(seq, seq_idx, total, tracker_names, device)
            all_seq_stats.append(stats)
            print_sequence_progress(seq_idx, total, stats)

        return compute_aggregate_stats(
            all_seq_stats,
            tracker_names=tracker_names,
            device=device,
            dataset=self._dataset.name,
            split=self._cfg.get("benchmark", {}).get("split", "unknown"),
            modality=self._cfg.get("benchmark", {}).get("modality", "IR"),
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_sequence(
        self,
        seq:          BaseSequence,
        seq_idx:      int,
        total:        int,
        tracker_names: list[str],
        device:       str,
    ) -> SequenceStats:
        frame0, bbox0, init_idx = seq.init_frame()

        for tracker in self._trackers:
            tracker.init(frame0, bbox0)

        records: list[FrameRecord] = []

        for frame_idx, (frame, gt) in enumerate(seq):
            if frame_idx <= init_idx:
                continue  # skip frames up to and including the init frame

            # Run all trackers
            results = [t.update(frame) for t in self._trackers]
            fused = self._fusion.fuse(results)

            # Metrics
            if gt.exists and gt.bbox is not None:
                iou = bbox_iou(fused.bbox, gt.bbox)
                c_pred = Centroid.from_bbox(fused.bbox)
                c_gt   = Centroid.from_bbox(gt.bbox)
                dist   = Centroid.distance(c_pred, c_gt)
            else:
                iou  = 0.0
                dist = math.inf

            records.append(FrameRecord(
                gt_exists=gt.exists,
                pred_conf=fused.confidence,
                iou=iou,
                center_dist=dist,
                latency_s=fused.latency_s,
            ))

        return compute_sequence_stats(seq.name, records)
