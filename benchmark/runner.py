"""BenchmarkRunner — sequences a dataset through tracker(s) and collects metrics."""
from __future__ import annotations

import math

from benchmark.datasets.base import BaseDataset, BaseSequence
from benchmark.metrics.standard import (
    AggregateStats,
    FrameRecord,
    SequenceStats,
    compute_aggregate_stats,
    compute_sequence_stats,
)
from benchmark.writer import print_sequence_progress
from core.centroid import Centroid
from core.iou import bbox_iou
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
        bench_cfg = self._cfg.get("benchmark", {})
        tracker_names = [
            spec["algorithm"]
            for spec in self._cfg.get("tracker", {}).get("algorithms", [])
        ]
        device = self._cfg.get("inference", {}).get("device", "cpu")
        all_seq_stats: list[SequenceStats] = []

        print(f"\nRunning benchmark: {self._dataset.name} "
              f"[{bench_cfg.get('split', '?')} / "
              f"{bench_cfg.get('modality', '?')}]")
        print(f"Trackers : {' + '.join(tracker_names)}  |  Device: {device}")
        print(f"Sequences: {total}\n")

        visualize = bench_cfg.get("visualize", False)

        visualizer = None
        if visualize:
            from benchmark.visualizer import BenchmarkVisualizer
            visualizer = BenchmarkVisualizer(
                width=bench_cfg.get("visualize_width", 800),
                height=bench_cfg.get("visualize_height", 600),
            )

        try:
            for seq_idx, seq in enumerate(sequences):
                stats, quit_requested = self._run_sequence(
                    seq, seq_idx, total, tracker_names, device, visualizer
                )
                all_seq_stats.append(stats)
                print_sequence_progress(seq_idx, total, stats)
                if quit_requested:
                    break
        finally:
            if visualizer is not None:
                visualizer.close()

        return compute_aggregate_stats(
            all_seq_stats,
            tracker_names=tracker_names,
            device=device,
            dataset=self._dataset.name,
            split=bench_cfg.get("split", "unknown"),
            modality=bench_cfg.get("modality", "IR"),
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_sequence(
        self,
        seq:           BaseSequence,
        seq_idx:       int,
        total:         int,
        tracker_names: list[str],
        device:        str,
        visualizer,                  # BenchmarkVisualizer | None
    ) -> tuple[SequenceStats, bool]:
        """Run one sequence; returns (stats, quit_requested)."""
        frame0, bbox0, init_idx = seq.init_frame()

        for tracker in self._trackers:
            tracker.init(frame0, bbox0)

        records: list[FrameRecord] = []
        n_frames = len(seq)

        # Running stats for the visualiser overlay
        running_iou_sum   = 0.0
        running_iou_count = 0

        quit_requested = False

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
                running_iou_sum   += iou
                running_iou_count += 1
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

            if visualizer is not None:
                running_iou = (
                    running_iou_sum / running_iou_count
                    if running_iou_count > 0 else 0.0
                )
                skip, quit_requested = visualizer.show(
                    frame_img=frame.image,
                    gt=gt,
                    result=fused,
                    seq_name=seq.name,
                    frame_idx=frame_idx,
                    n_frames=n_frames,
                    running_iou=running_iou,
                    iou=iou,
                    fps=seq.fps,
                )
                if skip or quit_requested:
                    break

        return compute_sequence_stats(seq.name, records), quit_requested
