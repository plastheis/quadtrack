"""Anti-UAV dataset loader.

Supports the Anti-UAV challenge format used in:
  https://github.com/ZhaoJ9014/Anti-UAV

Expected on-disk layout:
    <dataset_root>/
        <split>/                      # e.g. train / val / test
            <sequence_name>/
                infrared.mp4          # IR video
                visible.mp4           # RGB video (optional)
                infrared.json         # IR labels
                visible.json          # RGB labels

Label JSON schema:
    {
        "gt_rect": [[x, y, w, h], …],   // top-left + size; [0,0,0,0] when absent
        "exist":   [0, 1, 1, …]         // 1 = target visible, 0 = absent
    }
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterator

import cv2

from benchmark.datasets.base import BaseDataset, BaseSequence, GroundTruthFrame
from core.bbox import BBox
from core.frame import Frame


_MODALITY_FILE = {"IR": "infrared", "RGB": "visible"}


class AntiUAVSequence(BaseSequence):
    """One sequence from the Anti-UAV / Anti-UAV410 dataset."""

    def __init__(self, seq_dir: Path, modality: str) -> None:
        self._dir = seq_dir
        self._modality = modality

        stem = _MODALITY_FILE[modality]
        label_path = seq_dir / f"{stem}.json"
        with open(label_path) as f:
            data = json.load(f)

        self._gt_rects: list[list[float]] = data["gt_rect"]
        self._exists: list[int] = data["exist"]
        self._video_path: Path = seq_dir / f"{stem}.mp4"

        n_labels = len(self._exists)
        cap = cv2.VideoCapture(str(self._video_path))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if n_frames != n_labels:
            # Truncate to the shorter of the two (some datasets have off-by-one)
            n = min(n_frames, n_labels)
            self._exists = self._exists[:n]
            self._gt_rects = self._gt_rects[:n]
        self._n_frames = min(n_frames, n_labels)

    # ------------------------------------------------------------------
    # BaseSequence interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._dir.name

    @property
    def modality(self) -> str:
        return self._modality

    def __len__(self) -> int:
        return self._n_frames

    def __iter__(self) -> Iterator[tuple[Frame, GroundTruthFrame]]:
        cap = cv2.VideoCapture(str(self._video_path))
        try:
            for exist, rect in zip(self._exists, self._gt_rects):
                ok, image = cap.read()
                if not ok:
                    break
                frame = Frame(image=image, timestamp=time.perf_counter())
                gt = self._make_gt(exist, rect)
                yield frame, gt
        finally:
            cap.release()

    def init_frame(self) -> tuple[Frame, BBox, int]:
        cap = cv2.VideoCapture(str(self._video_path))
        try:
            for idx, (exist, rect) in enumerate(zip(self._exists, self._gt_rects)):
                ok, image = cap.read()
                if not ok:
                    break
                if exist:
                    frame = Frame(image=image, timestamp=time.perf_counter())
                    bbox = BBox.from_xywh(*rect)
                    return frame, bbox, idx
        finally:
            cap.release()
        raise ValueError(f"Sequence '{self.name}' has no visible frames.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_gt(exist: int, rect: list[float]) -> GroundTruthFrame:
        if exist and any(v != 0 for v in rect):
            return GroundTruthFrame(bbox=BBox.from_xywh(*rect), exists=True)
        return GroundTruthFrame(bbox=None, exists=False)


class AntiUAVDataset(BaseDataset):
    """Anti-UAV dataset (all versions share the same directory structure)."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        modality: str = "IR",
    ) -> None:
        self._root = Path(root)
        self._split = split
        self._modality = modality

        split_dir = self._root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"Dataset split directory not found: {split_dir}\n"
                f"Make sure the dataset is downloaded to '{self._root}' "
                f"and the split '{split}' exists."
            )

        label_filename = f"{_MODALITY_FILE[modality]}.json"
        self._seq_dirs: list[Path] = sorted(
            d for d in split_dir.iterdir()
            if d.is_dir() and (d / label_filename).exists()
        )
        if not self._seq_dirs:
            raise FileNotFoundError(
                f"No sequences with '{label_filename}' found under {split_dir}"
            )

    @property
    def name(self) -> str:
        return "anti-uav"

    def sequences(self) -> list[AntiUAVSequence]:
        return [
            AntiUAVSequence(d, self._modality) for d in self._seq_dirs
        ]
