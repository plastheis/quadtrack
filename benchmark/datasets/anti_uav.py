"""Anti-UAV dataset loader.

Supports the Anti-UAV challenge format used in:
  https://github.com/ZhaoJ9014/Anti-UAV

Expected on-disk layout:
    <dataset_root>/
        <split>/                      # e.g. train / val / test
            <sequence_name>/
                IR/                   # IR frames: 000001.jpg, 000002.jpg, …
                RGB/                  # RGB frames (optional)
                IR_label.json
                RGB_label.json

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


class AntiUAVSequence(BaseSequence):
    """One sequence from the Anti-UAV / Anti-UAV410 dataset."""

    def __init__(self, seq_dir: Path, modality: str) -> None:
        self._dir = seq_dir
        self._modality = modality

        label_path = seq_dir / f"{modality}_label.json"
        with open(label_path) as f:
            data = json.load(f)

        self._gt_rects: list[list[float]] = data["gt_rect"]
        self._exists: list[int] = data["exist"]

        # Collect sorted frame paths
        frame_dir = seq_dir / modality
        self._frame_paths: list[Path] = sorted(frame_dir.glob("*.jpg"))
        if not self._frame_paths:
            self._frame_paths = sorted(frame_dir.glob("*.png"))

        n_labels = len(self._exists)
        n_frames = len(self._frame_paths)
        if n_frames != n_labels:
            # Truncate to the shorter of the two (some datasets have off-by-one)
            self._frame_paths = self._frame_paths[:n_labels]
            self._exists = self._exists[:n_frames]
            self._gt_rects = self._gt_rects[:n_frames]

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
        return len(self._frame_paths)

    def __iter__(self) -> Iterator[tuple[Frame, GroundTruthFrame]]:
        for path, exist, rect in zip(
            self._frame_paths, self._exists, self._gt_rects
        ):
            image = cv2.imread(str(path))
            if image is None:
                raise RuntimeError(f"Failed to read frame: {path}")
            frame = Frame(image=image, timestamp=time.perf_counter())
            gt = self._make_gt(exist, rect)
            yield frame, gt

    def init_frame(self) -> tuple[Frame, BBox, int]:
        for idx, (path, exist, rect) in enumerate(
            zip(self._frame_paths, self._exists, self._gt_rects)
        ):
            if exist:
                image = cv2.imread(str(path))
                if image is None:
                    raise RuntimeError(f"Failed to read init frame: {path}")
                frame = Frame(image=image, timestamp=time.perf_counter())
                bbox = BBox.from_xywh(*rect)
                return frame, bbox, idx
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

        label_filename = f"{modality}_label.json"
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
