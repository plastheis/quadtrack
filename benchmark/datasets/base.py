"""Abstract base classes for benchmark datasets."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator, TYPE_CHECKING

if TYPE_CHECKING:
    from core.bbox import BBox
    from core.frame import Frame


@dataclass(frozen=True)
class GroundTruthFrame:
    """Ground-truth annotation for a single video frame."""

    bbox:   BBox | None  # bounding box in (cx, cy, w, h); None when target absent
    exists: bool         # True when the target is visible in this frame


class BaseSequence(ABC):
    """One labelled video clip inside a dataset."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique sequence identifier (e.g. directory name)."""

    @property
    @abstractmethod
    def modality(self) -> str:
        """Sensor modality used, e.g. 'IR' or 'RGB'."""

    @abstractmethod
    def __len__(self) -> int:
        """Total number of frames in the sequence."""

    @abstractmethod
    def __iter__(self) -> Iterator[tuple[Frame, GroundTruthFrame]]:
        """Yield (Frame, GroundTruthFrame) for every frame, index 0 onwards."""

    @abstractmethod
    def init_frame(self) -> tuple[Frame, BBox, int]:
        """Return (frame, bbox, frame_index) for the first visible frame.

        The returned frame and bbox are used to initialise all trackers.
        The frame_index indicates which iteration step to skip in the runner
        (we do not evaluate the init frame itself).

        Raises:
            ValueError: if no visible frame exists in the sequence.
        """


class BaseDataset(ABC):
    """A collection of labelled tracking sequences."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short dataset identifier, e.g. 'anti-uav'."""

    @abstractmethod
    def sequences(self) -> list[BaseSequence]:
        """Return all sequences in this dataset split."""
