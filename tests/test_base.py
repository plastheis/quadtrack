import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.bbox import BBox
from trackers.base import BaseTracker, TrackResult


class _Stub(BaseTracker):
    def init(self, frame, bbox): pass
    def update(self, frame): return TrackResult(BBox(0,0,1,1), 1.0, 0.0, "stub")
    def name(self): return "stub"


def test_result_age_default_zero():
    assert _Stub().result_age == 0


def test_is_async_default_false():
    assert _Stub().is_async is False


def test_async_submit_strategy_default():
    assert _Stub().async_submit_strategy == "on_completion"


def test_async_min_interval_default():
    assert _Stub().async_min_interval == 1
