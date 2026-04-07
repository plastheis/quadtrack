"""QuadTrack — mock ground station entry point.

Run with:  python main.py [config.yaml]

This file is a local mock of the ground station view. It runs the full
camera → tracker → overlay pipeline in a single while loop. When the radio
link is added, the tracker output will be streamed via ground-station/stream.py
instead of displayed here.

Keybindings
-----------
space   Lock on: initialise tracker with the centre rectangle as ROI
r       Release: stop tracking, return to crosshair view
[       Shrink the ROI rectangle
]       Grow the ROI rectangle
q       Quit
"""
from __future__ import annotations

import queue
import sys
import threading
import time

import cv2
import yaml

from tracker.camera import Camera
from tracker.kcf_tracker import KCFTracker
from tracker.nanotrack_tracker import NanoTracker

from ground_station.gui import draw_overlay

_DEFAULT_ROI_HALF = 80
_ROI_MIN          = 20
_ROI_MAX          = 300
_ROI_STEP         = 10


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _make_tracker(cfg: dict):
    algo = cfg["tracker"]["algorithm"].strip().lower()
    if algo == "kcf":
        return KCFTracker()
    if algo == "nanotrack":
        return NanoTracker(cfg)
    raise ValueError(f"Unknown tracker algorithm: {algo!r}")


def _start_capture_thread(cam: Camera) -> queue.Queue:
    """Open *cam* and start a daemon thread that feeds the latest frame into a
    1-slot queue (stale frames are dropped so the main loop always gets the
    most recent image without blocking on I/O).
    """
    frame_q: queue.Queue = queue.Queue(maxsize=1)
    stop_evt = threading.Event()

    cam.open()

    def _run() -> None:
        while not stop_evt.is_set():
            try:
                frame = cam.read()
            except RuntimeError:
                break
            if frame_q.full():
                try:
                    frame_q.get_nowait()
                except queue.Empty:
                    pass
            frame_q.put(frame)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return frame_q


def main(config_path: str = "config.yaml") -> None:
    cfg = _load_config(config_path)
    tracker = _make_tracker(cfg)
    cam = Camera(config_path)
    frame_q = _start_capture_thread(cam)

    tracking    = False
    bbox: tuple | None = None
    roi_half    = _DEFAULT_ROI_HALF
    prev_time   = time.perf_counter()
    fps         = 0.0

    print("QuadTrack started. Press SPACE to lock on, R to release, Q to quit.")

    try:
        while True:
            frame = frame_q.get()

            # FPS measurement
            now       = time.perf_counter()
            fps       = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            # Tracker step
            if tracking:
                bbox, ok = tracker.update(frame)
                if not ok:
                    tracking = False
                    bbox = None

            # Draw HUD
            display = draw_overlay(frame, bbox, tracking, roi_half, fps)
            cv2.imshow("QuadTrack", display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            elif key == ord("r"):
                tracking = False
                bbox = None

            elif key == ord("["):
                roi_half = max(_ROI_MIN, roi_half - _ROI_STEP)

            elif key == ord("]"):
                roi_half = min(_ROI_MAX, roi_half + _ROI_STEP)

            elif key == ord(" ") and not tracking:
                h, w = frame.shape[:2]
                cx, cy = w // 2, h // 2
                bbox = (cx - roi_half, cy - roi_half, roi_half * 2, roi_half * 2)
                tracker.init(frame, bbox)
                tracking = True

    finally:
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    main(config_path)
