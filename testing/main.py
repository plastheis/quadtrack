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
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from numpy import integer

from trackers.base import BaseTracker, TrackResult

import threading
import time

import cv2
import yaml

from trackers.factory import build_from_tracker_section
from core.bbox import BBox
from trackers.camera import Camera

from ground_station.gui import draw_overlay

_DEFAULT_ROI_HALF = 80
_ROI_MIN          = 20
_ROI_MAX          = 300
_ROI_STEP         = 10


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


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
    trackers, fusion = build_from_tracker_section(cfg)
    cam = Camera(config_path)
    frame_q = _start_capture_thread(cam)

    tracking    = False
    bbox: BBox | None = None
    roi_half    = _DEFAULT_ROI_HALF
    prev_time   = time.perf_counter()
    fps         = 0.
    
    # async fusion setup
    nframe = 0
    interval = 0
    if cfg["tracker"]["fusion"] == "async":
        interval = cfg["tracker"]["async_interval"]
        
    algo_names = " + ".join(
        s["algorithm"] for s in cfg["tracker"]["algorithms"]
    )
    print(f"QuadTrack started [{algo_names}]. "
          "Press SPACE to lock on, R to release, Q to quit.")

    try:
        while True:
            frame = frame_q.get()

            # FPS measurement
            now       = time.perf_counter()
            fps       = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            # Tracker step
            results = []
            if tracking:
                for t in trackers:
                    if t.is_async and nframe % interval != 0:
                        continue
                    results.append(t.update(frame))
                
                result  = fusion.fuse(results)
                bbox    = result.bbox
                nframe += 1
                if not result.confidence:
                    tracking = False
                    bbox = None
                    nframe = 0

            # Draw HUD
            display = draw_overlay(frame.image, bbox, tracking, roi_half, fps)
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
                h, w = frame.image.shape[:2]
                cx, cy = w // 2, h // 2
                bbox = BBox(cx=float(cx), cy=float(cy), w=float(roi_half * 2), h=float(roi_half * 2))
                for t in trackers:
                    t.init(frame, bbox)
                tracking = True
                nframe = 0

    finally:
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    main(config_path)
