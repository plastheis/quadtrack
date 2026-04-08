from __future__ import annotations

import sys
import time

import cv2
import numpy as np
import yaml

from core.frame import Frame


def _build_gstreamer_pipeline(cfg: dict) -> str:
    """
    Return a GStreamer pipeline string for CSI camera backends.

    If 'gstreamer_pipeline' is set in the camera config, it is returned as-is.
    Otherwise raise NotImplementedError — implement platform-specific construction
    here when migrating to the SBC, or supply the string directly in config.yaml.
    """
    if "gstreamer_pipeline" in cfg:
        return cfg["gstreamer_pipeline"]
    raise NotImplementedError(
        "CSI backend requires either a 'gstreamer_pipeline' string in config.yaml "
        "or a platform-specific implementation of _build_gstreamer_pipeline()."
    )


class Camera:
    """Abstracts USB webcam and CSI camera backends behind a single interface.

    Backend is selected via config.yaml camera.backend ('usb' or 'csi').
    """

    def __init__(self, config_path: str = "config.yaml") -> None:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        self._cfg: dict = config["camera"]
        self._backend: str = self._cfg["backend"].strip().lower()
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        if self._backend == "usb":
            self._cap = cv2.VideoCapture(self._cfg["index"])
        elif self._backend == "csi":
            pipeline = _build_gstreamer_pipeline(self._cfg)
            self._cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        else:
            raise ValueError(f"Unknown camera backend: {self._backend!r}")

        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open camera (backend={self._backend!r})")

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._cfg["width"])
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._cfg["height"])
        self._cap.set(cv2.CAP_PROP_FPS,          self._cfg["fps"])

        if self._cfg.get("autofocus", False):
            self._cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        else:
            self._cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            if "focus" in self._cfg:
                self._cap.set(cv2.CAP_PROP_FOCUS, self._cfg["focus"])

    def read(self) -> Frame:
        if self._cap is None:
            raise RuntimeError("Camera is not open. Call open() first.")
        ret, image = self._cap.read()
        if not ret:
            raise RuntimeError("Failed to read frame from camera.")
        return Frame(image=image, timestamp=time.perf_counter())

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> Camera:
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.release()
        return False


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"

    with Camera(config_path) as cam:
        print("Camera opened. Press 'q' to quit.")
        while True:
            frame = cam.read()
            cv2.imshow("QuadTrack - Camera Test", frame.image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()
