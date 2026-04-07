# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

QuadTrack is a real-time quadrotor tracking system that combines computer vision, sensor fusion, and flight controller communication. It is implemented in Python.

## Module Architecture

Five modules with distinct responsibilities:

- **tracker/** — Core tracking engine. `camera.py` handles frame capture; `kcf_tracker.py` and `nanotrack_tracker.py` implement two tracking algorithms (KCF and NanoTrack respectively); `fusion.py` fuses multiple tracking inputs; `utils.py` holds shared helpers.
- **flight-interface/** — Serial bridge to the flight controller. `uart.py` handles UART communication.
- **ground_station/** — Operator-facing layer. `gui.py` is the control interface; `stream.py` handles video/telemetry streaming.
- **benchmarks/** — Offline tooling. `benchmark.py` evaluates tracker performance; `calibrate-camera.py` calibrates camera intrinsics.
- **models/** — Pre-trained weights. `models/YOLO/` for YOLO detection models; `models/nanotrack/` for NanoTrack ONNX weights (git-ignored via `*.onnx`).

## Data Flow

```
Camera frames → tracker/camera.py → kcf_tracker.py ──────────────────┐
                                  → nanotrack_tracker.py ─────────────→ fusion.py → ground_station/{stream,gui}.py → Operator
flight-interface/uart.py (flight controller state) ───────────────────┘
```

## Configuration

`config.yaml` at the repo root is the single configuration file for runtime parameters (camera index, serial port, tracker selection, etc.).

## Dependencies

Python with OpenCV (cv2), PySerial, and ONNX Runtime expected. Model weights (`.onnx`) are excluded from git — download separately.

See `requirements.txt` for pinned versions. Install with `pip install -r requirements.txt`.

## Hardware Platforms

**Current (dev):** Windows PC, NVIDIA GPU, Logitech C920 USB webcam (index 0).

**Future (production):** Linux SBC with NPU, CSI camera module.

The codebase is designed so that only `config.yaml` needs to change between platforms — no code changes required:

| Setting | Windows (now) | Linux SBC (future) |
|---|---|---|
| `camera.backend` | `usb` | `csi` |
| `camera.index` | `0` | *(ignored)* |
| `camera.gstreamer_pipeline` | *(unused)* | set to platform pipeline string |
| `inference.device` | `cuda` | `npu` |
| `serial.port` | `COM3` | `/dev/ttyUSB0` or `/dev/ttyAMA0` |
| `onnxruntime` package | `onnxruntime-gpu` | vendor NPU wheel |

For the CSI backend, either set `camera.gstreamer_pipeline` in `config.yaml` directly, or implement `_build_gstreamer_pipeline()` in `tracker/camera.py`.
