# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

QuadTrack is a real-time quadrotor tracking system that combines computer vision, sensor fusion, and flight controller communication. It is implemented in Python.

## Module Architecture

Five modules with distinct responsibilities:

- **tracker/** — Core tracking engine. `camera.py` handles frame capture; `kcf_tracker.py` and `nanotrack_tracker.py` implement two tracking algorithms (KCF and NanoTrack respectively); `fusion.py` fuses multiple tracking inputs; `utils.py` holds shared helpers.
- **flight-interface/** — Serial bridge to the flight controller. `uart.py` handles UART communication.
- **ground-station/** — Operator-facing layer. `gui.py` is the control interface; `stream.py` handles video/telemetry streaming.
- **benchmarks/** — Offline tooling. `benchmark.py` evaluates tracker performance; `calibrate-camera.py` calibrates camera intrinsics.
- **models/** — Pre-trained weights. `models/YOLO/` for YOLO detection models; `models/nanotrack/` for NanoTrack ONNX weights (git-ignored via `*.onnx`).

## Data Flow

```
Camera frames → tracker/camera.py → kcf_tracker.py ──────────────────┐
                                  → nanotrack_tracker.py ─────────────→ fusion.py → ground-station/{stream,gui}.py → Operator
flight-interface/uart.py (flight controller state) ───────────────────┘
```

## Configuration

`config.yaml` at the repo root is the single configuration file for runtime parameters (camera index, serial port, tracker selection, etc.).

## Dependencies

Python with OpenCV (cv2), PySerial, and ONNX Runtime expected. Model weights (`.onnx`) are excluded from git — download separately.
