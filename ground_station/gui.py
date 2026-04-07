from __future__ import annotations

import cv2
import numpy as np

# Colours (BGR)
_WHITE  = (255, 255, 255)
_BLACK  = (0,   0,     0)
_GREEN  = (0,   255,   0)
_ORANGE = (0,   165, 255)
_RED    = (0,   0,   255)

_FONT       = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.55
_THICKNESS  = 1


def _text(img: np.ndarray, text: str, pos: tuple, scale: float = _FONT_SCALE) -> None:
    """Draw text with a black outline for legibility over any background."""
    x, y = pos
    cv2.putText(img, text, (x, y), _FONT, scale, _BLACK,  _THICKNESS + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), _FONT, scale, _WHITE,  _THICKNESS,     cv2.LINE_AA)


def _corner_brackets(img: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                     color: tuple, thickness: int = 2, arm: int = 18) -> None:
    """Draw L-shaped corner brackets inside a rectangle."""
    for px, py, sx, sy in [
        (x1, y1,  arm,  arm),
        (x2, y1, -arm,  arm),
        (x1, y2,  arm, -arm),
        (x2, y2, -arm, -arm),
    ]:
        cv2.line(img, (px, py), (px + sx, py),       color, thickness, cv2.LINE_AA)
        cv2.line(img, (px, py), (px,       py + sy), color, thickness, cv2.LINE_AA)


def draw_overlay(
    frame: np.ndarray,
    bbox: tuple | None,
    tracking: bool,
    roi_half: int,
    fps: float,
) -> np.ndarray:
    """Draw targeting HUD onto a copy of *frame* and return the annotated image."""
    out = frame.copy()
    h, w = out.shape[:2]
    cx, cy = w // 2, h // 2

    margin = 10

    # --- ROI corner brackets (always shown) ---
    rx1, ry1 = cx - roi_half, cy - roi_half
    rx2, ry2 = cx + roi_half, cy + roi_half

    # --- Crosshair lines — frame edge to ROI box edge only ---
    cv2.line(out, (0,  cy), (rx1, cy), _GREEN, 1, cv2.LINE_AA)  # left
    cv2.line(out, (rx2, cy), (w,  cy), _GREEN, 1, cv2.LINE_AA)  # right
    cv2.line(out, (cx,  0), (cx, ry1), _GREEN, 1, cv2.LINE_AA)  # top
    cv2.line(out, (cx, ry2), (cx,  h), _GREEN, 1, cv2.LINE_AA)  # bottom
    _corner_brackets(out, rx1, ry1, rx2, ry2, _GREEN, thickness=2, arm=max(10, roi_half // 3))

    # --- Tracking bbox (orange) with TRACKING label below, or nothing ---
    if tracking and bbox is not None:
        bx, by, bw, bh = bbox
        cv2.rectangle(out, (bx, by), (bx + bw, by + bh), _ORANGE, 2)
        label = "TRACKING"
        (tw, th), _ = cv2.getTextSize(label, _FONT, 0.55, 2)
        cv2.putText(out, label, (cx - tw // 2, ry2 + th + 6), _FONT, 0.55, _BLACK,  3, cv2.LINE_AA)
        cv2.putText(out, label, (cx - tw // 2, ry2 + th + 6), _FONT, 0.55, _ORANGE, 1, cv2.LINE_AA)

    # --- FPS (top-left) ---
    _text(out, f"{fps:.1f} FPS", (margin, margin + 18), scale=0.6)

    # --- Key legend (bottom-left) ---
    controls = ["SPACE lock on", "R  release", "[/] resize", "Q  quit"]
    line_h = 18
    y = h - margin - line_h * (len(controls) - 1)
    for text in controls:
        _text(out, text, (margin, y))
        y += line_h

    return out
