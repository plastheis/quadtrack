"""NanoTrack accelerated tracker — ONNX Runtime (CUDA/CPU) + RKNN (NPU).

Device selection via cfg["inference"]["device"]:
  cuda  → onnxruntime CUDAExecutionProvider (requires onnxruntime-gpu)
  npu   → RKNN via rknnlite (on-device) or rknn-toolkit2 (PC sim);
          falls back to CPU ONNX Runtime if RKNN libraries absent
  cpu   → onnxruntime CPUExecutionProvider

Tracking logic is a direct Python port of OpenCV's TrackerNano C++ implementation
(opencv_contrib/modules/tracking/src/trackerNano.cpp), so behaviour matches the
working cv2.TrackerNano baseline exactly.

Key design notes:
  - Backbone requires 255×255 input (fixed ONNX shape).  For the template,
    which the reference tracker runs at 127×127, we use the centre-crop trick:
    extract template_sz pixels → resize to 255 → backbone → take centre [4:12,4:12]
    of the 16×16 feature map → 8×8 features (same as backbone(127×127)).
  - Preprocessing: raw [0,255] float32, BGR→RGB only.  No ImageNet normalisation.
    This matches cv2.dnn.blobFromImage(scale=1.0, mean=Scalar(), swapRB=True).
  - Position is updated directly (no EMA), size is updated with EMA.
    This matches OpenCV TrackerNano exactly.
"""
from __future__ import annotations

import warnings

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Hyperparameters — matched to OpenCV TrackerNano source
# ---------------------------------------------------------------------------
EXEMPLAR_SIZE    = 127       # template model input side (px)
INSTANCE_SIZE    = 255       # search  model input side (px)
TOTAL_STRIDE     = 16        # backbone spatial stride
CONTEXT_AMOUNT   = 0.5

PENALTY_K        = 0.055     # scale/ratio change penalty coefficient
LR               = 0.37      # size EMA learning rate
WINDOW_INFLUENCE = 0.455     # Hanning window blend weight
SCORE_THRESHOLD  = 0.25      # below this → tracking lost
MIN_SIZE         = 10        # minimum bbox side (px)


class NanoTrackAccel:
    """NanoTrack tracker using ONNX Runtime (CUDA/CPU) or RKNN (NPU).

    Public interface (same as KCFTracker / NanoTracker):
        init(frame, bbox)              — first-frame initialisation
        update(frame) → (bbox, ok)    — per-frame tracking step
    """

    def __init__(self, cfg: dict) -> None:
        device = cfg.get("inference", {}).get("device", "cpu").strip().lower()

        self._use_rknn = False
        self._rknn_bb  = None
        self._rknn_hd  = None
        self._sess_bb  = None
        self._sess_hd  = None

        if device == "npu":
            self._rknn_bb, self._rknn_hd = self._try_rknn(cfg)
            if self._rknn_bb is not None:
                self._use_rknn = True
                self._score_size = 16          # default; RKNN shapes not introspectable
                self._t_lo, self._t_hi = 4, 12
            else:
                warnings.warn(
                    "[NanoTrackAccel] RKNN unavailable — falling back to CPU ONNX Runtime",
                    stacklevel=2,
                )
                device = "cpu"

        if not self._use_rknn:
            self._load_onnx_sessions(cfg, device)

        self._build_hanning_and_grid()

    # ------------------------------------------------------------------
    # Session / model loading
    # ------------------------------------------------------------------

    def _load_onnx_sessions(self, cfg: dict, device: str) -> None:
        import onnxruntime as ort

        providers = (
            ["CUDAExecutionProvider"]
            if device == "cuda"
            else ["CPUExecutionProvider"]
        )

        trk = cfg["tracker"]
        bb_path = trk.get("nanotrack_accel_backbone", trk["nanotrack_backbone"])
        hd_path = trk.get("nanotrack_accel_head",     trk["nanotrack_head"])

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._sess_bb = ort.InferenceSession(bb_path, sess_options=opts, providers=providers)
        self._sess_hd = ort.InferenceSession(hd_path, sess_options=opts, providers=providers)

        # Input / output names
        self._bb_in  = self._sess_bb.get_inputs()[0].name
        self._bb_out = self._sess_bb.get_outputs()[0].name
        self._hd_in1 = self._sess_hd.get_inputs()[0].name   # template
        self._hd_in2 = self._sess_hd.get_inputs()[1].name   # search
        self._hd_cls = self._sess_hd.get_outputs()[0].name
        self._hd_loc = self._sess_hd.get_outputs()[1].name

        # Derive score_size from head output shape
        self._score_size = int(self._sess_hd.get_outputs()[0].shape[2])
        feat_size        = int(self._sess_bb.get_outputs()[0].shape[2])
        self._t_lo = feat_size // 4
        self._t_hi = feat_size - self._t_lo

        active   = self._sess_bb.get_providers()
        on_gpu   = active[0] == "CUDAExecutionProvider"
        provider = "GPU (CUDA)" if on_gpu else "CPU"
        print(f"[NanoTrackAccel] running on {provider}  score_size={self._score_size}")

    def _try_rknn(self, cfg: dict):
        trk = cfg.get("tracker", {})
        bb_path = trk.get("nanotrack_accel_backbone_rknn")
        hd_path = trk.get("nanotrack_accel_head_rknn")
        if not bb_path or not hd_path:
            return None, None

        _RKNNCls = None
        is_lite  = False
        try:
            from rknnlite.api import RKNNLite
            _RKNNCls = RKNNLite; is_lite = True
        except ImportError:
            pass
        if _RKNNCls is None:
            try:
                from rknn.api import RKNN
                _RKNNCls = RKNN
            except ImportError:
                pass
        if _RKNNCls is None:
            return None, None

        def _load(path: str):
            m = _RKNNCls()
            if m.load_rknn(path) != 0:
                raise RuntimeError(f"RKNN load failed: {path}")
            ret = m.init_runtime() if is_lite else m.init_runtime(target=None)
            if ret != 0:
                raise RuntimeError(f"RKNN init_runtime failed: {path}")
            return m

        try:
            return _load(bb_path), _load(hd_path)
        except Exception as exc:
            warnings.warn(f"[NanoTrackAccel] RKNN init failed: {exc}", stacklevel=3)
            return None, None

    # ------------------------------------------------------------------
    # Pre-computed look-up tables
    # ------------------------------------------------------------------

    def _build_hanning_and_grid(self) -> None:
        S = self._score_size

        # Hanning window (same as OpenCV createHanningWindow)
        h = np.hanning(S + 2)[1:-1].astype(np.float32)
        self._window = np.outer(h, h)   # [S, S], NOT normalised (matches OpenCV)

        # Point grid — OpenCV formula (trackerNano.cpp ::generateGrids)
        # grid[i] = (i - S//2) * TOTAL_STRIDE + INSTANCE_SIZE//2
        half = S // 2
        xs = (np.arange(S, dtype=np.float32) - half) * TOTAL_STRIDE + INSTANCE_SIZE // 2
        self._grid_x, self._grid_y = np.meshgrid(xs, xs)  # [S,S] each

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def _run_backbone(self, tensor: np.ndarray) -> np.ndarray:
        if self._use_rknn:
            out = self._rknn_bb.inference(inputs=[tensor])
            return out[0]
        return self._sess_bb.run([self._bb_out], {self._bb_in: tensor})[0]

    def _run_head(self, t_feat: np.ndarray, s_feat: np.ndarray):
        if self._use_rknn:
            out = self._rknn_hd.inference(inputs=[t_feat, s_feat])
            return out[0], out[1]
        cls, loc = self._sess_hd.run(
            [self._hd_cls, self._hd_loc],
            {self._hd_in1: t_feat, self._hd_in2: s_feat},
        )
        return cls, loc

    # ------------------------------------------------------------------
    # Image preprocessing — matches OpenCV TrackerNano exactly
    # blobFromImage(crop, scalefactor=1.0, size=Size(), mean=Scalar(), swapRB=True)
    # → float32 NCHW, RGB channel order, values in [0, 255]
    # ------------------------------------------------------------------

    @staticmethod
    def _preprocess(patch: np.ndarray) -> np.ndarray:
        rgb = patch[:, :, ::-1].astype(np.float32)   # BGR→RGB, keep [0,255]
        return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])

    # ------------------------------------------------------------------
    # Patch extraction — matches OpenCV TrackerNano ::getSubwindow
    # ------------------------------------------------------------------

    @staticmethod
    def _get_subwindow(
        img: np.ndarray,
        cx: float,
        cy: float,
        original_sz: int,
        resize_sz: int,
    ) -> np.ndarray:
        """Crop a square patch of *original_sz* pixels centred at (cx,cy),
        pad with channel-mean at boundaries, then resize to *resize_sz*."""
        avg = np.mean(img, axis=(0, 1))   # (B, G, R)
        h, w = img.shape[:2]

        c        = (original_sz + 1) // 2
        xmin     = int(cx) - c
        ymin     = int(cy) - c
        xmax     = xmin + original_sz - 1
        ymax     = ymin + original_sz - 1

        pad_l = max(0, -xmin);  pad_t = max(0, -ymin)
        pad_r = max(0, xmax - w + 1)
        pad_b = max(0, ymax - h + 1)

        xmin += pad_l;  xmax += pad_l
        ymin += pad_t;  ymax += pad_t

        if pad_l or pad_t or pad_r or pad_b:
            src = cv2.copyMakeBorder(
                img, pad_t, pad_b, pad_l, pad_r,
                cv2.BORDER_CONSTANT, value=avg,
            )
        else:
            src = img

        crop = src[ymin : ymax + 1, xmin : xmax + 1]
        return cv2.resize(crop, (resize_sz, resize_sz))

    # ------------------------------------------------------------------
    # Size calculation — OpenCV TrackerNano ::sizeCal
    # ------------------------------------------------------------------

    @staticmethod
    def _size_cal(w, h):
        pad = (w + h) * 0.5
        return np.sqrt(np.maximum((w + pad) * (h + pad), 1e-6))

    # ------------------------------------------------------------------
    # Public tracker interface
    # ------------------------------------------------------------------

    def init(self, frame: np.ndarray, bbox: tuple) -> None:
        """Initialise on *frame* with *bbox* = (x, y, w, h)."""
        x, y, bw, bh = bbox
        self._cx = float(x + bw / 2)
        self._cy = float(y + bh / 2)
        self._w  = float(bw)
        self._h  = float(bh)

        # Template crop
        # OpenCV extracts sz pixels and runs backbone at EXEMPLAR_SIZE (127).
        # Since our backbone requires INSTANCE_SIZE (255), we instead extract
        # sx = sz × (255/127) pixels — the same region as the search window —
        # and centre-crop the resulting 16×16 features to 8×8.
        # Geometrically: the centre-cropped 8×8 represents sz pixels of the
        # original image at the correct scale, matching the reference template.
        s_sum = self._w + self._h
        wz    = self._w + CONTEXT_AMOUNT * s_sum
        hz    = self._h + CONTEXT_AMOUNT * s_sum
        sz    = int(np.sqrt(wz * hz))
        sx    = int(sz * (INSTANCE_SIZE / EXEMPLAR_SIZE))   # ≈ 2 × sz

        crop   = self._get_subwindow(frame, self._cx, self._cy, sx, INSTANCE_SIZE)
        feat   = self._run_backbone(self._preprocess(crop))   # [1,C,16,16]
        self._t_feat = feat[:, :, self._t_lo : self._t_hi, self._t_lo : self._t_hi]

    def update(self, frame: np.ndarray) -> tuple[tuple, bool]:
        """Track one frame.  Returns ((x,y,w,h), ok)."""
        h_img, w_img = frame.shape[:2]

        # Search window size — matches OpenCV update()
        s_sum  = self._w + self._h
        wc     = self._w + CONTEXT_AMOUNT * s_sum
        hc     = self._h + CONTEXT_AMOUNT * s_sum
        sz     = np.sqrt(wc * hc)
        scale_z = EXEMPLAR_SIZE / sz               # px-per-feature scale
        sx      = sz * (INSTANCE_SIZE / EXEMPLAR_SIZE)

        # Scale target size to search-image coordinates for penalty calc
        tw_s = self._w * scale_z
        th_s = self._h * scale_z

        crop   = self._get_subwindow(frame, self._cx, self._cy, int(sx), INSTANCE_SIZE)
        s_feat = self._run_backbone(self._preprocess(crop))   # [1,C,16,16]
        cls, loc = self._run_head(self._t_feat, s_feat)

        # --- Score (softmax, positive class) ---
        cls0 = cls[0]   # [2, S, S]
        mx   = cls0.max(axis=0, keepdims=True)
        exp  = np.exp(cls0 - mx)
        score = (exp / exp.sum(axis=0, keepdims=True))[1]   # [S, S]

        # --- Box decode (LTRB from grid — matches OpenCV predX1/X2/Y1/Y2) ---
        l, t, r, b = loc[0, 0], loc[0, 1], loc[0, 2], loc[0, 3]   # [S,S] each
        pred_x1 = self._grid_x - l
        pred_y1 = self._grid_y - t
        pred_x2 = self._grid_x + r
        pred_y2 = self._grid_y + b

        pred_w = pred_x2 - pred_x1   # [S,S] in search-image px
        pred_h = pred_y2 - pred_y1

        # --- Penalty (scale + ratio) — matches OpenCV sizeCal / elementReciprocalMax ---
        def _rmax(v):
            return np.maximum(v, 1.0 / np.maximum(v, 1e-6))

        sc = _rmax(self._size_cal(pred_w, pred_h) / self._size_cal(tw_s, th_s))
        rc = _rmax((pred_w / np.maximum(pred_h, 1e-6))
                   / (self._w   / max(self._h,   1e-6)))
        penalty = np.exp(-(sc * rc - 1.0) * PENALTY_K)

        # --- Hanning window + select best ---
        pscore    = score * penalty * (1.0 - WINDOW_INFLUENCE) + self._window * WINDOW_INFLUENCE
        best      = np.unravel_index(np.argmax(pscore), pscore.shape)
        bi, bj    = best

        best_score   = float(score[bi, bj])
        best_penalty = float(penalty[bi, bj])

        # --- Predicted centre / size in search-image space ---
        pred_xs = (pred_x1[bi, bj] + pred_x2[bi, bj]) / 2.0
        pred_ys = (pred_y1[bi, bj] + pred_y2[bi, bj]) / 2.0
        pred_bw = float(pred_w[bi, bj])
        pred_bh = float(pred_h[bi, bj])

        # --- Convert to original-image space (matches OpenCV diffXs/diffYs) ---
        # OpenCV uses integer division: instanceSize/2 = 127 (C++ int math)
        diff_x = (pred_xs - (INSTANCE_SIZE // 2)) / scale_z
        diff_y = (pred_ys - (INSTANCE_SIZE // 2)) / scale_z

        # Size in original-image space
        out_w = pred_bw / scale_z
        out_h = pred_bh / scale_z

        # --- Update state — position: direct; size: EMA (matches OpenCV) ---
        lr = best_penalty * best_score * LR

        new_cx = self._cx + diff_x
        new_cy = self._cy + diff_y
        new_w  = out_w * lr + self._w * (1.0 - lr)
        new_h  = out_h * lr + self._h * (1.0 - lr)

        # Clamp to image
        new_cx = float(np.clip(new_cx, 0, w_img))
        new_cy = float(np.clip(new_cy, 0, h_img))
        new_w  = float(np.clip(new_w,  MIN_SIZE, w_img))
        new_h  = float(np.clip(new_h,  MIN_SIZE, h_img))

        self._cx = new_cx
        self._cy = new_cy
        self._w  = new_w
        self._h  = new_h

        x = int(new_cx - new_w / 2)
        y = int(new_cy - new_h / 2)
        return (x, y, int(new_w), int(new_h)), best_score > SCORE_THRESHOLD
