"""Geometric transforms expressed as 3x3 affine matrices (output = M @ input).

Boxes and the image are transformed with the same matrix, so bbox and rotated
boxes stay consistent with the pixels.
"""
from __future__ import annotations

import math

import numpy as np


def translate(tx: float, ty: float) -> np.ndarray:
    return np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64)


def scale(sx: float, sy: float | None = None) -> np.ndarray:
    return np.array([[sx, 0, 0], [0, sx if sy is None else sy, 0], [0, 0, 1]], dtype=np.float64)


def about(m: np.ndarray, cx: float, cy: float) -> np.ndarray:
    """Apply m around the point (cx, cy)."""
    return translate(cx, cy) @ m @ translate(-cx, -cy)


def rotation(deg: float) -> np.ndarray:
    """Clockwise on screen for positive deg (y axis down)."""
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def shear(deg_x: float, deg_y: float) -> np.ndarray:
    return np.array([[1, math.tan(math.radians(deg_x)), 0],
                     [math.tan(math.radians(deg_y)), 1, 0], [0, 0, 1]], dtype=np.float64)


def flip_h(w: float) -> np.ndarray:
    return np.array([[-1, 0, w], [0, 1, 0], [0, 0, 1]], dtype=np.float64)


def flip_v(h: float) -> np.ndarray:
    return np.array([[1, 0, 0], [0, -1, h], [0, 0, 1]], dtype=np.float64)


def rot90(k: int, w: float, h: float) -> tuple[np.ndarray, float, float]:
    """Rotate by k*90 deg clockwise. Returns matrix and new (w, h)."""
    k %= 4
    if k == 0:
        return np.eye(3), w, h
    if k == 1:  # (x, y) -> (h - y, x)
        return np.array([[0, -1, h], [1, 0, 0], [0, 0, 1]], dtype=np.float64), h, w
    if k == 2:
        return np.array([[-1, 0, w], [0, -1, h], [0, 0, 1]], dtype=np.float64), w, h
    return np.array([[0, 1, 0], [-1, 0, w], [0, 0, 1]], dtype=np.float64), h, w  # (x,y)->(y, w-x)


def preprocess_matrix(w: int, h: int, mode: str, tw: int, th: int) -> tuple[np.ndarray, int, int]:
    """Resize preprocessing. mode: 'none' | 'stretch' | 'fit' (letterbox, keep aspect, pad black)."""
    if mode == "stretch":
        return scale(tw / w, th / h), tw, th
    if mode == "fit":
        r = min(tw / w, th / h)
        nw, nh = w * r, h * r
        return translate((tw - nw) / 2, (th - nh) / 2) @ scale(r), tw, th
    return np.eye(3), w, h


def sample_geometric(cfg, w: int, h: int, rng: np.random.Generator) -> tuple[np.ndarray, int, int]:
    """Randomly compose the enabled geometric augmentations. Returns (M, out_w, out_h)."""
    m = np.eye(3)
    cw, ch = float(w), float(h)

    if cfg.flip_h and rng.random() < 0.5:
        m = flip_h(cw) @ m
    if cfg.flip_v and rng.random() < 0.5:
        m = flip_v(ch) @ m
    if cfg.rot90:
        k = int(rng.integers(0, 4))
        r, cw, ch = rot90(k, cw, ch)
        m = r @ m
    if cfg.rotate > 0:
        m = about(rotation(rng.uniform(-cfg.rotate, cfg.rotate)), cw / 2, ch / 2) @ m
    if cfg.shear > 0:
        m = about(shear(rng.uniform(-cfg.shear, cfg.shear), rng.uniform(-cfg.shear, cfg.shear)),
                  cw / 2, ch / 2) @ m
    if cfg.scale > 0:
        s = rng.uniform(1 - cfg.scale / 100, 1 + cfg.scale / 100)
        m = about(scale(s), cw / 2, ch / 2) @ m
    if cfg.crop > 0:
        # crop a random window covering (1-p) of each side, then stretch it back to full size (zoom in)
        p = rng.uniform(0, cfg.crop / 100)
        kw, kh = cw * (1 - p), ch * (1 - p)
        x0, y0 = rng.uniform(0, cw - kw), rng.uniform(0, ch - kh)
        m = scale(cw / kw, ch / kh) @ translate(-x0, -y0) @ m
    return m, int(round(cw)), int(round(ch))
