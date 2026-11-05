from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields

import cv2
import numpy as np

from .. import geometry
from ..model import BBOX, Annotation
from .geometric import sample_geometric
from .photometric import apply_photometric


@dataclass
class AugConfig:
    # geometric
    flip_h: bool = False
    flip_v: bool = False
    rot90: bool = False
    rotate: float = 0.0       # +/- degrees
    shear: float = 0.0        # +/- degrees
    scale: float = 0.0        # +/- percent
    crop: float = 0.0         # max percent zoom-in crop
    # photometric
    brightness: float = 0.0   # +/- percent
    exposure: float = 0.0     # +/- percent
    contrast: float = 0.0     # +/- percent
    saturation: float = 0.0   # +/- percent
    hue: float = 0.0          # +/- degrees
    grayscale: float = 0.0    # probability percent
    blur: float = 0.0         # max gaussian sigma in px
    noise: float = 0.0        # max percent of pixels (salt & pepper)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AugConfig":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in names})

    def any_enabled(self) -> bool:
        return any(bool(v) for v in asdict(self).values())


def warp(img: np.ndarray, m: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    if np.allclose(m, np.eye(3)) and img.shape[1] == out_w and img.shape[0] == out_h:
        return img
    return cv2.warpAffine(img, m[:2], (out_w, out_h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


def transform_annotation(a: Annotation, m: np.ndarray, out_w: int, out_h: int,
                         min_visibility: float = 0.3) -> Annotation | None:
    """Transform a box through affine ``m``; clip to the output image.

    Returns None when less than ``min_visibility`` of the box stays inside.
    bbox -> bounding rect of transformed corners.  obb -> min-area rect.
    """
    tc = geometry.transform_points(a.corners(), m)
    if a.kind == BBOX:
        x0, y0, x1, y1 = geometry.aabb(tc)
        full = (x1 - x0) * (y1 - y0)
        cx0, cy0, cx1, cy1 = max(0.0, x0), max(0.0, y0), min(float(out_w), x1), min(float(out_h), y1)
        if full <= 0 or cx1 - cx0 < 1 or cy1 - cy0 < 1:
            return None
        if (cx1 - cx0) * (cy1 - cy0) / full < min_visibility:
            return None
        return Annotation(a.cls, (cx0 + cx1) / 2, (cy0 + cy1) / 2, cx1 - cx0, cy1 - cy0, 0.0, BBOX)

    full = geometry.polygon_area(tc)
    if full <= 0:
        return None
    clipped = geometry.clip_polygon(tc, out_w, out_h)
    if len(clipped) < 3:
        return None
    vis = geometry.polygon_area(clipped) / full
    if vis < min_visibility:
        return None
    d = tc[1] - tc[0]
    hint = math.degrees(math.atan2(d[1], d[0]))
    cx, cy, w, h, ang = geometry.fit_obb(tc if vis > 0.999 else clipped, hint)
    if w < 1 or h < 1:
        return None
    return Annotation(a.cls, cx, cy, w, h, ang, a.kind)


def transform_annotations(anns, m, out_w, out_h, min_visibility=0.3) -> list[Annotation]:
    out = []
    for a in anns:
        t = transform_annotation(a, m, out_w, out_h, min_visibility)
        if t is not None:
            out.append(t)
    return out


def augment(img: np.ndarray, anns: list[Annotation], cfg: AugConfig, rng: np.random.Generator,
            pre: np.ndarray | None = None, min_visibility: float = 0.3,
            size: tuple[int, int] | None = None) -> tuple[np.ndarray, list[Annotation]]:
    """Apply an optional preprocessing matrix ``pre`` (with output ``size``) followed by
    random augmentation, warping the image only once."""
    h, w = img.shape[:2]
    if pre is None:
        pre = np.eye(3)
    pw, ph = size if size else (w, h)
    g, ow, oh = sample_geometric(cfg, pw, ph, rng)
    m = g @ pre
    out = warp(img, m, ow, oh)
    out = apply_photometric(out, cfg, rng)
    return out, transform_annotations(anns, m, ow, oh, min_visibility)
