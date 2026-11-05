"""Core data model: every box is stored as (cx, cy, w, h, angle) in pixels."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np

from . import geometry

BBOX = "bbox"
OBB = "obb"

DEFAULT_COLORS = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#46f0f0",
    "#f032e6", "#bcf60c", "#fabebe", "#008080", "#e6beff", "#9a6324", "#fffac8",
    "#800000", "#aaffc3", "#808000", "#ffd8b1", "#000075", "#808080",
]


def default_color(index: int) -> str:
    return DEFAULT_COLORS[index % len(DEFAULT_COLORS)]


@dataclass
class Annotation:
    cls: int
    cx: float
    cy: float
    w: float
    h: float
    angle: float = 0.0
    kind: str = BBOX

    def corners(self) -> np.ndarray:
        return geometry.box_corners(self.cx, self.cy, self.w, self.h, self.angle)

    def aabb(self) -> tuple[float, float, float, float]:
        if self.kind == BBOX or self.angle == 0:
            return (self.cx - self.w / 2, self.cy - self.h / 2, self.cx + self.w / 2, self.cy + self.h / 2)
        return geometry.aabb(self.corners())

    def to_bbox(self) -> "Annotation":
        x0, y0, x1, y1 = self.aabb()
        return Annotation(self.cls, (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0, 0.0, BBOX)

    def to_obb(self) -> "Annotation":
        return Annotation(self.cls, self.cx, self.cy, self.w, self.h, self.angle, OBB)

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("cx", "cy", "w", "h", "angle"):
            d[k] = round(float(d[k]), 4)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Annotation":
        return cls(
            cls=int(d["cls"]), cx=float(d["cx"]), cy=float(d["cy"]), w=float(d["w"]), h=float(d["h"]),
            angle=float(d.get("angle", 0.0)), kind=d.get("kind", BBOX),
        )

    @classmethod
    def from_corners(cls, cls_id: int, pts, kind: str = OBB) -> "Annotation":
        if kind == BBOX:
            x0, y0, x1, y1 = geometry.aabb(pts)
            return cls(cls_id, (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0, 0.0, BBOX)
        p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        hint = math.degrees(math.atan2(*(p[1] - p[0])[::-1])) if len(p) >= 2 else None
        cx, cy, w, h, a = geometry.fit_obb(p, hint)
        return cls(cls_id, cx, cy, w, h, a, OBB)


@dataclass
class ClassDef:
    name: str
    color: str


@dataclass
class ImageAnnotations:
    width: int = 0
    height: int = 0
    annotations: list[Annotation] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"width": self.width, "height": self.height,
                "annotations": [a.to_dict() for a in self.annotations]}

    @classmethod
    def from_dict(cls, d: dict) -> "ImageAnnotations":
        return cls(int(d.get("width", 0)), int(d.get("height", 0)),
                   [Annotation.from_dict(a) for a in d.get("annotations", [])])
