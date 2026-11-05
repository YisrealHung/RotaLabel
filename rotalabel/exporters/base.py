from __future__ import annotations

from pathlib import Path

import numpy as np

from ..model import Annotation

SPLITS = ("train", "valid", "test")


class BaseWriter:
    """Writes <out>/<split>/images/<file> and format specific labels."""

    def __init__(self, out_dir: str | Path, class_names: list[str]):
        self.out = Path(out_dir)
        self.class_names = list(class_names)
        self.used_splits: set[str] = set()

    def image_dir(self, split: str) -> Path:
        d = self.out / split / "images"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def add(self, split: str, filename: str, image_bytes: bytes, width: int, height: int,
            anns: list[Annotation]) -> None:
        self.used_splits.add(split)
        (self.image_dir(split) / filename).write_bytes(image_bytes)
        self.write_labels(split, filename, width, height, anns)

    def write_labels(self, split, filename, width, height, anns) -> None:  # pragma: no cover
        raise NotImplementedError

    def finish(self) -> None:
        pass


def clipped_corners(ann: Annotation, width: int, height: int) -> np.ndarray:
    pts = ann.corners()
    pts[:, 0] = np.clip(pts[:, 0], 0, width)
    pts[:, 1] = np.clip(pts[:, 1], 0, height)
    return pts


def clipped_aabb(ann: Annotation, width: int, height: int) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = ann.aabb()
    return max(0.0, x0), max(0.0, y0), min(float(width), x1), min(float(height), y1)


def fmt(v: float) -> str:
    return f"{v:.6f}".rstrip("0").rstrip(".") if v != 0 else "0"
