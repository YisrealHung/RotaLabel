from __future__ import annotations

from pathlib import Path

import yaml

from .base import BaseWriter, clipped_aabb, clipped_corners, fmt


class YoloWriter(BaseWriter):
    """Ultralytics detect format: `cls cx cy w h` normalized. Rotated boxes use their bounding rect."""

    obb = False

    def label_path(self, split: str, filename: str) -> Path:
        d = self.out / split / "labels"
        d.mkdir(parents=True, exist_ok=True)
        return d / (Path(filename).stem + ".txt")

    def write_labels(self, split, filename, width, height, anns):
        lines = []
        for a in anns:
            if self.obb:
                pts = clipped_corners(a, width, height)
                coords = []
                for x, y in pts:
                    coords += [fmt(x / width), fmt(y / height)]
                lines.append(" ".join([str(a.cls)] + coords))
            else:
                x0, y0, x1, y1 = clipped_aabb(a, width, height)
                if x1 - x0 <= 0 or y1 - y0 <= 0:
                    continue
                lines.append(" ".join([str(a.cls), fmt((x0 + x1) / 2 / width), fmt((y0 + y1) / 2 / height),
                                       fmt((x1 - x0) / width), fmt((y1 - y0) / height)]))
        self.label_path(split, filename).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    def finish(self):
        def rel(split):
            return f"{split}/images"

        train = rel("train") if "train" in self.used_splits else None
        val = rel("valid") if "valid" in self.used_splits else None
        data = {
            "path": str(self.out.resolve()).replace("\\", "/"),
            "train": train or val,
            "val": val or train,
        }
        if "test" in self.used_splits:
            data["test"] = rel("test")
        data["nc"] = len(self.class_names)
        data["names"] = {i: n for i, n in enumerate(self.class_names)}
        if self.obb:
            data["task"] = "obb"
        (self.out / "data.yaml").write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
                                            encoding="utf-8")


class YoloObbWriter(YoloWriter):
    """Ultralytics OBB format: `cls x1 y1 x2 y2 x3 y3 x4 y4` normalized. Plain bboxes become angle-0 OBBs."""

    obb = True
