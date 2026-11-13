from __future__ import annotations

import json
from pathlib import Path

from ..geometry import aabb, polygon_area
from .base import BaseWriter, clipped_corners

# our split name -> folder / json name used by PaddleDetection (PP-YOLOE-R)
SPLIT_NAMES = {"train": "train", "valid": "val", "test": "test"}


class CocoObbWriter(BaseWriter):
    """COCO-OBB (rotated COCO JSON) as used by PaddleDetection PP-YOLOE-R::

        <out>/train/images, <out>/val/images, <out>/test/images
        <out>/annotations/train.json, val.json, test.json

    Each annotation has ``segmentation`` = [[x1, y1, ..., x4, y4]] (4 clockwise corners in pixels),
    ``bbox`` = the polygon's axis-aligned [min_x, min_y, w, h], ``area`` = polygon area and a 0-based
    ``category_id`` (= class index). Plain bboxes are written as 0-degree rotated boxes.
    """

    def __init__(self, out_dir, class_names):
        super().__init__(out_dir, class_names)
        self.data: dict[str, dict] = {}

    def image_dir(self, split: str) -> Path:
        d = self.out / SPLIT_NAMES.get(split, split) / "images"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _split(self, split):
        if split not in self.data:
            self.data[split] = {
                "info": {"description": "exported by rotalabel (COCO-OBB)"},
                "categories": [{"id": i, "name": n, "supercategory": "none"}
                               for i, n in enumerate(self.class_names)],
                "images": [], "annotations": [],
            }
        return self.data[split]

    def write_labels(self, split, filename, width, height, anns):
        d = self._split(split)
        img_id = len(d["images"]) + 1
        d["images"].append({"id": img_id, "file_name": filename, "width": width, "height": height})
        for a in anns:
            pts = clipped_corners(a, width, height)   # TL, TR, BR, BL = clockwise on screen (y down)
            area = polygon_area(pts)
            x0, y0, x1, y1 = aabb(pts)
            if area <= 0 or x1 - x0 <= 0 or y1 - y0 <= 0:
                continue
            d["annotations"].append({
                "id": len(d["annotations"]) + 1,
                "image_id": img_id,
                "category_id": a.cls,
                "segmentation": [[round(float(v), 2) for xy in pts for v in xy]],
                "bbox": [round(x0, 2), round(y0, 2), round(x1 - x0, 2), round(y1 - y0, 2)],
                "area": round(area, 2),
                "iscrowd": 0,
            })

    def finish(self):
        ann_dir = self.out / "annotations"
        ann_dir.mkdir(parents=True, exist_ok=True)
        for split, d in self.data.items():
            (ann_dir / f"{SPLIT_NAMES.get(split, split)}.json").write_text(
                json.dumps(d, ensure_ascii=False), encoding="utf-8")
