from __future__ import annotations

import json

from ..model import OBB
from .base import BaseWriter, clipped_aabb, clipped_corners


class CocoWriter(BaseWriter):
    """COCO detection JSON per split (`<split>/_annotations.coco.json`).

    Standard fields: bbox [x,y,w,h], segmentation = 4-corner polygon, area.
    Extra fields for rotated boxes: "rbox": [cx, cy, w, h, angle_deg] and "kind".
    Category ids are 1-based (category id = class index + 1).
    """

    def __init__(self, out_dir, class_names):
        super().__init__(out_dir, class_names)
        self.data: dict[str, dict] = {}

    def _split(self, split):
        if split not in self.data:
            self.data[split] = {
                "info": {"description": "exported by rotalabel"},
                "categories": [{"id": i + 1, "name": n, "supercategory": "none"}
                               for i, n in enumerate(self.class_names)],
                "images": [], "annotations": [],
            }
        return self.data[split]

    def write_labels(self, split, filename, width, height, anns):
        d = self._split(split)
        img_id = len(d["images"]) + 1
        d["images"].append({"id": img_id, "file_name": filename, "width": width, "height": height})
        for a in anns:
            x0, y0, x1, y1 = clipped_aabb(a, width, height)
            if x1 - x0 <= 0 or y1 - y0 <= 0:
                continue
            entry = {
                "id": len(d["annotations"]) + 1,
                "image_id": img_id,
                "category_id": a.cls + 1,
                "bbox": [round(x0, 2), round(y0, 2), round(x1 - x0, 2), round(y1 - y0, 2)],
                "segmentation": [[round(float(v), 2) for xy in clipped_corners(a, width, height) for v in xy]],
                "area": round(a.w * a.h if a.kind == OBB else (x1 - x0) * (y1 - y0), 2),
                "iscrowd": 0,
                "kind": a.kind,
            }
            if a.kind == OBB:
                entry["rbox"] = [round(a.cx, 2), round(a.cy, 2), round(a.w, 2), round(a.h, 2), round(a.angle, 3)]
            d["annotations"].append(entry)

    def finish(self):
        for split, d in self.data.items():
            p = self.out / split
            p.mkdir(parents=True, exist_ok=True)
            (p / "_annotations.coco.json").write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
