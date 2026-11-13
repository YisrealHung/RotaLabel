"""Import existing label files (YOLO, YOLO-OBB, DOTA) into a project.

The format is detected per line:
  5 tokens, numeric             -> YOLO bbox (normalized)
  9 tokens, numeric             -> YOLO-OBB  (normalized corners)
  >= 9 tokens, last 1-2 textual -> DOTA      (pixel corners + class name [+ difficult])
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import imageio
from .model import BBOX, OBB, Annotation, ImageAnnotations


def _is_num(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def _read_names(label_dir: Path) -> list[str] | None:
    for cand in (label_dir / "classes.txt", label_dir.parent / "classes.txt"):
        if cand.exists():
            return [l.strip() for l in cand.read_text(encoding="utf-8").splitlines() if l.strip()]
    for cand in (label_dir / "data.yaml", label_dir.parent / "data.yaml", label_dir.parent.parent / "data.yaml"):
        if cand.exists():
            import yaml
            d = yaml.safe_load(cand.read_text(encoding="utf-8")) or {}
            names = d.get("names")
            if isinstance(names, dict):
                return [names[k] for k in sorted(names)]
            if isinstance(names, list):
                return names
    return None


def parse_label_file(path: Path, width: int, height: int, resolve_class) -> list[Annotation]:
    anns = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        t = line.split()
        if not t or t[0].startswith(("imagesource", "gsd")):
            continue
        if len(t) == 5 and all(_is_num(v) for v in t):
            c = resolve_class(int(float(t[0])))
            cx, cy, w, h = (float(v) for v in t[1:])
            anns.append(Annotation(c, cx * width, cy * height, w * width, h * height, 0.0, BBOX))
        elif len(t) == 9 and all(_is_num(v) for v in t):
            c = resolve_class(int(float(t[0])))
            pts = np.array([float(v) for v in t[1:]]).reshape(4, 2) * [width, height]
            anns.append(Annotation.from_corners(c, pts, OBB))
        elif len(t) >= 9 and all(_is_num(v) for v in t[:8]) and not _is_num(t[8]):
            c = resolve_class(t[8])
            pts = np.array([float(v) for v in t[:8]]).reshape(4, 2)
            anns.append(Annotation.from_corners(c, pts, OBB))
    return anns


def import_labels(project, label_dir: str | Path, overwrite: bool = True) -> tuple[int, int]:
    """Returns (images_updated, boxes_imported)."""
    label_dir = Path(label_dir)
    names = _read_names(label_dir)

    def resolve_class(key):
        if isinstance(key, str):
            return project.class_index(key, create=True)
        if names:
            # ids are mapped by name; ids missing from the names file get their own class
            name = names[key] if 0 <= key < len(names) else f"class_{key}"
            return project.class_index(str(name), create=True)
        while key >= len(project.classes):
            project.add_class(f"class_{len(project.classes)}")
        return key

    updated = boxes = 0
    for name in project.images:
        f = label_dir / (Path(name).stem + ".txt")
        if not f.exists():
            continue
        ia = project.get(name)
        if not ia.width or not ia.height:
            img = imageio.imread(project.image_path(name))
            if img is None:
                continue
            ia.height, ia.width = img.shape[:2]
        anns = parse_label_file(f, ia.width, ia.height, resolve_class)
        ia.annotations = anns if overwrite else ia.annotations + anns
        project.save_annotations(name, keep_empty=True)  # an empty label file is a checked negative
        updated += 1
        boxes += len(anns)
    project.save_meta()
    return updated, boxes
