"""A project is an image folder; metadata lives in <folder>/rotalabel_project/.

    <folder>/rotalabel_project/project.json          classes + settings
    <folder>/rotalabel_project/annotations/<img>.json one file per image
    <folder>/rotalabel_project/versions/             generated dataset versions
    <folder>/<img stem>.txt                           DOTA copy of the labels, next to each image

The DOTA side files start with the DOTA header line ``imagesource:rotalabel``; a .txt that does not
start with it was not written by us and is never overwritten.

Folders labeled before the rename use <folder>/bboxer_project/; that folder keeps being
used as-is (not moved), so existing annotations and exported data.yaml paths stay valid.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .imageio import IMAGE_EXTS
from .model import ClassDef, ImageAnnotations, default_color

META_DIRNAME = "rotalabel_project"
SIDECAR_HEADER = "imagesource:rotalabel"   # DOTA header line marking files we own

LEGACY_META_DIRNAMES = ("bboxer_project",)  # project folder name before the rename


def meta_dir_for(root: Path) -> Path:
    new = root / META_DIRNAME
    if not new.exists():
        for name in LEGACY_META_DIRNAMES:
            if (root / name).is_dir():
                return root / name
    return new


def _fmt(v: float) -> str:
    s = f"{v:.2f}".rstrip("0").rstrip(".")
    return "0" if s in ("", "-0") else s


def _natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


class Project:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.meta_dir = meta_dir_for(self.root)
        self.ann_dir = self.meta_dir / "annotations"
        self.versions_dir = self.meta_dir / "versions"
        self.classes: list[ClassDef] = []
        self.images: list[str] = []
        self.annotations: dict[str, ImageAnnotations] = {}
        self.settings: dict = {}

    # ------------------------------------------------------------------ load/save
    @classmethod
    def open(cls, root: str | Path) -> "Project":
        p = cls(root)
        p.ann_dir.mkdir(parents=True, exist_ok=True)
        meta = p.meta_dir / "project.json"
        if meta.exists():
            d = json.loads(meta.read_text(encoding="utf-8"))
            p.classes = [ClassDef(c["name"], c.get("color") or default_color(i))
                         for i, c in enumerate(d.get("classes", []))]
            p.settings = d.get("settings", {})
        p.images = sorted((f.name for f in p.root.iterdir()
                           if f.is_file() and f.suffix.lower() in IMAGE_EXTS), key=_natural_key)
        for name in p.images:
            f = p.ann_dir / f"{name}.json"
            if f.exists():
                try:
                    p.annotations[name] = ImageAnnotations.from_dict(json.loads(f.read_text(encoding="utf-8")))
                except (ValueError, KeyError):
                    pass
        if not p.classes:
            p.classes.append(ClassDef("object", default_color(0)))
            p.save_meta()
        p.sync_label_files()  # backfill / refresh the DOTA side files
        return p

    def save_meta(self) -> None:
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        d = {"version": 1, "classes": [{"name": c.name, "color": c.color} for c in self.classes],
             "settings": self.settings}
        (self.meta_dir / "project.json").write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")

    def save_annotations(self, name: str) -> None:
        ia = self.annotations.get(name)
        f = self.ann_dir / f"{name}.json"
        if ia is None or (not ia.annotations and not f.exists()):
            return
        self.ann_dir.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(ia.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8")
        self.write_label_file(name)

    # ------------------------------------------------------------------ DOTA side files
    def label_file(self, name: str) -> Path:
        """<stem>.txt next to the image; <name>.txt if another image shares the stem (a.jpg / a.png)."""
        stem = Path(name).stem
        if sum(1 for n in self.images if Path(n).stem == stem) > 1:
            return self.root / f"{name}.txt"
        return self.root / f"{stem}.txt"

    def write_label_file(self, name: str) -> bool:
        """Write the image's boxes as DOTA (pixel corners, class name, difficult). Returns False when an
        existing .txt that we did not write is in the way (it is left untouched)."""
        ia = self.annotations.get(name)
        if ia is None:
            return False
        f = self.label_file(name)
        if f.exists():
            try:
                with f.open(encoding="utf-8", errors="ignore") as fh:
                    if fh.readline().strip() != SIDECAR_HEADER:
                        return False
            except OSError:
                return False
        lines = [SIDECAR_HEADER, "gsd:null"]
        for a in ia.annotations:
            cname = self.classes[a.cls].name if 0 <= a.cls < len(self.classes) else f"class_{a.cls}"
            coords = [_fmt(float(v)) for xy in a.corners() for v in xy]
            lines.append(" ".join(coords + [cname.replace(" ", "_"), "0"]))
        try:
            f.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except OSError:
            return False
        return True

    def sync_label_files(self) -> None:
        """Rewrite the side file of every image that has saved annotations (e.g. after a class rename)."""
        for name in self.images:
            if name in self.annotations and (self.ann_dir / f"{name}.json").exists():
                self.write_label_file(name)

    # ------------------------------------------------------------------ queries
    def image_path(self, name: str) -> Path:
        return self.root / name

    def get(self, name: str) -> ImageAnnotations:
        return self.annotations.setdefault(name, ImageAnnotations())

    def box_count(self, name: str) -> int:
        ia = self.annotations.get(name)
        return len(ia.annotations) if ia else 0

    def class_names(self) -> list[str]:
        return [c.name for c in self.classes]

    # ------------------------------------------------------------------ class edits
    def add_class(self, name: str) -> int:
        self.classes.append(ClassDef(name, default_color(len(self.classes))))
        self.save_meta()
        return len(self.classes) - 1

    def class_index(self, name: str, create: bool = True) -> int:
        for i, c in enumerate(self.classes):
            if c.name == name:
                return i
        if not create:
            return -1
        return self.add_class(name)

    def delete_class(self, idx: int) -> int:
        """Remove a class, drop its boxes and shift higher ids down. Returns #boxes removed."""
        removed = 0
        del self.classes[idx]
        for name, ia in self.annotations.items():
            before = len(ia.annotations)
            ia.annotations = [a for a in ia.annotations if a.cls != idx]
            removed += before - len(ia.annotations)
            changed = before != len(ia.annotations)
            for a in ia.annotations:
                if a.cls > idx:
                    a.cls -= 1
                    changed = True
            if changed:
                self.save_annotations(name)
        self.save_meta()
        return removed
