"""A project is an image folder; metadata lives in <folder>/rotalabel_project/.

    <folder>/rotalabel_project/project.json          classes + settings
    <folder>/rotalabel_project/annotations/<img>.json one file per image
    <folder>/rotalabel_project/versions/             generated dataset versions
    <folder>/<img stem>.txt                           DOTA copy of the labels, next to each image
                                                      (or in settings["label_dir"], see below)

The DOTA side files start with the DOTA header line ``imagesource:rotalabel``; a .txt that does not
start with it was not written by us and is never overwritten.

Label output folder: ``settings["label_dir"]`` (relative to the image folder, or absolute) sends the side
files to a folder of their own, in ``settings["label_format"]`` ("dota" or "yolo_obb"). That folder belongs
to RotaLabel, so its files are always rewritten; YOLO-OBB output also gets a classes.txt there.
Without label_dir the DOTA files go next to the images as described above.

YOLO-OBB folders: when a folder without RotaLabel annotations already holds YOLO-OBB .txt files next to
its images (e.g. from an auto-labeler), those files are loaded on open (class ids kept, names from
classes.txt) and are treated as read-only source data; edits go to ``<folder>/rotalabel_labels/`` as
YOLO-OBB (``cls x1 y1 ... x4 y4``, normalized). Format and folder can be changed later.

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
DOTA, YOLO_OBB = "dota", "yolo_obb"         # side file formats (settings["label_format"])
LABEL_FORMATS = {YOLO_OBB: "YOLO-OBB", DOTA: "DOTA"}
DEFAULT_LABEL_DIR = "rotalabel_labels"      # output folder used for adopted YOLO-OBB folders

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
        self.auto_imported: tuple[int, int] = (0, 0)  # (images, boxes) loaded from YOLO-OBB files on open

    @property
    def label_format(self) -> str:
        return self.settings.get("label_format", DOTA)

    @property
    def output_dir(self) -> Path | None:
        """Folder for the side files, or None = DOTA next to the images (default for older projects)."""
        d = self.settings.get("label_dir") or (DEFAULT_LABEL_DIR if self.label_format == YOLO_OBB else None)
        if not d:
            return None
        d = Path(d)
        return d if d.is_absolute() else self.root / d

    def set_label_output(self, fmt: str, label_dir: str | Path) -> int:
        """Change the side file format / folder and rewrite every labeled image there.
        Returns the number of files written. Files in a previous folder are left as they are."""
        if fmt not in LABEL_FORMATS:
            raise ValueError(f"unknown label format: {fmt}")
        label_dir = Path(label_dir)
        try:
            label_dir = label_dir.resolve().relative_to(self.root.resolve())  # keep it portable
        except ValueError:
            pass
        if str(label_dir) in ("", "."):
            raise ValueError("the output folder must not be the image folder itself")
        self.settings["label_format"] = fmt
        self.settings["label_dir"] = label_dir.as_posix()
        self.save_meta()
        return self.sync_label_files()

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
        if "label_format" not in p.settings and not p.annotations and p._has_yolo_obb_files():
            p._adopt_yolo_obb()
        if not p.classes:
            p.classes.append(ClassDef("object", default_color(0)))
            p.save_meta()
        p.sync_label_files()  # backfill / refresh the side files
        return p

    # ------------------------------------------------------------------ YOLO-OBB folders
    def _foreign_label_lines(self, name: str) -> list[list[str]] | None:
        """Token lines of the .txt next to an image, or None if missing / written by us as DOTA."""
        f = self._label_path(self.root, name)
        if not f.exists():
            return None
        try:
            lines = [l.split() for l in f.read_text(encoding="utf-8", errors="ignore").splitlines()]
        except OSError:
            return None
        if lines and lines[0] == [SIDECAR_HEADER]:
            return None
        return [t for t in lines if t]

    def _has_yolo_obb_files(self) -> bool:
        def is_obb_line(t):
            if len(t) != 9:
                return False
            try:
                [float(v) for v in t]
            except ValueError:
                return False
            return True
        for name in self.images:
            lines = self._foreign_label_lines(name)
            if lines and all(is_obb_line(t) for t in lines):
                return True
        return False

    def _adopt_yolo_obb(self) -> None:
        """Load the YOLO-OBB files next to the images (left untouched), keeping class ids;
        edits are written as YOLO-OBB to <folder>/rotalabel_labels/."""
        from . import importers  # local import: importers imports this module's siblings
        names = importers._read_names(self.root)
        default_only = [c.name for c in self.classes] in ([], ["object"])
        if default_only:
            self.classes = [ClassDef(n, default_color(i)) for i, n in enumerate(names or [])]
        self.settings["label_format"] = YOLO_OBB
        self.settings["label_dir"] = DEFAULT_LABEL_DIR
        self.settings["label_source"] = "."  # where the original labels came from (informational)
        self.save_meta()
        self.auto_imported = importers.import_labels(self, self.root, overwrite=True)

    def save_meta(self) -> None:
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        d = {"version": 1, "classes": [{"name": c.name, "color": c.color} for c in self.classes],
             "settings": self.settings}
        (self.meta_dir / "project.json").write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
        out = self.output_dir
        if self.label_format == YOLO_OBB and out is not None and self.classes:
            # keeps the YOLO class ids readable by other tools (and by our importer)
            out.mkdir(parents=True, exist_ok=True)
            (out / "classes.txt").write_text("\n".join(self.class_names()) + "\n", encoding="utf-8")

    def save_annotations(self, name: str, keep_empty: bool = False) -> None:
        """keep_empty: also save an image with no boxes that was never saved (e.g. an imported empty
        label file, which marks a checked negative image)."""
        ia = self.annotations.get(name)
        f = self.ann_dir / f"{name}.json"
        if ia is None or (not ia.annotations and not f.exists() and not keep_empty):
            return
        self.ann_dir.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(ia.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8")
        self.write_label_file(name)

    # ------------------------------------------------------------------ side files
    def _label_path(self, folder: Path, name: str) -> Path:
        """<stem>.txt; <name>.txt if another image shares the stem (a.jpg / a.png)."""
        stem = Path(name).stem
        if sum(1 for n in self.images if Path(n).stem == stem) > 1:
            return folder / f"{name}.txt"
        return folder / f"{stem}.txt"

    def label_file(self, name: str) -> Path:
        """Where the image's side file is written: the output folder, or next to the image."""
        return self._label_path(self.output_dir or self.root, name)

    def write_label_file(self, name: str) -> bool:
        """Write the image's boxes as DOTA (pixel corners, class name, difficult) or YOLO-OBB. Next to the
        images, returns False when an existing .txt that we did not write is in the way (it is left
        untouched); files in the output folder are always ours."""
        ia = self.annotations.get(name)
        if ia is None:
            return False
        out = self.output_dir
        if out is not None:
            try:
                out.mkdir(parents=True, exist_ok=True)
            except OSError:
                return False
        if self.label_format == YOLO_OBB:
            return self._write_yolo_obb(name, ia)
        f = self.label_file(name)
        if out is None and f.exists():
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

    def _write_yolo_obb(self, name: str, ia: ImageAnnotations) -> bool:
        from .exporters.base import clipped_corners, fmt
        if not ia.width or not ia.height:
            return False  # size unknown (image never loaded): keep the existing file
        lines = []
        for a in ia.annotations:
            coords = []
            for x, y in clipped_corners(a, ia.width, ia.height):
                coords += [fmt(x / ia.width), fmt(y / ia.height)]
            lines.append(" ".join([str(a.cls)] + coords))
        try:
            self.label_file(name).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        except OSError:
            return False
        return True

    def sync_label_files(self) -> int:
        """Rewrite the side file of every image that has saved annotations (e.g. after a class rename).
        Returns the number of files written."""
        written = 0
        for name in self.images:
            if name in self.annotations and (self.ann_dir / f"{name}.json").exists():
                written += bool(self.write_label_file(name))
        return written

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
