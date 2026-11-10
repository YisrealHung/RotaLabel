from __future__ import annotations

from pathlib import Path

from .base import BaseWriter, clipped_corners, fmt


class DotaWriter(BaseWriter):
    """DOTA v1 format: `x1 y1 x2 y2 x3 y3 x4 y4 classname difficult` in pixels."""

    def write_labels(self, split, filename, width, height, anns):
        d = self.out / split / "labelTxt"
        d.mkdir(parents=True, exist_ok=True)
        lines = []
        for a in anns:
            coords = [fmt(v) for xy in clipped_corners(a, width, height) for v in xy]
            name = self.class_names[a.cls].replace(" ", "_")
            lines.append(" ".join(coords + [name, "0"]))
        (d / (Path(filename).stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""),
                                                         encoding="utf-8")

    def finish(self):
        (self.out / "classes.txt").write_text(
            "\n".join(n.replace(" ", "_") for n in self.class_names) + "\n", encoding="utf-8")
