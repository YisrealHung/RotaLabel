"""Roboflow-style dataset version generation: split -> preprocess -> augment xN -> export."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .. import imageio
from ..exporters import FORMATS
from ..model import Annotation
from .geometric import preprocess_matrix
from .pipeline import AugConfig, augment, transform_annotations, warp


@dataclass
class VersionConfig:
    train: float = 70
    valid: float = 20
    test: float = 10
    seed: int = 42
    skip_unannotated: bool = True
    resize_mode: str = "none"     # none | stretch | fit
    resize_w: int = 640
    resize_h: int = 640
    grayscale: bool = False       # preprocessing (applies to every split)
    multiplier: int = 3           # outputs per train/valid image, including the original (max 100)
    min_visibility: float = 30    # percent of a box that must remain after augmentation
    formats: list[str] = field(default_factory=lambda: ["yolo_obb"])
    aug: AugConfig = field(default_factory=AugConfig)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["aug"] = self.aug.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "VersionConfig":
        d = dict(d or {})
        aug = AugConfig.from_dict(d.pop("aug", {}))
        names = set(cls.__dataclass_fields__)
        return cls(aug=aug, **{k: v for k, v in d.items() if k in names})


def split_items(names: list[str], cfg: VersionConfig) -> dict[str, list[str]]:
    rng = np.random.default_rng(cfg.seed)
    order = list(names)
    rng.shuffle(order)
    total = cfg.train + cfg.valid + cfg.test or 1
    n = len(order)
    n_train = int(round(n * cfg.train / total))
    n_valid = int(round(n * cfg.valid / total))
    n_valid = min(n_valid, n - n_train)
    return {"train": order[:n_train], "valid": order[n_train:n_train + n_valid],
            "test": order[n_train + n_valid:]}


TEST_TARGET = 10   # the test split is always exported as exactly this many images (when possible)
MAX_MULTIPLIER = 100


def split_copies(splits: dict[str, list[str]], cfg: VersionConfig) -> dict[str, list[int]]:
    """Number of output images (original included) for each source image of each split.

    Train / Valid are multiplied by ``cfg.multiplier``. Test is topped up with augmented copies until it
    holds TEST_TARGET images, or trimmed to TEST_TARGET originals when the split is larger than that.
    Without any augmentation enabled every image is exported once (test is still capped)."""
    aug_on = cfg.aug.any_enabled()
    mult = max(1, min(MAX_MULTIPLIER, int(cfg.multiplier))) if aug_on else 1
    out = {s: [mult] * len(splits[s]) for s in ("train", "valid")}
    n = len(splits["test"])
    if n == 0 or n >= TEST_TARGET or not aug_on:
        out["test"] = [1] * min(n, TEST_TARGET)
    else:
        base, extra = divmod(TEST_TARGET, n)
        out["test"] = [base + (i < extra) for i in range(n)]
    return out


def _prepare(img: np.ndarray, cfg: VersionConfig):
    h, w = img.shape[:2]
    pre, pw, ph = preprocess_matrix(w, h, cfg.resize_mode, cfg.resize_w, cfg.resize_h)
    if cfg.grayscale:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    return img, pre, pw, ph


def make_sample(img: np.ndarray, anns: list[Annotation], cfg: VersionConfig,
                rng: np.random.Generator | None, augmented: bool):
    """Return (image, annotations) for one output sample."""
    img, pre, pw, ph = _prepare(img, cfg)
    vis = cfg.min_visibility / 100
    if not augmented:
        return warp(img, pre, pw, ph), transform_annotations(anns, pre, pw, ph, 0.0)
    return augment(img, anns, cfg.aug, rng, pre=pre, min_visibility=vis, size=(pw, ph))


def _output_stems(stem: str, copies: int) -> list[str]:
    return [stem] + [f"{stem}_aug{k}" for k in range(1, copies)]


def _unique_stem(stem: str, used: set[str], copies: int = 1) -> str:
    """Pick a stem whose output names (stem, stem_aug1, ...) don't clash with any already written."""
    s, i = stem, 1
    while any(n in used for n in _output_stems(s, copies)):
        i += 1
        s = f"{stem}_{i}"
    used.update(_output_stems(s, copies))
    return s


def generate_version(project, cfg: VersionConfig, out_dir: str | Path,
                     progress: Callable[[int, int, str], None] | None = None,
                     cancelled: Callable[[], bool] | None = None) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    names = [n for n in project.images
             if not cfg.skip_unannotated or project.box_count(n) > 0]
    splits = split_items(names, cfg)
    copies_of = split_copies(splits, cfg)
    splits["test"] = splits["test"][:len(copies_of["test"])]   # test capped at TEST_TARGET images
    class_names = project.class_names()
    writers = {fmt: FORMATS[fmt](out / fmt if len(cfg.formats) > 1 else out, class_names)
               for fmt in cfg.formats}
    rng = np.random.default_rng(cfg.seed + 1)
    stats = {"train": 0, "valid": 0, "test": 0, "boxes": 0, "source_images": len(names)}
    total = sum(len(v) for v in splits.values())
    done = 0
    used: set[str] = set()

    for split in ("train", "valid", "test"):
        for name, copies in zip(splits[split], copies_of[split]):
            if cancelled and cancelled():
                stats["cancelled"] = True
                return stats
            if progress:
                progress(done, total, name)
            img = imageio.imread(project.image_path(name))
            done += 1
            if img is None:
                continue
            anns = list(project.get(name).annotations)
            stem = _unique_stem(Path(name).stem, used, copies)
            for k, out_stem in enumerate(_output_stems(stem, copies)):
                out_img, out_anns = make_sample(img, anns, cfg, rng, augmented=k > 0)
                fname = f"{out_stem}.jpg"
                data = imageio.encode(out_img, ".jpg")
                h, w = out_img.shape[:2]
                for wr in writers.values():
                    wr.add(split, fname, data, w, h, out_anns)
                stats[split] += 1
                stats["boxes"] += len(out_anns)

    for wr in writers.values():
        wr.finish()
    if progress:
        progress(total, total, "")
    meta = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "config": cfg.to_dict(), "stats": stats,
            "classes": class_names, "splits": splits}
    (out / "version.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def next_version_dir(project) -> Path:
    project.versions_dir.mkdir(parents=True, exist_ok=True)
    n = 1 + sum(1 for p in project.versions_dir.iterdir() if p.is_dir() and p.name.startswith("v"))
    return project.versions_dir / f"v{n}_{time.strftime('%Y%m%d_%H%M%S')}"


def draw_annotations(img: np.ndarray, anns: list[Annotation], colors: list[str], names: list[str]) -> np.ndarray:
    out = img.copy()
    thick = max(1, round(max(out.shape[:2]) / 500))
    for a in anns:
        hexc = colors[a.cls % len(colors)] if colors else "#00ff00"
        r, g, b = (int(hexc[i:i + 2], 16) for i in (1, 3, 5))
        pts = np.round(a.corners()).astype(np.int32)
        cv2.polylines(out, [pts], True, (b, g, r), thick, cv2.LINE_AA)
        label = names[a.cls] if a.cls < len(names) else str(a.cls)
        x, y = pts[:, 0].min(), pts[:, 1].min()
        cv2.putText(out, label, (int(x), int(max(12, y - 4))), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4 * thick + 0.2, (b, g, r), thick, cv2.LINE_AA)
    return out
