"""Mosaic (copy-paste) augmentation.

Every source image is cropped to the axis-aligned rectangle that holds all of its boxes (plus some
padding). For each mosaic, N +/- k of those patches are picked (reused with a different angle when the
split has fewer source images), transformed (flip / rotation / scale), recoloured and pasted onto a
canvas whose size the user chooses. The empty canvas area is filled with the mean colour of the used
patches' edges.

Fitting: patches may overlap each other's background, but never an annotation box (optional; otherwise
patches don't overlap at all). A patch that finds no free spot is shrunk step by step down to
``min_fit_scale`` percent before it is given up.

No image or canvas size is hard-coded: patches come from each image's own boxes, and a patch that is
larger than the canvas is scaled down until it fits.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .. import geometry, imageio
from ..model import BBOX, Annotation
from . import geometric as G
from .parallel import run_tasks
from .photometric import apply_photometric
from .pipeline import AugConfig, transform_annotations
from .version import (VersionConfig, main_commit, make_writers, prepare_dirs, source_names, split_items,
                      worker_export, write_meta)

SPLITS = ("train", "valid", "test")
GRID = 4                  # placement works on a 4 px grid (occupancy checked per 4x4 block, conservatively)
PLACE_TRIES = 200         # random positions tried per patch and size
LAYOUT_TRIES = 8          # whole layouts tried when not every patch could be placed
SHRINK_STEP = 0.85        # size factor per shrink step when a patch doesn't fit
REPEAT_MIN_ANGLE = 30.0   # a reused patch differs at least this many degrees from its other copies


@dataclass
class MosaicConfig:
    canvas_w: int = 1624          # initial values only; the user sets the canvas size
    canvas_h: int = 1240
    count_train: int = 100        # mosaics generated per split
    count_valid: int = 20
    count_test: int = 10
    patches: int = 12             # patches per mosaic ...
    patches_jitter: int = 2       # ... +/- this many (12 +/- 2 -> 10..14)
    padding: int = 10             # px added around the boxes when cropping a patch
    gap: int = 4                  # min px kept around boxes (overlap) / between patches (no overlap)
    allow_overlap: bool = True    # patches may cover each other's background, never a box
    min_fit_scale: float = 50     # percent; shrink a patch down to this size to make it fit (100 = never)
    flip_h: bool = True
    flip_v: bool = True
    rot90: bool = True
    rotate: float = 0.0           # +/- degrees, free rotation of each patch
    scale: float = 0.0            # +/- percent
    photometric_scope: str = "patch"   # patch: each patch recoloured on its own | canvas: whole mosaic
    aug: AugConfig = field(default_factory=AugConfig)   # only the colour / pixel fields are used

    def to_dict(self) -> dict:
        d = asdict(self)
        d["aug"] = self.aug.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "MosaicConfig":
        d = dict(d or {})
        aug = AugConfig.from_dict(d.pop("aug", {}))
        if "patches" not in d and "min_patches" in d:   # settings saved by the min~max version
            lo, hi = int(d["min_patches"]), int(d.get("max_patches", d["min_patches"]))
            d["patches"], d["patches_jitter"] = (lo + hi + 1) // 2, max(0, (hi - lo) // 2)
        names = {f.name for f in fields(cls)}
        return cls(aug=aug, **{k: v for k, v in d.items() if k in names})

    def counts(self) -> dict[str, int]:
        return {"train": self.count_train, "valid": self.count_valid, "test": self.count_test}

    def patch_range(self) -> tuple[int, int]:
        lo = max(1, int(self.patches) - max(0, int(self.patches_jitter)))
        return lo, max(lo, int(self.patches) + max(0, int(self.patches_jitter)))


# ---------------------------------------------------------------------- patches
def extract_patch(img: np.ndarray, anns: list[Annotation], padding: int = 0):
    """Crop the rectangle holding every box (+ padding, kept inside the image).
    Returns (patch, anns in patch coords) or None when the image has no usable box."""
    if not anns:
        return None
    h, w = img.shape[:2]
    pts = np.concatenate([a.corners() for a in anns])
    x0, y0, x1, y1 = geometry.aabb(pts)
    x0 = max(0, int(math.floor(x0)) - padding)
    y0 = max(0, int(math.floor(y0)) - padding)
    x1 = min(w, int(math.ceil(x1)) + padding)
    y1 = min(h, int(math.ceil(y1)) + padding)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    patch = img[y0:y1, x0:x1].copy()
    boxes = transform_annotations(anns, G.translate(-x0, -y0), x1 - x0, y1 - y0, 0.0)
    return (patch, boxes) if boxes else None


def similar_anns(anns: list[Annotation], m: np.ndarray) -> list[Annotation]:
    """Map boxes through a similarity transform (flip / rotation / uniform scale / shift).
    A rectangle stays a rectangle, so this is exact and needs no clipping or min-area-rect fitting,
    as long as the boxes end up inside the image (true for patches: they contain all their boxes)."""
    out = []
    for a in anns:
        c = geometry.transform_points(a.corners(), m)
        if a.kind == BBOX:
            x0, y0, x1, y1 = geometry.aabb(c)
            out.append(Annotation(a.cls, (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0, 0.0, BBOX))
            continue
        e1, e2 = c[1] - c[0], c[2] - c[1]
        cx, cy = c.mean(axis=0)
        ang = geometry.normalize_angle(math.degrees(math.atan2(e1[1], e1[0])))
        out.append(Annotation(a.cls, float(cx), float(cy), float(math.hypot(*e1)), float(math.hypot(*e2)), ang,
                              a.kind))
    return out


@dataclass
class Piece:
    """A patch picked for one mosaic, with its random transform."""
    patch: np.ndarray
    anns: list[Annotation]
    flip_h: bool = False
    flip_v: bool = False
    angle: float = 0.0     # degrees, clockwise
    scale: float = 1.0


def _angle_dist(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def _base_angle(cfg: MosaicConfig, rng: np.random.Generator) -> float:
    a = 90.0 * int(rng.integers(0, 4)) if cfg.rot90 else 0.0
    if cfg.rotate > 0:
        a += rng.uniform(-cfg.rotate, cfg.rotate)
    return a


def pick_patches(n_pool: int, k: int, cfg: MosaicConfig, rng: np.random.Generator) -> list[tuple[int, float]]:
    """k (pool index, angle) picks. Every patch is used once before any is reused; a reused patch gets
    an angle at least REPEAT_MIN_ANGLE away from its other copies (even when rotation is off)."""
    if n_pool <= 0 or k <= 0:
        return []
    if n_pool >= k:
        idx = [int(i) for i in rng.choice(n_pool, size=k, replace=False)]
    else:
        idx = [int(i) for i in rng.permutation(n_pool)] + [int(i) for i in rng.integers(0, n_pool, k - n_pool)]
    used: dict[int, list[float]] = {}
    out = []
    for i in idx:
        prev = used.setdefault(i, [])
        if not prev:
            ang = _base_angle(cfg, rng)
        else:
            best, best_d = 0.0, -1.0
            for _ in range(60):
                cand = rng.uniform(0, 360)
                d = min(_angle_dist(cand, p) for p in prev)
                if d > best_d:
                    best, best_d = cand, d
                if d >= REPEAT_MIN_ANGLE:
                    break
            ang = best
        prev.append(ang)
        out.append((i, ang))
    return out


def random_piece(patch, anns, cfg: MosaicConfig, rng: np.random.Generator, angle: float | None = None) -> Piece:
    fh = bool(cfg.flip_h and rng.random() < 0.5)
    fv = bool(cfg.flip_v and rng.random() < 0.5)
    s = rng.uniform(max(0.05, 1 - cfg.scale / 100), 1 + cfg.scale / 100) if cfg.scale > 0 else 1.0
    return Piece(patch, anns, fh, fv, _base_angle(cfg, rng) if angle is None else angle, s)


def piece_matrix(pc: Piece, max_w: float, max_h: float, shrink: float = 1.0) -> tuple[np.ndarray, int, int]:
    """Flip / rotate / scale the patch, shift it to start at (0, 0), scale it down if it would not fit
    into max_w x max_h, then apply ``shrink``. Returns (M, out_w, out_h)."""
    h, w = pc.patch.shape[:2]
    m = np.eye(3)
    if pc.flip_h:
        m = G.flip_h(w) @ m
    if pc.flip_v:
        m = G.flip_v(h) @ m
    m = G.scale(pc.scale) @ G.about(G.rotation(pc.angle), w / 2, h / 2) @ m
    x0, y0, x1, y1 = geometry.aabb(geometry.transform_points([[0, 0], [w, 0], [w, h], [0, h]], m))
    ow, oh = x1 - x0, y1 - y0
    f = min(1.0, max_w / ow, max_h / oh) * shrink
    m = G.scale(f) @ G.translate(-x0, -y0) @ m
    return m, max(1, int(math.ceil(ow * f - 1e-6))), max(1, int(math.ceil(oh * f - 1e-6)))


def render_piece(pc: Piece, max_w: float, max_h: float, shrink: float = 1.0):
    """Returns (image, alpha mask 0..1 float32, annotations) of the transformed patch."""
    h, w = pc.patch.shape[:2]
    m, ow, oh = piece_matrix(pc, max_w, max_h, shrink)
    img = cv2.warpAffine(pc.patch, m[:2], (ow, oh), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    mask = cv2.warpAffine(np.full((h, w), 255, np.uint8), m[:2], (ow, oh), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return img, mask.astype(np.float32) / 255.0, similar_anns(pc.anns, m)


def transform_patch(patch: np.ndarray, anns: list[Annotation], cfg: MosaicConfig,
                    rng: np.random.Generator, max_w: float, max_h: float):
    """Random transform of one patch. Returns (image, alpha, annotations)."""
    return render_piece(random_piece(patch, anns, cfg, rng), max_w, max_h)


def edge_mean_color(patches: list[np.ndarray]) -> np.ndarray:
    """Mean BGR of the 1-px border of every patch."""
    rows = []
    for p in patches:
        rows += [p[0], p[-1], p[:, 0], p[:, -1]]
    if not rows:
        return np.zeros(3, np.float32)
    return np.concatenate(rows).reshape(-1, 3).astype(np.float32).mean(axis=0)


# ---------------------------------------------------------------------- layout
@dataclass
class Placed:
    img: np.ndarray
    alpha: np.ndarray
    anns: list[Annotation]   # canvas coords
    x: int
    y: int


def _blocks(mask: np.ndarray, g: int = GRID) -> np.ndarray:
    """Bool mask -> per g x g block 'any' (conservative downsample)."""
    h, w = mask.shape
    hs, ws = -(-h // g), -(-w // g)
    pad = np.zeros((hs * g, ws * g), bool)
    pad[:h, :w] = mask
    return pad.reshape(hs, g, ws, g).any(axis=(1, 3))


class _Occupancy:
    """Protected canvas pixels (boxes, or whole patches without overlap) at full and block resolution."""

    def __init__(self, w: int, h: int):
        self.w, self.h = w, h
        self.hs, self.ws = -(-h // GRID), -(-w // GRID)
        self.full = np.zeros((self.hs * GRID, self.ws * GRID), np.uint8)
        self.small = np.zeros((self.hs, self.ws), bool)

    def free(self, foot_s: np.ndarray, gx: int, gy: int) -> bool:
        hs, ws = foot_s.shape
        return not (self.small[gy:gy + hs, gx:gx + ws] & foot_s).any()

    def _refresh(self, x0, y0, x1, y1):
        by0, bx0 = max(0, y0 // GRID), max(0, x0 // GRID)
        by1, bx1 = min(self.hs, -(-y1 // GRID)), min(self.ws, -(-x1 // GRID))
        if by1 > by0 and bx1 > bx0:
            blk = self.full[by0 * GRID:by1 * GRID, bx0 * GRID:bx1 * GRID]
            self.small[by0:by1, bx0:bx1] = blk.reshape(by1 - by0, GRID, bx1 - bx0, GRID).any(axis=(1, 3))

    def protect_boxes(self, anns: list[Annotation], gap: int):
        for a in anns:
            pts = np.round(a.corners()).astype(np.int32)
            cv2.fillPoly(self.full, [pts], 1)
            if gap > 0:
                cv2.polylines(self.full, [pts], True, 1, thickness=2 * gap + 1)
            x0, y0 = pts.min(axis=0) - gap - 1
            x1, y1 = pts.max(axis=0) + gap + 2
            self._refresh(int(x0), int(y0), int(x1), int(y1))

    def protect_footprint(self, foot: np.ndarray, x: int, y: int, gap: int):
        m = foot.astype(np.uint8)
        if gap > 0:
            m = cv2.dilate(np.pad(m, gap), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * gap + 1,) * 2))
            x, y = x - gap, y - gap
        h, w = m.shape
        sx0, sy0 = max(0, -x), max(0, -y)
        dx0, dy0 = max(0, x), max(0, y)
        dx1, dy1 = min(self.full.shape[1], x + w), min(self.full.shape[0], y + h)
        if dx1 > dx0 and dy1 > dy0:
            region = self.full[dy0:dy1, dx0:dx1]
            np.maximum(region, m[sy0:sy0 + dy1 - dy0, sx0:sx0 + dx1 - dx0], out=region)
            self._refresh(dx0, dy0, dx1, dy1)


def _shrink_levels(cfg: MosaicConfig) -> list[float]:
    floor = min(100.0, max(1.0, float(cfg.min_fit_scale))) / 100
    levels, f = [1.0], SHRINK_STEP
    while f >= floor - 1e-9:
        levels.append(f)
        f *= SHRINK_STEP
    return levels


def layout_mosaic(pieces: list[Piece], cfg: MosaicConfig, rng: np.random.Generator) -> list[Placed]:
    """Place the pieces on the canvas; returned in drawing order (later ones drawn on top).
    No piece ever covers a box of a piece placed before it; without ``allow_overlap`` pieces don't
    overlap at all. Pieces that don't fit even at the smallest size are left out."""
    W, H = int(cfg.canvas_w), int(cfg.canvas_h)
    gap = max(0, int(cfg.gap))
    levels = _shrink_levels(cfg)
    order = sorted(range(len(pieces)), key=lambda i: -pieces[i].patch.shape[0] * pieces[i].patch.shape[1])
    cache: dict = {}

    def rendered(i, f):
        if (i, f) not in cache:
            img, alpha, anns = render_piece(pieces[i], W, H, f)
            foot = alpha > 0.02
            cache[(i, f)] = (img, alpha, anns, foot, _blocks(foot))
        return cache[(i, f)]

    best: list[Placed] = []
    for _ in range(LAYOUT_TRIES):
        occ = _Occupancy(W, H)
        placed: list[Placed] = []
        for i in order:
            for f in levels:
                img, alpha, anns, foot, foot_s = rendered(i, f)
                h, w = img.shape[:2]
                if w > W or h > H:
                    continue
                gx_max, gy_max = (W - w) // GRID, (H - h) // GRID
                spot = None
                for _ in range(PLACE_TRIES):
                    gx, gy = int(rng.integers(0, gx_max + 1)), int(rng.integers(0, gy_max + 1))
                    if occ.free(foot_s, gx, gy):
                        spot = (gx * GRID, gy * GRID)
                        break
                if spot is None:
                    continue
                x, y = spot
                boxes = similar_anns(anns, G.translate(x, y))
                if cfg.allow_overlap:
                    occ.protect_boxes(boxes, gap)
                else:
                    occ.protect_footprint(foot, x, y, gap)
                placed.append(Placed(img, alpha, boxes, x, y))
                break
        if len(placed) > len(best):
            best = placed
        if len(best) == len(pieces):
            break
    return best


# ---------------------------------------------------------------------- compose
def _to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)


def make_mosaic(pool: list[tuple[np.ndarray, list[Annotation]]], cfg: MosaicConfig,
                rng: np.random.Generator, grayscale: bool = False):
    """One mosaic from a pool of (patch, anns). Returns (canvas BGR uint8, annotations)."""
    W, H = int(cfg.canvas_w), int(cfg.canvas_h)
    if not pool:
        return np.zeros((H, W, 3), np.uint8), []
    lo, hi = cfg.patch_range()
    k = int(rng.integers(lo, hi + 1))
    per_patch = cfg.photometric_scope != "canvas" and cfg.aug.any_enabled()

    pieces = []
    for i, ang in pick_patches(len(pool), k, cfg, rng):
        patch, anns = pool[i]
        if per_patch:
            patch = apply_photometric(patch, cfg.aug, rng)
        pieces.append(random_piece(patch, anns, cfg, rng, angle=ang))
    bg = edge_mean_color([p.patch for p in pieces])

    # uint8 canvas; blending is done only inside each patch's own region (much less memory traffic)
    color = np.clip(np.round(bg), 0, 255).astype(np.uint8)
    out = np.tile(np.tile(color, W), (H, 1)).reshape(H, W, 3)   # ~7x faster than out[:] = color
    boxes: list[Annotation] = []
    for p in layout_mosaic(pieces, cfg, rng):
        h, w = p.img.shape[:2]
        region = out[p.y:p.y + h, p.x:p.x + w]
        r = region.astype(np.float32)
        r += (p.img.astype(np.float32) - r) * p.alpha[..., None]
        np.copyto(region, (r + 0.5).astype(np.uint8))
        boxes += p.anns
    if cfg.photometric_scope == "canvas" and cfg.aug.any_enabled():
        out = apply_photometric(out, cfg.aug, rng)
    if grayscale:
        out = _to_gray(out)
    return out, boxes


# ---------------------------------------------------------------------- dataset
def build_pools(project, splits: dict[str, list[str]], padding: int):
    pools: dict[str, list] = {}
    for split in SPLITS:
        pool = []
        for name in splits.get(split, []):
            img = imageio.imread(project.image_path(name))
            if img is None:
                continue
            p = extract_patch(img, list(project.get(name).annotations), padding)
            if p is not None:
                pool.append(p)
        pools[split] = pool
    return pools


_worker_state: tuple | None = None   # (pools, MosaicConfig, grayscale, writers), set once per worker process


def _set_worker_state(pools, mcfg: MosaicConfig, grayscale: bool, writers: list | None = None) -> None:
    global _worker_state
    _worker_state = (pools, mcfg, grayscale, writers or [])


def _mosaic_task(task):
    """Worker: render one mosaic, write its JPG and per-image label files; returns the labels."""
    split, fname, ss = task
    pools, mcfg, grayscale, writers = _worker_state
    img, anns = make_mosaic(pools[split], mcfg, np.random.default_rng(ss), grayscale=grayscale)
    h, w = img.shape[:2]
    worker_export(writers, split, fname, imageio.encode(img, ".jpg"), w, h, anns)
    return split, fname, w, h, anns


def mosaic_splits(project, vcfg: VersionConfig) -> dict[str, list[str]]:
    """Source images per split, following the ratios / seed / skip setting of the normal page.
    Images without boxes never yield a patch, so they are always left out here."""
    names = [n for n in source_names(project, vcfg) if project.box_count(n) > 0]
    return split_items(names, vcfg)


def generate_mosaic(project, vcfg: VersionConfig, mcfg: MosaicConfig, out_dir: str | Path,
                    progress: Callable[[int, int, str], None] | None = None,
                    cancelled: Callable[[], bool] | None = None) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    splits = mosaic_splits(project, vcfg)
    if progress:
        progress(0, 1, "讀取圖塊…")
    pools = build_pools(project, splits, int(mcfg.padding))
    class_names = project.class_names()
    writers = make_writers(out, vcfg.formats, class_names)
    counts = {s: (mcfg.counts()[s] if pools[s] else 0) for s in SPLITS}
    stats = {"train": 0, "valid": 0, "test": 0, "boxes": 0,
             "source_images": sum(len(v) for v in splits.values()),
             "patches": {s: len(pools[s]) for s in SPLITS}}
    # one task per mosaic, each with its own seed: same output whatever the number of workers
    tasks = [(split, f"mosaic_{i + 1:05d}.jpg")
             for i, split in enumerate(s for s in SPLITS for _ in range(counts[s]))]
    seeds = np.random.SeedSequence(vcfg.seed + 2).spawn(len(tasks))
    tasks = [(split, fname, ss) for (split, fname), ss in zip(tasks, seeds)]
    total = len(tasks)
    if progress:
        progress(0, total, "")
    prepare_dirs(writers, {t[0] for t in tasks})
    done = 0
    for split, fname, w, h, anns in run_tasks(_mosaic_task, tasks, vcfg.workers, _set_worker_state,
                                              (pools, mcfg, vcfg.grayscale, list(writers.values())), cancelled):
        main_commit(writers, split, fname, w, h, anns)
        stats[split] += 1
        stats["boxes"] += len(anns)
        done += 1
        if progress:
            progress(done, total, fname)
    if done < total:
        stats["cancelled"] = True
        return stats
    for wr in writers.values():
        wr.finish()
    if progress:
        progress(total, total, "")
    write_meta(out, {"type": "mosaic", "config": vcfg.to_dict(), "mosaic_config": mcfg.to_dict(),
                     "stats": stats, "classes": class_names, "splits": splits})
    return stats
