import json

import cv2
import numpy as np
import pytest

from rotalabel import imageio
from rotalabel.augment.mosaic import (MosaicConfig, Placed, edge_mean_color, extract_patch, generate_mosaic,
                                      layout_mosaic, make_mosaic, mosaic_splits, pick_patches, random_piece,
                                      transform_patch)
from rotalabel.augment.pipeline import AugConfig
from rotalabel.augment.version import VersionConfig
from rotalabel.model import BBOX, OBB, Annotation
from rotalabel.project_io import Project


def _inside(a, w, h, tol=1.0):
    c = a.corners()
    return c[:, 0].min() >= -tol and c[:, 1].min() >= -tol and c[:, 0].max() <= w + tol and c[:, 1].max() <= h + tol


def test_extract_patch_crops_union_of_boxes():
    img = np.zeros((100, 200, 3), np.uint8)
    anns = [Annotation(0, 50, 50, 20, 10, 0, BBOX), Annotation(0, 120, 30, 20, 10, 90, OBB)]
    patch, boxes = extract_patch(img, anns, padding=5)
    # union: x 40..125, y 20..55 -> padded 35..130, 15..60
    assert patch.shape[:2] == (45, 95)
    assert boxes[0].corners() == pytest.approx(anns[0].corners() - [35, 15], abs=1e-6)
    assert boxes[1].corners().mean(axis=0) == pytest.approx([85, 15], abs=1e-6)
    # padding is clipped to the image
    patch, _ = extract_patch(img, [Annotation(0, 5, 5, 10, 10, 0, BBOX)], padding=50)
    assert patch.shape[:2] == (60, 60)
    assert extract_patch(img, [], 5) is None


def test_transform_patch_keeps_boxes_on_the_content():
    patch = np.zeros((40, 80, 3), np.uint8)
    patch[10:20, 10:30] = 255                       # white block = the box
    box = Annotation(0, 20, 15, 20, 10, 0, OBB)
    rng = np.random.default_rng(0)
    for rot in (0, 37):
        cfg = MosaicConfig(flip_h=True, flip_v=True, rot90=True, rotate=rot, scale=20)
        for _ in range(20):
            img, alpha, (b,) = transform_patch(patch, [box], cfg, rng, 1000, 1000)
            h, w = img.shape[:2]
            assert alpha.shape == (h, w) and _inside(b, w, h)
            ys, xs = np.nonzero(img[..., 0] > 128)
            assert b.cx == pytest.approx(xs.mean() + 0.5, abs=1.5)
            assert b.cy == pytest.approx(ys.mean() + 0.5, abs=1.5)


def test_similar_anns_matches_generic_transform():
    from rotalabel.augment import geometric as G
    from rotalabel.augment.mosaic import similar_anns
    from rotalabel.augment.pipeline import transform_annotations
    rng = np.random.default_rng(5)
    anns = [Annotation(0, 400, 300, 80, 30, 25, OBB), Annotation(1, 350, 330, 40, 20, 0, BBOX),
            Annotation(0, 420, 280, 60, 10, -70, OBB)]
    for _ in range(50):
        m = G.translate(*rng.uniform(-50, 50, 2)) @ G.scale(rng.uniform(0.5, 1.5)) @ \
            G.about(G.rotation(rng.uniform(-180, 180)), 400, 300)
        if rng.random() < 0.5:
            m = m @ G.flip_h(800)
        fast, ref = similar_anns(anns, m), transform_annotations(anns, m, 5000, 5000, 0.0)
        for a, b in zip(fast, ref):
            assert a.kind == b.kind and a.cls == b.cls
            ca, cb = a.corners(), b.corners()   # same rectangle, possibly another corner order
            for p in ca:
                assert np.min(np.hypot(*(cb - p).T)) < 0.05


def test_large_patch_is_scaled_to_fit():
    patch = np.zeros((500, 900, 3), np.uint8)
    img, _, (b,) = transform_patch(patch, [Annotation(0, 450, 250, 800, 400, 0, OBB)],
                                   MosaicConfig(flip_h=False, flip_v=False, rot90=False), np.random.default_rng(0),
                                   300, 200)
    assert img.shape[1] <= 300 and img.shape[0] <= 200
    assert _inside(b, img.shape[1], img.shape[0])


def _pieces(n, w=300, h=250, seed=0):
    """Patches whose background is gray and whose single box (the middle) is white."""
    patch = np.full((h, w, 3), 100, np.uint8)
    patch[h // 2 - 20:h // 2 + 20, w // 4:3 * w // 4] = 255
    box = Annotation(0, w / 2, h / 2, w / 2, 40, 0, OBB)
    rng = np.random.default_rng(seed)
    return [random_piece(patch, [box], MosaicConfig(rotate=30), rng) for _ in range(n)]


def _footprint(p: Placed, W, H):
    m = np.zeros((H, W), bool)
    h, w = p.alpha.shape
    m[p.y:p.y + h, p.x:p.x + w] = p.alpha > 0.02
    return m


def _box_mask(anns, W, H):
    m = np.zeros((H, W), np.uint8)
    for a in anns:
        cv2.fillPoly(m, [np.round(a.corners()).astype(np.int32)], 1)
    return m.astype(bool)


@pytest.mark.parametrize("overlap", [True, False])
def test_layout_never_covers_a_box(overlap):
    W, H = 900, 700
    cfg = MosaicConfig(canvas_w=W, canvas_h=H, allow_overlap=overlap, gap=2)
    placed = layout_mosaic(_pieces(14), cfg, np.random.default_rng(1))
    assert placed
    for i, p in enumerate(placed):
        assert 0 <= p.x and 0 <= p.y and p.x + p.alpha.shape[1] <= W and p.y + p.alpha.shape[0] <= H
        assert all(_inside(b, W, H) for b in p.anns)
        foot_i = _footprint(p, W, H)
        for q in placed[i + 1:]:   # drawn later = on top
            assert not (_footprint(q, W, H) & (_box_mask(p.anns, W, H) if overlap else foot_i)).any()


def test_overlap_and_shrink_fit_more_than_strict_layout():
    W, H = 900, 700                     # 14 patches of ~300x250 rotated can't all fit side by side
    strict = MosaicConfig(canvas_w=W, canvas_h=H, allow_overlap=False, min_fit_scale=100)
    loose = MosaicConfig(canvas_w=W, canvas_h=H, allow_overlap=True, min_fit_scale=40)
    n_strict = len(layout_mosaic(_pieces(14), strict, np.random.default_rng(2)))
    n_loose = len(layout_mosaic(_pieces(14), loose, np.random.default_rng(2)))
    assert n_strict < 14 and n_loose == 14
    # shrinking alone (no overlap) also places more than the strict layout
    shrink = MosaicConfig(canvas_w=W, canvas_h=H, allow_overlap=False, min_fit_scale=40)
    assert len(layout_mosaic(_pieces(14), shrink, np.random.default_rng(2))) > n_strict


def test_fixed_patch_count_is_reached():
    rng = np.random.default_rng(3)
    cfg = MosaicConfig(canvas_w=1624, canvas_h=1240, rotate=15)
    for _ in range(10):
        assert len(layout_mosaic(_pieces(12, seed=int(rng.integers(1000))), cfg, rng)) == 12


def test_patch_count_range_and_repeats_change_angle():
    cfg = MosaicConfig(patches=12, patches_jitter=2, rot90=False, rotate=0)
    assert cfg.patch_range() == (10, 14)
    assert MosaicConfig(patches=1, patches_jitter=5).patch_range() == (1, 6)
    rng = np.random.default_rng(4)
    picks = pick_patches(3, 14, cfg, rng)                 # only 3 source patches for 14 slots
    assert len(picks) == 14 and {i for i, _ in picks} == {0, 1, 2}
    for src in range(3):
        angles = [a for i, a in picks if i == src]
        assert len(angles) >= 2
        for j, a in enumerate(angles):
            for b in angles[j + 1:]:
                d = abs(a - b) % 360
                assert min(d, 360 - d) >= 30 - 1e-9
    # enough sources: each used once, no forced rotation
    picks = pick_patches(20, 12, cfg, rng)
    assert len({i for i, _ in picks}) == 12 and all(a == 0 for _, a in picks)


def test_old_min_max_settings_migrate():
    m = MosaicConfig.from_dict({"min_patches": 3, "max_patches": 8})
    assert (m.patches, m.patches_jitter, m.patch_range()) == (6, 2, (4, 8))


def test_edge_mean_color():
    p = np.zeros((10, 10, 3), np.uint8)
    p[:] = (10, 20, 30)
    p[3:7, 3:7] = 255                    # the inside doesn't count
    assert edge_mean_color([p]) == pytest.approx([10, 20, 30])


@pytest.mark.parametrize("canvas", [(800, 600), (1920, 1080)])
def test_make_mosaic_any_sizes(canvas):
    rng = np.random.default_rng(2)
    pool = []
    for w, h in ((200, 150), (640, 480), (408, 312), (1000, 900)):   # mixed sizes, one larger than a canvas side
        img = np.full((h, w, 3), 90, np.uint8)
        anns = [Annotation(0, w / 2, h / 2, w * 0.6, h * 0.3, 30, OBB)]
        pool.append(extract_patch(img, anns, 10))
    cfg = MosaicConfig(canvas_w=canvas[0], canvas_h=canvas[1], patches=4, patches_jitter=2, rotate=45, scale=20,
                       aug=AugConfig(brightness=20, hue=10))
    for scope in ("patch", "canvas"):
        cfg.photometric_scope = scope
        for _ in range(5):
            img, boxes = make_mosaic(pool, cfg, rng, grayscale=True)
            assert img.shape == (canvas[1], canvas[0], 3)
            assert boxes and all(_inside(b, *canvas) for b in boxes)
            assert np.array_equal(img[..., 0], img[..., 1])          # grayscale applied


def _project(tmp_path, n=10):
    root = tmp_path / "imgs"
    root.mkdir()
    for i in range(n):
        w, h = (300, 200) if i % 2 else (180, 240)                  # mixed sizes
        img = np.zeros((h, w, 3), np.uint8)
        cv2.rectangle(img, (40, 50), (90, 70), (0, 255, 0), -1)
        imageio.imwrite(root / f"img{i}.png", img)
    p = Project.open(root)
    for name in p.images:
        ia = p.get(name)
        ia.annotations = [Annotation(0, 65, 60, 50, 20, 0, OBB), Annotation(0, 120, 120, 40, 20, 45, OBB)]
        p.save_annotations(name)
    return p


def test_generate_mosaic(tmp_path):
    p = _project(tmp_path)
    vcfg = VersionConfig(formats=["yolo_obb", "coco_obb"])
    mcfg = MosaicConfig(canvas_w=640, canvas_h=480, count_train=6, count_valid=3, count_test=10,
                        rotate=30, aug=AugConfig(brightness=10))
    out = tmp_path / "v"
    stats = generate_mosaic(p, vcfg, mcfg, out)
    assert (stats["train"], stats["valid"], stats["test"]) == (6, 3, 10)
    assert stats["patches"] == {"train": 7, "valid": 2, "test": 1}
    imgs = sorted((out / "yolo_obb/train/images").iterdir())
    assert len(imgs) == 6 and imageio.imread(imgs[0]).shape[:2] == (480, 640)
    for f in (out / "yolo_obb/train/labels").iterdir():
        vals = np.array([[float(v) for v in l.split()[1:]] for l in f.read_text().splitlines()])
        assert len(vals) and vals.min() >= -1e-3 and vals.max() <= 1 + 1e-3
    assert len(json.loads((out / "coco_obb/annotations/test.json").read_text())["images"]) == 10
    meta = json.loads((out / "version.json").read_text(encoding="utf-8"))
    assert meta["type"] == "mosaic"
    # each split only draws from its own source images
    sp = mosaic_splits(p, vcfg)
    assert meta["splits"] == sp and not set(sp["train"]) & set(sp["test"])


def test_generate_mosaic_skips_empty_split(tmp_path):
    p = _project(tmp_path, 2)
    stats = generate_mosaic(p, VersionConfig(train=100, valid=0, test=0), MosaicConfig(count_train=2), tmp_path / "v")
    assert (stats["train"], stats["valid"], stats["test"]) == (2, 0, 0)
