import numpy as np
import pytest

from rotalabel import geometry
from rotalabel.augment import AugConfig, augment, transform_annotations
from rotalabel.augment import geometric as G
from rotalabel.model import BBOX, OBB, Annotation


def same_rect(a: Annotation, pts, tol=0.05):
    got = a.corners()
    for p in pts:
        assert np.min(np.hypot(*(got - p).T)) < tol


def test_flip_h_obb():
    a = Annotation(0, 30, 40, 20, 10, 25, OBB)
    m = G.flip_h(100)
    (t,) = transform_annotations([a], m, 100, 100)
    expected = geometry.transform_points(a.corners(), m)
    same_rect(t, expected)
    assert t.angle == pytest.approx(-25, abs=1e-3)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_rot90_bbox_and_obb(k):
    m, w, h = G.rot90(k, 200, 100)
    assert (w, h) == ((100, 200) if k % 2 else (200, 100))
    a = Annotation(0, 50, 30, 40, 20, 0, BBOX)
    o = Annotation(1, 50, 30, 40, 20, 15, OBB)
    tb, to = transform_annotations([a, o], m, w, h)
    same_rect(tb, geometry.transform_points(a.corners(), m))
    same_rect(to, geometry.transform_points(o.corners(), m))
    assert tb.kind == BBOX and tb.angle == 0
    # the box keeps its "width" axis: angle should follow the rotation
    assert geometry._angle_dist(to.angle, 15 + 90 * k) < 1e-3
    assert to.w == pytest.approx(40, abs=1e-3)


def test_arbitrary_rotation_obb_exact_bbox_grows():
    m = G.about(G.rotation(30), 100, 100)
    o = Annotation(0, 100, 100, 40, 20, 0, OBB)
    b = Annotation(0, 100, 100, 40, 20, 0, BBOX)
    to, tb = transform_annotations([o, b], m, 200, 200)
    assert to.angle == pytest.approx(30, abs=1e-3)
    assert (to.w, to.h) == pytest.approx((40, 20), abs=1e-3)
    assert tb.w > 40 and tb.h > 20 and tb.angle == 0


def test_box_outside_dropped_and_partial_clipped():
    m = G.translate(-90, 0)
    gone = Annotation(0, 50, 50, 20, 20, 0, BBOX)            # fully leaves the image
    part = Annotation(0, 90, 50, 20, 20, 0, OBB)             # x 80..100 -> -10..10, half stays
    out = transform_annotations([gone, part], m, 100, 100, min_visibility=0.3)
    assert len(out) == 1
    assert out[0].w == pytest.approx(10, abs=1e-3)
    assert transform_annotations([part], m, 100, 100, min_visibility=0.6) == []


def test_augment_pipeline_runs_and_keeps_boxes_inside():
    rng = np.random.default_rng(0)
    img = np.full((120, 160, 3), 128, np.uint8)
    anns = [Annotation(0, 80, 60, 40, 30, 20, OBB), Annotation(1, 40, 40, 20, 20, 0, BBOX)]
    cfg = AugConfig(flip_h=True, flip_v=True, rot90=True, rotate=20, shear=5, scale=10, crop=20,
                    brightness=20, exposure=10, contrast=20, saturation=30, hue=20, grayscale=20,
                    blur=2, noise=2)
    for _ in range(30):
        out, boxes = augment(img, anns, cfg, rng)
        h, w = out.shape[:2]
        assert out.dtype == np.uint8 and {w, h} == {160, 120}
        for b in boxes:
            x0, y0, x1, y1 = b.aabb()
            assert -1 <= b.cx <= w + 1 and -1 <= b.cy <= h + 1
            if b.kind == BBOX:
                assert x0 >= -1e-6 and y0 >= -1e-6 and x1 <= w + 1e-6 and y1 <= h + 1e-6


def test_preprocess_fit_letterbox():
    m, w, h = G.preprocess_matrix(200, 100, "fit", 100, 100)
    a = Annotation(0, 100, 50, 200, 100, 0, BBOX)  # whole image
    (t,) = transform_annotations([a], m, w, h, 0)
    assert (t.cx, t.cy, t.w, t.h) == pytest.approx((50, 50, 100, 50))


def test_translate_keeps_boxes_inside_by_default():
    rng = np.random.default_rng(1)
    img = np.full((100, 200, 3), 128, np.uint8)
    anns = [Annotation(0, 30, 50, 40, 20, 0, BBOX), Annotation(1, 160, 40, 30, 20, 30, OBB)]
    cfg = AugConfig(translate_x=150, translate_y=80)
    moved = set()
    for _ in range(50):
        out, boxes = augment(img, anns, cfg, rng, min_visibility=0.0)
        assert out.shape == img.shape and len(boxes) == 2
        for a, b in zip(anns, boxes):
            # whole box still inside, size unchanged (pure shift)
            assert b.w == pytest.approx(a.w, abs=1e-3) and b.h == pytest.approx(a.h, abs=1e-3)
            x0, y0, x1, y1 = geometry.aabb(b.corners())
            assert x0 >= -1e-6 and y0 >= -1e-6 and x1 <= 200 + 1e-6 and y1 <= 100 + 1e-6
        moved.add(round(boxes[0].cx - 30, 3))
    assert len(moved) > 10  # actually shifts
    # leftmost box edge at x=10 -> may move left at most 10 px
    assert min(moved) >= -10 - 1e-6


def test_translate_allow_out_can_push_boxes_out():
    rng = np.random.default_rng(2)
    img = np.full((100, 100, 3), 128, np.uint8)
    anns = [Annotation(0, 50, 50, 20, 20, 0, BBOX)]
    cfg = AugConfig(translate_x=100, translate_allow_out=True)
    counts = [len(augment(img, anns, cfg, rng, min_visibility=0.5)[1]) for _ in range(100)]
    assert 0 in counts and 1 in counts


def test_translate_shift_matches_pixels():
    m, _, _ = G.sample_geometric(AugConfig(translate_x=30, translate_y=10), 100, 100,
                                 np.random.default_rng(3))
    assert np.allclose(m[:2, :2], np.eye(2))
    assert abs(m[0, 2]) <= 30 and abs(m[1, 2]) <= 10


def test_allow_out_alone_is_not_an_augmentation():
    assert not AugConfig(translate_allow_out=True).any_enabled()
    assert AugConfig(translate_y=5).any_enabled()
