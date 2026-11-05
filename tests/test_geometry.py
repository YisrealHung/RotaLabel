import numpy as np
import pytest

from rotalabel import geometry
from rotalabel.model import OBB, Annotation


@pytest.mark.parametrize("angle", [0, 10, 33.3, 45, 89, -60, -89.5])
def test_corners_fit_roundtrip(angle):
    cx, cy, w, h = 120.0, 80.0, 60.0, 25.0
    pts = geometry.box_corners(cx, cy, w, h, angle)
    fcx, fcy, fw, fh, fa = geometry.fit_obb(pts, hint_angle=angle)
    assert fcx == pytest.approx(cx, abs=1e-2)
    assert fcy == pytest.approx(cy, abs=1e-2)
    assert fw == pytest.approx(w, abs=1e-2)
    assert fh == pytest.approx(h, abs=1e-2)
    assert geometry._angle_dist(fa, angle) < 1e-2
    # corners describe the same rectangle
    back = geometry.box_corners(fcx, fcy, fw, fh, fa)
    for p in pts:
        assert np.min(np.hypot(*(back - p).T)) < 1e-2


def test_corner_order_clockwise_on_screen():
    pts = geometry.box_corners(0, 0, 2, 2, 90)
    # TL corner (-1,-1) rotated 90deg clockwise on screen -> (1,-1)
    assert pts[0] == pytest.approx([1, -1])


def test_normalize_angle():
    assert geometry.normalize_angle(90) == -90
    assert geometry.normalize_angle(-90) == -90
    assert geometry.normalize_angle(180) == 0
    assert geometry.normalize_angle(135) == -45


def test_clip_polygon_area():
    square = geometry.box_corners(0, 0, 10, 10, 0)  # half outside on x and y
    clipped = geometry.clip_polygon(square, 100, 100)
    assert geometry.polygon_area(clipped) == pytest.approx(25)


def test_annotation_to_bbox():
    a = Annotation(0, 50, 50, 20, 10, 90, OBB)
    b = a.to_bbox()
    assert (b.w, b.h) == pytest.approx((10, 20))
    assert b.angle == 0
