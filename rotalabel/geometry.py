"""Geometry helpers shared by the canvas, exporters and augmentation.

Angle convention: degrees, positive = clockwise on screen (image y axis points
down). This matches Qt's QGraphicsItem.setRotation.  A box (cx, cy, w, h, a)
has its local x axis along (cos a, sin a) and local y axis along (-sin a, cos a).
"""
from __future__ import annotations

import math

import cv2
import numpy as np


def box_corners(cx: float, cy: float, w: float, h: float, angle: float = 0.0) -> np.ndarray:
    """Return the 4 corners (4x2) in order TL, TR, BR, BL (in the box's local frame)."""
    t = math.radians(angle)
    c, s = math.cos(t), math.sin(t)
    local = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]], dtype=np.float64)
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    return local @ rot.T + np.array([cx, cy], dtype=np.float64)


def normalize_angle(angle: float) -> float:
    """Map an angle to [-90, 90). A rectangle rotated by 180 deg is the same rectangle."""
    a = (angle + 90.0) % 180.0 - 90.0
    # guard against -0.0 / float noise near the boundary
    if a >= 90.0 - 1e-9:
        a -= 180.0
    return a + 0.0


def _angle_dist(a: float, b: float) -> float:
    """Distance between two angles modulo 180."""
    d = (a - b) % 180.0
    return min(d, 180.0 - d)


def fit_obb(points, hint_angle: float | None = None) -> tuple[float, float, float, float, float]:
    """Fit the minimum-area rotated rectangle around ``points``.

    Independent of OpenCV's angle convention: we read the rectangle from
    cv2.boxPoints and derive (w, h, angle) ourselves.  When ``hint_angle`` is
    given, pick between (w, h, a) and (h, w, a+90) the one whose angle is
    closest to the hint, so a box keeps its notion of "width" after rotation.
    """
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    rect = cv2.minAreaRect(pts)
    bp = cv2.boxPoints(rect).astype(np.float64)
    e1 = bp[1] - bp[0]
    e2 = bp[2] - bp[1]
    w = float(np.hypot(*e1))
    h = float(np.hypot(*e2))
    ang = math.degrees(math.atan2(e1[1], e1[0]))
    cx, cy = (float(v) for v in bp.mean(axis=0))
    if hint_angle is not None and _angle_dist(ang + 90.0, hint_angle) < _angle_dist(ang, hint_angle):
        w, h, ang = h, w, ang + 90.0
    return cx, cy, w, h, normalize_angle(ang)


def aabb(points) -> tuple[float, float, float, float]:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    return float(x0), float(y0), float(x1), float(y1)


def polygon_area(points) -> float:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 3:
        return 0.0
    x, y = pts[:, 0], pts[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def clip_polygon(points, width: float, height: float) -> np.ndarray:
    """Sutherland-Hodgman clip of a polygon against the rect [0,width]x[0,height]."""
    poly = [tuple(p) for p in np.asarray(points, dtype=np.float64).reshape(-1, 2)]

    def clip(poly, inside, intersect):
        out = []
        if not poly:
            return out
        prev = poly[-1]
        for cur in poly:
            if inside(cur):
                if not inside(prev):
                    out.append(intersect(prev, cur))
                out.append(cur)
            elif inside(prev):
                out.append(intersect(prev, cur))
            prev = cur
        return out

    def ix(x):
        return lambda p, q: (x, p[1] + (q[1] - p[1]) * (x - p[0]) / (q[0] - p[0]))

    def iy(y):
        return lambda p, q: (p[0] + (q[0] - p[0]) * (y - p[1]) / (q[1] - p[1]), y)

    poly = clip(poly, lambda p: p[0] >= 0, ix(0.0))
    poly = clip(poly, lambda p: p[0] <= width, ix(float(width)))
    poly = clip(poly, lambda p: p[1] >= 0, iy(0.0))
    poly = clip(poly, lambda p: p[1] <= height, iy(float(height)))
    return np.array(poly, dtype=np.float64).reshape(-1, 2)


def transform_points(points, m: np.ndarray) -> np.ndarray:
    """Apply a 3x3 (or 2x3) affine matrix to Nx2 points."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    m = np.asarray(m, dtype=np.float64)
    return pts @ m[:2, :2].T + m[:2, 2]
