//! Geometry helpers shared by the canvas, exporters, and augmentation.
//!
//! Angle convention: degrees, positive = clockwise on screen (image y axis points
//! down). A box `(cx, cy, w, h, a)` has its local x axis along `(cos a, sin a)`
//! and local y axis along `(-sin a, cos a)`.

/// 3x3 row-major affine matrix.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat3(pub [[f64; 3]; 3]);

impl Mat3 {
    pub const fn identity() -> Self {
        Self([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    }

    pub fn mul(&self, rhs: &Self) -> Self {
        let mut out = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                out[i][j] = self.0[i][0] * rhs.0[0][j]
                    + self.0[i][1] * rhs.0[1][j]
                    + self.0[i][2] * rhs.0[2][j];
            }
        }
        Self(out)
    }

    pub fn is_identity(&self, tol: f64) -> bool {
        let id = Self::identity();
        for i in 0..3 {
            for j in 0..3 {
                if (self.0[i][j] - id.0[i][j]).abs() > tol {
                    return false;
                }
            }
        }
        true
    }

    pub fn invert(&self) -> Option<Self> {
        let m = &self.0;
        let a = m[0][0];
        let b = m[0][1];
        let c = m[0][2];
        let d = m[1][0];
        let e = m[1][1];
        let f = m[1][2];
        let g = m[2][0];
        let h = m[2][1];
        let i = m[2][2];

        let c00 = e * i - f * h;
        let c01 = -(d * i - f * g);
        let c02 = d * h - e * g;
        let c10 = -(b * i - c * h);
        let c11 = a * i - c * g;
        let c12 = -(a * h - b * g);
        let c20 = b * f - c * e;
        let c21 = -(a * f - c * d);
        let c22 = a * e - b * d;

        let det = a * c00 + b * c01 + c * c02;
        if det.abs() < 1e-15 {
            return None;
        }
        let inv_det = 1.0 / det;
        Some(Self([
            [c00 * inv_det, c10 * inv_det, c20 * inv_det],
            [c01 * inv_det, c11 * inv_det, c21 * inv_det],
            [c02 * inv_det, c12 * inv_det, c22 * inv_det],
        ]))
    }

    #[inline]
    pub fn transform_point(&self, p: [f64; 2]) -> [f64; 2] {
        [
            self.0[0][0] * p[0] + self.0[0][1] * p[1] + self.0[0][2],
            self.0[1][0] * p[0] + self.0[1][1] * p[1] + self.0[1][2],
        ]
    }

    pub fn transform_points(&self, pts: &[[f64; 2]]) -> Vec<[f64; 2]> {
        pts.iter().map(|&p| self.transform_point(p)).collect()
    }
}

/// Return the 4 corners in order TL, TR, BR, BL (clockwise on screen).
pub fn box_corners(cx: f64, cy: f64, w: f64, h: f64, angle: f64) -> [[f64; 2]; 4] {
    let t = angle.to_radians();
    let (s, c) = t.sin_cos();
    let hw = w / 2.0;
    let hh = h / 2.0;
    let locals = [[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]];
    let mut out = [[0.0; 2]; 4];
    for (idx, [lx, ly]) in locals.iter().enumerate() {
        out[idx] = [lx * c - ly * s + cx, lx * s + ly * c + cy];
    }
    out
}

/// Map an angle to [-90, 90). A rectangle rotated by 180 deg is the same rectangle.
pub fn normalize_angle(angle: f64) -> f64 {
    let mut a = (angle + 90.0).rem_euclid(180.0) - 90.0;
    if a >= 90.0 - 1e-9 {
        a -= 180.0;
    }
    if a == 0.0 {
        0.0
    } else {
        a
    }
}

/// Keep angles in (-180, 180] for display/editing.
pub fn norm180(angle: f64) -> f64 {
    let a = (angle + 180.0).rem_euclid(360.0) - 180.0;
    if (a + 180.0).abs() < 1e-12 {
        180.0
    } else if a == 0.0 {
        0.0
    } else {
        a
    }
}

/// Distance between two angles modulo 180.
pub fn angle_dist(a: f64, b: f64) -> f64 {
    let d = (a - b).rem_euclid(180.0);
    d.min(180.0 - d)
}

fn cross(o: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
}

/// 2D convex hull using Andrew's monotone chain algorithm.
pub fn convex_hull(points: &[[f64; 2]]) -> Vec<[f64; 2]> {
    if points.len() <= 1 {
        return points.to_vec();
    }
    let mut pts = points.to_vec();
    pts.sort_by(|a, b| {
        a[0].total_cmp(&b[0])
            .then_with(|| a[1].total_cmp(&b[1]))
    });
    pts.dedup_by(|a, b| (a[0] - b[0]).abs() < 1e-12 && (a[1] - b[1]).abs() < 1e-12);
    if pts.len() <= 2 {
        return pts;
    }
    let mut lower: Vec<[f64; 2]> = Vec::with_capacity(pts.len());
    for &p in &pts {
        while lower.len() >= 2 && cross(lower[lower.len() - 2], lower[lower.len() - 1], p) <= 0.0 {
            lower.pop();
        }
        lower.push(p);
    }
    let mut upper: Vec<[f64; 2]> = Vec::with_capacity(pts.len());
    for &p in pts.iter().rev() {
        while upper.len() >= 2 && cross(upper[upper.len() - 2], upper[upper.len() - 1], p) <= 0.0 {
            upper.pop();
        }
        upper.push(p);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

/// Fit the minimum-area rotated rectangle around `points`.
/// Returns `(cx, cy, w, h, angle_deg)`.
pub fn fit_obb(points: &[[f64; 2]], hint_angle: Option<f64>) -> (f64, f64, f64, f64, f64) {
    if points.is_empty() {
        return (0.0, 0.0, 0.0, 0.0, 0.0);
    }
    let hull = convex_hull(points);
    if hull.len() == 1 {
        return (hull[0][0], hull[0][1], 0.0, 0.0, 0.0);
    }
    if hull.len() == 2 {
        let cx = 0.5 * (hull[0][0] + hull[1][0]);
        let cy = 0.5 * (hull[0][1] + hull[1][1]);
        let dx = hull[1][0] - hull[0][0];
        let dy = hull[1][1] - hull[0][1];
        let mut w = dx.hypot(dy);
        let mut h = 0.0;
        let mut ang = dy.atan2(dx).to_degrees();
        if let Some(hint) = hint_angle {
            if angle_dist(ang + 90.0, hint) < angle_dist(ang, hint) {
                std::mem::swap(&mut w, &mut h);
                ang += 90.0;
            }
        }
        return (cx, cy, w, h, normalize_angle(ang));
    }

    let n = hull.len();
    let mut best_area = f64::INFINITY;
    let mut best = (0.0, 0.0, 0.0, 0.0, 0.0);

    for i in 0..n {
        let p0 = hull[i];
        let p1 = hull[(i + 1) % n];
        let dx = p1[0] - p0[0];
        let dy = p1[1] - p0[1];
        let len = dx.hypot(dy);
        if len < 1e-12 {
            continue;
        }
        let ux = dx / len;
        let uy = dy / len;
        let vx = -uy;
        let vy = ux;

        let mut min_u = f64::INFINITY;
        let mut max_u = f64::NEG_INFINITY;
        let mut min_v = f64::INFINITY;
        let mut max_v = f64::NEG_INFINITY;
        for &p in &hull {
            let pu = p[0] * ux + p[1] * uy;
            let pv = p[0] * vx + p[1] * vy;
            min_u = min_u.min(pu);
            max_u = max_u.max(pu);
            min_v = min_v.min(pv);
            max_v = max_v.max(pv);
        }
        let ew = max_u - min_u;
        let eh = max_v - min_v;
        let area = ew * eh;
        if area < best_area {
            best_area = area;
            let cu = 0.5 * (min_u + max_u);
            let cv = 0.5 * (min_v + max_v);
            let cx = cu * ux + cv * vx;
            let cy = cu * uy + cv * vy;
            let ang = uy.atan2(ux).to_degrees();
            best = (cx, cy, ew, eh, ang);
        }
    }

    let (cx, cy, mut w, mut h, mut ang) = best;
    if let Some(hint) = hint_angle {
        if angle_dist(ang + 90.0, hint) < angle_dist(ang, hint) {
            std::mem::swap(&mut w, &mut h);
            ang += 90.0;
        }
    }
    (cx, cy, w, h, normalize_angle(ang))
}

/// Axis-aligned bounding box `[min_x, min_y, max_x, max_y]`.
pub fn aabb(points: &[[f64; 2]]) -> [f64; 4] {
    let mut x0 = f64::INFINITY;
    let mut y0 = f64::INFINITY;
    let mut x1 = f64::NEG_INFINITY;
    let mut y1 = f64::NEG_INFINITY;
    for &[x, y] in points {
        x0 = x0.min(x);
        y0 = y0.min(y);
        x1 = x1.max(x);
        y1 = y1.max(y);
    }
    if x0.is_infinite() {
        [0.0, 0.0, 0.0, 0.0]
    } else {
        [x0, y0, x1, y1]
    }
}

/// Shoelace polygon area.
pub fn polygon_area(points: &[[f64; 2]]) -> f64 {
    let n = points.len();
    if n < 3 {
        return 0.0;
    }
    let mut sum = 0.0;
    for i in 0..n {
        let j = (i + 1) % n;
        sum += points[i][0] * points[j][1] - points[i][1] * points[j][0];
    }
    sum.abs() * 0.5
}

fn clip_edge<F, G>(poly: Vec<[f64; 2]>, inside: F, intersect: G) -> Vec<[f64; 2]>
where
    F: Fn([f64; 2]) -> bool,
    G: Fn([f64; 2], [f64; 2]) -> [f64; 2],
{
    let mut out = Vec::with_capacity(poly.len() + 2);
    if poly.is_empty() {
        return out;
    }
    let mut prev = *poly.last().unwrap();
    for cur in poly {
        if inside(cur) {
            if !inside(prev) {
                out.push(intersect(prev, cur));
            }
            out.push(cur);
        } else if inside(prev) {
            out.push(intersect(prev, cur));
        }
        prev = cur;
    }
    out
}

/// Sutherland-Hodgman clip of a polygon against `[0, width] x [0, height]`.
pub fn clip_polygon(points: &[[f64; 2]], width: f64, height: f64) -> Vec<[f64; 2]> {
    let mut poly = points.to_vec();
    poly = clip_edge(
        poly,
        |p| p[0] >= 0.0,
        |p, q| [0.0, p[1] + (q[1] - p[1]) * (0.0 - p[0]) / (q[0] - p[0])],
    );
    poly = clip_edge(
        poly,
        |p| p[0] <= width,
        |p, q| [width, p[1] + (q[1] - p[1]) * (width - p[0]) / (q[0] - p[0])],
    );
    poly = clip_edge(
        poly,
        |p| p[1] >= 0.0,
        |p, q| [p[0] + (q[0] - p[0]) * (0.0 - p[1]) / (q[1] - p[1]), 0.0],
    );
    poly = clip_edge(
        poly,
        |p| p[1] <= height,
        |p, q| [p[0] + (q[0] - p[0]) * (height - p[1]) / (q[1] - p[1]), height],
    );
    poly
}
