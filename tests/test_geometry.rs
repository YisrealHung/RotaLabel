use rotalabel::geometry;
use rotalabel::model::{Annotation, BoxKind};

fn approx_eq(a: f64, b: f64, eps: f64) {
    assert!(
        (a - b).abs() <= eps,
        "expected {a} ≈ {b} (diff {}, eps {eps})",
        (a - b).abs()
    );
}

#[test]
fn test_corners_fit_roundtrip() {
    for angle in [0.0, 10.0, 33.3, 45.0, 89.0, -60.0, -89.5] {
        let (cx, cy, w, h) = (120.0, 80.0, 60.0, 25.0);
        let pts = geometry::box_corners(cx, cy, w, h, angle);
        let (fcx, fcy, fw, fh, fa) = geometry::fit_obb(&pts, Some(angle));
        approx_eq(fcx, cx, 1e-2);
        approx_eq(fcy, cy, 1e-2);
        approx_eq(fw, w, 1e-2);
        approx_eq(fh, h, 1e-2);
        assert!(geometry::angle_dist(fa, angle) < 1e-2);
        let back = geometry::box_corners(fcx, fcy, fw, fh, fa);
        for p in pts {
            let min_d = back
                .iter()
                .map(|q| (q[0] - p[0]).hypot(q[1] - p[1]))
                .fold(f64::INFINITY, f64::min);
            assert!(min_d < 1e-2);
        }
    }
}

#[test]
fn test_corner_order_clockwise_on_screen() {
    let pts = geometry::box_corners(0.0, 0.0, 2.0, 2.0, 90.0);
    approx_eq(pts[0][0], 1.0, 1e-6);
    approx_eq(pts[0][1], -1.0, 1e-6);
}

#[test]
fn test_normalize_angle() {
    approx_eq(geometry::normalize_angle(90.0), -90.0, 1e-9);
    approx_eq(geometry::normalize_angle(-90.0), -90.0, 1e-9);
    approx_eq(geometry::normalize_angle(180.0), 0.0, 1e-9);
    approx_eq(geometry::normalize_angle(135.0), -45.0, 1e-9);
}

#[test]
fn test_clip_polygon_area() {
    let square = geometry::box_corners(0.0, 0.0, 10.0, 10.0, 0.0);
    let clipped = geometry::clip_polygon(&square, 100.0, 100.0);
    approx_eq(geometry::polygon_area(&clipped), 25.0, 1e-6);
}

#[test]
fn test_annotation_to_bbox() {
    let a = Annotation::new(0, 50.0, 50.0, 20.0, 10.0, 90.0, BoxKind::Obb);
    let b = a.to_bbox();
    approx_eq(b.w, 10.0, 1e-6);
    approx_eq(b.h, 20.0, 1e-6);
    approx_eq(b.angle, 0.0, 1e-6);
    assert_eq!(b.kind, BoxKind::Bbox);
}
