use image::{Rgb, RgbImage};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use rotalabel::augment::geometric as G;
use rotalabel::augment::{augment, transform_annotations, AugConfig};
use rotalabel::geometry;
use rotalabel::model::{Annotation, BoxKind};

fn same_rect(a: &Annotation, pts: &[[f64; 2]], tol: f64) {
    let got = a.corners();
    for p in pts {
        let min_d = got
            .iter()
            .map(|q| (q[0] - p[0]).hypot(q[1] - p[1]))
            .fold(f64::INFINITY, f64::min);
        assert!(min_d < tol, "min_d={min_d} >= {tol}");
    }
}

#[test]
fn test_flip_h_obb() {
    let a = Annotation::new(0, 30.0, 40.0, 20.0, 10.0, 25.0, BoxKind::Obb);
    let m = G::flip_h(100.0);
    let out = transform_annotations(&[a.clone()], &m, 100, 100, 0.3);
    assert_eq!(out.len(), 1);
    let expected = m.transform_points(&a.corners());
    same_rect(&out[0], &expected, 0.05);
    assert!((out[0].angle - (-25.0)).abs() < 1e-3);
}

#[test]
fn test_rot90_bbox_and_obb() {
    for k in [1, 2, 3] {
        let (m, w, h) = G::rot90(k, 200.0, 100.0);
        if k % 2 == 1 {
            assert_eq!((w as u32, h as u32), (100, 200));
        } else {
            assert_eq!((w as u32, h as u32), (200, 100));
        }
        let a = Annotation::new(0, 50.0, 30.0, 40.0, 20.0, 0.0, BoxKind::Bbox);
        let o = Annotation::new(1, 50.0, 30.0, 40.0, 20.0, 15.0, BoxKind::Obb);
        let res = transform_annotations(&[a.clone(), o.clone()], &m, w as u32, h as u32, 0.3);
        assert_eq!(res.len(), 2);
        let (tb, to) = (&res[0], &res[1]);
        same_rect(tb, &m.transform_points(&a.corners()), 0.05);
        same_rect(to, &m.transform_points(&o.corners()), 0.05);
        assert_eq!(tb.kind, BoxKind::Bbox);
        assert_eq!(tb.angle, 0.0);
        assert!(geometry::angle_dist(to.angle, 15.0 + 90.0 * k as f64) < 1e-3);
        assert!((to.w - 40.0).abs() < 1e-3);
    }
}

#[test]
fn test_arbitrary_rotation_obb_exact_bbox_grows() {
    let m = G::about(G::rotation(30.0), 100.0, 100.0);
    let o = Annotation::new(0, 100.0, 100.0, 40.0, 20.0, 0.0, BoxKind::Obb);
    let b = Annotation::new(0, 100.0, 100.0, 40.0, 20.0, 0.0, BoxKind::Bbox);
    let res = transform_annotations(&[o, b], &m, 200, 200, 0.3);
    let (to, tb) = (&res[0], &res[1]);
    assert!((to.angle - 30.0).abs() < 1e-3);
    assert!((to.w - 40.0).abs() < 1e-3 && (to.h - 20.0).abs() < 1e-3);
    assert!(tb.w > 40.0 && tb.h > 20.0 && tb.angle == 0.0);
}

#[test]
fn test_box_outside_dropped_and_partial_clipped() {
    let m = G::translate(-90.0, 0.0);
    let gone = Annotation::new(0, 50.0, 50.0, 20.0, 20.0, 0.0, BoxKind::Bbox);
    let part = Annotation::new(0, 90.0, 50.0, 20.0, 20.0, 0.0, BoxKind::Obb);
    let out = transform_annotations(&[gone, part.clone()], &m, 100, 100, 0.3);
    assert_eq!(out.len(), 1);
    assert!((out[0].w - 10.0).abs() < 1e-3);
    assert!(transform_annotations(&[part], &m, 100, 100, 0.6).is_empty());
}

#[test]
fn test_augment_pipeline_runs_and_keeps_boxes_inside() {
    let mut rng = ChaCha8Rng::seed_from_u64(0);
    let img = RgbImage::from_pixel(160, 120, Rgb([128, 128, 128]));
    let anns = vec![
        Annotation::new(0, 80.0, 60.0, 40.0, 30.0, 20.0, BoxKind::Obb),
        Annotation::new(1, 40.0, 40.0, 20.0, 20.0, 0.0, BoxKind::Bbox),
    ];
    let cfg = AugConfig {
        flip_h: true,
        flip_v: true,
        rot90: true,
        rotate: 20.0,
        shear: 5.0,
        scale: 10.0,
        crop: 20.0,
        brightness: 20.0,
        exposure: 10.0,
        contrast: 20.0,
        saturation: 30.0,
        hue: 20.0,
        grayscale: 20.0,
        blur: 2.0,
        noise: 2.0,
        ..Default::default()
    };
    for _ in 0..30 {
        let (out, boxes) = augment(&img, &anns, &cfg, &mut rng, None, 0.3, None);
        let w = out.width();
        let h = out.height();
        assert!((w == 160 && h == 120) || (w == 120 && h == 160));
        for b in boxes {
            let [x0, y0, x1, y1] = b.aabb();
            assert!(-1.0 <= b.cx && b.cx <= w as f64 + 1.0);
            assert!(-1.0 <= b.cy && b.cy <= h as f64 + 1.0);
            if b.kind == BoxKind::Bbox {
                assert!(x0 >= -1e-6 && y0 >= -1e-6 && x1 <= w as f64 + 1e-6 && y1 <= h as f64 + 1e-6);
            }
        }
    }
}

#[test]
fn test_preprocess_fit_letterbox() {
    let (m, w, h) = G::preprocess_matrix(200, 100, "fit", 100, 100);
    let a = Annotation::new(0, 100.0, 50.0, 200.0, 100.0, 0.0, BoxKind::Bbox);
    let t = &transform_annotations(&[a], &m, w, h, 0.0)[0];
    assert!((t.cx - 50.0).abs() < 1e-6);
    assert!((t.cy - 50.0).abs() < 1e-6);
    assert!((t.w - 100.0).abs() < 1e-6);
    assert!((t.h - 50.0).abs() < 1e-6);
}

#[test]
fn test_translate_keeps_boxes_inside_by_default() {
    let mut rng = ChaCha8Rng::seed_from_u64(1);
    let img = RgbImage::from_pixel(200, 100, Rgb([128, 128, 128]));
    let anns = vec![
        Annotation::new(0, 30.0, 50.0, 40.0, 20.0, 0.0, BoxKind::Bbox),
        Annotation::new(1, 160.0, 40.0, 30.0, 20.0, 30.0, BoxKind::Obb),
    ];
    let cfg = AugConfig {
        translate_x: 150.0,
        translate_y: 80.0,
        ..Default::default()
    };
    let mut moved = std::collections::HashSet::new();
    for _ in 0..50 {
        let (out, boxes) = augment(&img, &anns, &cfg, &mut rng, None, 0.0, None);
        assert_eq!((out.width(), out.height()), (200, 100));
        assert_eq!(boxes.len(), 2);
        for (a, b) in anns.iter().zip(boxes.iter()) {
            assert!((b.w - a.w).abs() < 1e-3 && (b.h - a.h).abs() < 1e-3);
            let [x0, y0, x1, y1] = geometry::aabb(&b.corners());
            assert!(x0 >= -1e-6 && y0 >= -1e-6 && x1 <= 200.0 + 1e-6 && y1 <= 100.0 + 1e-6);
        }
        moved.insert(((boxes[0].cx - 30.0) * 1000.0).round() as i64);
    }
    assert!(moved.len() > 10);
    let min_dx = *moved.iter().min().unwrap() as f64 / 1000.0;
    assert!(min_dx >= -10.0 - 1e-6);
}

#[test]
fn test_translate_allow_out_can_push_boxes_out() {
    let mut rng = ChaCha8Rng::seed_from_u64(2);
    let img = RgbImage::from_pixel(100, 100, Rgb([128, 128, 128]));
    let anns = vec![Annotation::new(0, 50.0, 50.0, 20.0, 20.0, 0.0, BoxKind::Bbox)];
    let cfg = AugConfig {
        translate_x: 100.0,
        translate_allow_out: true,
        ..Default::default()
    };
    let counts: Vec<usize> = (0..100)
        .map(|_| augment(&img, &anns, &cfg, &mut rng, None, 0.5, None).1.len())
        .collect();
    assert!(counts.contains(&0) && counts.contains(&1));
}

#[test]
fn test_translate_shift_matches_pixels() {
    let mut rng = ChaCha8Rng::seed_from_u64(3);
    let cfg = AugConfig {
        translate_x: 30.0,
        translate_y: 10.0,
        ..Default::default()
    };
    let (m, _, _) = G::sample_geometric(&cfg, 100, 100, &mut rng, None);
    assert!((m.0[0][0] - 1.0).abs() < 1e-9 && m.0[0][1].abs() < 1e-9);
    assert!(m.0[0][2].abs() <= 30.0 && m.0[1][2].abs() <= 10.0);
}

#[test]
fn test_allow_out_alone_is_not_an_augmentation() {
    let cfg = AugConfig {
        translate_allow_out: true,
        ..Default::default()
    };
    assert!(!cfg.any_enabled());
    let cfg2 = AugConfig {
        translate_y: 5.0,
        ..Default::default()
    };
    assert!(cfg2.any_enabled());
}
