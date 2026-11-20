use std::collections::HashSet;

use image::{Rgb, RgbImage};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use tempfile::TempDir;

use rotalabel::augment::geometric as G;
use rotalabel::augment::mosaic::{
    edge_mean_color, extract_patch, generate_mosaic, layout_mosaic, make_mosaic, mosaic_splits,
    pick_patches, random_piece, similar_anns, transform_patch, MosaicConfig, Piece, Placed,
};
use rotalabel::augment::pipeline::transform_annotations;
use rotalabel::augment::version::VersionConfig;
use rotalabel::augment::AugConfig;
use rotalabel::imageio;
use rotalabel::model::{Annotation, BoxKind};
use rotalabel::project_io::Project;

fn inside(a: &Annotation, w: u32, h: u32, tol: f64) -> bool {
    let c = a.corners();
    let min_x = c.iter().map(|p| p[0]).fold(f64::INFINITY, f64::min);
    let min_y = c.iter().map(|p| p[1]).fold(f64::INFINITY, f64::min);
    let max_x = c.iter().map(|p| p[0]).fold(f64::NEG_INFINITY, f64::max);
    let max_y = c.iter().map(|p| p[1]).fold(f64::NEG_INFINITY, f64::max);
    min_x >= -tol && min_y >= -tol && max_x <= w as f64 + tol && max_y <= h as f64 + tol
}

#[test]
fn test_extract_patch_crops_union_of_boxes() {
    let img = RgbImage::new(200, 100);
    let anns = vec![
        Annotation::new(0, 50.0, 50.0, 20.0, 10.0, 0.0, BoxKind::Bbox),
        Annotation::new(0, 120.0, 30.0, 20.0, 10.0, 90.0, BoxKind::Obb),
    ];
    let (patch, boxes) = extract_patch(&img, &anns, 5).unwrap();
    assert_eq!((patch.height(), patch.width()), (45, 95));
    let c0 = boxes[0].corners();
    let orig0 = anns[0].corners();
    for i in 0..4 {
        assert!((c0[i][0] - (orig0[i][0] - 35.0)).abs() < 1e-6);
        assert!((c0[i][1] - (orig0[i][1] - 15.0)).abs() < 1e-6);
    }

    let (patch2, _) = extract_patch(
        &img,
        &[Annotation::new(0, 5.0, 5.0, 10.0, 10.0, 0.0, BoxKind::Bbox)],
        50,
    )
    .unwrap();
    assert_eq!((patch2.height(), patch2.width()), (60, 60));
    assert!(extract_patch(&img, &[], 5).is_none());
}

#[test]
fn test_transform_patch_keeps_boxes_on_the_content() {
    let mut patch = RgbImage::new(80, 40);
    for y in 10..20 {
        for x in 10..30 {
            patch.put_pixel(x, y, Rgb([255, 255, 255]));
        }
    }
    let b = Annotation::new(0, 20.0, 15.0, 20.0, 10.0, 0.0, BoxKind::Obb);
    let mut rng = ChaCha8Rng::seed_from_u64(0);
    for rot in [0.0, 37.0] {
        let cfg = MosaicConfig {
            flip_h: true,
            flip_v: true,
            rot90: true,
            rotate: rot,
            scale: 20.0,
            ..Default::default()
        };
        for _ in 0..20 {
            let (img, alpha, boxes) =
                transform_patch(&patch, &[b.clone()], &cfg, &mut rng, 1000.0, 1000.0);
            let w = img.width();
            let h = img.height();
            assert_eq!(alpha.len(), (w * h) as usize);
            assert!(inside(&boxes[0], w, h, 1.0));
            let mut sum_x = 0.0;
            let mut sum_y = 0.0;
            let mut cnt = 0usize;
            for y in 0..h {
                for x in 0..w {
                    if img.get_pixel(x, y)[0] > 128 {
                        sum_x += x as f64;
                        sum_y += y as f64;
                        cnt += 1;
                    }
                }
            }
            assert!(cnt > 0);
            assert!((boxes[0].cx - (sum_x / cnt as f64 + 0.5)).abs() < 1.5);
            assert!((boxes[0].cy - (sum_y / cnt as f64 + 0.5)).abs() < 1.5);
        }
    }
}

#[test]
fn test_similar_anns_matches_generic_transform() {
    use rand::Rng;
    let mut rng = ChaCha8Rng::seed_from_u64(5);
    let anns = vec![
        Annotation::new(0, 400.0, 300.0, 80.0, 30.0, 25.0, BoxKind::Obb),
        Annotation::new(1, 350.0, 330.0, 40.0, 20.0, 0.0, BoxKind::Bbox),
        Annotation::new(0, 420.0, 280.0, 60.0, 10.0, -70.0, BoxKind::Obb),
    ];
    for _ in 0..50 {
        let mut m = G::translate(rng.gen_range(-50.0..50.0), rng.gen_range(-50.0..50.0))
            .mul(&G::scale(rng.gen_range(0.5..1.5)))
            .mul(&G::about(
                G::rotation(rng.gen_range(-180.0..180.0)),
                400.0,
                300.0,
            ));
        if rng.gen::<f64>() < 0.5 {
            m = m.mul(&G::flip_h(800.0));
        }
        let fast = similar_anns(&anns, &m);
        let reference = transform_annotations(&anns, &m, 5000, 5000, 0.0);
        for (a, b) in fast.iter().zip(reference.iter()) {
            assert_eq!(a.kind, b.kind);
            assert_eq!(a.cls, b.cls);
            let ca = a.corners();
            let cb = b.corners();
            for p in ca {
                let min_d = cb
                    .iter()
                    .map(|q| (q[0] - p[0]).hypot(q[1] - p[1]))
                    .fold(f64::INFINITY, f64::min);
                assert!(min_d < 0.05);
            }
        }
    }
}

#[test]
fn test_large_patch_is_scaled_to_fit() {
    let patch = RgbImage::new(900, 500);
    let cfg = MosaicConfig {
        flip_h: false,
        flip_v: false,
        rot90: false,
        ..Default::default()
    };
    let mut rng = ChaCha8Rng::seed_from_u64(0);
    let (img, _, boxes) = transform_patch(
        &patch,
        &[Annotation::new(0, 450.0, 250.0, 800.0, 400.0, 0.0, BoxKind::Obb)],
        &cfg,
        &mut rng,
        300.0,
        200.0,
    );
    assert!(img.width() <= 300 && img.height() <= 200);
    assert!(inside(&boxes[0], img.width(), img.height(), 1.0));
}

fn make_pieces(n: usize, w: u32, h: u32, seed: u64) -> Vec<Piece> {
    let patch = RgbImage::from_pixel(w, h, Rgb([100, 100, 100]));
    let bx = Annotation::new(0, w as f64 / 2.0, h as f64 / 2.0, w as f64 / 2.0, 40.0, 0.0, BoxKind::Obb);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let cfg = MosaicConfig {
        rotate: 30.0,
        ..Default::default()
    };
    (0..n)
        .map(|_| random_piece(patch.clone(), vec![bx.clone()], &cfg, &mut rng, None))
        .collect()
}

fn footprint(p: &Placed, w: u32, h: u32) -> Vec<bool> {
    let mut m = vec![false; (w * h) as usize];
    let pw = p.img.width();
    let ph = p.img.height();
    for py in 0..ph {
        for px in 0..pw {
            if p.alpha[(py * pw + px) as usize] > 0.02 {
                m[((p.y + py) * w + (p.x + px)) as usize] = true;
            }
        }
    }
    m
}

fn box_mask(anns: &[Annotation], w: u32, h: u32) -> Vec<bool> {
    let mut m = vec![false; (w * h) as usize];
    for a in anns {
        let (s, c) = a.angle.to_radians().sin_cos();
        let [fx0, fy0, fx1, fy1] = a.aabb();
        let x0 = (fx0.floor() as i32).max(0);
        let y0 = (fy0.floor() as i32).max(0);
        let x1 = (fx1.ceil() as i32).min(w as i32);
        let y1 = (fy1.ceil() as i32).min(h as i32);
        for py in y0..y1 {
            for px in x0..x1 {
                let dx = px as f64 - a.cx;
                let dy = py as f64 - a.cy;
                let lx = dx * c + dy * s;
                let ly = -dx * s + dy * c;
                if lx.abs() <= a.w / 2.0 && ly.abs() <= a.h / 2.0 {
                    m[py as usize * w as usize + px as usize] = true;
                }
            }
        }
    }
    m
}

#[test]
fn test_layout_never_covers_a_box() {
    for overlap in [true, false] {
        let (w, h) = (900, 700);
        let cfg = MosaicConfig {
            canvas_w: w,
            canvas_h: h,
            allow_overlap: overlap,
            gap: 2,
            ..Default::default()
        };
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let placed = layout_mosaic(&make_pieces(14, 300, 250, 0), &cfg, &mut rng);
        assert!(!placed.is_empty());
        for (i, p) in placed.iter().enumerate() {
            assert!(p.x + p.img.width() <= w && p.y + p.img.height() <= h);
            assert!(p.anns.iter().all(|b| inside(b, w, h, 1.0)));
            let target = if overlap {
                box_mask(&p.anns, w, h)
            } else {
                footprint(p, w, h)
            };
            for q in &placed[i + 1..] {
                let fq = footprint(q, w, h);
                assert!(!fq.iter().zip(target.iter()).any(|(&a, &b)| a && b));
            }
        }
    }
}

#[test]
fn test_overlap_and_shrink_fit_more_than_strict_layout() {
    let (w, h) = (900, 700);
    let strict = MosaicConfig {
        canvas_w: w,
        canvas_h: h,
        allow_overlap: false,
        min_fit_scale: 100.0,
        ..Default::default()
    };
    let loose = MosaicConfig {
        canvas_w: w,
        canvas_h: h,
        allow_overlap: true,
        min_fit_scale: 40.0,
        ..Default::default()
    };
    let shrink = MosaicConfig {
        canvas_w: w,
        canvas_h: h,
        allow_overlap: false,
        min_fit_scale: 40.0,
        ..Default::default()
    };
    let pieces = make_pieces(14, 300, 250, 0);
    let n_strict = layout_mosaic(&pieces, &strict, &mut ChaCha8Rng::seed_from_u64(2)).len();
    let n_loose = layout_mosaic(&pieces, &loose, &mut ChaCha8Rng::seed_from_u64(2)).len();
    let n_shrink = layout_mosaic(&pieces, &shrink, &mut ChaCha8Rng::seed_from_u64(2)).len();
    assert!(n_strict < 14 && n_loose == 14);
    assert!(n_shrink > n_strict);
}

#[test]
fn test_fixed_patch_count_is_reached() {
    let mut rng = ChaCha8Rng::seed_from_u64(3);
    let cfg = MosaicConfig {
        canvas_w: 1624,
        canvas_h: 1240,
        rotate: 15.0,
        ..Default::default()
    };
    for s in 0..10 {
        let placed = layout_mosaic(&make_pieces(12, 300, 250, s), &cfg, &mut rng);
        assert_eq!(placed.len(), 12);
    }
}

#[test]
fn test_patch_count_range_and_repeats_change_angle() {
    let cfg = MosaicConfig {
        patches: 12,
        patches_jitter: 2,
        rot90: false,
        rotate: 0.0,
        ..Default::default()
    };
    assert_eq!(cfg.patch_range(), (10, 14));
    assert_eq!(
        MosaicConfig {
            patches: 1,
            patches_jitter: 5,
            ..Default::default()
        }
        .patch_range(),
        (1, 6)
    );
    let mut rng = ChaCha8Rng::seed_from_u64(4);
    let picks = pick_patches(3, 14, &cfg, &mut rng);
    assert_eq!(picks.len(), 14);
    let unique_idx: HashSet<usize> = picks.iter().map(|&(i, _)| i).collect();
    assert_eq!(unique_idx, HashSet::from([0, 1, 2]));
    for src in 0..3 {
        let angles: Vec<f64> = picks
            .iter()
            .filter(|&&(i, _)| i == src)
            .map(|&(_, a)| a)
            .collect();
        assert!(angles.len() >= 2);
        for (j, &a) in angles.iter().enumerate() {
            for &b in &angles[j + 1..] {
                let d = (a - b).abs().rem_euclid(360.0);
                assert!(d.min(360.0 - d) >= 30.0 - 1e-9);
            }
        }
    }
    let picks2 = pick_patches(20, 12, &cfg, &mut rng);
    assert_eq!(picks2.iter().map(|&(i, _)| i).collect::<HashSet<_>>().len(), 12);
    assert!(picks2.iter().all(|&(_, a)| a == 0.0));
}

#[test]
fn test_old_min_max_settings_migrate() {
    let m = MosaicConfig::from_json_value(&serde_json::json!({"min_patches": 3, "max_patches": 8}));
    assert_eq!((m.patches, m.patches_jitter, m.patch_range()), (6, 2, (4, 8)));
}

#[test]
fn test_edge_mean_color() {
    let mut p = RgbImage::from_pixel(10, 10, Rgb([10, 20, 30]));
    for y in 3..7 {
        for x in 3..7 {
            p.put_pixel(x, y, Rgb([255, 255, 255]));
        }
    }
    let c = edge_mean_color(&[&p]);
    assert!((c[0] - 10.0).abs() < 1e-3);
    assert!((c[1] - 20.0).abs() < 1e-3);
    assert!((c[2] - 30.0).abs() < 1e-3);
}

#[test]
fn test_make_mosaic_any_sizes() {
    let mut rng = ChaCha8Rng::seed_from_u64(2);
    let mut pool = Vec::new();
    for (w, h) in [(200, 150), (640, 480), (408, 312), (1000, 900)] {
        let img = RgbImage::from_pixel(w, h, Rgb([90, 90, 90]));
        let anns = vec![Annotation::new(
            0,
            w as f64 / 2.0,
            h as f64 / 2.0,
            w as f64 * 0.6,
            h as f64 * 0.3,
            30.0,
            BoxKind::Obb,
        )];
        pool.push(extract_patch(&img, &anns, 10).unwrap());
    }
    for canvas in [(800, 600), (1920, 1080)] {
        for scope in ["patch", "canvas"] {
            let cfg = MosaicConfig {
                canvas_w: canvas.0,
                canvas_h: canvas.1,
                patches: 4,
                patches_jitter: 2,
                rotate: 45.0,
                scale: 20.0,
                photometric_scope: scope.into(),
                aug: AugConfig {
                    brightness: 20.0,
                    hue: 10.0,
                    ..Default::default()
                },
                ..Default::default()
            };
            for _ in 0..3 {
                let (img, boxes) = make_mosaic(&pool, &cfg, &mut rng, true);
                assert_eq!((img.width(), img.height()), canvas);
                assert!(!boxes.is_empty());
                assert!(boxes.iter().all(|b| inside(b, canvas.0, canvas.1, 1.0)));
                assert!(img.pixels().all(|p| p[0] == p[1] && p[1] == p[2]));
            }
        }
    }
}

fn project_for_mosaic(tmp: &std::path::Path, n: usize) -> Project {
    let root = tmp.join("imgs");
    std::fs::create_dir_all(&root).unwrap();
    for i in 0..n {
        let (w, h) = if i % 2 == 1 { (300, 200) } else { (180, 240) };
        let mut img = RgbImage::new(w, h);
        for y in 50..70 {
            for x in 40..90 {
                img.put_pixel(x, y, Rgb([0, 255, 0]));
            }
        }
        imageio::imwrite(&root.join(format!("img{i}.png")), &img, 95).unwrap();
    }
    let mut p = Project::open(&root).unwrap();
    let names = p.images.clone();
    for name in names {
        p.get_mut(&name).annotations = vec![
            Annotation::new(0, 65.0, 60.0, 50.0, 20.0, 0.0, BoxKind::Obb),
            Annotation::new(0, 120.0, 120.0, 40.0, 20.0, 45.0, BoxKind::Obb),
        ];
        p.save_annotations(&name, false);
    }
    p
}

#[test]
fn test_generate_mosaic() {
    let tmp = TempDir::new().unwrap();
    let p = project_for_mosaic(tmp.path(), 10);
    let vcfg = VersionConfig {
        formats: vec!["yolo_obb".into(), "coco_obb".into()],
        ..Default::default()
    };
    let mcfg = MosaicConfig {
        canvas_w: 640,
        canvas_h: 480,
        count_train: 6,
        count_valid: 3,
        count_test: 10,
        rotate: 30.0,
        aug: AugConfig {
            brightness: 10.0,
            ..Default::default()
        },
        ..Default::default()
    };
    let out = tmp.path().join("v");
    let stats = generate_mosaic(&p, &vcfg, &mcfg, &out, None, None).unwrap();
    assert_eq!((stats.train, stats.valid, stats.test), (6, 3, 10));
    let patches = stats.patches.unwrap();
    assert_eq!((patches["train"], patches["valid"], patches["test"]), (7, 2, 1));

    let imgs: Vec<_> = std::fs::read_dir(out.join("yolo_obb/train/images"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    assert_eq!(imgs.len(), 6);
    let first = imageio::imread(&imgs[0]).unwrap();
    assert_eq!((first.width(), first.height()), (640, 480));

    for entry in std::fs::read_dir(out.join("yolo_obb/train/labels")).unwrap() {
        let txt = std::fs::read_to_string(entry.unwrap().path()).unwrap();
        let lines: Vec<&str> = txt.lines().collect();
        assert!(!lines.is_empty());
        for l in lines {
            for v in l.split_whitespace().skip(1) {
                let f: f64 = v.parse().unwrap();
                assert!((-1e-3..=1.0 + 1e-3).contains(&f));
            }
        }
    }

    let coco_test: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(out.join("coco_obb/annotations/test.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(coco_test["images"].as_array().unwrap().len(), 10);

    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(out.join("version.json")).unwrap()).unwrap();
    assert_eq!(meta["type"], "mosaic");
    let sp = mosaic_splits(&p, &vcfg);
    let tr: HashSet<&str> = sp["train"].iter().map(|s| s.as_str()).collect();
    let te: HashSet<&str> = sp["test"].iter().map(|s| s.as_str()).collect();
    assert!(tr.is_disjoint(&te));
}

#[test]
fn test_generate_mosaic_skips_empty_split() {
    let tmp = TempDir::new().unwrap();
    let p = project_for_mosaic(tmp.path(), 2);
    let vcfg = VersionConfig {
        train: 100.0,
        valid: 0.0,
        test: 0.0,
        ..Default::default()
    };
    let mcfg = MosaicConfig {
        count_train: 2,
        ..Default::default()
    };
    let stats = generate_mosaic(&p, &vcfg, &mcfg, &tmp.path().join("v"), None, None).unwrap();
    assert_eq!((stats.train, stats.valid, stats.test), (2, 0, 0));
}
