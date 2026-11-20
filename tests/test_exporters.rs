use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

use image::{Rgb, RgbImage};
use tempfile::TempDir;

use rotalabel::augment::version::{generate_version, VersionConfig};
use rotalabel::augment::AugConfig;
use rotalabel::exporters::{CocoObbWriter, CocoWriter, DotaWriter, YoloWriter};
use rotalabel::geometry;
use rotalabel::imageio;
use rotalabel::importers;
use rotalabel::model::{Annotation, BoxKind};
use rotalabel::project_io::Project;

const W: u32 = 200;
const H: u32 = 100;

fn box_ann() -> Annotation {
    Annotation::new(0, 50.0, 50.0, 40.0, 20.0, 0.0, BoxKind::Bbox)
}

fn rot_ann() -> Annotation {
    Annotation::new(1, 120.0, 50.0, 40.0, 20.0, 90.0, BoxKind::Obb)
}

fn floats(line: &str) -> Vec<f64> {
    line.split_whitespace()
        .map(|v| v.parse::<f64>().unwrap())
        .collect()
}

fn approx_slice(a: &[f64], b: &[f64], eps: f64) {
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b.iter()) {
        assert!((x - y).abs() <= eps, "{x} != {y}");
    }
}

#[test]
fn test_yolo_bbox() {
    let tmp = TempDir::new().unwrap();
    let mut w = YoloWriter::new(tmp.path(), vec!["a".into(), "b".into()], false);
    w.image_dir("train").unwrap();
    w.write_labels("train", "x.jpg", W, H, &[box_ann(), rot_ann()])
        .unwrap();
    w.used_splits.insert("train".into());
    w.finish().unwrap();

    let raw = std::fs::read_to_string(tmp.path().join("train/labels/x.txt")).unwrap();
    let lines: Vec<&str> = raw.lines().collect();
    approx_slice(&floats(lines[0]), &[0.0, 0.25, 0.5, 0.2, 0.2], 1e-5);
    approx_slice(&floats(lines[1]), &[1.0, 0.6, 0.5, 0.1, 0.4], 1e-5);

    let yml: serde_yaml::Value =
        serde_yaml::from_str(&std::fs::read_to_string(tmp.path().join("data.yaml")).unwrap())
            .unwrap();
    assert_eq!(yml["train"].as_str().unwrap(), "train/images");
    assert_eq!(yml["val"].as_str().unwrap(), "train/images");
    assert_eq!(yml["names"][0].as_str().unwrap(), "a");
    assert_eq!(yml["names"][1].as_str().unwrap(), "b");
}

#[test]
fn test_yolo_obb_and_dota() {
    let tmp = TempDir::new().unwrap();
    let y = YoloWriter::new(tmp.path().join("y"), vec!["a".into(), "b".into()], true);
    let d = DotaWriter::new(tmp.path().join("d"), vec!["a".into(), "b".into()]);
    y.write_labels("valid", "x.jpg", W, H, &[rot_ann()]).unwrap();
    d.write_labels("valid", "x.jpg", W, H, &[rot_ann()]).unwrap();
    y.finish().unwrap();
    d.finish().unwrap();

    let yv = floats(
        &std::fs::read_to_string(tmp.path().join("y/valid/labels/x.txt")).unwrap(),
    );
    let d_raw = std::fs::read_to_string(tmp.path().join("d/valid/labelTxt/x.txt")).unwrap();
    let dv: Vec<&str> = d_raw.split_whitespace().collect();
    assert_eq!(&dv[8..], ["b", "0"]);
    let rc = rot_ann().corners();
    for i in 0..4 {
        let px = yv[1 + i * 2] * W as f64;
        let py = yv[2 + i * 2] * H as f64;
        let dx: f64 = dv[i * 2].parse().unwrap();
        let dy: f64 = dv[1 + i * 2].parse().unwrap();
        assert!((px - dx).abs() < 1e-3 && (py - dy).abs() < 1e-3);
        assert!((px - rc[i][0]).abs() < 1e-3 && (py - rc[i][1]).abs() < 1e-3);
    }
}

#[test]
fn test_coco() {
    let tmp = TempDir::new().unwrap();
    let mut c = CocoWriter::new(tmp.path(), vec!["a".into(), "b".into()]);
    c.write_labels("train", "x.jpg", W, H, &[box_ann(), rot_ann()]);
    c.finish().unwrap();
    let d: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(tmp.path().join("train/_annotations.coco.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(d["annotations"][0]["bbox"], serde_json::json!([30.0, 40.0, 40.0, 20.0]));
    assert_eq!(d["annotations"][0]["category_id"], 1);
    assert_eq!(
        d["annotations"][1]["rbox"],
        serde_json::json!([120.0, 50.0, 40.0, 20.0, 90.0])
    );
    assert_eq!(
        d["annotations"][1]["segmentation"][0].as_array().unwrap().len(),
        8
    );
}

#[test]
fn test_coco_obb() {
    let tmp = TempDir::new().unwrap();
    let mut c = CocoObbWriter::new(tmp.path(), vec!["a".into(), "b".into()]);
    c.image_dir("train").unwrap();
    c.image_dir("valid").unwrap();
    c.write_labels("train", "x.jpg", W, H, &[box_ann(), rot_ann()]);
    c.write_labels("valid", "y.jpg", W, H, &[rot_ann()]);
    c.finish().unwrap();

    assert!(tmp.path().join("train/images").exists());
    assert!(tmp.path().join("val/images").exists());
    assert!(!tmp.path().join("valid").exists());

    let d: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(tmp.path().join("annotations/train.json")).unwrap(),
    )
    .unwrap();
    let val: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(tmp.path().join("annotations/val.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(val["images"][0]["file_name"], "y.jpg");
    assert_eq!(d["categories"][0]["id"], 0);
    assert_eq!(d["categories"][1]["id"], 1);

    let b = &d["annotations"][0];
    let r = &d["annotations"][1];
    assert_eq!(b["category_id"], 0);
    assert_eq!(r["category_id"], 1);
    assert_eq!(b["bbox"], serde_json::json!([30.0, 40.0, 40.0, 20.0]));
    assert_eq!(b["area"], 800.0);
    assert_eq!(
        r["segmentation"],
        serde_json::json!([[130.0, 30.0, 130.0, 70.0, 110.0, 70.0, 110.0, 30.0]])
    );
    assert_eq!(r["bbox"], serde_json::json!([110.0, 30.0, 20.0, 40.0]));
    assert_eq!(r["area"], 800.0);

    // clockwise on screen (y down) => positive signed shoelace area
    for a in [b, r] {
        let seg: Vec<f64> = a["segmentation"][0]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let mut s = 0.0;
        for i in 0..4 {
            let j = (i + 1) % 4;
            s += seg[i * 2] * seg[j * 2 + 1] - seg[i * 2 + 1] * seg[j * 2];
        }
        assert!(s > 0.0);
    }
}

fn make_project(tmp: &Path, n: usize) -> Project {
    let root = tmp.join("imgs 中文");
    std::fs::create_dir_all(&root).unwrap();
    for i in 0..n {
        let mut img = RgbImage::new(W, H);
        for y in 40..60 {
            for x in 30..70 {
                img.put_pixel(x, y, Rgb([0, 255, 0]));
            }
        }
        imageio::imwrite(&root.join(format!("img{i}.png")), &img, 95).unwrap();
    }
    let mut p = Project::open(&root).unwrap();
    p.add_class("b");
    let names = p.images.clone();
    for name in names {
        {
            let ia = p.get_mut(&name);
            ia.width = W;
            ia.height = H;
            ia.annotations = vec![box_ann(), rot_ann()];
        }
        p.save_annotations(&name, false);
    }
    p
}

#[test]
fn test_project_roundtrip() {
    let tmp = TempDir::new().unwrap();
    let p = make_project(tmp.path(), 2);
    let mut q = Project::open(&p.root).unwrap();
    assert_eq!(q.class_names(), vec!["object", "b"]);
    assert_eq!(q.get("img0.png").unwrap().annotations[1], rot_ann());
    q.delete_class(0);
    let q2 = Project::open(&p.root).unwrap();
    let cls_list: Vec<usize> = q2
        .get("img1.png")
        .unwrap()
        .annotations
        .iter()
        .map(|a| a.cls)
        .collect();
    assert_eq!(cls_list, vec![0]);
}

#[test]
fn test_dota_side_file_next_to_image() {
    let tmp = TempDir::new().unwrap();
    let p = make_project(tmp.path(), 2);
    let raw = std::fs::read_to_string(p.root.join("img0.txt")).unwrap();
    let lines: Vec<&str> = raw.lines().collect();
    assert_eq!(&lines[..2], ["imagesource:rotalabel", "gsd:null"]);
    assert_eq!(lines[2], "30 40 70 40 70 60 30 60 object 0");
    assert!(lines[3].ends_with("b 0"));

    let mut q = Project::open(&p.root).unwrap();
    let (upd, n) = importers::import_labels(&mut q, &p.root, true);
    assert_eq!((upd, n), (2, 4));
    let c0 = q.get("img0.png").unwrap().annotations[0].corners();
    let bc = box_ann().corners();
    for i in 0..4 {
        assert!((c0[i][0] - bc[i][0]).abs() < 1e-2 && (c0[i][1] - bc[i][1]).abs() < 1e-2);
    }

    q.get_mut("img1.png").annotations.clear();
    q.save_annotations("img1.png", false);
    let raw1 = std::fs::read_to_string(p.root.join("img1.txt")).unwrap();
    assert_eq!(
        raw1.lines().collect::<Vec<_>>(),
        ["imagesource:rotalabel", "gsd:null"]
    );
    q.classes[0].name = "car".into();
    q.sync_label_files();
    let raw0 = std::fs::read_to_string(p.root.join("img0.txt")).unwrap();
    assert!(raw0.lines().nth(2).unwrap().ends_with("car 0"));
}

#[test]
fn test_side_file_overwrites_in_place() {
    let tmp = TempDir::new().unwrap();
    let root = tmp.path().join("imgs");
    std::fs::create_dir_all(&root).unwrap();
    for n in ["a.png", "b.jpg", "b.png"] {
        imageio::imwrite(&root.join(n), &RgbImage::new(W, H), 95).unwrap();
    }
    std::fs::write(root.join("a.txt"), "0 0.5 0.5 0.1 0.1\n").unwrap();
    let mut p = Project::open(&root).unwrap();
    let imgs = p.images.clone();
    for n in imgs {
        p.get_mut(&n).annotations = vec![box_ann()];
        p.save_annotations(&n, false);
    }
    assert!(std::fs::read_to_string(root.join("a.txt"))
        .unwrap()
        .starts_with("imagesource:rotalabel"));
    assert!(root.join("b.jpg.txt").exists());
    assert!(root.join("b.png.txt").exists());
    assert!(!root.join("b.txt").exists());
}

fn yolo_obb_folder(tmp: &Path) -> std::path::PathBuf {
    let root = tmp.join("imgs");
    std::fs::create_dir_all(&root).unwrap();
    for n in ["a.png", "b.png", "c.png"] {
        imageio::imwrite(&root.join(n), &RgbImage::new(W, H), 95).unwrap();
    }
    std::fs::write(root.join("classes.txt"), "nail\nscrew\n").unwrap();
    std::fs::write(
        root.join("a.txt"),
        "1 0.1 0.2 0.3 0.2 0.3 0.4 0.1 0.4\n0 0.5 0.5 0.7 0.5 0.7 0.6 0.5 0.6\n",
    )
    .unwrap();
    std::fs::write(root.join("b.txt"), "").unwrap();
    root
}

#[test]
fn test_yolo_obb_folder_is_loaded_and_edits_overwrite_in_place() {
    let tmp = TempDir::new().unwrap();
    let root = yolo_obb_folder(tmp.path());
    let mut p = Project::open(&root).unwrap();
    assert_eq!(p.label_format(), "yolo_obb");
    assert!(p.output_dir().is_none());
    assert!(!root.join("rotalabel_labels").exists());
    assert_eq!(p.class_names(), vec!["nail", "screw"]);
    assert_eq!(p.auto_imported, (2, 2));
    let anns = p.get("a.png").unwrap().annotations.clone();
    assert_eq!(anns.iter().map(|a| a.cls).collect::<Vec<_>>(), vec![1, 0]);

    p.get_mut("a.png").annotations = anns[1..].to_vec();
    p.save_annotations("a.png", false);
    {
        let ic = p.get_mut("c.png");
        ic.width = W;
        ic.height = H;
        ic.annotations = vec![box_ann()];
    }
    p.save_annotations("c.png", false);

    assert_eq!(
        std::fs::read_to_string(root.join("a.txt"))
            .unwrap()
            .split_whitespace()
            .collect::<Vec<_>>(),
        ["0", "0.5", "0.5", "0.7", "0.5", "0.7", "0.6", "0.5", "0.6"]
    );
    let c_toks: Vec<String> = std::fs::read_to_string(root.join("c.txt"))
        .unwrap()
        .split_whitespace()
        .map(|s| s.to_string())
        .collect();
    assert_eq!(c_toks[0], "0");
    assert_eq!(c_toks.len(), 9);
    assert_eq!(std::fs::read_to_string(root.join("b.txt")).unwrap(), "");
    assert_eq!(
        std::fs::read_to_string(root.join("classes.txt"))
            .unwrap()
            .split_whitespace()
            .collect::<Vec<_>>(),
        ["nail", "screw"]
    );
    assert!(!root.join("rotalabel_labels").exists());

    let mut q = Project::open(&root).unwrap();
    assert_eq!(q.label_format(), "yolo_obb");
    assert_eq!(q.auto_imported, (0, 0));
    assert_eq!(q.get("a.png").unwrap().annotations.len(), 1);
    q.add_class("bolt");
    assert_eq!(
        std::fs::read_to_string(root.join("classes.txt"))
            .unwrap()
            .split_whitespace()
            .collect::<Vec<_>>(),
        ["nail", "screw", "bolt"]
    );
}

#[test]
fn test_dota_folder_is_loaded_and_edits_overwrite_in_place() {
    for with_classes_txt in [true, false] {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("imgs");
        std::fs::create_dir_all(&root).unwrap();
        for n in ["a.jpg", "b.jpg", "c.jpg"] {
            imageio::imwrite(&root.join(n), &RgbImage::new(W, H), 95).unwrap();
        }
        if with_classes_txt {
            std::fs::write(root.join("classes.txt"), "nail\nscrew\n").unwrap();
        }
        std::fs::write(
            root.join("a.txt"),
            "30 40 70 40 70 60 30 60 screw 0\n110 30 130 30 130 70 110 70 nail 0\n",
        )
        .unwrap();
        std::fs::write(
            root.join("b.txt"),
            "imagesource:GoogleEarth\ngsd:0.5\n10 10 50 10 50 30 10 30 nail 1\n",
        )
        .unwrap();

        let mut p = Project::open(&root).unwrap();
        assert_eq!(p.label_format(), "dota");
        assert!(p.output_dir().is_none());
        assert!(!root.join("rotalabel_labels").exists());
        assert_eq!(p.auto_imported, (2, 3));
        assert_eq!(
            p.class_names(),
            if with_classes_txt {
                vec!["nail", "screw"]
            } else {
                vec!["screw", "nail"]
            }
        );

        let a = p.get("a.jpg").unwrap().annotations.clone();
        p.get_mut("a.jpg").annotations = a[..1].to_vec();
        p.save_annotations("a.jpg", false);
        let raw = std::fs::read_to_string(root.join("a.txt")).unwrap();
        let lines: Vec<&str> = raw.lines().collect();
        assert_eq!(lines[0], "imagesource:rotalabel");
        assert_eq!(lines.len(), 3);
        assert!(lines[2].ends_with("screw 0"));

        let q = Project::open(&root).unwrap();
        assert_eq!(q.label_format(), "dota");
        assert_eq!(q.auto_imported, (0, 0));
        assert_eq!(q.get("a.jpg").unwrap().annotations.len(), 1);
    }
}

#[test]
fn test_unknown_txt_next_to_images_is_not_adopted() {
    let tmp = TempDir::new().unwrap();
    let root = tmp.path().join("imgs");
    std::fs::create_dir_all(&root).unwrap();
    imageio::imwrite(&root.join("a.png"), &RgbImage::new(W, H), 95).unwrap();
    std::fs::write(root.join("a.txt"), "this is just a note about the photo\n").unwrap();
    let p = Project::open(&root).unwrap();
    assert!(!p.settings.contains_key("label_format"));
    assert_eq!(p.auto_imported, (0, 0));
}

#[test]
fn test_label_output_format_and_folder_can_change() {
    let tmp = TempDir::new().unwrap();
    let root = yolo_obb_folder(tmp.path());
    let mut p = Project::open(&root).unwrap();
    let n = p
        .set_label_output("dota", &tmp.path().join("dota_labels"))
        .unwrap();
    assert_eq!(n, 2);
    let lines: Vec<String> = std::fs::read_to_string(tmp.path().join("dota_labels/a.txt"))
        .unwrap()
        .lines()
        .map(|s| s.to_string())
        .collect();
    assert_eq!(&lines[..2], ["imagesource:rotalabel", "gsd:null"]);
    assert!(lines[2].ends_with("screw 0"));

    p.set_label_output("yolo_obb", &root.join("v2")).unwrap();
    assert_eq!(p.settings["label_dir"], "v2");
    assert!(std::fs::read_to_string(root.join("v2/a.txt"))
        .unwrap()
        .starts_with('1'));
    assert_eq!(
        Project::open(&root).unwrap().output_dir(),
        Some(root.join("v2"))
    );

    p.set_label_output("yolo_obb", &root).unwrap();
    assert_eq!(p.settings["label_dir"], "");
    assert!(p.output_dir().is_none());
}

#[test]
fn test_yolo_obb_without_names_keeps_ids() {
    let tmp = TempDir::new().unwrap();
    let root = tmp.path().join("imgs");
    std::fs::create_dir_all(&root).unwrap();
    imageio::imwrite(&root.join("a.png"), &RgbImage::new(W, H), 95).unwrap();
    std::fs::write(root.join("a.txt"), "2 0.1 0.2 0.3 0.2 0.3 0.4 0.1 0.4\n").unwrap();
    let p = Project::open(&root).unwrap();
    assert_eq!(p.class_names(), vec!["class_0", "class_1", "class_2"]);
    assert_eq!(p.get("a.png").unwrap().annotations[0].cls, 2);
}

#[test]
fn test_legacy_project_folder_still_opens() {
    let tmp = TempDir::new().unwrap();
    let p = make_project(tmp.path(), 2);
    std::fs::rename(
        p.root.join("rotalabel_project"),
        p.root.join("bboxer_project"),
    )
    .unwrap();
    let q = Project::open(&p.root).unwrap();
    assert_eq!(q.class_names(), vec!["object", "b"]);
    assert_eq!(q.get("img0.png").unwrap().annotations.len(), 2);
}

#[test]
fn test_generate_version_deterministic_across_workers_and_cancel() {
    let tmp = TempDir::new().unwrap();
    let p = make_project(tmp.path(), 10);
    let mut cfg = VersionConfig {
        train: 70.0,
        valid: 20.0,
        test: 10.0,
        multiplier: 2,
        formats: vec!["yolo_obb".into(), "coco_obb".into()],
        aug: AugConfig {
            flip_h: true,
            rotate: 15.0,
            brightness: 15.0,
            ..Default::default()
        },
        workers: 1,
        ..Default::default()
    };

    let out1 = tmp.path().join("v_serial");
    let s1 = generate_version(&p, &cfg, &out1, None, None).unwrap();
    assert_eq!(s1.train, 14);
    assert_eq!(s1.valid, 4);
    assert_eq!(s1.test, 10);
    assert!(out1.join("yolo_obb/data.yaml").exists());
    assert!(out1.join("coco_obb/annotations/val.json").exists());
    assert!(out1.join("version.json").exists());

    cfg.workers = 4;
    let out4 = tmp.path().join("v_parallel");
    let s4 = generate_version(&p, &cfg, &out4, None, None).unwrap();
    assert_eq!((s1.train, s1.valid, s1.test, s1.boxes), (s4.train, s4.valid, s4.test, s4.boxes));

    // Same label outputs regardless of worker count
    let lbl1 = std::fs::read_to_string(out1.join("yolo_obb/train/labels/img0.txt")).ok();
    let lbl4 = std::fs::read_to_string(out4.join("yolo_obb/train/labels/img0.txt")).ok();
    assert_eq!(lbl1, lbl4);

    // Cancellation stops cleanly
    let cancel = AtomicBool::new(true);
    cancel.store(true, Ordering::Relaxed);
    let sc = generate_version(&p, &cfg, &tmp.path().join("v_cancel"), None, Some(&cancel)).unwrap();
    assert!(sc.cancelled);
    assert!(!geometry::aabb(&[[0.0, 0.0]]).iter().any(|v| v.is_nan()));
}
