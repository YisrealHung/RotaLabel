use std::collections::HashSet;
use std::path::Path;

use image::RgbImage;
use tempfile::TempDir;

use rotalabel::augment::pipeline::AugConfig;
use rotalabel::augment::version::{default_workers, generate_version, output_stems, unique_stem, VersionConfig};
use rotalabel::exporters::{CocoWriter, DotaWriter};
use rotalabel::imageio;
use rotalabel::importers;
use rotalabel::model::{Annotation, BoxKind};
use rotalabel::project_io::Project;
use rotalabel::ui::canvas::{CanvasState, DragState, HandleId, LabelMode};
use rotalabel::ui::theme::{apply_theme, dark_palette, light_palette};
use rotalabel::ui::version_dialog::preview_visibility;

fn make_images(folder: &Path, names: &[&str]) {
    std::fs::create_dir_all(folder).unwrap();
    for n in names {
        imageio::imwrite(&folder.join(n), &RgbImage::new(100, 100), 95).unwrap();
    }
}

#[test]
fn test_unique_stem_avoids_aug_name_clash() {
    let mut used = HashSet::new();
    assert_eq!(unique_stem("a", &mut used, 3), "a");
    assert_eq!(unique_stem("a_aug1", &mut used, 1), "a_aug1_2");
    let mut used2 = HashSet::from(["a_aug1".to_string()]);
    assert_eq!(unique_stem("a", &mut used2, 2), "a_2");
    assert_eq!(output_stems("x", 3), vec!["x", "x_aug1", "x_aug2"]);
}

#[test]
fn test_import_ids_outside_names_file_get_own_class() {
    let tmp = TempDir::new().unwrap();
    make_images(&tmp.path().join("img"), &["x.png"]);
    let labels = tmp.path().join("labels");
    std::fs::create_dir_all(&labels).unwrap();
    std::fs::write(labels.join("classes.txt"), "cat\n").unwrap();
    std::fs::write(
        labels.join("x.txt"),
        "0 0.5 0.5 0.2 0.2\n1 0.5 0.5 0.1 0.1\n",
    )
    .unwrap();

    let mut p = Project::open(tmp.path().join("img")).unwrap();
    p.add_class("cat");
    importers::import_labels(&mut p, &labels, true);
    let names = p.class_names();
    let got: Vec<&str> = p
        .get("x.png")
        .unwrap()
        .annotations
        .iter()
        .map(|a| names[a.cls].as_str())
        .collect();
    assert_eq!(got, vec!["cat", "class_1"]);
}

#[test]
fn test_dota_and_coco_corners_stay_inside_image() {
    let tmp = TempDir::new().unwrap();
    let a = Annotation::new(0, 5.0, 50.0, 60.0, 20.0, 30.0, BoxKind::Obb);
    let d = DotaWriter::new(tmp.path().join("d"), vec!["a".into()]);
    d.write_labels("train", "x.jpg", 200, 100, &[a.clone()])
        .unwrap();
    let raw = std::fs::read_to_string(tmp.path().join("d/train/labelTxt/x.txt")).unwrap();
    let vals: Vec<f64> = raw
        .split_whitespace()
        .take(8)
        .map(|v| v.parse().unwrap())
        .collect();
    let xs: Vec<f64> = vals.iter().step_by(2).copied().collect();
    assert!(xs.iter().all(|&x| (0.0..=200.0).contains(&x)));

    let mut c = CocoWriter::new(tmp.path().join("c"), vec!["a".into()]);
    c.write_labels("train", "x.jpg", 200, 100, &[a]);
    c.finish().unwrap();
    let coco: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(tmp.path().join("c/train/_annotations.coco.json")).unwrap(),
    )
    .unwrap();
    let seg: Vec<f64> = coco["annotations"][0]["segmentation"][0]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    assert!(seg.iter().step_by(2).all(|&x| x >= 0.0));
}

#[test]
fn test_canvas_delete_selected_and_last_box() {
    let mut c = CanvasState::default();
    c.load_image(
        100,
        100,
        vec![
            Annotation::new(0, 20.0, 20.0, 15.0, 15.0, 0.0, BoxKind::Bbox),
            Annotation::new(0, 60.0, 60.0, 15.0, 15.0, 0.0, BoxKind::Bbox),
        ],
        true,
    );
    c.select(Some(0));
    assert!(c.delete_selected());
    assert_eq!(c.items.len(), 1);
    assert_eq!(c.selected, Some(0));
    assert!(c.delete_selected());
    assert!(c.items.is_empty());
    assert_eq!(c.selected, None);
    // deleting when empty is a safe no-op
    assert!(!c.delete_selected());
    // stale out-of-range selection is safely cleared
    c.selected = Some(99);
    assert!(!c.delete_selected());
    assert_eq!(c.selected, None);
}

#[test]
fn test_canvas_delete_while_drawing_cancels_drag() {
    let mut c = CanvasState::default();
    c.load_image(100, 100, Vec::new(), true);
    c.drag = Some(DragState::Draw {
        start_img: [10.0, 10.0],
        end_img: [40.0, 40.0],
    });
    assert!(!c.cancel_or_delete());
    assert!(c.drag.is_none());
    assert!(c.items.is_empty());
}

#[test]
fn test_canvas_delete_while_resizing_clears_drag() {
    let mut c = CanvasState::default();
    let a = Annotation::new(0, 50.0, 50.0, 20.0, 20.0, 0.0, BoxKind::Bbox);
    c.load_image(100, 100, vec![a.clone()], true);
    c.select(Some(0));
    c.drag = Some(DragState::Resize {
        idx: 0,
        handle: HandleId::Br,
        orig: a.clone(),
        before: vec![a],
    });
    assert!(c.delete_selected());
    assert!(c.drag.is_none());
    assert!(c.items.is_empty());
    // finishing drag after deletion must not crash or resurrect box
    assert!(!c.finish_drag());
    assert!(c.items.is_empty());
}

#[test]
fn test_canvas_label_display_modes() {
    let mut c = CanvasState::default();
    c.load_image(
        100,
        100,
        vec![
            Annotation::new(0, 50.0, 50.0, 20.0, 20.0, 0.0, BoxKind::Bbox),
            Annotation::new(0, 50.0, 50.0, 20.0, 20.0, 0.0, BoxKind::Bbox),
        ],
        true,
    );
    assert!(c.label_visible(0) && c.label_visible(1));
    c.label_mode = c.label_mode.cycle(); // Selected
    assert!(!c.label_visible(0) && !c.label_visible(1));
    c.select(Some(1));
    assert!(c.label_visible(1) && !c.label_visible(0));
    c.label_mode = c.label_mode.cycle(); // None
    assert!(!c.label_visible(1) && c.box_visible(0) && c.box_visible(1));
    c.label_mode = c.label_mode.cycle(); // Hidden
    assert!(c.box_visible(1) && !c.box_visible(0));
    c.select(None);
    assert!(!c.box_visible(0) && !c.box_visible(1));
    assert!(c.item_at([50.0, 50.0]).is_none());
    c.label_mode = c.label_mode.cycle(); // All
    assert_eq!(c.label_mode, LabelMode::All);
    assert!(c.item_at([50.0, 50.0]).is_some());
}

#[test]
fn test_in_place_overwrite_same_and_separate_folder() {
    let tmp = TempDir::new().unwrap();
    // Case 1: images and labels in same folder
    let same = tmp.path().join("same");
    make_images(&same, &["a.png"]);
    std::fs::write(same.join("a.txt"), "0 0.1 0.1 0.5 0.1 0.5 0.3 0.1 0.3\n").unwrap();
    let mut p = Project::open(&same).unwrap();
    assert!(!same.join("rotalabel_labels").exists());
    p.get_mut("a.png")
        .annotations
        .push(Annotation::new(0, 50.0, 50.0, 20.0, 20.0, 0.0, BoxKind::Bbox));
    p.save_annotations("a.png", false);
    assert_eq!(
        std::fs::read_to_string(same.join("a.txt"))
            .unwrap()
            .trim()
            .lines()
            .count(),
        2
    );

    // Case 2: images and labels in separate folders
    let imgs = tmp.path().join("imgs");
    let lbls = tmp.path().join("imgs_label");
    make_images(&imgs, &["b.png"]);
    std::fs::create_dir_all(&lbls).unwrap();
    std::fs::write(lbls.join("classes.txt"), "nail\n").unwrap();
    std::fs::write(lbls.join("b.txt"), "0 0.1 0.1 0.5 0.1 0.5 0.3 0.1 0.3\n").unwrap();
    let mut p2 = Project::open(&imgs).unwrap();
    importers::import_labels(&mut p2, &lbls, true);
    p2.get_mut("b.png")
        .annotations
        .push(Annotation::new(0, 50.0, 50.0, 20.0, 20.0, 0.0, BoxKind::Bbox));
    p2.save_annotations("b.png", false);
    assert_eq!(
        std::fs::read_to_string(lbls.join("b.txt"))
            .unwrap()
            .trim()
            .lines()
            .count(),
        2
    );
    assert!(!imgs.join("b.txt").exists());
    assert!(!imgs.join("rotalabel_labels").exists());
}

#[test]
fn test_theme_dark_and_light_override_system_theme() {
    let ctx = egui::Context::default();

    let dpal = apply_theme(&ctx, true);
    let _ = ctx.run(
        egui::RawInput {
            system_theme: Some(egui::Theme::Light),
            ..Default::default()
        },
        |_| {},
    );
    assert_eq!(ctx.theme(), egui::Theme::Dark);
    let dv = ctx.style().visuals.clone();
    assert!(dv.dark_mode);
    assert_eq!(dv.panel_fill, dark_palette().bg);
    assert_eq!(dv.window_fill, dpal.panel);
    assert_eq!(dv.widgets.noninteractive.weak_bg_fill, dpal.panel);
    assert_eq!(dv.widgets.inactive.weak_bg_fill, dpal.panel2);
    assert_eq!(dv.widgets.open.weak_bg_fill, dpal.panel2);
    assert_eq!(dv.selection.stroke.color, dpal.text);

    let lpal = apply_theme(&ctx, false);
    let _ = ctx.run(
        egui::RawInput {
            system_theme: Some(egui::Theme::Dark),
            ..Default::default()
        },
        |_| {},
    );
    assert_eq!(ctx.theme(), egui::Theme::Light);
    let lv = ctx.style().visuals.clone();
    assert!(!lv.dark_mode);
    assert_eq!(lv.panel_fill, light_palette().bg);
    assert_eq!(lv.window_fill, lpal.panel);
    assert_eq!(lv.widgets.noninteractive.weak_bg_fill, lpal.panel);
    assert_eq!(lv.widgets.inactive.weak_bg_fill, lpal.panel2);
    assert_eq!(lv.widgets.open.weak_bg_fill, lpal.panel2);
    assert_eq!(lv.selection.stroke.color, lpal.text);
}

#[test]
fn test_preview_visibility_matches_label_mode() {
    assert_eq!(preview_visibility(LabelMode::All, false), (true, true));
    assert_eq!(preview_visibility(LabelMode::All, true), (true, true));
    assert_eq!(preview_visibility(LabelMode::Selected, false), (true, false));
    assert_eq!(preview_visibility(LabelMode::Selected, true), (true, true));
    assert_eq!(preview_visibility(LabelMode::None, false), (true, false));
    assert_eq!(preview_visibility(LabelMode::None, true), (true, false));
    assert_eq!(preview_visibility(LabelMode::Hidden, false), (false, false));
    assert_eq!(preview_visibility(LabelMode::Hidden, true), (false, false));
}

#[test]
fn test_canvas_copy_paste_and_small_dataset_parallel_determinism() {
    let mut c = CanvasState::default();
    let a = Annotation::new(0, 30.0, 40.0, 20.0, 10.0, 15.0, BoxKind::Obb);
    c.load_image(100, 100, vec![a], true);
    c.select(Some(0));
    c.copy_selected();
    assert_eq!(c.clipboard.len(), 1);
    assert!(c.paste(1));
    assert_eq!(c.items.len(), 2);
    assert_eq!(c.selected, Some(1));

    let total_threads = std::thread::available_parallelism()
        .map(|v| v.get())
        .unwrap_or(1);
    if total_threads >= 2 {
        assert_eq!(default_workers(), total_threads - 1);
    } else {
        assert_eq!(default_workers(), 1);
    }

    let tmp = TempDir::new().unwrap();
    let imgs = tmp.path().join("imgs");
    make_images(&imgs, &["a.png", "b.png"]);
    let mut p = Project::open(&imgs).unwrap();
    p.add_class("obj");
    for name in ["a.png", "b.png"] {
        p.get_mut(name)
            .annotations
            .push(Annotation::new(0, 50.0, 50.0, 30.0, 20.0, 10.0, BoxKind::Obb));
    }
    let mut cfg = VersionConfig {
        train: 100.0,
        valid: 0.0,
        test: 0.0,
        multiplier: 5,
        formats: vec!["yolo_obb".into()],
        aug: AugConfig {
            flip_h: true,
            rotate: 20.0,
            exposure: 15.0,
            ..Default::default()
        },
        workers: 1,
        ..Default::default()
    };
    let out1 = tmp.path().join("v1");
    let s1 = generate_version(&p, &cfg, &out1, None, None).unwrap();
    cfg.workers = 4;
    let out4 = tmp.path().join("v4");
    let s4 = generate_version(&p, &cfg, &out4, None, None).unwrap();
    assert_eq!(s1.train, 10);
    assert_eq!((s1.train, s1.boxes), (s4.train, s4.boxes));
    assert_eq!(
        std::fs::read_to_string(out1.join("train/labels/a_aug3.txt")).unwrap(),
        std::fs::read_to_string(out4.join("train/labels/a_aug3.txt")).unwrap()
    );
}
