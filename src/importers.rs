//! Import existing label files (YOLO, YOLO-OBB, DOTA) into a project.
//!
//! Format is detected per line:
//!   5 tokens, numeric             -> YOLO bbox (normalized)
//!   9 tokens, numeric             -> YOLO-OBB  (normalized corners)
//!   >= 9 tokens, last 1-2 textual -> DOTA      (pixel corners + class name [+ difficult])

use std::path::Path;

use crate::imageio;
use crate::model::{Annotation, BoxKind};
use crate::project_io::{clean_path, Project, DOTA, YOLO_OBB};

fn is_num(s: &str) -> bool {
    s.parse::<f64>().is_ok()
}

pub enum ClassKey<'a> {
    Id(usize),
    Name(&'a str),
}

pub fn read_names(label_dir: &Path) -> Option<Vec<String>> {
    let mut txt_cands = vec![label_dir.join("classes.txt")];
    if let Some(p) = label_dir.parent() {
        txt_cands.push(p.join("classes.txt"));
    }
    for cand in txt_cands {
        if cand.exists() {
            if let Ok(raw) = std::fs::read_to_string(&cand) {
                let names: Vec<String> = raw
                    .lines()
                    .map(|l| l.trim().to_string())
                    .filter(|l| !l.is_empty())
                    .collect();
                return Some(names);
            }
        }
    }

    let mut yaml_cands = vec![label_dir.join("data.yaml")];
    if let Some(p) = label_dir.parent() {
        yaml_cands.push(p.join("data.yaml"));
        if let Some(pp) = p.parent() {
            yaml_cands.push(pp.join("data.yaml"));
        }
    }
    for cand in yaml_cands {
        if cand.exists() {
            if let Ok(raw) = std::fs::read_to_string(&cand) {
                if let Ok(val) = serde_yaml::from_str::<serde_yaml::Value>(&raw) {
                    if let Some(names) = val.get("names") {
                        match names {
                            serde_yaml::Value::Mapping(map) => {
                                let mut entries: Vec<(i64, String, String)> = map
                                    .iter()
                                    .filter_map(|(k, v)| {
                                        let ks = match k {
                                            serde_yaml::Value::Number(n) => {
                                                return Some((
                                                    n.as_i64().unwrap_or(0),
                                                    String::new(),
                                                    yaml_val_to_string(v),
                                                ));
                                            }
                                            serde_yaml::Value::String(s) => s.clone(),
                                            _ => return None,
                                        };
                                        let ki = ks.parse::<i64>().unwrap_or(i64::MAX);
                                        Some((ki, ks, yaml_val_to_string(v)))
                                    })
                                    .collect();
                                entries.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
                                return Some(entries.into_iter().map(|e| e.2).collect());
                            }
                            serde_yaml::Value::Sequence(seq) => {
                                return Some(seq.iter().map(yaml_val_to_string).collect());
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
    }
    None
}

fn yaml_val_to_string(v: &serde_yaml::Value) -> String {
    match v {
        serde_yaml::Value::String(s) => s.clone(),
        serde_yaml::Value::Number(n) => n.to_string(),
        serde_yaml::Value::Bool(b) => b.to_string(),
        _ => String::new(),
    }
}

pub fn detect_format(path: &Path) -> Option<&'static str> {
    let raw = std::fs::read_to_string(path).ok()?;
    let lines: Vec<Vec<&str>> = raw
        .lines()
        .map(|l| l.split_whitespace().collect::<Vec<_>>())
        .filter(|t| !t.is_empty() && !t[0].starts_with("imagesource") && !t[0].starts_with("gsd"))
        .collect();
    if lines.is_empty() {
        return None;
    }
    if lines.iter().all(|t| t.len() == 9 && t.iter().all(|v| is_num(v))) {
        return Some(YOLO_OBB);
    }
    if lines.iter().all(|t| {
        t.len() >= 9 && t[..8].iter().all(|v| is_num(v)) && !is_num(t[8])
    }) {
        return Some(DOTA);
    }
    None
}

pub fn parse_label_file<F>(
    path: &Path,
    width: u32,
    height: u32,
    mut resolve_class: F,
) -> Vec<Annotation>
where
    F: FnMut(ClassKey<'_>) -> usize,
{
    let Ok(raw) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    let fw = width as f64;
    let fh = height as f64;
    let mut anns = Vec::new();
    for line in raw.lines() {
        let t: Vec<&str> = line.split_whitespace().collect();
        if t.is_empty() || t[0].starts_with("imagesource") || t[0].starts_with("gsd") {
            continue;
        }
        if t.len() == 5 && t.iter().all(|v| is_num(v)) {
            let cls_num = t[0].parse::<f64>().unwrap_or(0.0).max(0.0) as usize;
            let c = resolve_class(ClassKey::Id(cls_num));
            let cx = t[1].parse::<f64>().unwrap_or(0.0) * fw;
            let cy = t[2].parse::<f64>().unwrap_or(0.0) * fh;
            let w = t[3].parse::<f64>().unwrap_or(0.0) * fw;
            let h = t[4].parse::<f64>().unwrap_or(0.0) * fh;
            anns.push(Annotation::new(c, cx, cy, w, h, 0.0, BoxKind::Bbox));
        } else if t.len() == 9 && t.iter().all(|v| is_num(v)) {
            let cls_num = t[0].parse::<f64>().unwrap_or(0.0).max(0.0) as usize;
            let c = resolve_class(ClassKey::Id(cls_num));
            let mut pts = [[0.0; 2]; 4];
            for i in 0..4 {
                pts[i][0] = t[1 + i * 2].parse::<f64>().unwrap_or(0.0) * fw;
                pts[i][1] = t[2 + i * 2].parse::<f64>().unwrap_or(0.0) * fh;
            }
            anns.push(Annotation::from_corners(c, &pts, BoxKind::Obb));
        } else if t.len() >= 9 && t[..8].iter().all(|v| is_num(v)) && !is_num(t[8]) {
            let c = resolve_class(ClassKey::Name(t[8]));
            let mut pts = [[0.0; 2]; 4];
            for i in 0..4 {
                pts[i][0] = t[i * 2].parse::<f64>().unwrap_or(0.0);
                pts[i][1] = t[1 + i * 2].parse::<f64>().unwrap_or(0.0);
            }
            anns.push(Annotation::from_corners(c, &pts, BoxKind::Obb));
        }
    }
    anns
}

/// Returns `(images_updated, boxes_imported)`.
pub fn import_labels(project: &mut Project, label_dir: &Path, overwrite: bool) -> (usize, usize) {
    let resolved = clean_path(label_dir);
    let root_clean = clean_path(&project.root);
    let dir_val = if let Ok(rel) = resolved.strip_prefix(&root_clean) {
        let s = rel.to_string_lossy().replace('\\', "/");
        if s.is_empty() || s == "." {
            String::new()
        } else {
            s
        }
    } else {
        resolved.to_string_lossy().to_string()
    };
    project
        .settings
        .insert("label_dir".to_string(), serde_json::Value::String(dir_val));

    let names = read_names(label_dir);

    for name in &project.images {
        let stem = Path::new(name)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(name);
        let f = label_dir.join(format!("{stem}.txt"));
        if f.exists() {
            if let Some(fmt) = detect_format(&f) {
                project.settings.insert(
                    "label_format".to_string(),
                    serde_json::Value::String(fmt.to_string()),
                );
                break;
            }
        }
    }

    let images = project.images.clone();
    let mut updated = 0usize;
    let mut boxes = 0usize;

    for name in images {
        let stem = Path::new(&name)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(&name);
        let f = label_dir.join(format!("{stem}.txt"));
        if !f.exists() {
            continue;
        }
        let (mut width, mut height) = {
            let ia = project.get_mut(&name);
            (ia.width, ia.height)
        };
        if width == 0 || height == 0 {
            let Some(img) = imageio::imread(&project.image_path(&name)) else {
                continue;
            };
            width = img.width();
            height = img.height();
            let ia = project.get_mut(&name);
            ia.width = width;
            ia.height = height;
        }

        let anns = parse_label_file(&f, width, height, |key| match key {
            ClassKey::Name(cname) => project.class_index(cname, true).unwrap_or(0),
            ClassKey::Id(id) => {
                if let Some(ref ns) = names {
                    let cname = ns
                        .get(id)
                        .cloned()
                        .unwrap_or_else(|| format!("class_{id}"));
                    project.class_index(&cname, true).unwrap_or(0)
                } else {
                    while id >= project.classes.len() {
                        let cname = format!("class_{}", project.classes.len());
                        project.add_class(cname);
                    }
                    id
                }
            }
        });

        let count = anns.len();
        {
            let ia = project.get_mut(&name);
            if overwrite {
                ia.annotations = anns;
            } else {
                ia.annotations.extend(anns);
            }
        }
        project.save_annotations(&name, true);
        updated += 1;
        boxes += count;
    }

    project.save_meta();
    (updated, boxes)
}
