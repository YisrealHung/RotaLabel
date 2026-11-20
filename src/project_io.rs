//! A project is an image folder; metadata lives in `<folder>/rotalabel_project/`.
//!
//!   <folder>/rotalabel_project/project.json          classes + settings
//!   <folder>/rotalabel_project/annotations/<img>.json one file per image
//!   <folder>/rotalabel_project/versions/             generated dataset versions
//!   <folder>/<img stem>.txt (or <label_dir>/...)     side label files (overwritten in-place)

use std::cmp::Ordering;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::exporters::base::{clipped_corners, fmt as fmt6};
use crate::imageio;
use crate::importers;
use crate::model::{default_color, BoxKind, ClassDef, ImageAnnotations};

pub const META_DIRNAME: &str = "rotalabel_project";
pub const LEGACY_META_DIRNAMES: &[&str] = &["bboxer_project"];
pub const SIDECAR_HEADER: &str = "imagesource:rotalabel";
pub const DOTA: &str = "dota";
pub const YOLO_OBB: &str = "yolo_obb";
pub const DEFAULT_LABEL_DIR: &str = "rotalabel_labels";

pub const LABEL_FORMATS: [(&str, &str); 2] = [(YOLO_OBB, "YOLO-OBB"), (DOTA, "DOTA")];

pub fn clean_path(p: &Path) -> PathBuf {
    let abs = std::fs::canonicalize(p).unwrap_or_else(|_| {
        if p.is_absolute() {
            p.to_path_buf()
        } else {
            std::env::current_dir().map(|c| c.join(p)).unwrap_or_else(|_| p.to_path_buf())
        }
    });
    let s = abs.to_string_lossy();
    if let Some(stripped) = s.strip_prefix(r"\\?\UNC\") {
        PathBuf::from(format!(r"\\{stripped}"))
    } else if let Some(stripped) = s.strip_prefix(r"\\?\") {
        PathBuf::from(stripped)
    } else {
        abs
    }
}

pub fn meta_dir_for(root: &Path) -> PathBuf {
    let new = root.join(META_DIRNAME);
    if !new.exists() {
        for name in LEGACY_META_DIRNAMES {
            let legacy = root.join(name);
            if legacy.is_dir() {
                return legacy;
            }
        }
    }
    new
}

fn fmt2(v: f64) -> String {
    let s = format!("{:.2}", v);
    let trimmed = s.trim_end_matches('0').trim_end_matches('.');
    if trimmed.is_empty() || trimmed == "-0" {
        "0".to_string()
    } else {
        trimmed.to_string()
    }
}

#[derive(Debug, PartialEq, Eq)]
enum NatChunk {
    Num(u128, usize),
    Text(String),
}

fn natural_chunks(s: &str) -> Vec<NatChunk> {
    let mut chunks = Vec::new();
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i].is_ascii_digit() {
            let start = i;
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                i += 1;
            }
            let slice = &s[start..i];
            let val = slice.parse::<u128>().unwrap_or(u128::MAX);
            chunks.push(NatChunk::Num(val, slice.len()));
        } else {
            let start = i;
            while i < bytes.len() && !bytes[i].is_ascii_digit() {
                i += 1;
            }
            chunks.push(NatChunk::Text(s[start..i].to_lowercase()));
        }
    }
    chunks
}

pub fn natural_cmp(a: &str, b: &str) -> Ordering {
    let ca = natural_chunks(a);
    let cb = natural_chunks(b);
    for (xa, xb) in ca.iter().zip(cb.iter()) {
        let ord = match (xa, xb) {
            (NatChunk::Num(na, la), NatChunk::Num(nb, lb)) => na.cmp(nb).then_with(|| la.cmp(lb)),
            (NatChunk::Text(ta), NatChunk::Text(tb)) => ta.cmp(tb),
            (NatChunk::Num(..), NatChunk::Text(_)) => Ordering::Less,
            (NatChunk::Text(_), NatChunk::Num(..)) => Ordering::Greater,
        };
        if ord != Ordering::Equal {
            return ord;
        }
    }
    ca.len().cmp(&cb.len()).then_with(|| a.cmp(b))
}

#[derive(Debug, Deserialize, Serialize)]
struct ProjectMetaFile {
    #[serde(default = "default_version")]
    version: u32,
    #[serde(default)]
    classes: Vec<ClassDef>,
    #[serde(default)]
    settings: Map<String, Value>,
}

fn default_version() -> u32 {
    1
}

#[derive(Debug, Clone)]
pub struct Project {
    pub root: PathBuf,
    pub meta_dir: PathBuf,
    pub ann_dir: PathBuf,
    pub versions_dir: PathBuf,
    pub classes: Vec<ClassDef>,
    pub images: Vec<String>,
    pub stem_counts: HashMap<String, usize>,
    pub annotations: HashMap<String, ImageAnnotations>,
    pub settings: Map<String, Value>,
    pub auto_imported: (usize, usize),
}

impl Project {
    pub fn new(root: impl AsRef<Path>) -> Self {
        let root_path = clean_path(root.as_ref());
        let meta_dir = meta_dir_for(&root_path);
        let ann_dir = meta_dir.join("annotations");
        let versions_dir = meta_dir.join("versions");
        Self {
            root: root_path,
            meta_dir,
            ann_dir,
            versions_dir,
            classes: Vec::new(),
            images: Vec::new(),
            stem_counts: HashMap::new(),
            annotations: HashMap::new(),
            settings: Map::new(),
            auto_imported: (0, 0),
        }
    }

    pub fn open(root: impl AsRef<Path>) -> std::io::Result<Self> {
        let mut p = Self::new(root);
        std::fs::create_dir_all(&p.ann_dir)?;
        let meta = p.meta_dir.join("project.json");
        if meta.exists() {
            if let Ok(raw) = std::fs::read_to_string(&meta) {
                if let Ok(d) = serde_json::from_str::<ProjectMetaFile>(&raw) {
                    p.classes = d
                        .classes
                        .into_iter()
                        .enumerate()
                        .map(|(i, c)| ClassDef {
                            name: c.name,
                            color: if c.color.is_empty() {
                                default_color(i)
                            } else {
                                c.color
                            },
                        })
                        .collect();
                    p.settings = d.settings;
                }
            }
        }

        let mut images = Vec::new();
        for entry in std::fs::read_dir(&p.root)? {
            let entry = entry?;
            let path = entry.path();
            if imageio::is_image_path(&path) {
                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                    images.push(name.to_string());
                }
            }
        }
        images.sort_by(|a, b| natural_cmp(a, b));
        for name in &images {
            let stem = Path::new(name)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or(name)
                .to_string();
            *p.stem_counts.entry(stem).or_insert(0) += 1;
        }
        p.images = images;

        for name in &p.images {
            let f = p.ann_dir.join(format!("{name}.json"));
            if f.exists() {
                if let Ok(raw) = std::fs::read_to_string(&f) {
                    if let Ok(ia) = serde_json::from_str::<ImageAnnotations>(&raw) {
                        p.annotations.insert(name.clone(), ia);
                    }
                }
            }
        }

        if !p.settings.contains_key("label_format") && p.annotations.is_empty() {
            if let Some(fmt) = p.detect_label_files() {
                p.adopt_label_files(fmt);
            }
        }
        if p.classes.is_empty() {
            p.classes.push(ClassDef {
                name: "object".to_string(),
                color: default_color(0),
            });
            p.save_meta();
        }
        p.sync_label_files();
        Ok(p)
    }

    pub fn label_format(&self) -> &str {
        self.settings
            .get("label_format")
            .and_then(|v| v.as_str())
            .unwrap_or(DOTA)
    }

    /// Folder for side files, or `None` = next to the images (`self.root`).
    pub fn output_dir(&self) -> Option<PathBuf> {
        let d = self.settings.get("label_dir").and_then(|v| v.as_str())?;
        if d.is_empty() || d == "." {
            return None;
        }
        let path = PathBuf::from(d);
        if path.is_absolute() {
            Some(path)
        } else {
            Some(self.root.join(path))
        }
    }

    pub fn set_label_output(&mut self, fmt: &str, label_dir: &Path) -> Result<usize, String> {
        if fmt != YOLO_OBB && fmt != DOTA {
            return Err(format!("unknown label format: {fmt}"));
        }
        let resolved = clean_path(label_dir);
        let root_clean = clean_path(&self.root);
        let val = if let Ok(rel) = resolved.strip_prefix(&root_clean) {
            let s = rel.to_string_lossy().replace('\\', "/");
            if s.is_empty() || s == "." {
                String::new()
            } else {
                s
            }
        } else {
            resolved.to_string_lossy().to_string()
        };
        self.settings
            .insert("label_format".to_string(), Value::String(fmt.to_string()));
        self.settings
            .insert("label_dir".to_string(), Value::String(val));
        self.save_meta();
        Ok(self.sync_label_files())
    }

    fn foreign_label_lines(&self, name: &str) -> Option<Vec<Vec<String>>> {
        let f = self.label_path(&self.root, name);
        if !f.exists() {
            return None;
        }
        let text = std::fs::read_to_string(&f).ok()?;
        let lines: Vec<Vec<String>> = text
            .lines()
            .map(|l| l.split_whitespace().map(|t| t.to_string()).collect::<Vec<_>>())
            .collect();
        if let Some(first) = lines.first() {
            if first.len() == 1 && first[0] == SIDECAR_HEADER {
                return None;
            }
        }
        Some(lines.into_iter().filter(|t| !t.is_empty()).collect())
    }

    fn detect_label_files(&self) -> Option<&'static str> {
        let is_nums = |ts: &[String]| ts.iter().all(|v| v.parse::<f64>().is_ok());
        for name in &self.images {
            let Some(lines) = self.foreign_label_lines(name) else {
                continue;
            };
            let lines: Vec<Vec<String>> = lines
                .into_iter()
                .filter(|t| !t[0].starts_with("imagesource") && !t[0].starts_with("gsd"))
                .collect();
            if lines.is_empty() {
                continue;
            }
            if lines.iter().all(|t| t.len() == 9 && is_nums(t)) {
                return Some(YOLO_OBB);
            }
            if lines.iter().all(|t| {
                (t.len() == 9 || t.len() == 10) && is_nums(&t[..8]) && t[8].parse::<f64>().is_err()
            }) {
                return Some(DOTA);
            }
        }
        None
    }

    fn adopt_label_files(&mut self, fmt: &str) {
        let names = importers::read_names(&self.root);
        let default_only = self.classes.is_empty()
            || (self.classes.len() == 1 && self.classes[0].name == "object");
        if default_only {
            self.classes = names
                .unwrap_or_default()
                .into_iter()
                .enumerate()
                .map(|(i, n)| ClassDef {
                    name: n,
                    color: default_color(i),
                })
                .collect();
        }
        self.settings
            .insert("label_format".to_string(), Value::String(fmt.to_string()));
        self.settings
            .insert("label_dir".to_string(), Value::String(String::new()));
        self.settings
            .insert("label_source".to_string(), Value::String(".".to_string()));
        self.save_meta();
        let root_clone = self.root.clone();
        self.auto_imported = importers::import_labels(self, &root_clone, true);
    }

    pub fn save_meta(&self) {
        let _ = std::fs::create_dir_all(&self.meta_dir);
        let d = ProjectMetaFile {
            version: 1,
            classes: self.classes.clone(),
            settings: self.settings.clone(),
        };
        if let Ok(json) = serde_json::to_string_pretty(&d) {
            let _ = std::fs::write(self.meta_dir.join("project.json"), json);
        }
        if self.label_format() == YOLO_OBB && !self.classes.is_empty() {
            let out = self.output_dir().unwrap_or_else(|| self.root.clone());
            if std::fs::create_dir_all(&out).is_ok() {
                let mut content = self.class_names().join("\n");
                content.push('\n');
                let _ = std::fs::write(out.join("classes.txt"), content);
            }
        }
    }

    pub fn save_annotations(&self, name: &str, keep_empty: bool) {
        let Some(ia) = self.annotations.get(name) else {
            return;
        };
        let f = self.ann_dir.join(format!("{name}.json"));
        if ia.annotations.is_empty() && !f.exists() && !keep_empty {
            return;
        }
        if std::fs::create_dir_all(&self.ann_dir).is_ok() {
            if let Ok(json) = serde_json::to_string_pretty(ia) {
                if std::fs::write(&f, json).is_err() {
                    return;
                }
            }
        }
        self.write_label_file(name);
    }

    pub fn label_path(&self, folder: &Path, name: &str) -> PathBuf {
        let stem = Path::new(name)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(name);
        if self.stem_counts.get(stem).copied().unwrap_or(0) > 1 {
            folder.join(format!("{name}.txt"))
        } else {
            folder.join(format!("{stem}.txt"))
        }
    }

    pub fn label_file(&self, name: &str) -> PathBuf {
        let folder = self.output_dir().unwrap_or_else(|| self.root.clone());
        self.label_path(&folder, name)
    }

    pub fn write_label_file(&self, name: &str) -> bool {
        let Some(ia) = self.annotations.get(name) else {
            return false;
        };
        if let Some(out) = self.output_dir() {
            if std::fs::create_dir_all(&out).is_err() {
                return false;
            }
        }
        if self.label_format() == YOLO_OBB {
            return self.write_yolo_obb(name, ia);
        }
        let f = self.label_file(name);
        let mut lines = vec![SIDECAR_HEADER.to_string(), "gsd:null".to_string()];
        for a in &ia.annotations {
            let cname = self
                .classes
                .get(a.cls)
                .map(|c| c.name.clone())
                .unwrap_or_else(|| format!("class_{}", a.cls));
            let mut parts = Vec::with_capacity(10);
            for [x, y] in a.corners() {
                parts.push(fmt2(x));
                parts.push(fmt2(y));
            }
            parts.push(cname.replace(' ', "_"));
            parts.push("0".to_string());
            lines.push(parts.join(" "));
        }
        let mut content = lines.join("\n");
        content.push('\n');
        std::fs::write(f, content).is_ok()
    }

    fn write_yolo_obb(&self, name: &str, ia: &ImageAnnotations) -> bool {
        if ia.width == 0 || ia.height == 0 {
            return false;
        }
        let fw = ia.width as f64;
        let fh = ia.height as f64;
        let mut lines = Vec::with_capacity(ia.annotations.len());
        for a in &ia.annotations {
            let mut parts = Vec::with_capacity(9);
            parts.push(a.cls.to_string());
            for [x, y] in clipped_corners(a, ia.width, ia.height) {
                parts.push(fmt6(x / fw));
                parts.push(fmt6(y / fh));
            }
            lines.push(parts.join(" "));
        }
        let mut content = lines.join("\n");
        if !lines.is_empty() {
            content.push('\n');
        }
        std::fs::write(self.label_file(name), content).is_ok()
    }

    pub fn sync_label_files(&self) -> usize {
        let mut written = 0;
        for name in &self.images {
            if self.annotations.contains_key(name)
                && self.ann_dir.join(format!("{name}.json")).exists()
                && self.write_label_file(name)
            {
                written += 1;
            }
        }
        written
    }

    pub fn image_path(&self, name: &str) -> PathBuf {
        self.root.join(name)
    }

    pub fn get_mut(&mut self, name: &str) -> &mut ImageAnnotations {
        self.annotations.entry(name.to_string()).or_default()
    }

    pub fn get(&self, name: &str) -> Option<&ImageAnnotations> {
        self.annotations.get(name)
    }

    pub fn box_count(&self, name: &str) -> usize {
        self.annotations
            .get(name)
            .map(|ia| ia.annotations.len())
            .unwrap_or(0)
    }

    pub fn class_names(&self) -> Vec<String> {
        self.classes.iter().map(|c| c.name.clone()).collect()
    }

    pub fn add_class(&mut self, name: impl Into<String>) -> usize {
        let idx = self.classes.len();
        self.classes.push(ClassDef {
            name: name.into(),
            color: default_color(idx),
        });
        self.save_meta();
        idx
    }

    pub fn class_index(&mut self, name: &str, create: bool) -> Option<usize> {
        if let Some(i) = self.classes.iter().position(|c| c.name == name) {
            return Some(i);
        }
        if create {
            Some(self.add_class(name))
        } else {
            None
        }
    }

    pub fn delete_class(&mut self, idx: usize) -> usize {
        if idx >= self.classes.len() {
            return 0;
        }
        let mut removed = 0;
        let names = self.images.clone();
        for name in names {
            let mut changed = false;
            if let Some(ia) = self.annotations.get_mut(&name) {
                let before = ia.annotations.len();
                let mut kept = Vec::with_capacity(before);
                for mut a in ia.annotations.drain(..) {
                    if a.cls == idx {
                        removed += 1;
                        changed = true;
                    } else {
                        if a.cls > idx {
                            a.cls -= 1;
                            changed = true;
                        }
                        kept.push(a);
                    }
                }
                ia.annotations = kept;
            }
            if changed {
                self.save_annotations(&name, false);
            }
        }
        self.classes.remove(idx);
        self.save_meta();
        self.sync_label_files();
        removed
    }

    pub fn label_kind_for_new_box(&self) -> BoxKind {
        if self.label_format() == YOLO_OBB {
            BoxKind::Obb
        } else {
            BoxKind::Bbox
        }
    }
}
