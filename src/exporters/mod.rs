pub mod base;
pub mod coco;
pub mod coco_obb;
pub mod dota;
pub mod yolo;

use std::path::{Path, PathBuf};

pub use coco::CocoWriter;
pub use coco_obb::CocoObbWriter;
pub use dota::DotaWriter;
pub use yolo::YoloWriter;

use crate::model::Annotation;

pub const FORMAT_OPTIONS: [(&str, &str); 5] = [
    ("yolo", "YOLO (bbox)"),
    ("yolo_obb", "YOLO-OBB (旋轉框)"),
    ("dota", "DOTA (旋轉框)"),
    ("coco", "COCO JSON"),
    ("coco_obb", "COCO-OBB (旋轉框，PP-YOLOE-R)"),
];

#[derive(Debug, Clone)]
pub enum Writer {
    Yolo(YoloWriter),
    YoloObb(YoloWriter),
    Dota(DotaWriter),
    Coco(CocoWriter),
    CocoObb(CocoObbWriter),
}

impl Writer {
    pub fn from_key(key: &str, out: PathBuf, class_names: Vec<String>) -> Option<Self> {
        match key {
            "yolo" => Some(Writer::Yolo(YoloWriter::new(out, class_names, false))),
            "yolo_obb" => Some(Writer::YoloObb(YoloWriter::new(out, class_names, true))),
            "dota" => Some(Writer::Dota(DotaWriter::new(out, class_names))),
            "coco" => Some(Writer::Coco(CocoWriter::new(out, class_names))),
            "coco_obb" => Some(Writer::CocoObb(CocoObbWriter::new(out, class_names))),
            _ => None,
        }
    }

    pub fn per_image_labels(&self) -> bool {
        matches!(self, Writer::Yolo(_) | Writer::YoloObb(_) | Writer::Dota(_))
    }

    pub fn image_dir(&self, split: &str) -> std::io::Result<PathBuf> {
        match self {
            Writer::Yolo(w) | Writer::YoloObb(w) => w.image_dir(split),
            Writer::Dota(w) => w.image_dir(split),
            Writer::Coco(w) => w.image_dir(split),
            Writer::CocoObb(w) => w.image_dir(split),
        }
    }

    /// Worker-safe write: writes encoded image bytes and (if per-image) label file.
    pub fn write_worker(
        &self,
        split: &str,
        filename: &str,
        image_bytes: &[u8],
        width: u32,
        height: u32,
        anns: &[Annotation],
    ) -> std::io::Result<()> {
        let dir = self.image_dir(split)?;
        std::fs::write(dir.join(filename), image_bytes)?;
        match self {
            Writer::Yolo(w) | Writer::YoloObb(w) => {
                w.write_labels(split, filename, width, height, anns)?;
            }
            Writer::Dota(w) => {
                w.write_labels(split, filename, width, height, anns)?;
            }
            Writer::Coco(_) | Writer::CocoObb(_) => {}
        }
        Ok(())
    }

    /// Main-thread commit: records `used_splits` and accumulates non-per-image labels (COCO).
    pub fn main_commit(
        &mut self,
        split: &str,
        filename: &str,
        width: u32,
        height: u32,
        anns: &[Annotation],
    ) {
        match self {
            Writer::Yolo(w) | Writer::YoloObb(w) => {
                w.used_splits.insert(split.to_string());
            }
            Writer::Dota(w) => {
                w.used_splits.insert(split.to_string());
            }
            Writer::Coco(w) => {
                w.used_splits.insert(split.to_string());
                w.write_labels(split, filename, width, height, anns);
            }
            Writer::CocoObb(w) => {
                w.used_splits.insert(split.to_string());
                w.write_labels(split, filename, width, height, anns);
            }
        }
    }

    /// Single-step convenience helper (matching Python `BaseWriter.add`).
    pub fn add(
        &mut self,
        split: &str,
        filename: &str,
        image_bytes: &[u8],
        width: u32,
        height: u32,
        anns: &[Annotation],
    ) -> std::io::Result<()> {
        self.write_worker(split, filename, image_bytes, width, height, anns)?;
        self.main_commit(split, filename, width, height, anns);
        Ok(())
    }

    pub fn finish(&self) -> std::io::Result<()> {
        match self {
            Writer::Yolo(w) | Writer::YoloObb(w) => w.finish(),
            Writer::Dota(w) => w.finish(),
            Writer::Coco(w) => w.finish(),
            Writer::CocoObb(w) => w.finish(),
        }
    }
}

pub fn make_writers(
    out: &Path,
    formats: &[String],
    class_names: &[String],
) -> Vec<(String, Writer)> {
    let multi = formats.len() > 1;
    let mut writers = Vec::new();
    for fmt in formats {
        let dir = if multi { out.join(fmt) } else { out.to_path_buf() };
        if let Some(w) = Writer::from_key(fmt, dir, class_names.to_vec()) {
            writers.push((fmt.clone(), w));
        }
    }
    writers
}
