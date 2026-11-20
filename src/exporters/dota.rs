use std::collections::HashSet;
use std::path::{Path, PathBuf};

use crate::exporters::base::{clipped_corners, fmt};
use crate::model::Annotation;

#[derive(Debug, Clone)]
pub struct DotaWriter {
    pub out: PathBuf,
    pub class_names: Vec<String>,
    pub used_splits: HashSet<String>,
}

impl DotaWriter {
    pub fn new(out: impl Into<PathBuf>, class_names: Vec<String>) -> Self {
        Self {
            out: out.into(),
            class_names,
            used_splits: HashSet::new(),
        }
    }

    pub fn image_dir(&self, split: &str) -> std::io::Result<PathBuf> {
        let d = self.out.join(split).join("images");
        std::fs::create_dir_all(&d)?;
        Ok(d)
    }

    pub fn write_labels(
        &self,
        split: &str,
        filename: &str,
        width: u32,
        height: u32,
        anns: &[Annotation],
    ) -> std::io::Result<()> {
        let d = self.out.join(split).join("labelTxt");
        std::fs::create_dir_all(&d)?;
        let mut lines = Vec::with_capacity(anns.len());
        for a in anns {
            let pts = clipped_corners(a, width, height);
            let mut parts = Vec::with_capacity(10);
            for [x, y] in pts {
                parts.push(fmt(x));
                parts.push(fmt(y));
            }
            let cname = self
                .class_names
                .get(a.cls)
                .map(|s| s.replace(' ', "_"))
                .unwrap_or_else(|| format!("class_{}", a.cls));
            parts.push(cname);
            parts.push("0".to_string());
            lines.push(parts.join(" "));
        }
        let mut content = lines.join("\n");
        if !lines.is_empty() {
            content.push('\n');
        }
        let stem = Path::new(filename)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(filename);
        std::fs::write(d.join(format!("{stem}.txt")), content)
    }

    pub fn finish(&self) -> std::io::Result<()> {
        std::fs::create_dir_all(&self.out)?;
        let mut content = self
            .class_names
            .iter()
            .map(|n| n.replace(' ', "_"))
            .collect::<Vec<_>>()
            .join("\n");
        content.push('\n');
        std::fs::write(self.out.join("classes.txt"), content)
    }
}
