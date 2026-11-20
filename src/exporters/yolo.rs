use std::collections::HashSet;
use std::path::{Path, PathBuf};

use serde_yaml::{Mapping, Value};

use crate::exporters::base::{clipped_aabb, clipped_corners, fmt};
use crate::model::Annotation;

#[derive(Debug, Clone)]
pub struct YoloWriter {
    pub out: PathBuf,
    pub class_names: Vec<String>,
    pub used_splits: HashSet<String>,
    pub obb: bool,
}

impl YoloWriter {
    pub fn new(out: impl Into<PathBuf>, class_names: Vec<String>, obb: bool) -> Self {
        Self {
            out: out.into(),
            class_names,
            used_splits: HashSet::new(),
            obb,
        }
    }

    pub fn image_dir(&self, split: &str) -> std::io::Result<PathBuf> {
        let d = self.out.join(split).join("images");
        std::fs::create_dir_all(&d)?;
        Ok(d)
    }

    pub fn label_path(&self, split: &str, filename: &str) -> std::io::Result<PathBuf> {
        let d = self.out.join(split).join("labels");
        std::fs::create_dir_all(&d)?;
        let stem = Path::new(filename)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(filename);
        Ok(d.join(format!("{stem}.txt")))
    }

    pub fn write_labels(
        &self,
        split: &str,
        filename: &str,
        width: u32,
        height: u32,
        anns: &[Annotation],
    ) -> std::io::Result<()> {
        let fw = width as f64;
        let fh = height as f64;
        let mut lines = Vec::with_capacity(anns.len());
        for a in anns {
            if self.obb {
                let pts = clipped_corners(a, width, height);
                let mut parts = Vec::with_capacity(9);
                parts.push(a.cls.to_string());
                for [x, y] in pts {
                    parts.push(fmt(x / fw));
                    parts.push(fmt(y / fh));
                }
                lines.push(parts.join(" "));
            } else {
                let [x0, y0, x1, y1] = clipped_aabb(a, width, height);
                if x1 - x0 <= 0.0 || y1 - y0 <= 0.0 {
                    continue;
                }
                let line = [
                    a.cls.to_string(),
                    fmt((x0 + x1) / 2.0 / fw),
                    fmt((y0 + y1) / 2.0 / fh),
                    fmt((x1 - x0) / fw),
                    fmt((y1 - y0) / fh),
                ]
                .join(" ");
                lines.push(line);
            }
        }
        let mut content = lines.join("\n");
        if !lines.is_empty() {
            content.push('\n');
        }
        let path = self.label_path(split, filename)?;
        std::fs::write(path, content)
    }

    pub fn finish(&self) -> std::io::Result<()> {
        std::fs::create_dir_all(&self.out)?;
        let abs_out = std::fs::canonicalize(&self.out).unwrap_or_else(|_| self.out.clone());
        let abs_str = abs_out.to_string_lossy().replace('\\', "/");
        // Strip Windows extended-length prefix \\?\ if canonicalize added it
        let clean_path = abs_str.strip_prefix("//?/").unwrap_or(&abs_str).to_string();

        let train = self.used_splits.contains("train").then(|| "train/images".to_string());
        let val = self.used_splits.contains("valid").then(|| "valid/images".to_string());

        let mut data = Mapping::new();
        data.insert(Value::String("path".into()), Value::String(clean_path));
        data.insert(
            Value::String("train".into()),
            train
                .clone()
                .or_else(|| val.clone())
                .map(Value::String)
                .unwrap_or(Value::Null),
        );
        data.insert(
            Value::String("val".into()),
            val.or(train)
                .map(Value::String)
                .unwrap_or(Value::Null),
        );
        if self.used_splits.contains("test") {
            data.insert(
                Value::String("test".into()),
                Value::String("test/images".into()),
            );
        }
        data.insert(
            Value::String("nc".into()),
            Value::Number((self.class_names.len() as u64).into()),
        );
        let mut names = Mapping::new();
        for (i, n) in self.class_names.iter().enumerate() {
            names.insert(Value::Number((i as u64).into()), Value::String(n.clone()));
        }
        data.insert(Value::String("names".into()), Value::Mapping(names));
        if self.obb {
            data.insert(Value::String("task".into()), Value::String("obb".into()));
        }

        let yaml_str = serde_yaml::to_string(&data)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(self.out.join("data.yaml"), yaml_str)
    }
}
