use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use serde::Serialize;

use crate::exporters::base::clipped_corners;
use crate::geometry::{aabb, polygon_area};
use crate::model::Annotation;

fn round2(v: f64) -> f64 {
    let r = (v * 100.0).round() / 100.0;
    if r == 0.0 { 0.0 } else { r }
}

pub fn split_folder_name(split: &str) -> &str {
    match split {
        "valid" => "val",
        other => other,
    }
}

#[derive(Debug, Clone, Serialize)]
struct CocoObbInfo {
    description: &'static str,
}

#[derive(Debug, Clone, Serialize)]
struct CocoObbCategory {
    id: usize,
    name: String,
    supercategory: &'static str,
}

#[derive(Debug, Clone, Serialize)]
struct CocoObbImage {
    id: usize,
    file_name: String,
    width: u32,
    height: u32,
}

#[derive(Debug, Clone, Serialize)]
struct CocoObbAnnotation {
    id: usize,
    image_id: usize,
    category_id: usize,
    segmentation: Vec<Vec<f64>>,
    bbox: [f64; 4],
    area: f64,
    iscrowd: u8,
}

#[derive(Debug, Clone, Serialize)]
struct CocoObbDataset {
    info: CocoObbInfo,
    categories: Vec<CocoObbCategory>,
    images: Vec<CocoObbImage>,
    annotations: Vec<CocoObbAnnotation>,
}

#[derive(Debug, Clone)]
pub struct CocoObbWriter {
    pub out: PathBuf,
    pub class_names: Vec<String>,
    pub used_splits: HashSet<String>,
    data: HashMap<String, CocoObbDataset>,
}

impl CocoObbWriter {
    pub fn new(out: impl Into<PathBuf>, class_names: Vec<String>) -> Self {
        Self {
            out: out.into(),
            class_names,
            used_splits: HashSet::new(),
            data: HashMap::new(),
        }
    }

    pub fn image_dir(&self, split: &str) -> std::io::Result<PathBuf> {
        let d = self.out.join(split_folder_name(split)).join("images");
        std::fs::create_dir_all(&d)?;
        Ok(d)
    }

    fn split_mut(&mut self, split: &str) -> &mut CocoObbDataset {
        if !self.data.contains_key(split) {
            let categories = self
                .class_names
                .iter()
                .enumerate()
                .map(|(i, n)| CocoObbCategory {
                    id: i,
                    name: n.clone(),
                    supercategory: "none",
                })
                .collect();
            self.data.insert(
                split.to_string(),
                CocoObbDataset {
                    info: CocoObbInfo {
                        description: "exported by rotalabel (COCO-OBB)",
                    },
                    categories,
                    images: Vec::new(),
                    annotations: Vec::new(),
                },
            );
        }
        self.data.get_mut(split).unwrap()
    }

    pub fn write_labels(
        &mut self,
        split: &str,
        filename: &str,
        width: u32,
        height: u32,
        anns: &[Annotation],
    ) {
        let d = self.split_mut(split);
        let img_id = d.images.len() + 1;
        d.images.push(CocoObbImage {
            id: img_id,
            file_name: filename.to_string(),
            width,
            height,
        });
        for a in anns {
            let pts = clipped_corners(a, width, height);
            let area = polygon_area(&pts);
            let [x0, y0, x1, y1] = aabb(&pts);
            if area <= 0.0 || x1 - x0 <= 0.0 || y1 - y0 <= 0.0 {
                continue;
            }
            let seg: Vec<f64> = pts
                .into_iter()
                .flat_map(|[x, y]| [round2(x), round2(y)])
                .collect();
            let ann_id = d.annotations.len() + 1;
            d.annotations.push(CocoObbAnnotation {
                id: ann_id,
                image_id: img_id,
                category_id: a.cls,
                segmentation: vec![seg],
                bbox: [round2(x0), round2(y0), round2(x1 - x0), round2(y1 - y0)],
                area: round2(area),
                iscrowd: 0,
            });
        }
    }

    pub fn finish(&self) -> std::io::Result<()> {
        let ann_dir = self.out.join("annotations");
        std::fs::create_dir_all(&ann_dir)?;
        for (split, d) in &self.data {
            let json = serde_json::to_string(d)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
            std::fs::write(
                ann_dir.join(format!("{}.json", split_folder_name(split))),
                json,
            )?;
        }
        Ok(())
    }
}
