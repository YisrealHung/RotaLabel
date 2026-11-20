use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use serde::Serialize;

use crate::exporters::base::{clipped_aabb, clipped_corners};
use crate::model::{Annotation, BoxKind};

fn round2(v: f64) -> f64 {
    let r = (v * 100.0).round() / 100.0;
    if r == 0.0 { 0.0 } else { r }
}

fn round3(v: f64) -> f64 {
    let r = (v * 1000.0).round() / 1000.0;
    if r == 0.0 { 0.0 } else { r }
}

#[derive(Debug, Clone, Serialize)]
struct CocoInfo {
    description: &'static str,
}

#[derive(Debug, Clone, Serialize)]
struct CocoCategory {
    id: usize,
    name: String,
    supercategory: &'static str,
}

#[derive(Debug, Clone, Serialize)]
struct CocoImage {
    id: usize,
    file_name: String,
    width: u32,
    height: u32,
}

#[derive(Debug, Clone, Serialize)]
struct CocoAnnotation {
    id: usize,
    image_id: usize,
    category_id: usize,
    bbox: [f64; 4],
    segmentation: Vec<Vec<f64>>,
    area: f64,
    iscrowd: u8,
    kind: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    rbox: Option<[f64; 5]>,
}

#[derive(Debug, Clone, Serialize)]
struct CocoDataset {
    info: CocoInfo,
    categories: Vec<CocoCategory>,
    images: Vec<CocoImage>,
    annotations: Vec<CocoAnnotation>,
}

#[derive(Debug, Clone)]
pub struct CocoWriter {
    pub out: PathBuf,
    pub class_names: Vec<String>,
    pub used_splits: HashSet<String>,
    data: HashMap<String, CocoDataset>,
}

impl CocoWriter {
    pub fn new(out: impl Into<PathBuf>, class_names: Vec<String>) -> Self {
        Self {
            out: out.into(),
            class_names,
            used_splits: HashSet::new(),
            data: HashMap::new(),
        }
    }

    pub fn image_dir(&self, split: &str) -> std::io::Result<PathBuf> {
        let d = self.out.join(split).join("images");
        std::fs::create_dir_all(&d)?;
        Ok(d)
    }

    fn split_mut(&mut self, split: &str) -> &mut CocoDataset {
        if !self.data.contains_key(split) {
            let categories = self
                .class_names
                .iter()
                .enumerate()
                .map(|(i, n)| CocoCategory {
                    id: i + 1,
                    name: n.clone(),
                    supercategory: "none",
                })
                .collect();
            self.data.insert(
                split.to_string(),
                CocoDataset {
                    info: CocoInfo {
                        description: "exported by rotalabel",
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
        d.images.push(CocoImage {
            id: img_id,
            file_name: filename.to_string(),
            width,
            height,
        });
        for a in anns {
            let [x0, y0, x1, y1] = clipped_aabb(a, width, height);
            if x1 - x0 <= 0.0 || y1 - y0 <= 0.0 {
                continue;
            }
            let seg: Vec<f64> = clipped_corners(a, width, height)
                .into_iter()
                .flat_map(|[x, y]| [round2(x), round2(y)])
                .collect();
            let area = if a.kind == BoxKind::Obb {
                round2(a.w * a.h)
            } else {
                round2((x1 - x0) * (y1 - y0))
            };
            let rbox = (a.kind == BoxKind::Obb).then(|| {
                [
                    round2(a.cx),
                    round2(a.cy),
                    round2(a.w),
                    round2(a.h),
                    round3(a.angle),
                ]
            });
            let ann_id = d.annotations.len() + 1;
            d.annotations.push(CocoAnnotation {
                id: ann_id,
                image_id: img_id,
                category_id: a.cls + 1,
                bbox: [round2(x0), round2(y0), round2(x1 - x0), round2(y1 - y0)],
                segmentation: vec![seg],
                area,
                iscrowd: 0,
                kind: a.kind.as_str(),
                rbox,
            });
        }
    }

    pub fn finish(&self) -> std::io::Result<()> {
        for (split, d) in &self.data {
            let p = self.out.join(split);
            std::fs::create_dir_all(&p)?;
            let json = serde_json::to_string(d)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
            std::fs::write(p.join("_annotations.coco.json"), json)?;
        }
        Ok(())
    }
}
