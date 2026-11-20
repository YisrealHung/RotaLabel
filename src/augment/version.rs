//! Dataset version generation: split -> preprocess -> augment xN -> parallel export.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use chrono::Local;
use image::RgbImage;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::augment::geometric::preprocess_matrix;
use crate::augment::pipeline::{augment, transform_annotations, warp, AugConfig};
use crate::exporters::{make_writers, Writer};
use crate::geometry::Mat3;
use crate::imageio;
use crate::model::Annotation;
use crate::project_io::Project;

pub const TEST_TARGET: usize = 10;
pub const MAX_MULTIPLIER: usize = 100;

fn d_train() -> f64 { 70.0 }
fn d_valid() -> f64 { 20.0 }
fn d_test() -> f64 { 10.0 }
fn d_seed() -> u64 { 42 }
fn d_true() -> bool { true }
fn d_resize_mode() -> String { "none".to_string() }
fn d_resize_dim() -> u32 { 640 }
fn d_mult() -> usize { 3 }
fn d_min_vis() -> f64 { 30.0 }
fn d_formats() -> Vec<String> { vec!["yolo_obb".to_string()] }

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct VersionConfig {
    #[serde(default = "d_train")]
    pub train: f64,
    #[serde(default = "d_valid")]
    pub valid: f64,
    #[serde(default = "d_test")]
    pub test: f64,
    #[serde(default = "d_seed")]
    pub seed: u64,
    #[serde(default = "d_true")]
    pub skip_unannotated: bool,
    #[serde(default = "d_resize_mode")]
    pub resize_mode: String,
    #[serde(default = "d_resize_dim")]
    pub resize_w: u32,
    #[serde(default = "d_resize_dim")]
    pub resize_h: u32,
    pub grayscale: bool,
    #[serde(default = "d_mult")]
    pub multiplier: usize,
    #[serde(default = "d_min_vis")]
    pub min_visibility: f64,
    #[serde(default = "d_formats")]
    pub formats: Vec<String>,
    pub workers: usize,
    pub aug: AugConfig,
}

impl Default for VersionConfig {
    fn default() -> Self {
        Self {
            train: d_train(),
            valid: d_valid(),
            test: d_test(),
            seed: d_seed(),
            skip_unannotated: true,
            resize_mode: d_resize_mode(),
            resize_w: d_resize_dim(),
            resize_h: d_resize_dim(),
            grayscale: false,
            multiplier: d_mult(),
            min_visibility: d_min_vis(),
            formats: d_formats(),
            workers: 0,
            aug: AugConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct VersionStats {
    pub train: usize,
    pub valid: usize,
    pub test: usize,
    pub boxes: usize,
    pub source_images: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub patches: Option<HashMap<String, usize>>,
    #[serde(skip_serializing_if = "is_false")]
    pub cancelled: bool,
}

fn is_false(b: &bool) -> bool {
    !*b
}

pub fn default_workers() -> usize {
    let n = std::thread::available_parallelism()
        .map(|v| v.get())
        .unwrap_or(1);
    if n >= 4 {
        (n / 2).max(1)
    } else {
        n.max(1)
    }
}

pub fn resolve_workers(workers: usize, n_tasks: usize) -> usize {
    let w = if workers == 0 {
        default_workers()
    } else {
        workers
    };
    w.min(n_tasks.max(1)).max(1)
}

pub fn split_items(names: &[String], cfg: &VersionConfig) -> HashMap<String, Vec<String>> {
    let mut rng = ChaCha8Rng::seed_from_u64(cfg.seed);
    let mut order = names.to_vec();
    order.shuffle(&mut rng);
    let total_raw = cfg.train + cfg.valid + cfg.test;
    let total = if total_raw == 0.0 { 1.0 } else { total_raw };
    let n = order.len();
    let n_train = ((n as f64 * cfg.train / total).round() as usize).min(n);
    let n_valid = ((n as f64 * cfg.valid / total).round() as usize).min(n - n_train);
    let mut out = HashMap::new();
    out.insert("train".to_string(), order[..n_train].to_vec());
    out.insert("valid".to_string(), order[n_train..n_train + n_valid].to_vec());
    out.insert("test".to_string(), order[n_train + n_valid..].to_vec());
    out
}

pub fn split_copies(
    splits: &HashMap<String, Vec<String>>,
    cfg: &VersionConfig,
) -> HashMap<String, Vec<usize>> {
    let aug_on = cfg.aug.any_enabled();
    let mult = if aug_on {
        cfg.multiplier.clamp(1, MAX_MULTIPLIER)
    } else {
        1
    };
    let mut out = HashMap::new();
    for s in ["train", "valid"] {
        let len = splits.get(s).map(|v| v.len()).unwrap_or(0);
        out.insert(s.to_string(), vec![mult; len]);
    }
    let n = splits.get("test").map(|v| v.len()).unwrap_or(0);
    let test_copies = if n == 0 || n >= TEST_TARGET || !aug_on {
        vec![1; n.min(TEST_TARGET)]
    } else {
        let base = TEST_TARGET / n;
        let extra = TEST_TARGET % n;
        (0..n).map(|i| base + usize::from(i < extra)).collect()
    };
    out.insert("test".to_string(), test_copies);
    out
}

pub fn to_gray_rgb(img: &RgbImage) -> RgbImage {
    let mut out = img.clone();
    for px in out.pixels_mut() {
        let y = (0.299 * px[0] as f32 + 0.587 * px[1] as f32 + 0.114 * px[2] as f32)
            .round()
            .clamp(0.0, 255.0) as u8;
        px[0] = y;
        px[1] = y;
        px[2] = y;
    }
    out
}

fn prepare(img: &RgbImage, cfg: &VersionConfig) -> (RgbImage, Mat3, u32, u32) {
    let (pre, pw, ph) = preprocess_matrix(
        img.width(),
        img.height(),
        &cfg.resize_mode,
        cfg.resize_w,
        cfg.resize_h,
    );
    let prepared = if cfg.grayscale {
        to_gray_rgb(img)
    } else {
        img.clone()
    };
    (prepared, pre, pw, ph)
}

pub fn make_sample(
    img: &RgbImage,
    anns: &[Annotation],
    cfg: &VersionConfig,
    rng: &mut ChaCha8Rng,
    augmented: bool,
) -> (RgbImage, Vec<Annotation>) {
    let (prepared, pre, pw, ph) = prepare(img, cfg);
    let vis = cfg.min_visibility / 100.0;
    if !augmented {
        (
            warp(&prepared, &pre, pw, ph),
            transform_annotations(anns, &pre, pw, ph, 0.0),
        )
    } else {
        augment(&prepared, anns, &cfg.aug, rng, Some(&pre), vis, Some((pw, ph)))
    }
}

pub fn output_stems(stem: &str, copies: usize) -> Vec<String> {
    let mut out = Vec::with_capacity(copies.max(1));
    out.push(stem.to_string());
    for k in 1..copies {
        out.push(format!("{stem}_aug{k}"));
    }
    out
}

pub fn unique_stem(stem: &str, used: &mut HashSet<String>, copies: usize) -> String {
    let mut s = stem.to_string();
    let mut i = 1usize;
    while output_stems(&s, copies).iter().any(|n| used.contains(n)) {
        i += 1;
        s = format!("{stem}_{i}");
    }
    for n in output_stems(&s, copies) {
        used.insert(n);
    }
    s
}

pub fn source_names(project: &Project, cfg: &VersionConfig) -> Vec<String> {
    project
        .images
        .iter()
        .filter(|n| !cfg.skip_unannotated || project.box_count(n) > 0)
        .cloned()
        .collect()
}

pub fn prepare_dirs(writers: &[(String, Writer)], splits: &HashSet<String>) -> std::io::Result<()> {
    for s in splits {
        for (_, wr) in writers {
            wr.image_dir(s)?;
        }
    }
    Ok(())
}

pub fn write_meta(out: &Path, meta: serde_json::Value) -> std::io::Result<()> {
    let created = Local::now().format("%Y-%m-%d %H:%M:%S").to_string();
    let mut obj = match meta {
        serde_json::Value::Object(m) => m,
        _ => serde_json::Map::new(),
    };
    obj.insert("created".to_string(), serde_json::Value::String(created));
    let json = serde_json::to_string_pretty(&serde_json::Value::Object(obj))
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    std::fs::write(out.join("version.json"), json)
}

pub fn next_version_dir(project: &Project) -> PathBuf {
    let _ = std::fs::create_dir_all(&project.versions_dir);
    let count = std::fs::read_dir(&project.versions_dir)
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .filter(|e| {
                    e.path().is_dir()
                        && e.file_name()
                            .to_str()
                            .map(|s| s.starts_with('v'))
                            .unwrap_or(false)
                })
                .count()
        })
        .unwrap_or(0);
    let n = 1 + count;
    let ts = Local::now().format("%Y%m%d_%H%M%S");
    project.versions_dir.join(format!("v{n}_{ts}"))
}

struct TaskItem {
    split: String,
    name: String,
    path: PathBuf,
    anns: Vec<Annotation>,
    stems: Vec<String>,
    seed: u64,
}

type TaskResult = (String, Vec<(String, u32, u32, Vec<Annotation>)>);

pub fn generate_version(
    project: &Project,
    cfg: &VersionConfig,
    out_dir: &Path,
    progress: Option<&(dyn Fn(usize, usize, &str) + Sync)>,
    cancel_flag: Option<&AtomicBool>,
) -> std::io::Result<VersionStats> {
    std::fs::create_dir_all(out_dir)?;
    let names = source_names(project, cfg);
    let mut splits = split_items(&names, cfg);
    let copies_of = split_copies(&splits, cfg);
    let test_keep = copies_of.get("test").map(|v| v.len()).unwrap_or(0);
    if let Some(tv) = splits.get_mut("test") {
        tv.truncate(test_keep);
    }

    let class_names = project.class_names();
    let mut writers = make_writers(out_dir, &cfg.formats, &class_names);
    let mut stats = VersionStats {
        source_images: names.len(),
        ..Default::default()
    };

    let mut used = HashSet::new();
    let mut tasks = Vec::new();
    let mut idx = 0u64;
    for split in ["train", "valid", "test"] {
        let sp_names = splits.get(split).cloned().unwrap_or_default();
        let sp_copies = copies_of.get(split).cloned().unwrap_or_default();
        for (name, copies) in sp_names.into_iter().zip(sp_copies.into_iter()) {
            let file_stem = Path::new(&name)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or(&name);
            let stem = unique_stem(file_stem, &mut used, copies);
            let anns = project
                .get(&name)
                .map(|ia| ia.annotations.clone())
                .unwrap_or_default();
            idx += 1;
            let seed = cfg
                .seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(idx.wrapping_mul(1442695040888963407));
            tasks.push(TaskItem {
                split: split.to_string(),
                path: project.image_path(&name),
                name,
                anns,
                stems: output_stems(&stem, copies),
                seed,
            });
        }
    }

    let total = tasks.len();
    if let Some(cb) = progress {
        cb(0, total, "");
    }
    let used_split_keys: HashSet<String> = tasks.iter().map(|t| t.split.clone()).collect();
    prepare_dirs(&writers, &used_split_keys)?;

    let n_workers = resolve_workers(cfg.workers, total);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_workers)
        .build()
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

    let done_counter = AtomicUsize::new(0);
    let task_results: Vec<Option<std::io::Result<TaskResult>>> = pool.install(|| {
        tasks
            .par_iter()
            .map(|t| {
                if cancel_flag.map(|f| f.load(Ordering::Relaxed)).unwrap_or(false) {
                    return None;
                }
                let res = (|| -> std::io::Result<TaskResult> {
                    let Some(img) = imageio::imread(&t.path) else {
                        return Ok((t.split.clone(), Vec::new()));
                    };
                    let mut rng = ChaCha8Rng::seed_from_u64(t.seed);
                    let mut out_items = Vec::with_capacity(t.stems.len());
                    for (k, out_stem) in t.stems.iter().enumerate() {
                        if cancel_flag.map(|f| f.load(Ordering::Relaxed)).unwrap_or(false) {
                            break;
                        }
                        let (out_img, out_anns) = make_sample(&img, &t.anns, cfg, &mut rng, k > 0);
                        let w = out_img.width();
                        let h = out_img.height();
                        let fname = format!("{out_stem}.jpg");
                        let jpg = imageio::encode_jpeg(&out_img, 95)?;
                        for (_, wr) in &writers {
                            wr.write_worker(&t.split, &fname, &jpg, w, h, &out_anns)?;
                        }
                        out_items.push((fname, w, h, out_anns));
                    }
                    Ok((t.split.clone(), out_items))
                })();
                let done = done_counter.fetch_add(1, Ordering::Relaxed) + 1;
                if let Some(cb) = progress {
                    cb(done, total, &t.name);
                }
                Some(res)
            })
            .collect()
    });

    if cancel_flag.map(|f| f.load(Ordering::Relaxed)).unwrap_or(false)
        || task_results.iter().any(|r| r.is_none())
    {
        stats.cancelled = true;
        return Ok(stats);
    }

    for entry in task_results {
        let (split, results) = entry.unwrap()?;
        for (fname, w, h, out_anns) in results {
            for (_, wr) in &mut writers {
                wr.main_commit(&split, &fname, w, h, &out_anns);
            }
            match split.as_str() {
                "train" => stats.train += 1,
                "valid" => stats.valid += 1,
                "test" => stats.test += 1,
                _ => {}
            }
            stats.boxes += out_anns.len();
        }
    }

    for (_, wr) in &writers {
        wr.finish()?;
    }
    if let Some(cb) = progress {
        cb(total, total, "");
    }

    let meta = serde_json::json!({
        "config": cfg,
        "stats": stats,
        "classes": class_names,
        "splits": splits,
    });
    write_meta(out_dir, meta)?;
    Ok(stats)
}
