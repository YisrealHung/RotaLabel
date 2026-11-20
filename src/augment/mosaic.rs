//! Mosaic (copy-paste) augmentation.

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use image::RgbImage;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::augment::geometric as G;
use crate::augment::photometric::apply_photometric;
use crate::augment::pipeline::{transform_annotations, warp, warp_mask, AugConfig};
use crate::augment::version::{
    prepare_dirs, resolve_workers, source_names, split_items, to_gray_rgb, write_meta,
    VersionConfig, VersionStats,
};
use crate::exporters::make_writers;
use crate::geometry::{self, Mat3};
use crate::imageio;
use crate::model::{Annotation, BoxKind};
use crate::project_io::Project;

pub const SPLITS: [&str; 3] = ["train", "valid", "test"];
pub const GRID: usize = 4;
pub const PLACE_TRIES: usize = 200;
pub const LAYOUT_TRIES: usize = 8;
pub const SHRINK_STEP: f64 = 0.85;
pub const REPEAT_MIN_ANGLE: f64 = 30.0;

fn d_cw() -> u32 { 1624 }
fn d_ch() -> u32 { 1240 }
fn d_ctrain() -> usize { 100 }
fn d_cvalid() -> usize { 20 }
fn d_ctest() -> usize { 10 }
fn d_patches() -> usize { 12 }
fn d_jitter() -> usize { 2 }
fn d_padding() -> i32 { 10 }
fn d_gap() -> i32 { 4 }
fn d_true() -> bool { true }
fn d_min_fit() -> f64 { 50.0 }
fn d_scope() -> String { "patch".to_string() }

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MosaicConfig {
    #[serde(default = "d_cw")]
    pub canvas_w: u32,
    #[serde(default = "d_ch")]
    pub canvas_h: u32,
    #[serde(default = "d_ctrain")]
    pub count_train: usize,
    #[serde(default = "d_cvalid")]
    pub count_valid: usize,
    #[serde(default = "d_ctest")]
    pub count_test: usize,
    #[serde(default = "d_patches")]
    pub patches: usize,
    #[serde(default = "d_jitter")]
    pub patches_jitter: usize,
    #[serde(default = "d_padding")]
    pub padding: i32,
    #[serde(default = "d_gap")]
    pub gap: i32,
    #[serde(default = "d_true")]
    pub allow_overlap: bool,
    #[serde(default = "d_min_fit")]
    pub min_fit_scale: f64,
    #[serde(default = "d_true")]
    pub flip_h: bool,
    #[serde(default = "d_true")]
    pub flip_v: bool,
    #[serde(default = "d_true")]
    pub rot90: bool,
    pub rotate: f64,
    pub scale: f64,
    #[serde(default = "d_scope")]
    pub photometric_scope: String,
    pub aug: AugConfig,
}

impl Default for MosaicConfig {
    fn default() -> Self {
        Self {
            canvas_w: d_cw(),
            canvas_h: d_ch(),
            count_train: d_ctrain(),
            count_valid: d_cvalid(),
            count_test: d_ctest(),
            patches: d_patches(),
            patches_jitter: d_jitter(),
            padding: d_padding(),
            gap: d_gap(),
            allow_overlap: true,
            min_fit_scale: d_min_fit(),
            flip_h: true,
            flip_v: true,
            rot90: true,
            rotate: 0.0,
            scale: 0.0,
            photometric_scope: d_scope(),
            aug: AugConfig::default(),
        }
    }
}

impl MosaicConfig {
    pub fn from_json_value(val: &serde_json::Value) -> Self {
        let mut obj = val.as_object().cloned().unwrap_or_default();
        if !obj.contains_key("patches") {
            if let Some(lo) = obj.get("min_patches").and_then(|v| v.as_i64()) {
                let hi = obj
                    .get("max_patches")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(lo);
                obj.insert("patches".into(), serde_json::json!((lo + hi + 1) / 2));
                obj.insert(
                    "patches_jitter".into(),
                    serde_json::json!(0.max((hi - lo) / 2)),
                );
            }
        }
        serde_json::from_value(serde_json::Value::Object(obj)).unwrap_or_default()
    }

    pub fn counts(&self) -> HashMap<&'static str, usize> {
        let mut m = HashMap::new();
        m.insert("train", self.count_train);
        m.insert("valid", self.count_valid);
        m.insert("test", self.count_test);
        m
    }

    pub fn patch_range(&self) -> (usize, usize) {
        let lo = self.patches.saturating_sub(self.patches_jitter).max(1);
        let hi = lo.max(self.patches.saturating_add(self.patches_jitter));
        (lo, hi)
    }
}

pub fn extract_patch(
    img: &RgbImage,
    anns: &[Annotation],
    padding: i32,
) -> Option<(RgbImage, Vec<Annotation>)> {
    if anns.is_empty() {
        return None;
    }
    let w = img.width() as i32;
    let h = img.height() as i32;
    let pts: Vec<[f64; 2]> = anns.iter().flat_map(|a| a.corners()).collect();
    let [fx0, fy0, fx1, fy1] = geometry::aabb(&pts);
    let x0 = (fx0.floor() as i32 - padding).max(0);
    let y0 = (fy0.floor() as i32 - padding).max(0);
    let x1 = (fx1.ceil() as i32 + padding).min(w);
    let y1 = (fy1.ceil() as i32 + padding).min(h);
    if x1 - x0 < 2 || y1 - y0 < 2 {
        return None;
    }
    let pw = (x1 - x0) as u32;
    let ph = (y1 - y0) as u32;
    let patch = image::imageops::crop_imm(img, x0 as u32, y0 as u32, pw, ph).to_image();
    let boxes = transform_annotations(anns, &G::translate(-x0 as f64, -y0 as f64), pw, ph, 0.0);
    if boxes.is_empty() {
        None
    } else {
        Some((patch, boxes))
    }
}

pub fn similar_anns(anns: &[Annotation], m: &Mat3) -> Vec<Annotation> {
    anns.iter()
        .map(|a| {
            let c = m.transform_points(&a.corners());
            if a.kind == BoxKind::Bbox {
                let [x0, y0, x1, y1] = geometry::aabb(&c);
                return Annotation::new(
                    a.cls,
                    (x0 + x1) / 2.0,
                    (y0 + y1) / 2.0,
                    x1 - x0,
                    y1 - y0,
                    0.0,
                    BoxKind::Bbox,
                );
            }
            let e1 = [c[1][0] - c[0][0], c[1][1] - c[0][1]];
            let e2 = [c[2][0] - c[1][0], c[2][1] - c[1][1]];
            let cx = (c[0][0] + c[1][0] + c[2][0] + c[3][0]) / 4.0;
            let cy = (c[0][1] + c[1][1] + c[2][1] + c[3][1]) / 4.0;
            let ang = geometry::normalize_angle(e1[1].atan2(e1[0]).to_degrees());
            Annotation::new(a.cls, cx, cy, e1[0].hypot(e1[1]), e2[0].hypot(e2[1]), ang, a.kind)
        })
        .collect()
}

#[derive(Debug, Clone)]
pub struct Piece {
    pub patch: RgbImage,
    pub anns: Vec<Annotation>,
    pub flip_h: bool,
    pub flip_v: bool,
    pub angle: f64,
    pub scale: f64,
}

fn angle_dist_360(a: f64, b: f64) -> f64 {
    let d = (a - b).abs().rem_euclid(360.0);
    d.min(360.0 - d)
}

fn base_angle<R: Rng>(cfg: &MosaicConfig, rng: &mut R) -> f64 {
    let mut a = if cfg.rot90 {
        90.0 * (rng.gen_range(0..4) as f64)
    } else {
        0.0
    };
    if cfg.rotate > 0.0 {
        a += rng.gen_range(-cfg.rotate..cfg.rotate);
    }
    a
}

pub fn pick_patches<R: Rng>(
    n_pool: usize,
    k: usize,
    cfg: &MosaicConfig,
    rng: &mut R,
) -> Vec<(usize, f64)> {
    if n_pool == 0 || k == 0 {
        return Vec::new();
    }
    let mut idx = Vec::with_capacity(k);
    let mut perm: Vec<usize> = (0..n_pool).collect();
    perm.shuffle(rng);
    if n_pool >= k {
        idx.extend_from_slice(&perm[..k]);
    } else {
        idx.extend_from_slice(&perm);
        for _ in 0..(k - n_pool) {
            idx.push(rng.gen_range(0..n_pool));
        }
    }

    let mut used: HashMap<usize, Vec<f64>> = HashMap::new();
    let mut out = Vec::with_capacity(k);
    for i in idx {
        let ang = match used.get(&i) {
            None => base_angle(cfg, rng),
            Some(prev) => {
                let mut best = 0.0;
                let mut best_d = -1.0;
                for _ in 0..60 {
                    let cand = rng.gen_range(0.0..360.0);
                    let d = prev
                        .iter()
                        .map(|&p| angle_dist_360(cand, p))
                        .fold(f64::INFINITY, f64::min);
                    if d > best_d {
                        best = cand;
                        best_d = d;
                    }
                    if d >= REPEAT_MIN_ANGLE {
                        break;
                    }
                }
                best
            }
        };
        used.entry(i).or_default().push(ang);
        out.push((i, ang));
    }
    out
}

pub fn random_piece<R: Rng>(
    patch: RgbImage,
    anns: Vec<Annotation>,
    cfg: &MosaicConfig,
    rng: &mut R,
    angle: Option<f64>,
) -> Piece {
    let fh = cfg.flip_h && rng.gen::<f64>() < 0.5;
    let fv = cfg.flip_v && rng.gen::<f64>() < 0.5;
    let s = if cfg.scale > 0.0 {
        let lo = (1.0 - cfg.scale / 100.0).max(0.05);
        let hi = 1.0 + cfg.scale / 100.0;
        if hi > lo { rng.gen_range(lo..hi) } else { lo }
    } else {
        1.0
    };
    let ang = angle.unwrap_or_else(|| base_angle(cfg, rng));
    Piece {
        patch,
        anns,
        flip_h: fh,
        flip_v: fv,
        angle: ang,
        scale: s,
    }
}

pub fn piece_matrix(pc: &Piece, max_w: f64, max_h: f64, shrink: f64) -> (Mat3, u32, u32) {
    let w = pc.patch.width() as f64;
    let h = pc.patch.height() as f64;
    let mut m = Mat3::identity();
    if pc.flip_h {
        m = G::flip_h(w).mul(&m);
    }
    if pc.flip_v {
        m = G::flip_v(h).mul(&m);
    }
    m = G::scale(pc.scale)
        .mul(&G::about(G::rotation(pc.angle), w / 2.0, h / 2.0))
        .mul(&m);
    let corners = m.transform_points(&[[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]]);
    let [x0, y0, x1, y1] = geometry::aabb(&corners);
    let ow = (x1 - x0).max(1e-6);
    let oh = (y1 - y0).max(1e-6);
    let f = (1.0_f64).min(max_w / ow).min(max_h / oh) * shrink;
    m = G::scale(f).mul(&G::translate(-x0, -y0)).mul(&m);
    let out_w = ((ow * f - 1e-6).ceil() as i64).max(1) as u32;
    let out_h = ((oh * f - 1e-6).ceil() as i64).max(1) as u32;
    (m, out_w, out_h)
}

pub fn render_piece(
    pc: &Piece,
    max_w: f64,
    max_h: f64,
    shrink: f64,
) -> (RgbImage, Vec<f32>, Vec<Annotation>) {
    let (m, ow, oh) = piece_matrix(pc, max_w, max_h, shrink);
    let img = warp(&pc.patch, &m, ow, oh);
    let alpha = warp_mask(pc.patch.width(), pc.patch.height(), &m, ow, oh);
    let anns = similar_anns(&pc.anns, &m);
    (img, alpha, anns)
}

pub fn transform_patch<R: Rng>(
    patch: &RgbImage,
    anns: &[Annotation],
    cfg: &MosaicConfig,
    rng: &mut R,
    max_w: f64,
    max_h: f64,
) -> (RgbImage, Vec<f32>, Vec<Annotation>) {
    let pc = random_piece(patch.clone(), anns.to_vec(), cfg, rng, None);
    render_piece(&pc, max_w, max_h, 1.0)
}

pub fn edge_mean_color(patches: &[&RgbImage]) -> [f32; 3] {
    let mut sum = [0.0_f64; 3];
    let mut count = 0usize;
    for p in patches {
        let w = p.width();
        let h = p.height();
        if w == 0 || h == 0 {
            continue;
        }
        for x in 0..w {
            let top = p.get_pixel(x, 0);
            let bot = p.get_pixel(x, h - 1);
            for c in 0..3 {
                sum[c] += top[c] as f64 + bot[c] as f64;
            }
            count += 2;
        }
        for y in 0..h {
            let left = p.get_pixel(0, y);
            let right = p.get_pixel(w - 1, y);
            for c in 0..3 {
                sum[c] += left[c] as f64 + right[c] as f64;
            }
            count += 2;
        }
    }
    if count == 0 {
        [0.0; 3]
    } else {
        let fc = count as f64;
        [
            (sum[0] / fc) as f32,
            (sum[1] / fc) as f32,
            (sum[2] / fc) as f32,
        ]
    }
}

#[derive(Debug, Clone)]
pub struct Placed {
    pub img: RgbImage,
    pub alpha: Vec<f32>,
    pub anns: Vec<Annotation>,
    pub x: u32,
    pub y: u32,
}

fn blocks(mask: &[bool], w: usize, h: usize) -> (Vec<bool>, usize, usize) {
    let hs = (h + GRID - 1) / GRID;
    let ws = (w + GRID - 1) / GRID;
    let mut out = vec![false; hs * ws];
    for y in 0..h {
        let by = y / GRID;
        let row_m = y * w;
        let row_o = by * ws;
        for x in 0..w {
            if mask[row_m + x] {
                out[row_o + x / GRID] = true;
            }
        }
    }
    (out, ws, hs)
}

struct Occupancy {
    full_w: usize,
    full_h: usize,
    hs: usize,
    ws: usize,
    full: Vec<u8>,
    small: Vec<bool>,
}

impl Occupancy {
    fn new(w: usize, h: usize) -> Self {
        let hs = (h + GRID - 1) / GRID;
        let ws = (w + GRID - 1) / GRID;
        let full_w = ws * GRID;
        let full_h = hs * GRID;
        Self {
            full_w,
            full_h,
            hs,
            ws,
            full: vec![0; full_w * full_h],
            small: vec![false; hs * ws],
        }
    }

    fn free(&self, foot_s: &[bool], fws: usize, fhs: usize, gx: usize, gy: usize) -> bool {
        for by in 0..fhs {
            let s_row = (gy + by) * self.ws + gx;
            let f_row = by * fws;
            for bx in 0..fws {
                if foot_s[f_row + bx] && self.small[s_row + bx] {
                    return false;
                }
            }
        }
        true
    }

    fn refresh(&mut self, x0: i32, y0: i32, x1: i32, y1: i32) {
        let bx0 = (x0.max(0) as usize) / GRID;
        let by0 = (y0.max(0) as usize) / GRID;
        let bx1 = ((x1.max(0) as usize + GRID - 1) / GRID).min(self.ws);
        let by1 = ((y1.max(0) as usize + GRID - 1) / GRID).min(self.hs);
        for by in by0..by1 {
            for bx in bx0..bx1 {
                let mut any = false;
                'blk: for dy in 0..GRID {
                    let y = by * GRID + dy;
                    let row = y * self.full_w + bx * GRID;
                    for dx in 0..GRID {
                        if self.full[row + dx] != 0 {
                            any = true;
                            break 'blk;
                        }
                    }
                }
                self.small[by * self.ws + bx] = any;
            }
        }
    }

    fn protect_boxes(&mut self, anns: &[Annotation], gap: i32) {
        let g = gap.max(0) as f64;
        let tol_sq = if gap > 0 { (g + 0.5).powi(2) } else { 0.25 };
        for a in anns {
            let [fx0, fy0, fx1, fy1] = a.aabb();
            let ix0 = (fx0.floor() as i32 - gap - 1).max(0);
            let iy0 = (fy0.floor() as i32 - gap - 1).max(0);
            let ix1 = (fx1.ceil() as i32 + gap + 2).min(self.full_w as i32);
            let iy1 = (fy1.ceil() as i32 + gap + 2).min(self.full_h as i32);
            if ix1 <= ix0 || iy1 <= iy0 {
                continue;
            }
            let (s, c) = a.angle.to_radians().sin_cos();
            let hw = a.w / 2.0;
            let hh = a.h / 2.0;
            for py in iy0..iy1 {
                let dy = py as f64 - a.cy;
                let row = (py as usize) * self.full_w;
                for px in ix0..ix1 {
                    let dx = px as f64 - a.cx;
                    let lx = dx * c + dy * s;
                    let ly = -dx * s + dy * c;
                    let ox = (lx.abs() - hw).max(0.0);
                    let oy = (ly.abs() - hh).max(0.0);
                    let inside = if gap > 0 {
                        ox * ox + oy * oy <= tol_sq
                    } else {
                        lx.abs() <= hw + 0.5 && ly.abs() <= hh + 0.5
                    };
                    if inside {
                        self.full[row + px as usize] = 1;
                    }
                }
            }
            self.refresh(ix0, iy0, ix1, iy1);
        }
    }

    fn protect_footprint(&mut self, foot: &[bool], fw: usize, fh: usize, x: i32, y: i32, gap: i32) {
        let g = gap.max(0);
        let dx0 = (x - g).max(0);
        let dy0 = (y - g).max(0);
        let dx1 = (x + fw as i32 + g).min(self.full_w as i32);
        let dy1 = (y + fh as i32 + g).min(self.full_h as i32);
        if dx1 <= dx0 || dy1 <= dy0 {
            return;
        }
        if g == 0 {
            for py in dy0..dy1 {
                let sy = (py - y) as usize;
                let row_d = (py as usize) * self.full_w;
                let row_s = sy * fw;
                for px in dx0..dx1 {
                    let sx = (px - x) as usize;
                    if foot[row_s + sx] {
                        self.full[row_d + px as usize] = 1;
                    }
                }
            }
        } else {
            let mut offsets = Vec::new();
            let g2 = g * g;
            for oy in -g..=g {
                for ox in -g..=g {
                    if ox * ox + oy * oy <= g2 {
                        offsets.push((ox, oy));
                    }
                }
            }
            for sy in 0..fh as i32 {
                let row_s = sy as usize * fw;
                for sx in 0..fw as i32 {
                    if !foot[row_s + sx as usize] {
                        continue;
                    }
                    let cx = x + sx;
                    let cy = y + sy;
                    for &(ox, oy) in &offsets {
                        let px = cx + ox;
                        let py = cy + oy;
                        if px >= 0
                            && py >= 0
                            && (px as usize) < self.full_w
                            && (py as usize) < self.full_h
                        {
                            self.full[py as usize * self.full_w + px as usize] = 1;
                        }
                    }
                }
            }
        }
        self.refresh(dx0, dy0, dx1, dy1);
    }
}

fn shrink_levels(cfg: &MosaicConfig) -> Vec<f64> {
    let floor = cfg.min_fit_scale.clamp(1.0, 100.0) / 100.0;
    let mut levels = vec![1.0];
    let mut f = SHRINK_STEP;
    while f >= floor - 1e-9 {
        levels.push(f);
        f *= SHRINK_STEP;
    }
    levels
}

struct RenderedPiece {
    img: RgbImage,
    alpha: Vec<f32>,
    anns: Vec<Annotation>,
    foot: Vec<bool>,
    foot_s: Vec<bool>,
    fws: usize,
    fhs: usize,
}

pub fn layout_mosaic<R: Rng>(
    pieces: &[Piece],
    cfg: &MosaicConfig,
    rng: &mut R,
) -> Vec<Placed> {
    let cw = cfg.canvas_w as usize;
    let ch = cfg.canvas_h as usize;
    let gap = cfg.gap.max(0);
    let levels = shrink_levels(cfg);
    let mut order: Vec<usize> = (0..pieces.len()).collect();
    order.sort_by_key(|&i| {
        std::cmp::Reverse(pieces[i].patch.width() as usize * pieces[i].patch.height() as usize)
    });

    let mut cache: HashMap<(usize, usize), RenderedPiece> = HashMap::new();
    let mut best: Vec<Placed> = Vec::new();

    for _ in 0..LAYOUT_TRIES {
        let mut occ = Occupancy::new(cw, ch);
        let mut placed: Vec<Placed> = Vec::new();
        for &i in &order {
            for (fi, &f) in levels.iter().enumerate() {
                if !cache.contains_key(&(i, fi)) {
                    let (img, alpha, anns) = render_piece(&pieces[i], cw as f64, ch as f64, f);
                    let foot: Vec<bool> = alpha.iter().map(|&a| a > 0.02).collect();
                    let (foot_s, fws, fhs) =
                        blocks(&foot, img.width() as usize, img.height() as usize);
                    cache.insert(
                        (i, fi),
                        RenderedPiece {
                            img,
                            alpha,
                            anns,
                            foot,
                            foot_s,
                            fws,
                            fhs,
                        },
                    );
                }
                let rp = cache.get(&(i, fi)).unwrap();
                let w = rp.img.width() as usize;
                let h = rp.img.height() as usize;
                if w > cw || h > ch {
                    continue;
                }
                let gx_max = (cw - w) / GRID;
                let gy_max = (ch - h) / GRID;
                let mut spot: Option<(usize, usize)> = None;
                for _ in 0..PLACE_TRIES {
                    let gx = rng.gen_range(0..=gx_max);
                    let gy = rng.gen_range(0..=gy_max);
                    if occ.free(&rp.foot_s, rp.fws, rp.fhs, gx, gy) {
                        spot = Some((gx * GRID, gy * GRID));
                        break;
                    }
                }
                let Some((x, y)) = spot else {
                    continue;
                };
                let boxes = similar_anns(&rp.anns, &G::translate(x as f64, y as f64));
                if cfg.allow_overlap {
                    occ.protect_boxes(&boxes, gap);
                } else {
                    occ.protect_footprint(&rp.foot, w, h, x as i32, y as i32, gap);
                }
                placed.push(Placed {
                    img: rp.img.clone(),
                    alpha: rp.alpha.clone(),
                    anns: boxes,
                    x: x as u32,
                    y: y as u32,
                });
                break;
            }
        }
        if placed.len() > best.len() {
            best = placed;
        }
        if best.len() == pieces.len() {
            break;
        }
    }
    best
}

pub fn make_mosaic<R: Rng>(
    pool: &[(RgbImage, Vec<Annotation>)],
    cfg: &MosaicConfig,
    rng: &mut R,
    grayscale: bool,
) -> (RgbImage, Vec<Annotation>) {
    let cw = cfg.canvas_w;
    let ch = cfg.canvas_h;
    if pool.is_empty() {
        return (RgbImage::new(cw, ch), Vec::new());
    }
    let (lo, hi) = cfg.patch_range();
    let k = rng.gen_range(lo..=hi);
    let per_patch = cfg.photometric_scope != "canvas" && cfg.aug.any_enabled();

    let picks = pick_patches(pool.len(), k, cfg, rng);
    let mut pieces = Vec::with_capacity(picks.len());
    for (i, ang) in picks {
        let (patch, anns) = &pool[i];
        let p_img = if per_patch {
            apply_photometric(patch, &cfg.aug, rng)
        } else {
            patch.clone()
        };
        pieces.push(random_piece(p_img, anns.clone(), cfg, rng, Some(ang)));
    }
    let patch_refs: Vec<&RgbImage> = pieces.iter().map(|p| &p.patch).collect();
    let bg = edge_mean_color(&patch_refs);
    let color = [
        bg[0].round().clamp(0.0, 255.0) as u8,
        bg[1].round().clamp(0.0, 255.0) as u8,
        bg[2].round().clamp(0.0, 255.0) as u8,
    ];

    let mut raw = Vec::with_capacity((cw as usize) * (ch as usize) * 3);
    for _ in 0..(cw as usize * ch as usize) {
        raw.extend_from_slice(&color);
    }
    let mut out = RgbImage::from_raw(cw, ch, raw).expect("valid buffer");
    let mut boxes = Vec::new();
    for p in layout_mosaic(&pieces, cfg, rng) {
        let pw = p.img.width();
        let ph = p.img.height();
        for py in 0..ph {
            for px in 0..pw {
                let a = p.alpha[(py * pw + px) as usize];
                if a <= 0.0 {
                    continue;
                }
                let src = p.img.get_pixel(px, py);
                let dst = out.get_pixel_mut(p.x + px, p.y + py);
                for c in 0..3 {
                    let r = dst[c] as f32;
                    let val = r + (src[c] as f32 - r) * a;
                    dst[c] = val.round().clamp(0.0, 255.0) as u8;
                }
            }
        }
        boxes.extend(p.anns);
    }
    if cfg.photometric_scope == "canvas" && cfg.aug.any_enabled() {
        out = apply_photometric(&out, &cfg.aug, rng);
    }
    if grayscale {
        out = to_gray_rgb(&out);
    }
    (out, boxes)
}

pub fn build_pools(
    project: &Project,
    splits: &HashMap<String, Vec<String>>,
    padding: i32,
) -> HashMap<String, Vec<(RgbImage, Vec<Annotation>)>> {
    let mut pools = HashMap::new();
    for split in SPLITS {
        let pool = if let Some(names) = splits.get(split) {
            names
                .par_iter()
                .filter_map(|name| {
                    let img = imageio::imread(&project.image_path(name))?;
                    let anns = project
                        .get(name)
                        .map(|ia| ia.annotations.as_slice())
                        .unwrap_or(&[]);
                    extract_patch(&img, anns, padding)
                })
                .collect()
        } else {
            Vec::new()
        };
        pools.insert(split.to_string(), pool);
    }
    pools
}

pub fn mosaic_splits(project: &Project, vcfg: &VersionConfig) -> HashMap<String, Vec<String>> {
    let names: Vec<String> = source_names(project, vcfg)
        .into_iter()
        .filter(|n| project.box_count(n) > 0)
        .collect();
    split_items(&names, vcfg)
}

struct MosaicTask {
    split: String,
    fname: String,
    seed: u64,
}

type MosaicResult = (String, String, u32, u32, Vec<Annotation>);

pub fn generate_mosaic(
    project: &Project,
    vcfg: &VersionConfig,
    mcfg: &MosaicConfig,
    out_dir: &Path,
    progress: Option<&(dyn Fn(usize, usize, &str) + Sync)>,
    cancel_flag: Option<&AtomicBool>,
) -> std::io::Result<VersionStats> {
    std::fs::create_dir_all(out_dir)?;
    let splits = mosaic_splits(project, vcfg);
    if let Some(cb) = progress {
        cb(0, 1, "讀取圖塊…");
    }
    let pools = build_pools(project, &splits, mcfg.padding);
    let class_names = project.class_names();
    let mut writers = make_writers(out_dir, &vcfg.formats, &class_names);

    let m_counts = mcfg.counts();
    let mut counts: HashMap<&str, usize> = HashMap::new();
    let mut patch_counts: HashMap<String, usize> = HashMap::new();
    for s in SPLITS {
        let plen = pools.get(s).map(|v| v.len()).unwrap_or(0);
        patch_counts.insert(s.to_string(), plen);
        counts.insert(s, if plen > 0 { m_counts[s] } else { 0 });
    }

    let mut stats = VersionStats {
        source_images: splits.values().map(|v| v.len()).sum(),
        patches: Some(patch_counts),
        ..Default::default()
    };

    let mut tasks = Vec::new();
    let mut i = 0usize;
    for split in SPLITS {
        for _ in 0..counts[split] {
            i += 1;
            let seed = vcfg
                .seed
                .wrapping_add(2)
                .wrapping_mul(6364136223846793005)
                .wrapping_add((i as u64).wrapping_mul(1442695040888963407));
            tasks.push(MosaicTask {
                split: split.to_string(),
                fname: format!("mosaic_{i:05}.jpg"),
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

    let n_workers = resolve_workers(vcfg.workers, total);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_workers)
        .build()
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

    let done_counter = AtomicUsize::new(0);
    let task_results: Vec<Option<std::io::Result<MosaicResult>>> = pool.install(|| {
        tasks
            .par_iter()
            .map(|t| {
                if cancel_flag.map(|f| f.load(Ordering::Relaxed)).unwrap_or(false) {
                    return None;
                }
                let res = (|| -> std::io::Result<MosaicResult> {
                    let sp_pool = pools.get(&t.split).map(|v| v.as_slice()).unwrap_or(&[]);
                    let mut rng = ChaCha8Rng::seed_from_u64(t.seed);
                    let (img, anns) =
                        make_mosaic(sp_pool, mcfg, &mut rng, vcfg.grayscale);
                    let w = img.width();
                    let h = img.height();
                    let jpg = imageio::encode_jpeg(&img, 95)?;
                    for (_, wr) in &writers {
                        wr.write_worker(&t.split, &t.fname, &jpg, w, h, &anns)?;
                    }
                    Ok((t.split.clone(), t.fname.clone(), w, h, anns))
                })();
                let done = done_counter.fetch_add(1, Ordering::Relaxed) + 1;
                if let Some(cb) = progress {
                    cb(done, total, &t.fname);
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
        let (split, fname, w, h, anns) = entry.unwrap()?;
        for (_, wr) in &mut writers {
            wr.main_commit(&split, &fname, w, h, &anns);
        }
        match split.as_str() {
            "train" => stats.train += 1,
            "valid" => stats.valid += 1,
            "test" => stats.test += 1,
            _ => {}
        }
        stats.boxes += anns.len();
    }

    for (_, wr) in &writers {
        wr.finish()?;
    }
    if let Some(cb) = progress {
        cb(total, total, "");
    }

    let meta = serde_json::json!({
        "type": "mosaic",
        "config": vcfg,
        "mosaic_config": mcfg,
        "stats": stats,
        "classes": class_names,
        "splits": splits,
    });
    write_meta(out_dir, meta)?;
    Ok(stats)
}
