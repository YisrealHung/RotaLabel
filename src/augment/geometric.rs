//! Geometric transforms expressed as 3x3 affine matrices (`output = M * input`).

use rand::Rng;

use crate::augment::pipeline::AugConfig;
use crate::geometry::{self, Mat3};

pub fn translate(tx: f64, ty: f64) -> Mat3 {
    Mat3([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]])
}

pub fn scale(s: f64) -> Mat3 {
    scale_xy(s, s)
}

pub fn scale_xy(sx: f64, sy: f64) -> Mat3 {
    Mat3([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]])
}

pub fn about(m: Mat3, cx: f64, cy: f64) -> Mat3 {
    translate(cx, cy).mul(&m).mul(&translate(-cx, -cy))
}

pub fn rotation(deg: f64) -> Mat3 {
    let (s, c) = deg.to_radians().sin_cos();
    Mat3([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
}

pub fn shear(deg_x: f64, deg_y: f64) -> Mat3 {
    Mat3([
        [1.0, deg_x.to_radians().tan(), 0.0],
        [deg_y.to_radians().tan(), 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
}

pub fn flip_h(w: f64) -> Mat3 {
    Mat3([[-1.0, 0.0, w], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
}

pub fn flip_v(h: f64) -> Mat3 {
    Mat3([[1.0, 0.0, 0.0], [0.0, -1.0, h], [0.0, 0.0, 1.0]])
}

pub fn rot90(k: i32, w: f64, h: f64) -> (Mat3, f64, f64) {
    match k.rem_euclid(4) {
        0 => (Mat3::identity(), w, h),
        1 => (
            Mat3([[0.0, -1.0, h], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
            h,
            w,
        ),
        2 => (
            Mat3([[-1.0, 0.0, w], [0.0, -1.0, h], [0.0, 0.0, 1.0]]),
            w,
            h,
        ),
        _ => (
            Mat3([[0.0, 1.0, 0.0], [-1.0, 0.0, w], [0.0, 0.0, 1.0]]),
            h,
            w,
        ),
    }
}

pub fn preprocess_matrix(w: u32, h: u32, mode: &str, tw: u32, th: u32) -> (Mat3, u32, u32) {
    let fw = w.max(1) as f64;
    let fh = h.max(1) as f64;
    let ftw = tw.max(1) as f64;
    let fth = th.max(1) as f64;
    match mode {
        "stretch" => (scale_xy(ftw / fw, fth / fh), tw, th),
        "fit" => {
            let r = (ftw / fw).min(fth / fh);
            let nw = fw * r;
            let nh = fh * r;
            (
                translate((ftw - nw) / 2.0, (fth - nh) / 2.0).mul(&scale(r)),
                tw,
                th,
            )
        }
        _ => (Mat3::identity(), w, h),
    }
}

fn shift_range(limit: f64, lo_edge: f64, hi_edge: f64, size: f64) -> (f64, f64) {
    (
        (-limit).max(0.0_f64.min(-lo_edge)),
        limit.min(0.0_f64.max(size - hi_edge)),
    )
}

fn uniform<R: Rng>(rng: &mut R, lo: f64, hi: f64) -> f64 {
    if hi <= lo {
        lo
    } else {
        rng.gen_range(lo..hi)
    }
}

pub fn sample_geometric<R: Rng>(
    cfg: &AugConfig,
    w: u32,
    h: u32,
    rng: &mut R,
    points: Option<&[[f64; 2]]>,
) -> (Mat3, u32, u32) {
    let mut m = Mat3::identity();
    let mut cw = w as f64;
    let mut ch = h as f64;

    if cfg.flip_h && rng.gen::<f64>() < 0.5 {
        m = flip_h(cw).mul(&m);
    }
    if cfg.flip_v && rng.gen::<f64>() < 0.5 {
        m = flip_v(ch).mul(&m);
    }
    if cfg.rot90 {
        let k = rng.gen_range(0..4);
        let (r, nw, nh) = rot90(k, cw, ch);
        cw = nw;
        ch = nh;
        m = r.mul(&m);
    }
    if cfg.rotate > 0.0 {
        let deg = uniform(rng, -cfg.rotate, cfg.rotate);
        m = about(rotation(deg), cw / 2.0, ch / 2.0).mul(&m);
    }
    if cfg.shear > 0.0 {
        let dx = uniform(rng, -cfg.shear, cfg.shear);
        let dy = uniform(rng, -cfg.shear, cfg.shear);
        m = about(shear(dx, dy), cw / 2.0, ch / 2.0).mul(&m);
    }
    if cfg.scale > 0.0 {
        let s = uniform(rng, (1.0 - cfg.scale / 100.0).max(0.05), 1.0 + cfg.scale / 100.0);
        m = about(scale(s), cw / 2.0, ch / 2.0).mul(&m);
    }
    if cfg.crop > 0.0 {
        let p = uniform(rng, 0.0, (cfg.crop / 100.0).min(0.95));
        let kw = cw * (1.0 - p);
        let kh = ch * (1.0 - p);
        let x0 = uniform(rng, 0.0, (cw - kw).max(0.0));
        let y0 = uniform(rng, 0.0, (ch - kh).max(0.0));
        m = scale_xy(cw / kw, ch / kh).mul(&translate(-x0, -y0)).mul(&m);
    }
    let tx_max = cfg.translate_x;
    let ty_max = cfg.translate_y;
    if tx_max > 0.0 || ty_max > 0.0 {
        let (mut x_lo, mut x_hi, mut y_lo, mut y_hi) = (-tx_max, tx_max, -ty_max, ty_max);
        if !cfg.translate_allow_out {
            if let Some(pts) = points {
                if !pts.is_empty() {
                    let tpts = m.transform_points(pts);
                    let [bx0, by0, bx1, by1] = geometry::aabb(&tpts);
                    (x_lo, x_hi) = shift_range(tx_max, bx0, bx1, cw);
                    (y_lo, y_hi) = shift_range(ty_max, by0, by1, ch);
                }
            }
        }
        let tx = if x_hi > x_lo { uniform(rng, x_lo, x_hi) } else { 0.0 };
        let ty = if y_hi > y_lo { uniform(rng, y_lo, y_hi) } else { 0.0 };
        m = translate(tx, ty).mul(&m);
    }

    (m, cw.round() as u32, ch.round() as u32)
}
