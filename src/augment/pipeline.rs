use image::RgbImage;
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::augment::geometric::sample_geometric;
use crate::augment::photometric::apply_photometric;
use crate::geometry::{self, Mat3};
use crate::model::{Annotation, BoxKind};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct AugConfig {
    // geometric
    pub flip_h: bool,
    pub flip_v: bool,
    pub rot90: bool,
    pub rotate: f64,
    pub shear: f64,
    pub scale: f64,
    pub crop: f64,
    pub translate_x: f64,
    pub translate_y: f64,
    pub translate_allow_out: bool,
    // photometric
    pub brightness: f64,
    pub exposure: f64,
    pub contrast: f64,
    pub saturation: f64,
    pub hue: f64,
    pub grayscale: f64,
    pub blur: f64,
    pub noise: f64,
}

impl AugConfig {
    /// Options that only modify another augmentation don't enable anything by themselves.
    pub fn any_enabled(&self) -> bool {
        self.flip_h
            || self.flip_v
            || self.rot90
            || self.rotate > 0.0
            || self.shear > 0.0
            || self.scale > 0.0
            || self.crop > 0.0
            || self.translate_x > 0.0
            || self.translate_y > 0.0
            || self.brightness > 0.0
            || self.exposure > 0.0
            || self.contrast > 0.0
            || self.saturation > 0.0
            || self.hue > 0.0
            || self.grayscale > 0.0
            || self.blur > 0.0
            || self.noise > 0.0
    }
}

/// Bilinear affine warp matching `cv2.warpAffine(..., INTER_LINEAR, BORDER_CONSTANT, (0,0,0))`.
pub fn warp(img: &RgbImage, m: &Mat3, out_w: u32, out_h: u32) -> RgbImage {
    if m.is_identity(1e-9) && img.width() == out_w && img.height() == out_h {
        return img.clone();
    }
    let Some(inv) = m.invert() else {
        return RgbImage::new(out_w, out_h);
    };
    let sw = img.width() as i32;
    let sh = img.height() as i32;
    let src = img.as_raw();
    let sw_u = sw as usize;
    let mut dst = vec![0u8; (out_w as usize) * (out_h as usize) * 3];

    let a = inv.0[0][0];
    let b = inv.0[0][1];
    let tx = inv.0[0][2];
    let c = inv.0[1][0];
    let d = inv.0[1][1];
    let ty = inv.0[1][2];

    for y in 0..out_h as usize {
        let fy = y as f64;
        let base_sx = b * fy + tx;
        let base_sy = d * fy + ty;
        let row_out = y * (out_w as usize) * 3;
        for x in 0..out_w as usize {
            let fx = x as f64;
            let sx = a * fx + base_sx;
            let sy = c * fx + base_sy;
            let x0 = sx.floor() as i32;
            let y0 = sy.floor() as i32;
            let x1 = x0 + 1;
            let y1 = y0 + 1;
            if x1 < 0 || x0 >= sw || y1 < 0 || y0 >= sh {
                continue;
            }
            let wx1 = (sx - x0 as f64) as f32;
            let wy1 = (sy - y0 as f64) as f32;
            let wx0 = 1.0 - wx1;
            let wy0 = 1.0 - wy1;

            let sample = |px: i32, py: i32| -> [f32; 3] {
                if px >= 0 && px < sw && py >= 0 && py < sh {
                    let i = (py as usize * sw_u + px as usize) * 3;
                    [src[i] as f32, src[i + 1] as f32, src[i + 2] as f32]
                } else {
                    [0.0, 0.0, 0.0]
                }
            };

            let p00 = sample(x0, y0);
            let p10 = sample(x1, y0);
            let p01 = sample(x0, y1);
            let p11 = sample(x1, y1);
            let w00 = wx0 * wy0;
            let w10 = wx1 * wy0;
            let w01 = wx0 * wy1;
            let w11 = wx1 * wy1;

            let out_i = row_out + x * 3;
            for ch in 0..3 {
                let v = p00[ch] * w00 + p10[ch] * w10 + p01[ch] * w01 + p11[ch] * w11;
                dst[out_i + ch] = v.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    RgbImage::from_raw(out_w, out_h, dst).expect("valid buffer")
}

/// Bilinearly warp a solid `1.0` mask of size `src_w x src_h` for mosaic edge blending.
pub fn warp_mask(src_w: u32, src_h: u32, m: &Mat3, out_w: u32, out_h: u32) -> Vec<f32> {
    let Some(inv) = m.invert() else {
        return vec![0.0; (out_w as usize) * (out_h as usize)];
    };
    let sw = src_w as i32;
    let sh = src_h as i32;
    let mut dst = vec![0.0_f32; (out_w as usize) * (out_h as usize)];

    let a = inv.0[0][0];
    let b = inv.0[0][1];
    let tx = inv.0[0][2];
    let c = inv.0[1][0];
    let d = inv.0[1][1];
    let ty = inv.0[1][2];

    for y in 0..out_h as usize {
        let fy = y as f64;
        let base_sx = b * fy + tx;
        let base_sy = d * fy + ty;
        let row_out = y * (out_w as usize);
        for x in 0..out_w as usize {
            let fx = x as f64;
            let sx = a * fx + base_sx;
            let sy = c * fx + base_sy;
            let x0 = sx.floor() as i32;
            let y0 = sy.floor() as i32;
            let x1 = x0 + 1;
            let y1 = y0 + 1;
            if x1 < 0 || x0 >= sw || y1 < 0 || y0 >= sh {
                continue;
            }
            let wx1 = (sx - x0 as f64) as f32;
            let wy1 = (sy - y0 as f64) as f32;
            let wx0 = 1.0 - wx1;
            let wy0 = 1.0 - wy1;
            let inside = |px: i32, py: i32| -> f32 {
                if px >= 0 && px < sw && py >= 0 && py < sh {
                    1.0
                } else {
                    0.0
                }
            };
            dst[row_out + x] = inside(x0, y0) * wx0 * wy0
                + inside(x1, y0) * wx1 * wy0
                + inside(x0, y1) * wx0 * wy1
                + inside(x1, y1) * wx1 * wy1;
        }
    }
    dst
}

pub fn transform_annotation(
    a: &Annotation,
    m: &Mat3,
    out_w: u32,
    out_h: u32,
    min_visibility: f64,
) -> Option<Annotation> {
    let tc = m.transform_points(&a.corners());
    let fw = out_w as f64;
    let fh = out_h as f64;
    if a.kind == BoxKind::Bbox {
        let [x0, y0, x1, y1] = geometry::aabb(&tc);
        let full = (x1 - x0) * (y1 - y0);
        let cx0 = x0.max(0.0);
        let cy0 = y0.max(0.0);
        let cx1 = x1.min(fw);
        let cy1 = y1.min(fh);
        if full <= 0.0 || cx1 - cx0 < 1.0 || cy1 - cy0 < 1.0 {
            return None;
        }
        if (cx1 - cx0) * (cy1 - cy0) / full < min_visibility {
            return None;
        }
        return Some(Annotation::new(
            a.cls,
            (cx0 + cx1) / 2.0,
            (cy0 + cy1) / 2.0,
            cx1 - cx0,
            cy1 - cy0,
            0.0,
            BoxKind::Bbox,
        ));
    }

    let full = geometry::polygon_area(&tc);
    if full <= 0.0 {
        return None;
    }
    let clipped = geometry::clip_polygon(&tc, fw, fh);
    if clipped.len() < 3 {
        return None;
    }
    let vis = geometry::polygon_area(&clipped) / full;
    if vis < min_visibility {
        return None;
    }
    let dx = tc[1][0] - tc[0][0];
    let dy = tc[1][1] - tc[0][1];
    let hint = Some(dy.atan2(dx).to_degrees());
    let pts = if vis > 0.999 { &tc } else { &clipped };
    let (cx, cy, w, h, ang) = geometry::fit_obb(pts, hint);
    if w < 1.0 || h < 1.0 {
        return None;
    }
    Some(Annotation::new(a.cls, cx, cy, w, h, ang, a.kind))
}

pub fn transform_annotations(
    anns: &[Annotation],
    m: &Mat3,
    out_w: u32,
    out_h: u32,
    min_visibility: f64,
) -> Vec<Annotation> {
    anns.iter()
        .filter_map(|a| transform_annotation(a, m, out_w, out_h, min_visibility))
        .collect()
}

pub fn augment<R: Rng>(
    img: &RgbImage,
    anns: &[Annotation],
    cfg: &AugConfig,
    rng: &mut R,
    pre: Option<&Mat3>,
    min_visibility: f64,
    size: Option<(u32, u32)>,
) -> (RgbImage, Vec<Annotation>) {
    let pre_m = pre.copied().unwrap_or_else(Mat3::identity);
    let (pw, ph) = size.unwrap_or_else(|| (img.width(), img.height()));
    let all_pts: Vec<[f64; 2]> = if !anns.is_empty() {
        let raw_pts: Vec<[f64; 2]> = anns.iter().flat_map(|a| a.corners()).collect();
        pre_m.transform_points(&raw_pts)
    } else {
        Vec::new()
    };
    let pts_opt = (!all_pts.is_empty()).then_some(all_pts.as_slice());
    let (g, ow, oh) = sample_geometric(cfg, pw, ph, rng, pts_opt);
    let m = g.mul(&pre_m);
    let out = warp(img, &m, ow, oh);
    let out = apply_photometric(&out, cfg, rng);
    let out_anns = transform_annotations(anns, &m, ow, oh, min_visibility);
    (out, out_anns)
}
