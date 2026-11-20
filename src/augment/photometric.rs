//! Pixel-level augmentations on RGB uint8 images. Boxes are unaffected.

use image::RgbImage;
use rand::Rng;

use crate::augment::pipeline::AugConfig;

fn uniform<R: Rng>(rng: &mut R, lo: f64, hi: f64) -> f64 {
    if hi <= lo {
        lo
    } else {
        rng.gen_range(lo..hi)
    }
}

fn rgb_to_hsv(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let d = max - min;
    let v = max;
    let s = if max > 0.0 { d / max } else { 0.0 };
    let mut h = if d <= 1e-6 {
        0.0
    } else if (max - r).abs() < 1e-6 {
        60.0 * (((g - b) / d) % 6.0)
    } else if (max - g).abs() < 1e-6 {
        60.0 * ((b - r) / d + 2.0)
    } else {
        60.0 * ((r - g) / d + 4.0)
    };
    if h < 0.0 {
        h += 360.0;
    }
    (h, s, v)
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
    let h = h.rem_euclid(360.0);
    let s = s.clamp(0.0, 1.0);
    let v = v.clamp(0.0, 1.0);
    let c = v * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = v - c;
    let (r1, g1, b1) = if h < 60.0 {
        (c, x, 0.0)
    } else if h < 120.0 {
        (x, c, 0.0)
    } else if h < 180.0 {
        (0.0, c, x)
    } else if h < 240.0 {
        (0.0, x, c)
    } else if h < 300.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };
    (r1 + m, g1 + m, b1 + m)
}

fn gaussian_blur_rgb(buf: &mut [f32], w: usize, h: usize, sigma: f32) {
    let radius = (sigma * 3.0).ceil().max(1.0) as isize;
    let ksize = (2 * radius + 1) as usize;
    let mut kernel = vec![0.0_f32; ksize];
    let inv2s2 = 1.0 / (2.0 * sigma * sigma);
    let mut ksum = 0.0_f32;
    for i in 0..ksize {
        let x = (i as isize - radius) as f32;
        let val = (-x * x * inv2s2).exp();
        kernel[i] = val;
        ksum += val;
    }
    for v in &mut kernel {
        *v /= ksum;
    }

    let mut tmp = vec![0.0_f32; buf.len()];
    // Horizontal pass with edge clamping
    for y in 0..h {
        let row_off = y * w * 3;
        for x in 0..w {
            let mut r = 0.0_f32;
            let mut g = 0.0_f32;
            let mut b = 0.0_f32;
            for (ki, &kv) in kernel.iter().enumerate() {
                let sx = (x as isize + ki as isize - radius).clamp(0, w as isize - 1) as usize;
                let idx = row_off + sx * 3;
                r += buf[idx] * kv;
                g += buf[idx + 1] * kv;
                b += buf[idx + 2] * kv;
            }
            let dst = row_off + x * 3;
            tmp[dst] = r;
            tmp[dst + 1] = g;
            tmp[dst + 2] = b;
        }
    }
    // Vertical pass with edge clamping
    for y in 0..h {
        for x in 0..w {
            let mut r = 0.0_f32;
            let mut g = 0.0_f32;
            let mut b = 0.0_f32;
            for (ki, &kv) in kernel.iter().enumerate() {
                let sy = (y as isize + ki as isize - radius).clamp(0, h as isize - 1) as usize;
                let idx = (sy * w + x) * 3;
                r += tmp[idx] * kv;
                g += tmp[idx + 1] * kv;
                b += tmp[idx + 2] * kv;
            }
            let dst = (y * w + x) * 3;
            buf[dst] = r;
            buf[dst + 1] = g;
            buf[dst + 2] = b;
        }
    }
}

pub fn apply_photometric<R: Rng>(img: &RgbImage, cfg: &AugConfig, rng: &mut R) -> RgbImage {
    let w = img.width() as usize;
    let h = img.height() as usize;
    let raw = img.as_raw();
    let mut out: Vec<f32> = raw.iter().map(|&v| v as f32).collect();

    if cfg.brightness > 0.0 {
        let delta = (uniform(rng, -cfg.brightness, cfg.brightness) / 100.0 * 255.0) as f32;
        for v in &mut out {
            *v += delta;
        }
    }

    if cfg.exposure > 0.0 {
        let e = uniform(rng, -cfg.exposure, cfg.exposure) / 100.0;
        let gamma = 2.0_f64.powf(-e * 2.0) as f32;
        for v in &mut out {
            let norm = (*v).clamp(0.0, 255.0) / 255.0;
            *v = 255.0 * norm.powf(gamma);
        }
    }

    if cfg.contrast > 0.0 && !out.is_empty() {
        let c = (1.0 + uniform(rng, -cfg.contrast, cfg.contrast) / 100.0) as f32;
        let mean: f32 = out.iter().sum::<f32>() / out.len() as f32;
        for v in &mut out {
            *v = (*v - mean) * c + mean;
        }
    }

    if cfg.saturation > 0.0 || cfg.hue > 0.0 {
        let h_shift = if cfg.hue > 0.0 {
            uniform(rng, -cfg.hue, cfg.hue) as f32
        } else {
            0.0
        };
        let s_scale = if cfg.saturation > 0.0 {
            (1.0 + uniform(rng, -cfg.saturation, cfg.saturation) / 100.0) as f32
        } else {
            1.0
        };
        for px in out.chunks_exact_mut(3) {
            let r = px[0].clamp(0.0, 255.0) / 255.0;
            let g = px[1].clamp(0.0, 255.0) / 255.0;
            let b = px[2].clamp(0.0, 255.0) / 255.0;
            let (mut hh, mut ss, vv) = rgb_to_hsv(r, g, b);
            hh = (hh + h_shift).rem_euclid(360.0);
            ss = (ss * s_scale).clamp(0.0, 1.0);
            let (nr, ng, nb) = hsv_to_rgb(hh, ss, vv);
            px[0] = nr * 255.0;
            px[1] = ng * 255.0;
            px[2] = nb * 255.0;
        }
    }

    if cfg.grayscale > 0.0 && rng.gen::<f64>() < cfg.grayscale / 100.0 {
        for px in out.chunks_exact_mut(3) {
            let y = 0.299 * px[0].clamp(0.0, 255.0)
                + 0.587 * px[1].clamp(0.0, 255.0)
                + 0.114 * px[2].clamp(0.0, 255.0);
            px[0] = y;
            px[1] = y;
            px[2] = y;
        }
    }

    if cfg.blur > 0.0 && w > 0 && h > 0 {
        let sigma = uniform(rng, 0.0, cfg.blur) as f32;
        if sigma > 0.3 {
            gaussian_blur_rgb(&mut out, w, h, sigma);
        }
    }

    let mut bytes: Vec<u8> = out.into_iter().map(|v| v.round().clamp(0.0, 255.0) as u8).collect();

    if cfg.noise > 0.0 {
        let frac = uniform(rng, 0.0, cfg.noise / 100.0);
        let half = frac * 0.5;
        for px in bytes.chunks_exact_mut(3) {
            let r: f64 = rng.gen();
            if r < half {
                px[0] = 0;
                px[1] = 0;
                px[2] = 0;
            } else if r < frac {
                px[0] = 255;
                px[1] = 255;
                px[2] = 255;
            }
        }
    }

    RgbImage::from_raw(img.width(), img.height(), bytes).expect("buffer length matches dimensions")
}
