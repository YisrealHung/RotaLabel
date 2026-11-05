"""Pixel-level augmentations on BGR uint8 images. Boxes are unaffected."""
from __future__ import annotations

import cv2
import numpy as np


def apply_photometric(img: np.ndarray, cfg, rng: np.random.Generator) -> np.ndarray:
    out = img.astype(np.float32)

    if cfg.brightness > 0:
        out += rng.uniform(-cfg.brightness, cfg.brightness) / 100 * 255
    if cfg.exposure > 0:
        # gamma-like exposure change: >1 darkens, <1 brightens
        e = rng.uniform(-cfg.exposure, cfg.exposure) / 100
        gamma = 2.0 ** (-e * 2)
        out = 255.0 * np.power(np.clip(out, 0, 255) / 255.0, gamma)
    if cfg.contrast > 0:
        c = 1 + rng.uniform(-cfg.contrast, cfg.contrast) / 100
        mean = out.mean()
        out = (out - mean) * c + mean

    if cfg.saturation > 0 or cfg.hue > 0:
        hsv = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        if cfg.hue > 0:
            hsv[..., 0] = (hsv[..., 0] + rng.uniform(-cfg.hue, cfg.hue) / 2) % 180  # OpenCV hue is 0..180
        if cfg.saturation > 0:
            hsv[..., 1] *= 1 + rng.uniform(-cfg.saturation, cfg.saturation) / 100
        out = cv2.cvtColor(np.clip(hsv, 0, [179, 255, 255]).astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

    if cfg.grayscale > 0 and rng.random() < cfg.grayscale / 100:
        g = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        out = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR).astype(np.float32)

    if cfg.blur > 0:
        sigma = rng.uniform(0, cfg.blur)
        if sigma > 0.3:
            out = cv2.GaussianBlur(out, (0, 0), sigma)

    out = np.clip(out, 0, 255).astype(np.uint8)

    if cfg.noise > 0:
        frac = rng.uniform(0, cfg.noise / 100)
        mask = rng.random(out.shape[:2])
        out[mask < frac / 2] = 0
        out[(mask >= frac / 2) & (mask < frac)] = 255
    return out
