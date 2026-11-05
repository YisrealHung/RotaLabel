"""Image read/write that works with non-ASCII Windows paths (cv2.imread does not)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def imread(path: str | Path) -> np.ndarray | None:
    """Read as BGR uint8. EXIF orientation is applied (cv2 default), matching Qt's autoTransform."""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def encode(img: np.ndarray, ext: str = ".jpg", quality: int = 95) -> bytes:
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if ext.lower() in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        raise IOError(f"failed to encode image as {ext}")
    return buf.tobytes()


def imwrite(path: str | Path, img: np.ndarray, quality: int = 95) -> None:
    Path(path).write_bytes(encode(img, Path(path).suffix, quality))
