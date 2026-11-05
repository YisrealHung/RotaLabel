"""Dataset writers. Each writer receives already-encoded image bytes plus boxes in
pixel coordinates of that (possibly augmented) image."""
from .base import BaseWriter
from .coco import CocoWriter
from .dota import DotaWriter
from .yolo import YoloObbWriter, YoloWriter

FORMATS: dict[str, type[BaseWriter]] = {
    "yolo": YoloWriter,
    "yolo_obb": YoloObbWriter,
    "dota": DotaWriter,
    "coco": CocoWriter,
}

FORMAT_LABELS = {
    "yolo": "YOLO (bbox)",
    "yolo_obb": "YOLO-OBB (旋轉框)",
    "dota": "DOTA (旋轉框)",
    "coco": "COCO JSON",
}

__all__ = ["BaseWriter", "FORMATS", "FORMAT_LABELS", "YoloWriter", "YoloObbWriter", "DotaWriter", "CocoWriter"]
