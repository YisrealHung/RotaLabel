"""Generation gives byte-identical output whatever the number of worker processes, and can be cancelled."""
import cv2
import numpy as np

from rotalabel import imageio
from rotalabel.augment.mosaic import MosaicConfig, generate_mosaic
from rotalabel.augment.parallel import resolve_workers, run_tasks
from rotalabel.augment.pipeline import AugConfig
from rotalabel.augment.version import VersionConfig, generate_version
from rotalabel.model import OBB, Annotation
from rotalabel.project_io import Project


def _square(x):
    return x * x


def test_run_tasks_keeps_order_serial_and_parallel():
    assert list(run_tasks(_square, list(range(20)), workers=1)) == [i * i for i in range(20)]
    assert list(run_tasks(_square, list(range(20)), workers=3)) == [i * i for i in range(20)]
    assert list(run_tasks(_square, [], workers=3)) == []
    assert resolve_workers(0, 5) >= 1 and resolve_workers(8, 3) == 3 and resolve_workers(1, 100) == 1


def test_run_tasks_cancel():
    n = 0

    def cancelled():
        return n >= 5

    out = []
    for r in run_tasks(_square, list(range(50)), workers=2, cancelled=cancelled):
        out.append(r)
        n += 1
    assert len(out) == 5


def _project(tmp_path, n=8):
    root = tmp_path / "imgs"
    root.mkdir()
    for i in range(n):
        img = np.full((160 + 20 * (i % 3), 220, 3), 60, np.uint8)
        cv2.rectangle(img, (40, 50), (100, 70), (255, 255, 255), -1)
        imageio.imwrite(root / f"img{i}.png", img)
    p = Project.open(root)
    for name in p.images:
        p.get(name).annotations = [Annotation(0, 70, 60, 60, 20, 0, OBB), Annotation(0, 150, 110, 50, 16, 40, OBB)]
        p.save_annotations(name)
    return p


def _tree(root):
    return {str(f.relative_to(root)): f.read_bytes() for f in sorted(root.rglob("*"))
            if f.is_file() and f.name not in ("version.json", "data.yaml")}   # these hold the folder path / time


def test_generate_version_same_output_any_workers(tmp_path):
    p = _project(tmp_path)
    outs = []
    for workers in (1, 4):
        cfg = VersionConfig(multiplier=3, formats=["yolo_obb", "coco_obb"], workers=workers,
                            aug=AugConfig(rotate=20, brightness=15, noise=1))
        out = tmp_path / f"v{workers}"
        stats = generate_version(p, cfg, out)
        assert not stats.get("cancelled")
        outs.append(_tree(out))
    assert outs[0] and outs[0] == outs[1]


def test_generate_mosaic_same_output_any_workers(tmp_path):
    p = _project(tmp_path)
    outs = []
    for workers in (1, 4):
        vcfg = VersionConfig(formats=["yolo_obb"], workers=workers)
        mcfg = MosaicConfig(canvas_w=500, canvas_h=400, count_train=6, count_valid=2, count_test=3,
                            patches=5, patches_jitter=1, rotate=30, aug=AugConfig(hue=10))
        out = tmp_path / f"m{workers}"
        generate_mosaic(p, vcfg, mcfg, out)
        outs.append(_tree(out))
    assert outs[0] and outs[0] == outs[1]


def test_generate_cancel(tmp_path):
    p = _project(tmp_path)
    stats = generate_mosaic(p, VersionConfig(workers=2), MosaicConfig(count_train=50), tmp_path / "m",
                            cancelled=lambda: True)
    assert stats["cancelled"] and stats["train"] == 0
    stats = generate_version(p, VersionConfig(workers=2), tmp_path / "v", cancelled=lambda: True)
    assert stats["cancelled"]
