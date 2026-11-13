import json

import cv2
import numpy as np
import pytest
import yaml

from rotalabel import imageio, importers
from rotalabel.augment.pipeline import AugConfig
from rotalabel.augment.version import VersionConfig, generate_version
from rotalabel.exporters import CocoObbWriter, CocoWriter, DotaWriter, YoloObbWriter, YoloWriter
from rotalabel.model import BBOX, OBB, Annotation
from rotalabel.project_io import Project

W, H = 200, 100
BOX = Annotation(0, 50, 50, 40, 20, 0, BBOX)
ROT = Annotation(1, 120, 50, 40, 20, 90, OBB)


def _floats(line):
    return [float(v) for v in line.split()]


def test_yolo_bbox(tmp_path):
    w = YoloWriter(tmp_path, ["a", "b"])
    w.add("train", "x.jpg", b"img", W, H, [BOX, ROT])
    w.finish()
    lines = (tmp_path / "train/labels/x.txt").read_text().splitlines()
    assert _floats(lines[0]) == pytest.approx([0, 0.25, 0.5, 0.2, 0.2])
    # rotated 90 -> bounding rect is 20 x 40
    assert _floats(lines[1]) == pytest.approx([1, 0.6, 0.5, 0.1, 0.4])
    data = yaml.safe_load((tmp_path / "data.yaml").read_text())
    assert data["train"] == "train/images" and data["val"] == "train/images"
    assert data["names"] == {0: "a", 1: "b"}


def test_yolo_obb_and_dota(tmp_path):
    y = YoloObbWriter(tmp_path / "y", ["a", "b"])
    d = DotaWriter(tmp_path / "d", ["a", "b"])
    for wr in (y, d):
        wr.add("valid", "x.jpg", b"img", W, H, [ROT])
        wr.finish()
    yv = _floats((tmp_path / "y/valid/labels/x.txt").read_text())
    dv = (tmp_path / "d/valid/labelTxt/x.txt").read_text().split()
    pix = np.array(yv[1:]).reshape(4, 2) * [W, H]
    assert pix == pytest.approx(np.array([float(v) for v in dv[:8]]).reshape(4, 2), abs=1e-3)
    assert dv[8:] == ["b", "0"]
    assert pix == pytest.approx(ROT.corners(), abs=1e-3)


def test_coco(tmp_path):
    c = CocoWriter(tmp_path, ["a", "b"])
    c.add("train", "x.jpg", b"img", W, H, [BOX, ROT])
    c.finish()
    d = json.loads((tmp_path / "train/_annotations.coco.json").read_text())
    assert d["annotations"][0]["bbox"] == [30, 40, 40, 20]
    assert d["annotations"][0]["category_id"] == 1
    assert d["annotations"][1]["rbox"] == [120, 50, 40, 20, 90]
    assert len(d["annotations"][1]["segmentation"][0]) == 8


def test_coco_obb(tmp_path):
    c = CocoObbWriter(tmp_path, ["a", "b"])
    c.add("train", "x.jpg", b"img", W, H, [BOX, ROT])
    c.add("valid", "y.jpg", b"img", W, H, [ROT])
    c.finish()
    assert (tmp_path / "train/images/x.jpg").exists() and (tmp_path / "val/images/y.jpg").exists()
    assert not (tmp_path / "valid").exists()
    d = json.loads((tmp_path / "annotations/train.json").read_text())
    assert json.loads((tmp_path / "annotations/val.json").read_text())["images"][0]["file_name"] == "y.jpg"
    assert [c["id"] for c in d["categories"]] == [0, 1]
    box, rot = d["annotations"]
    assert box["category_id"] == 0 and rot["category_id"] == 1
    assert box["bbox"] == [30, 40, 40, 20] and box["area"] == 800
    assert rot["segmentation"] == [[130, 30, 130, 70, 110, 70, 110, 30]]
    assert rot["bbox"] == [110, 30, 20, 40] and rot["area"] == 800
    for a in (box, rot):  # clockwise on screen (y down) <=> positive signed shoelace area
        p = np.array(a["segmentation"][0]).reshape(4, 2)
        x, y = p[:, 0], p[:, 1]
        assert np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)) > 0


def _make_project(tmp_path, n=10):
    root = tmp_path / "imgs 中文"
    root.mkdir()
    for i in range(n):
        img = np.zeros((H, W, 3), np.uint8)
        cv2.rectangle(img, (30, 40), (70, 60), (0, 255, 0), -1)
        imageio.imwrite(root / f"img{i}.png", img)
    p = Project.open(root)
    p.add_class("b")
    for name in p.images:
        ia = p.get(name)
        ia.width, ia.height = W, H
        ia.annotations = [BOX, ROT]
        p.save_annotations(name)
    return p


def test_project_roundtrip(tmp_path):
    p = _make_project(tmp_path, 2)
    q = Project.open(p.root)
    assert q.class_names() == ["object", "b"]
    assert q.get("img0.png").annotations[1] == ROT
    q.delete_class(0)
    assert [a.cls for a in Project.open(p.root).get("img1.png").annotations] == [0]


def test_legacy_project_folder_still_opens(tmp_path):
    p = _make_project(tmp_path, 2)
    (p.root / "rotalabel_project").rename(p.root / "bboxer_project")  # folder labeled before the rename
    q = Project.open(p.root)
    assert q.meta_dir.name == "bboxer_project"
    assert q.get("img0.png").annotations[1] == ROT
    assert not (p.root / "rotalabel_project").exists()


def test_split_copies_test_target_and_cap():
    from rotalabel.augment.version import split_copies
    aug = AugConfig(flip_h=True)
    sp = {"train": ["a", "b"], "valid": ["c"], "test": ["d", "e", "f"]}
    cp = split_copies(sp, VersionConfig(multiplier=100, aug=aug))
    assert cp["train"] == [100, 100] and cp["valid"] == [100] and cp["test"] == [4, 3, 3]
    big = {"train": [], "valid": [], "test": [str(i) for i in range(15)]}
    assert split_copies(big, VersionConfig(aug=aug))["test"] == [1] * 10
    # without augmentation nothing is duplicated; test is still capped at 10
    assert split_copies(sp, VersionConfig(multiplier=5))["test"] == [1, 1, 1]
    assert split_copies(big, VersionConfig())["test"] == [1] * 10
    assert split_copies({"train": [], "valid": [], "test": []}, VersionConfig(aug=aug))["test"] == []


def test_generate_version_and_reimport(tmp_path):
    p = _make_project(tmp_path, 10)
    cfg = VersionConfig(multiplier=3, formats=["yolo", "yolo_obb", "dota", "coco", "coco_obb"],
                        resize_mode="fit", resize_w=128, resize_h=128,
                        aug=AugConfig(flip_h=True, rotate=15, brightness=10))
    out = tmp_path / "v1"
    stats = generate_version(p, cfg, out)
    # train/valid multiplied, the single test image topped up to 10
    assert stats["train"] == 7 * 3 and stats["valid"] == 2 * 3 and stats["test"] == 10
    assert len(list((out / "yolo_obb/test/images").iterdir())) == 10
    imgs = list((out / "yolo_obb/train/images").iterdir())
    assert len(imgs) == 21
    assert imageio.imread(imgs[0]).shape[:2] == (128, 128)
    assert (out / "version.json").exists()
    for split, n in (("train", 21), ("val", 6), ("test", 10)):
        js = json.loads((out / f"coco_obb/annotations/{split}.json").read_text(encoding="utf-8"))
        assert len(js["images"]) == n == len(list((out / f"coco_obb/{split}/images").iterdir()))
    # re-import the valid labels (originals + augmented copies) into a fresh project
    tmp2 = tmp_path / "re"
    tmp2.mkdir()
    for f in (out / "dota/valid/images").iterdir():
        (tmp2 / f.name).write_bytes(f.read_bytes())
    q = Project.open(tmp2)
    upd, n = importers.import_labels(q, out / "dota/valid/labelTxt")
    assert upd == 6 and n >= 2
    assert set(q.class_names()) >= {"object", "b"}
