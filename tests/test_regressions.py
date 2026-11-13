"""Regression tests for data-safety bugs (class panel sync, unreadable images, export edge cases)."""
import json
import os

import cv2
import numpy as np
import pytest

from rotalabel import importers
from rotalabel.augment.version import _output_stems, _unique_stem
from rotalabel.exporters import CocoWriter, DotaWriter
from rotalabel.model import BBOX, OBB, Annotation
from rotalabel.project_io import Project


def _make_images(folder, *names):
    folder.mkdir(parents=True, exist_ok=True)
    for n in names:
        cv2.imwrite(str(folder / n), np.zeros((100, 100, 3), np.uint8))


# ---------------------------------------------------------------- non-UI
def test_unique_stem_avoids_aug_name_clash():
    used: set[str] = set()
    assert _unique_stem("a", used, 3) == "a"
    # an image literally named a_aug1 must not overwrite a's first augmented copy
    assert _unique_stem("a_aug1", used, 1) == "a_aug1_2"
    used = {"a_aug1"}
    assert _unique_stem("a", used, 2) == "a_2"
    assert _output_stems("x", 3) == ["x", "x_aug1", "x_aug2"]


def test_import_ids_outside_names_file_get_own_class(tmp_path):
    _make_images(tmp_path / "img", "x.png")
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "classes.txt").write_text("cat\n", encoding="utf-8")
    (labels / "x.txt").write_text("0 0.5 0.5 0.2 0.2\n1 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    p = Project.open(tmp_path / "img")
    p.add_class("cat")  # project: object, cat
    importers.import_labels(p, labels)
    names = p.class_names()
    got = [names[a.cls] for a in p.get("x.png").annotations]
    assert got == ["cat", "class_1"]  # id 1 is not in classes.txt -> must not map to project class 1 ("cat")


def test_dota_and_coco_corners_stay_inside_image(tmp_path):
    a = Annotation(0, 5, 50, 60, 20, 30.0, OBB)  # sticks out of the left edge
    d = DotaWriter(tmp_path / "d", ["a"])
    d.add("train", "x.jpg", b"img", 200, 100, [a])
    vals = [float(v) for v in (tmp_path / "d/train/labelTxt/x.txt").read_text().split()[:8]]
    assert min(vals[0::2]) >= 0 and max(vals[0::2]) <= 200
    c = CocoWriter(tmp_path / "c", ["a"])
    c.add("train", "x.jpg", b"img", 200, 100, [a])
    c.finish()
    seg = json.loads((tmp_path / "c/train/_annotations.coco.json").read_text())["annotations"][0]["segmentation"][0]
    assert min(seg[0::2]) >= 0


# ---------------------------------------------------------------- UI (offscreen Qt)
@pytest.fixture
def window(tmp_path, monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    import rotalabel.ui  # noqa: F401
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    QSettings.setDefaultFormat(QSettings.IniFormat)
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, str(tmp_path / "settings"))
    from rotalabel.ui.main_window import MainWindow
    w = MainWindow()
    yield w
    w.close()
    app.processEvents()


def _add_box(w, cls=0, select=True):
    c = w.canvas
    before = c._snapshot()
    it = c._add_item(Annotation(cls, 50, 50, 20, 20, 0, BBOX))
    c._commit(before, "add")
    if select:
        c.select(it)
    return it


def _saved(folder, name):
    f = folder / "rotalabel_project" / "annotations" / f"{name}.json"
    return json.loads(f.read_text(encoding="utf-8"))["annotations"] if f.exists() else None


def test_switching_folder_does_not_leak_selected_box(window, tmp_path):
    a, b = tmp_path / "A", tmp_path / "B"
    _make_images(a, "img1.png")
    _make_images(b, "img1.png")
    window.open_project(str(a))
    window.project.add_class("cat")
    window.class_panel.reload()
    _add_box(window, cls=1)
    window.open_project(str(b))
    assert _saved(b, "img1.png") is None
    assert _saved(a, "img1.png")[0]["cls"] == 1


def test_adding_class_keeps_selected_box_class(window, tmp_path, monkeypatch):
    _make_images(tmp_path / "A", "img1.png")
    window.open_project(str(tmp_path / "A"))
    it = _add_box(window, cls=0)
    from PySide6.QtWidgets import QInputDialog
    monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **k: ("dog", True)))
    window.class_panel.add()
    assert it.ann.cls == 0
    assert window.canvas.current_class == 1  # new class becomes the drawing class


def test_label_output_dialog_switches_format(window, tmp_path, monkeypatch):
    folder = tmp_path / "A"
    _make_images(folder, "a.png")
    (folder / "a.txt").write_text("0 0.1 0.1 0.5 0.1 0.5 0.3 0.1 0.3\n", encoding="utf-8")
    window.open_project(str(folder))
    assert "YOLO-OBB → rotalabel_labels/" in window.windowTitle()
    from PySide6.QtWidgets import QComboBox, QDialog, QLineEdit, QMessageBox

    def fake_exec(dlg):
        dlg.findChild(QComboBox).setCurrentIndex(dlg.findChild(QComboBox).findData("dota"))
        dlg.findChild(QLineEdit).setText("out_dota")  # relative -> inside the image folder
        return QDialog.Accepted
    monkeypatch.setattr(QDialog, "exec", fake_exec)
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: None))
    window.edit_label_output()
    assert (folder / "out_dota" / "a.txt").read_text(encoding="utf-8").startswith("imagesource:rotalabel")
    assert "DOTA → out_dota/" in window.windowTitle()
    assert (folder / "a.txt").read_text(encoding="utf-8") == "0 0.1 0.1 0.5 0.1 0.5 0.3 0.1 0.3\n"


def test_unreadable_image_keeps_annotations(window, tmp_path):
    folder = tmp_path / "A"
    _make_images(folder, "a.png", "b.png")
    window.open_project(str(folder))
    window.step(1)
    _add_box(window, select=False)
    window.step(-1)
    (folder / "b.png").write_bytes(b"corrupt")
    window.step(1)
    window.step(-1)
    assert len(_saved(folder, "b.png")) == 1
    assert window.project.get("b.png").width == 100
