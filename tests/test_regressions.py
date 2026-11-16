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


def test_label_display_modes(window, tmp_path):
    from PySide6.QtCore import QPointF

    from rotalabel.ui.canvas import LABELS_ALL, LABELS_NONE, LABELS_SELECTED, SHOW_NOTHING
    _make_images(tmp_path / "A", "img1.png")
    window.open_project(str(tmp_path / "A"))
    c = window.canvas
    a = _add_box(window, select=False)
    b = _add_box(window, select=False)
    window.set_label_mode(LABELS_ALL)
    assert c.label_visible(a) and c.label_visible(b)
    window.toggle_labels()  # L: all -> tag on selected only
    assert c.label_mode == LABELS_SELECTED
    assert not c.label_visible(a) and not c.label_visible(b)
    c.select(b)
    assert c.label_visible(b) and not c.label_visible(a)
    window.toggle_labels()  # -> boxes only, no tags even for the selected box
    assert c.label_mode == LABELS_NONE and not c.label_visible(b)
    assert c.box_visible(a) and c.box_visible(b)
    window.toggle_labels()  # -> nothing: boxes hidden too, except the one being edited
    assert c.label_mode == SHOW_NOTHING
    assert window.label_mode_actions[SHOW_NOTHING].isChecked()
    assert c.box_visible(b) and not c.box_visible(a)
    c.select(None)
    assert not c.box_visible(a) and not c.box_visible(b)
    assert c._item_at(QPointF(50, 50)) is None  # hidden boxes can't be grabbed
    window.toggle_labels()  # -> back to all
    assert c.label_mode == LABELS_ALL
    assert c._item_at(QPointF(50, 50)) is not None
    # the choice is remembered for the next window
    window.set_label_mode(LABELS_SELECTED)
    from rotalabel.ui.main_window import MainWindow
    w2 = MainWindow()
    assert w2.canvas.label_mode == LABELS_SELECTED
    w2.close()


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


def test_delete_selected_and_last_box(window, tmp_path):
    folder = tmp_path / "A"
    _make_images(folder, "a.png")
    window.open_project(str(folder))
    c = window.canvas
    it1 = _add_box(window, cls=0)
    it2 = _add_box(window, cls=0)
    assert len(c.items) == 2
    c.select(it1)
    c.delete_selected()
    assert len(c.items) == 1
    assert c.selected_item is it2
    # delete the last remaining box
    c.delete_selected()
    assert len(c.items) == 0
    assert c.selected_item is None
    # calling delete when empty is a safe no-op
    c.delete_selected()
    assert len(c.items) == 0


def test_delete_while_drawing_cancels_drag(window, tmp_path):
    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtGui import QKeyEvent
    folder = tmp_path / "A"
    _make_images(folder, "a.png")
    window.open_project(str(folder))
    c = window.canvas
    # simulate active draw drag
    c._drag = {"type": "draw", "start": QPointF(10, 10), "end": QPointF(40, 40)}
    ev = QKeyEvent(QKeyEvent.KeyPress, Qt.Key_Delete, Qt.NoModifier)
    c.keyPressEvent(ev)
    assert c._drag is None
    assert len(c.items) == 0


def test_delete_while_resizing_clears_drag(window, tmp_path):
    from PySide6.QtCore import QPoint, Qt
    folder = tmp_path / "A"
    _make_images(folder, "a.png")
    window.open_project(str(folder))
    c = window.canvas
    it = _add_box(window, cls=0)
    c._drag = {"type": "resize", "handle": "br", "item": it, "orig": it.ann, "before": c._snapshot()}
    c.delete_selected()
    assert c._drag is None
    assert len(c.items) == 0

    class DummyEvent:
        def position(self):
            class Pos:
                def toPoint(self): return QPoint(60, 60)
            return Pos()
        def modifiers(self): return Qt.NoModifier

    # subsequent mouse move / release should not crash or resurrect the box
    c.mouseMoveEvent(DummyEvent())
    c.mouseReleaseEvent(DummyEvent())
    assert len(c.items) == 0


def test_open_empty_folder_clears_canvas_state(window, tmp_path):
    a, empty = tmp_path / "A", tmp_path / "empty"
    _make_images(a, "img1.png")
    empty.mkdir()
    window.open_project(str(a))
    _add_box(window, cls=0)
    assert len(window.canvas.items) == 1
    window.open_project(str(empty))
    c = window.canvas
    assert len(c.items) == 0
    assert c.pixmap_item is None
    assert c.selected_item is None
    assert c.hover_item is None
    assert c.undo_stack.count() == 0
    # fit_view and undo on empty canvas should be safe no-ops
    c.fit_view()
    c.undo_stack.undo()


def test_version_dialog_reject_while_worker_running(window, tmp_path):
    _make_images(tmp_path / "A", "img1.png")
    window.open_project(str(tmp_path / "A"))
    from rotalabel.ui.version_dialog import VersionDialog

    dlg = VersionDialog(window.project, window)

    class FakeWorker:
        def __init__(self):
            self.cancelled = False
            self.running = True

        def isRunning(self):
            return self.running

        def cancel(self):
            self.cancelled = True

        def wait(self):
            self.running = False

    dlg.worker = FakeWorker()
    dlg.reject()  # Esc while worker is running must cancel, not close/destroy thread
    assert dlg.worker.cancelled is True
    dlg.worker.running = False
    dlg.close()
