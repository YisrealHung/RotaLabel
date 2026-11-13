from __future__ import annotations

import copy
import os
from pathlib import Path

from PySide6.QtCore import QSettings, QSize, Qt
from PySide6.QtGui import QAction, QActionGroup, QColor, QKeySequence
from PySide6.QtWidgets import (QApplication, QComboBox, QDialog, QDialogButtonBox, QDockWidget, QFileDialog,
                               QFormLayout, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
                               QMainWindow, QMenu, QMessageBox, QProgressBar, QPushButton, QSizePolicy, QToolBar,
                               QToolButton, QVBoxLayout, QWidget)

from .. import importers
from ..model import BBOX, OBB
from ..project_io import DEFAULT_LABEL_DIR, LABEL_FORMATS, Project
from . import theme
from .canvas import LABEL_MODES, LABELS_ALL, LABELS_NONE, LABELS_SELECTED, SHOW_NOTHING, Canvas
from .class_panel import ClassPanel
from .delegates import DATA, AnnotationDelegate, ImageDelegate

ICONS = {}  # action -> icon name, so icons can be redrawn when the theme changes

LABEL_MODE_TEXT = {
    LABELS_ALL: "全部顯示（框＋標籤）",
    LABELS_SELECTED: "只顯示選取框的標籤",
    LABELS_NONE: "只顯示框（隱藏標籤）",
    SHOW_NOTHING: "全部隱藏（只看圖片）",
}
LABEL_MODE_SHORT = {LABELS_ALL: "全部", LABELS_SELECTED: "選取", LABELS_NONE: "只有框", SHOW_NOTHING: "隱藏"}

HELP = """
<h3>標註</h3>
<table cellpadding=3>
<tr><td><b>W</b></td><td>畫 BBox（拖曳）</td></tr>
<tr><td><b>E</b></td><td>畫旋轉框（拖曳方式與 BBox 相同，畫完拖黃色把手旋轉）</td></tr>
<tr><td><b>V</b></td><td>選取 / 編輯模式</td></tr>
<tr><td><b>Ctrl + 左鍵</b>、右鍵、雙擊</td><td>在繪製模式下直接選取框</td></tr>
<tr><td><b>Z / X</b></td><td>選取框逆 / 順時針旋轉 1°（Shift：15°）；BBox 會自動轉成旋轉框</td></tr>
<tr><td>拖曳旋轉把手 + <b>Shift</b></td><td>以 15° 為單位吸附</td></tr>
<tr><td><b>T</b></td><td>選取框在 BBox / 旋轉框之間切換</td></tr>
<tr><td><b>1 – 9</b></td><td>切換類別（若有選取框則同時變更其類別）</td></tr>
<tr><td><b>Delete</b></td><td>刪除選取框</td></tr>
<tr><td><b>方向鍵</b></td><td>微調位置 1px（Shift：10px）</td></tr>
<tr><td><b>Ctrl+C / Ctrl+V</b></td><td>複製 / 貼上框（可跨圖片）</td></tr>
<tr><td><b>Ctrl+Z / Ctrl+Y</b></td><td>復原 / 重做</td></tr>
<tr><td><b>A / D</b></td><td>上一張 / 下一張</td></tr>
<tr><td><b>F</b></td><td>圖片符合視窗</td></tr>
<tr><td>滾輪 / Shift+滾輪</td><td>上下 / 左右捲動</td></tr>
<tr><td>Ctrl+滾輪</td><td>縮放</td></tr>
<tr><td>中鍵或空白鍵拖曳</td><td>平移</td></tr>
<tr><td><b>L</b></td><td>切換顯示：全部 → 只顯示選取框的標籤 → 只顯示框 → 全部隱藏（只看圖片）</td></tr>
<tr><td><b>Esc</b></td><td>取消選取</td></tr>
</table>
<p>標註會自動儲存到 <code>&lt;圖片資料夾&gt;/rotalabel_project/</code>。</p>
"""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RotaLabel 標註工具")
        self.resize(1440, 900)
        self.settings = QSettings("rotalabel", "rotalabel")
        self._migrate_settings()
        self.project: Project | None = None
        self.index = -1
        theme.apply_theme(QApplication.instance(), self.settings.value("theme", "dark"))

        self.canvas = Canvas()
        self.setCentralWidget(self.canvas)
        self.canvas.changed.connect(self._on_canvas_changed)
        self.canvas.selection_changed.connect(self._on_selection)
        self.canvas.cursor_moved.connect(self._on_cursor)
        self.canvas.zoom_changed.connect(lambda z: self.zoom_label.setText(f"縮放 {z * 100:.0f}%"))

        self._build_docks()
        self._build_actions()
        self._build_status()
        self._apply_theme_extras()

        last = self.settings.value("last_project")
        if last and Path(last).is_dir():
            self.open_project(last)

    def _migrate_settings(self):
        """Carry over last folder / theme saved under the app's old name (BBoxer)."""
        if self.settings.value("migrated_from_bboxer"):
            return
        old = QSettings("bboxer", "bboxer")
        for key in ("last_project", "theme"):
            if self.settings.value(key) is None and old.value(key) is not None:
                self.settings.setValue(key, old.value(key))
        self.settings.setValue("migrated_from_bboxer", True)

    # ------------------------------------------------------------------ UI
    def _dock(self, title: str, name: str, widget: QWidget, area) -> QDockWidget:
        d = QDockWidget(title, self)
        d.setObjectName(name)
        d.setWidget(widget)
        d.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.addDockWidget(area, d)
        return d

    def _build_docks(self):
        # image list
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(8, 4, 8, 8)
        lay.setSpacing(6)
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("🔍  搜尋檔名…")
        self.filter_edit.setClearButtonEnabled(True)
        self.filter_edit.textChanged.connect(self._apply_filter)
        self.image_list = QListWidget()
        self.image_list.setItemDelegate(ImageDelegate(self.image_list))
        self.image_list.setMouseTracking(True)
        self.image_list.setUniformItemSizes(True)
        self.image_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.image_list.currentRowChanged.connect(self._on_image_row)
        self.progress_label = QLabel()
        self.progress_label.setObjectName("muted")
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(6)
        lay.addWidget(self.filter_edit)
        lay.addWidget(self.image_list, 1)
        lay.addWidget(self.progress_label)
        lay.addWidget(self.progress_bar)
        d = self._dock("圖片", "images", w, Qt.LeftDockWidgetArea)
        d.setMinimumWidth(240)

        # classes
        self.class_panel = ClassPanel()
        self.class_panel.class_selected.connect(self._on_class_selected)
        self.class_panel.current_class_changed.connect(self._on_current_class)
        self.class_panel.classes_changed.connect(self.canvas.refresh)
        self.class_panel.classes_changed.connect(self._refresh_ann_list)
        self.class_panel.about_to_delete.connect(self.flush)
        self.class_panel.class_deleted.connect(self._on_class_deleted)
        d = self._dock("類別", "classes", self.class_panel, Qt.RightDockWidgetArea)
        d.setMinimumWidth(280)

        # annotations of current image
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(6, 2, 6, 6)
        self.ann_list = QListWidget()
        self.ann_list.setItemDelegate(AnnotationDelegate(self.ann_list))
        self.ann_list.setMouseTracking(True)
        self.ann_list.currentRowChanged.connect(self._on_ann_row)
        self.ann_empty = QLabel("尚無標註\n按 W 畫 BBox，按 E 畫旋轉框")
        self.ann_empty.setObjectName("hint")
        self.ann_empty.setAlignment(Qt.AlignCenter)
        lay.addWidget(self.ann_list, 1)
        lay.addWidget(self.ann_empty, 1)
        self.ann_dock = self._dock("此圖標註", "annotations", w, Qt.RightDockWidgetArea)

    def _act(self, text, slot, shortcut=None, checkable=False, tip=None, icon=None):
        a = QAction(text, self)
        if shortcut:
            a.setShortcut(QKeySequence(shortcut))
            a.setShortcutContext(Qt.WindowShortcut)
        a.setCheckable(checkable)
        key = QKeySequence(shortcut).toString(QKeySequence.NativeText) if shortcut else ""
        a.setToolTip(f"{tip or text}  ({key})" if key else (tip or text))
        if icon:
            ICONS[a] = icon
        a.triggered.connect(slot)
        self.addAction(a)
        return a

    def _build_actions(self):
        c = self.canvas
        self.a_open = self._act("開啟資料夾", self.choose_folder, "Ctrl+O", icon="folder")
        self.a_save = self._act("儲存", self.flush, "Ctrl+S", icon="save")
        self.a_prev = self._act("上一張", lambda: self.step(-1), "A", icon="prev")
        self.a_next = self._act("下一張", lambda: self.step(1), "D", icon="next")
        self.a_select = self._act("選取", lambda: self.set_mode("select"), "V", True, "選取 / 編輯框", "select")
        self.a_bbox = self._act("BBox", lambda: self.set_mode(BBOX), "W", True, "畫 BBox", "bbox")
        self.a_obb = self._act("旋轉框", lambda: self.set_mode(OBB), "E", True, "畫旋轉框", "obb")
        group = QActionGroup(self)
        for a in (self.a_select, self.a_bbox, self.a_obb):
            group.addAction(a)
        self.a_bbox.setChecked(True)
        self.a_fit = self._act("符合視窗", c.fit_view, "F", icon="fit")
        self.a_undo = c.undo_stack.createUndoAction(self, "復原")
        self.a_undo.setShortcut(QKeySequence.Undo)
        self.a_undo.setToolTip("復原 (Ctrl+Z)")
        ICONS[self.a_undo] = "undo"
        self.a_redo = c.undo_stack.createRedoAction(self, "重做")
        self.a_redo.setShortcuts([QKeySequence("Ctrl+Y"), QKeySequence("Ctrl+Shift+Z")])
        self.a_redo.setToolTip("重做 (Ctrl+Y)")
        ICONS[self.a_redo] = "redo"
        self.addActions([self.a_undo, self.a_redo])
        self.a_version = self._act("產生資料集", self.open_version_dialog, "Ctrl+E",
                                   tip="產生資料集版本：切分、擴增、匯出", icon="layers")
        self.a_import = self._act("匯入標註 (YOLO / YOLO-OBB / DOTA)…", self.import_labels, icon="import")
        self.a_label_out = self._act("標註輸出設定…", self.edit_label_output,
                                     tip="修改後的標註要寫到哪個資料夾、用什麼格式")
        self.a_theme = self._act("切換深色 / 淺色主題", self.toggle_theme, icon="theme")
        self.a_help = self._act("快捷鍵說明", self.show_help, "F1", icon="help")

        self._act("刪除框", c.delete_selected, "Delete")
        self._act("刪除框", c.delete_selected, "Backspace")
        self._act("逆時針 1°", lambda: c.rotate_selected(-1), "Z")
        self._act("順時針 1°", lambda: c.rotate_selected(1), "X")
        self._act("逆時針 15°", lambda: c.rotate_selected(-15), "Shift+Z")
        self._act("順時針 15°", lambda: c.rotate_selected(15), "Shift+X")
        self._act("切換框類型", c.toggle_selected_kind, "T")
        self._act("複製", c.copy_selected, "Ctrl+C")
        self._act("貼上", c.paste, "Ctrl+V")
        # label display: L cycles, the View menu / toolbar dropdown pick a mode directly
        self.a_labels = self._act("標籤", self.toggle_labels, "L",
                                  tip="切換顯示：全部 → 只顯示選取框的標籤 → 只顯示框 → 全部隱藏", icon="tag")
        self.label_mode_actions = {}
        label_group = QActionGroup(self)
        self.label_menu = QMenu("標註顯示", self)
        for mode in LABEL_MODES:
            a = QAction(LABEL_MODE_TEXT[mode], self)
            a.setCheckable(True)
            a.triggered.connect(lambda _=False, m=mode: self.set_label_mode(m))
            label_group.addAction(a)
            self.label_menu.addAction(a)
            self.label_mode_actions[mode] = a
        self.a_labels.setMenu(self.label_menu)
        for i in range(9):
            self._act(f"類別 {i + 1}", lambda _=False, i=i: self.pick_class(i), str(i + 1))

        tb = QToolBar("工具")
        tb.setObjectName("tools")
        tb.setMovable(False)
        tb.setIconSize(QSize(20, 20))
        tb.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.addToolBar(tb)
        tb.addActions([self.a_open, self.a_save])
        tb.addSeparator()
        tb.addActions([self.a_prev, self.a_next])
        tb.addSeparator()
        tb.addActions([self.a_select, self.a_bbox, self.a_obb])
        tb.addSeparator()
        for a in (self.a_undo, self.a_redo, self.a_fit):
            tb.addAction(a)
            tb.widgetForAction(a).setToolButtonStyle(Qt.ToolButtonIconOnly)
        tb.addSeparator()
        tb.addAction(self.a_labels)
        tb.widgetForAction(self.a_labels).setPopupMode(QToolButton.MenuButtonPopup)
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        tb.addWidget(spacer)
        for a in (self.a_theme, self.a_help):
            tb.addAction(a)
            tb.widgetForAction(a).setToolButtonStyle(Qt.ToolButtonIconOnly)
        tb.addAction(self.a_version)
        btn = tb.widgetForAction(self.a_version)
        btn.setObjectName("primary")
        btn.setCursor(Qt.PointingHandCursor)
        self.toolbar = tb

        m = self.menuBar().addMenu("檔案")
        m.addActions([self.a_open, self.a_save, self.a_import, self.a_label_out])
        m.addSeparator()
        m.addAction(self.a_version)
        m.addAction(self._act("開啟版本資料夾", self.open_versions_folder))
        m.addSeparator()
        m.addAction(self._act("結束", self.close))
        m = self.menuBar().addMenu("編輯")
        m.addActions([self.a_undo, self.a_redo])
        m.addSeparator()
        m.addActions([self.a_select, self.a_bbox, self.a_obb])
        m = self.menuBar().addMenu("檢視")
        m.addActions([self.a_fit, self.a_theme])
        m.addSeparator()
        m.addMenu(self.label_menu)
        self.set_label_mode(self.settings.value("label_mode", LABELS_ALL))
        m = self.menuBar().addMenu("說明")
        m.addAction(self.a_help)

    def _build_status(self):
        self.pos_label = QLabel()
        self.zoom_label = QLabel()
        self.info_label = QLabel()
        self.mode_label = QLabel()
        sb = self.statusBar()
        sb.setSizeGripEnabled(False)
        sb.addWidget(self.mode_label)
        sb.addWidget(self.info_label, 1)
        sb.addPermanentWidget(self.pos_label)
        sb.addPermanentWidget(self.zoom_label)

    def _apply_theme_extras(self):
        """Things a stylesheet cannot reach: icons, canvas colors, list repaint."""
        for a, name in ICONS.items():
            a.setIcon(theme.make_icon(name))
        self.canvas.setBackgroundBrush(QColor(theme.current["canvas"]))
        self.canvas.viewport().update()
        for lst in (self.image_list, self.ann_list, self.class_panel.list):
            lst.viewport().update()
        self._update_mode_label()

    def toggle_theme(self):
        name = "light" if theme.current_name == "dark" else "dark"
        theme.apply_theme(QApplication.instance(), name)
        self.settings.setValue("theme", name)
        self._apply_theme_extras()

    def _update_mode_label(self):
        mode = getattr(self.canvas, "mode", BBOX)
        text = {"select": "選取", BBOX: "BBox", OBB: "旋轉框"}[mode]
        p = theme.current
        self.mode_label.setText(text)
        self.mode_label.setStyleSheet(
            f"background:{p['accent']}; color:{p['accent_text']}; border-radius:8px; padding:1px 10px;"
            f"font-weight:bold; margin:3px 4px;")

    # ------------------------------------------------------------------ project
    def choose_folder(self):
        start = self.settings.value("last_project") or str(Path.home())
        d = QFileDialog.getExistingDirectory(self, "選擇圖片資料夾", start)
        if d:
            self.open_project(d)

    def open_project(self, folder: str):
        self.flush()
        try:
            project = Project.open(folder)
        except OSError as e:
            QMessageBox.critical(self, "開啟失敗", str(e))
            return
        # detach the old image first so nothing below can edit it or save it into the new project
        self.canvas.select(None)
        self.index = -1
        self.project = project
        self.settings.setValue("last_project", folder)
        self._update_title()
        n_img, n_box = self.project.auto_imported
        if n_img:
            self.statusBar().showMessage(
                f"已載入資料夾內的 YOLO-OBB 標註：{n_img} 張圖片、{n_box} 個框（原檔不會被修改）；"
                f"修改後的標註存到 {self._output_text()}", 15000)
        self.canvas.classes = self.project.classes
        self.class_panel.set_project(self.project)
        self.image_list.blockSignals(True)
        self.image_list.clear()
        for name in self.project.images:
            self.image_list.addItem(QListWidgetItem(name))
        self.image_list.blockSignals(False)
        self._refresh_image_items()
        if not self.project.images:
            self.canvas.scene().clear()
            self.info_label.setText("此資料夾沒有圖片")
            return
        last = self.project.settings.get("last_image")
        start = self.project.images.index(last) if last in self.project.images else 0
        self.image_list.setCurrentRow(start)

    def current_name(self) -> str | None:
        if self.project and 0 <= self.index < len(self.project.images):
            return self.project.images[self.index]
        return None

    def flush(self):
        """Write the canvas boxes back to the project and save."""
        name = self.current_name()
        if not name or self.canvas.pixmap_item is None:
            return  # image could not be loaded: keep its stored annotations untouched
        ia = self.project.get(name)
        ia.annotations = copy.deepcopy(self.canvas.annotations())
        ia.width, ia.height = self.canvas.img_w, self.canvas.img_h
        self.project.save_annotations(name)
        self._refresh_image_item(self.index)

    def load_index(self, idx: int):
        if not self.project or not (0 <= idx < len(self.project.images)):
            return
        if idx != self.index:
            self.flush()
        self.index = idx
        name = self.project.images[idx]
        ia = self.project.get(name)
        w, h = self.canvas.load_image(str(self.project.image_path(name)), ia.annotations)
        self.project.settings["last_image"] = name
        self._refresh_ann_list()
        self._update_info()
        if w:
            ia.width, ia.height = w, h
        else:
            self.info_label.setText(f"無法讀取：{name}（此圖的標註保留不變）")
        self.canvas.setFocus()

    def _reload_current(self):
        idx, self.index = self.index, -1
        self.canvas.classes = self.project.classes
        self.load_index(idx)
        self._refresh_image_items()

    def step(self, d: int):
        if not self.project:
            return
        row = self.image_list.currentRow()
        r = row + d
        while 0 <= r < self.image_list.count() and self.image_list.item(r).isHidden():
            r += d
        if 0 <= r < self.image_list.count():
            self.image_list.setCurrentRow(r)

    def closeEvent(self, e):
        self.flush()
        if self.project:
            self.project.save_meta()
        super().closeEvent(e)

    # ------------------------------------------------------------------ lists
    def _refresh_image_item(self, row: int):
        if not self.project or not (0 <= row < self.image_list.count()):
            return
        name = self.project.images[row]
        n = self.project.box_count(name)
        it = self.image_list.item(row)
        it.setText(name)
        it.setData(DATA, n)
        it.setToolTip(f"{name}\n{n} 個框" if n else f"{name}\n尚未標註")
        self._update_progress()

    def _refresh_image_items(self):
        for r in range(self.image_list.count()):
            self._refresh_image_item(r)

    def _update_progress(self):
        if not self.project:
            return
        total = len(self.project.images)
        done = sum(1 for n in self.project.images if self.project.box_count(n))
        boxes = sum(self.project.box_count(n) for n in self.project.images)
        pct = done * 100 // total if total else 0
        self.progress_label.setText(f"已標註 {done} / {total} 張（{pct}%）· {boxes} 個框")
        self.progress_bar.setMaximum(max(total, 1))
        self.progress_bar.setValue(done)

    def _apply_filter(self, text: str):
        t = text.lower()
        for r in range(self.image_list.count()):
            self.image_list.item(r).setHidden(bool(t) and t not in self.project.images[r].lower())

    def _on_image_row(self, row: int):
        if row >= 0:
            self.load_index(row)

    def _ann_data(self, i, a) -> dict:
        return {"n": i + 1, "color": self.canvas.class_color(a.cls).name(), "name": self.canvas.class_name(a.cls),
                "kind": a.kind, "angle": a.angle, "w": a.w, "h": a.h}

    def _refresh_ann_list(self):
        self.ann_list.blockSignals(True)
        self.ann_list.clear()
        anns = self.canvas.annotations()
        for i, a in enumerate(anns):
            it = QListWidgetItem(self.canvas.class_name(a.cls))
            it.setData(DATA, self._ann_data(i, a))
            self.ann_list.addItem(it)
        self.ann_list.setCurrentRow(self.canvas.selected_index())
        self.ann_list.blockSignals(False)
        self.ann_list.setVisible(bool(anns))
        self.ann_empty.setVisible(not anns)
        self.ann_dock.setWindowTitle(f"此圖標註（{len(anns)}）" if anns else "此圖標註")

    def _on_ann_row(self, row: int):
        self.canvas.select_index(row)
        self.canvas.setFocus()

    # ------------------------------------------------------------------ canvas events
    def _on_canvas_changed(self):
        self.flush()  # autosave every edit
        self._refresh_ann_list()
        self._update_info()

    def _on_selection(self, idx: int):
        self.ann_list.blockSignals(True)
        self.ann_list.setCurrentRow(idx)
        anns = self.canvas.annotations()
        if 0 <= idx < self.ann_list.count():
            self.ann_list.item(idx).setData(DATA, self._ann_data(idx, anns[idx]))
        self.ann_list.blockSignals(False)
        if 0 <= idx < len(anns):
            self.class_panel.list.blockSignals(True)
            self.class_panel.set_current(anns[idx].cls)
            self.class_panel.list.blockSignals(False)
            self.canvas.current_class = anns[idx].cls
        self._update_info()

    def _on_cursor(self, x, y):
        self.pos_label.setText(f"x {x:.0f}   y {y:.0f}")

    def _update_info(self):
        name = self.current_name()
        if not name:
            return
        s = f"{self.index + 1} / {len(self.project.images)}   ·   {name}   ·   {self.canvas.img_w}×{self.canvas.img_h}" \
            f"   ·   類別：{self.canvas.class_name(self.canvas.current_class)}"
        it = self.canvas.selected_item
        if it:
            a = it.ann
            s += f"   ·   選取 cx {a.cx:.0f}  cy {a.cy:.0f}  w {a.w:.0f}  h {a.h:.0f}  ∠ {a.angle:.1f}°"
        self.info_label.setText(s)

    def _on_current_class(self, idx: int):
        self.canvas.current_class = idx
        self._update_info()

    def _on_class_deleted(self):
        self.canvas.clipboard = []  # copied boxes may refer to class ids that no longer exist
        self._reload_current()

    def _on_class_selected(self, idx: int):
        self.canvas.current_class = idx
        self.canvas.set_selected_class(idx)
        self._update_info()

    def pick_class(self, i: int):
        if self.project and i < len(self.project.classes):
            self.class_panel.set_current(i)

    def set_mode(self, mode: str):
        self.canvas.set_mode(mode)
        {"select": self.a_select, BBOX: self.a_bbox, OBB: self.a_obb}[mode].setChecked(True)
        self._update_mode_label()
        self._update_info()

    def toggle_labels(self):
        """L / toolbar button: all -> selected tag only -> boxes only -> nothing -> all."""
        i = LABEL_MODES.index(self.canvas.label_mode)
        self.set_label_mode(LABEL_MODES[(i + 1) % len(LABEL_MODES)])

    def set_label_mode(self, mode: str):
        if mode not in LABEL_MODES:
            mode = LABELS_ALL
        self.canvas.set_label_mode(mode)
        self.label_mode_actions[mode].setChecked(True)
        self.a_labels.setText(f"顯示：{LABEL_MODE_SHORT[mode]}")
        self.settings.setValue("label_mode", mode)
        if hasattr(self, "info_label"):
            self.statusBar().showMessage(f"標註顯示：{LABEL_MODE_TEXT[mode]}", 2000)

    # ------------------------------------------------------------------ dialogs
    def open_version_dialog(self):
        if not self.project:
            return
        self.flush()
        from .version_dialog import VersionDialog
        VersionDialog(self.project, self).exec()

    def open_versions_folder(self):
        if self.project:
            self.project.versions_dir.mkdir(parents=True, exist_ok=True)
            os.startfile(str(self.project.versions_dir))

    def import_labels(self):
        if not self.project:
            return
        d = QFileDialog.getExistingDirectory(self, "選擇標註資料夾（.txt，檔名需與圖片相同）",
                                             str(self.project.root))
        if not d:
            return
        r = QMessageBox.question(self, "匯入標註", "對有對應標註檔的圖片：\n是 = 取代現有標註\n否 = 附加到現有標註",
                                 QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
        if r == QMessageBox.Cancel:
            return
        self.flush()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            n_img, n_box = importers.import_labels(self.project, d, overwrite=(r == QMessageBox.Yes))
        finally:
            QApplication.restoreOverrideCursor()
        self.class_panel.reload()
        self._reload_current()
        QMessageBox.information(self, "匯入標註", f"已匯入 {n_img} 張圖片、{n_box} 個框")

    # ------------------------------------------------------------------ label output
    def _output_text(self) -> str:
        p = self.project
        out = p.output_dir
        where = "圖片旁" if out is None else (
            f"{out.relative_to(p.root).as_posix()}/" if out.is_relative_to(p.root) else str(out))
        return f"{LABEL_FORMATS[p.label_format]} → {where}"

    def _update_title(self):
        self.setWindowTitle(f"RotaLabel — {self.project.root}  [標註輸出：{self._output_text()}]")

    def edit_label_output(self):
        if not self.project:
            return
        p = self.project
        dlg = QDialog(self)
        dlg.setWindowTitle("標註輸出設定")
        form = QFormLayout(dlg)
        fmt = QComboBox()
        for key, label in LABEL_FORMATS.items():
            fmt.addItem(label, key)
        fmt.setCurrentIndex(max(0, fmt.findData(p.label_format)))
        path = QLineEdit(str(p.output_dir or p.root / DEFAULT_LABEL_DIR))
        browse = QPushButton("瀏覽…")

        def pick():
            d = QFileDialog.getExistingDirectory(dlg, "選擇輸出資料夾", path.text() or str(p.root))
            if d:
                path.setText(d)
        browse.clicked.connect(pick)
        row = QHBoxLayout()
        row.addWidget(path, 1)
        row.addWidget(browse)
        form.addRow("格式", fmt)
        form.addRow("資料夾", row)
        hint = QLabel("按確定後，所有已標註的圖片會以新設定重新寫出一次。\n"
                      "圖片旁原本的標註檔不會被修改；舊輸出資料夾的檔案會保留。")
        hint.setObjectName("muted")
        form.addRow(hint)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        form.addRow(buttons)
        dlg.resize(560, dlg.sizeHint().height())
        if dlg.exec() != QDialog.Accepted:
            return
        target = Path(path.text().strip() or DEFAULT_LABEL_DIR)
        if not target.is_absolute():
            target = p.root / target
        self.flush()
        try:
            n = p.set_label_output(fmt.currentData(), target)
        except (ValueError, OSError) as e:
            QMessageBox.warning(self, "標註輸出設定", f"無法套用：{e}")
            return
        self._update_title()
        QMessageBox.information(self, "標註輸出設定", f"已將 {n} 張圖片的標註寫到 {self._output_text()}")

    def show_help(self):
        QMessageBox.information(self, "快捷鍵", HELP)
