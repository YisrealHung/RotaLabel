"""Roboflow-style "Generate version" dialog: split, preprocess, augment, preview, export."""
from __future__ import annotations

import os

import cv2
import numpy as np
from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFormLayout, QGridLayout,
                               QGroupBox, QHBoxLayout, QLabel, QMessageBox, QProgressBar, QPushButton,
                               QScrollArea, QSizePolicy, QSpinBox, QVBoxLayout, QWidget)

from .. import imageio
from ..augment.pipeline import AugConfig
from ..augment.version import (VersionConfig, draw_annotations, generate_version, make_sample,
                               next_version_dir, split_items)
from ..exporters import FORMAT_LABELS
from . import theme

# key, label, unit, max, default-when-enabled, decimals
GEOMETRIC = [
    ("rotate", "任意角度旋轉", "± 度", 180, 15, 0),
    ("shear", "剪切 Shear", "± 度", 45, 10, 0),
    ("scale", "縮放", "± %", 50, 20, 0),
    ("crop", "隨機裁切 (放大)", "最多 %", 80, 20, 0),
]
PHOTOMETRIC = [
    ("brightness", "亮度", "± %", 100, 25, 0),
    ("exposure", "曝光", "± %", 100, 15, 0),
    ("contrast", "對比", "± %", 100, 20, 0),
    ("saturation", "飽和度", "± %", 100, 25, 0),
    ("hue", "色相", "± 度", 180, 15, 0),
    ("grayscale", "灰階", "機率 %", 100, 15, 0),
    ("blur", "模糊", "最多 px", 10, 1.5, 1),
    ("noise", "雜訊", "最多 % 像素", 25, 2, 1),
]
FLAGS = [("flip_h", "水平翻轉"), ("flip_v", "垂直翻轉"), ("rot90", "90° 旋轉（隨機 0/90/180/270）")]


def _hint(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setObjectName("hint")
    lab.setWordWrap(True)
    return lab


def to_qpixmap(img: np.ndarray) -> QPixmap:
    img = np.ascontiguousarray(img)
    h, w = img.shape[:2]
    q = QImage(img.data, w, h, img.strides[0], QImage.Format_BGR888)
    return QPixmap.fromImage(q.copy())


class Worker(QThread):
    progress = Signal(int, int, str)
    finished_ok = Signal(dict, str)
    failed = Signal(str)

    def __init__(self, project, cfg, out_dir):
        super().__init__()
        self.project, self.cfg, self.out_dir = project, cfg, out_dir
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            stats = generate_version(self.project, self.cfg, self.out_dir,
                                     progress=lambda a, b, n: self.progress.emit(a, b, n),
                                     cancelled=lambda: self._cancel)
            self.finished_ok.emit(stats, str(self.out_dir))
        except Exception as e:  # noqa: BLE001 - surface any failure to the user
            self.failed.emit(f"{type(e).__name__}: {e}")


class VersionDialog(QDialog):
    def __init__(self, project, parent=None):
        super().__init__(parent)
        self.setWindowTitle("產生資料集版本")
        self.resize(1250, 820)
        self.project = project
        self.worker: Worker | None = None
        self.preview_seed = 0
        self.preview_name: str | None = None
        self.aug_rows: dict[str, tuple[QCheckBox, QDoubleSpinBox | None]] = {}

        cfg = VersionConfig.from_dict(project.settings.get("version_config", {}))

        form_widget = QWidget()
        form = QVBoxLayout(form_widget)
        form.setContentsMargins(0, 0, 10, 0)
        form.setSpacing(4)
        form.addWidget(self._split_group(cfg))
        form.addWidget(self._pre_group(cfg))
        form.addWidget(self._aug_group("3. 幾何擴增 · 框會跟著轉換", FLAGS, GEOMETRIC, cfg.aug))
        form.addWidget(self._aug_group("4. 色彩 / 像素擴增", [], PHOTOMETRIC, cfg.aug))
        form.addWidget(self._output_group(cfg))
        form.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(form_widget)
        scroll.setMinimumWidth(560)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # preview
        title = QLabel("擴增預覽")
        title.setObjectName("summary")
        hint = QLabel("左上為前處理後原圖，其餘為隨機擴增樣本")
        hint.setObjectName("hint")
        self.preview = QLabel("預覽")
        self.preview.setObjectName("preview")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        b_img = QPushButton("換一張圖片")
        b_img.clicked.connect(self.next_preview_image)
        b_rand = QPushButton("重新隨機")
        b_rand.clicked.connect(self.reroll)
        heads = QVBoxLayout()
        heads.setSpacing(0)
        heads.addWidget(title)
        heads.addWidget(hint)
        prow = QHBoxLayout()
        prow.addLayout(heads)
        prow.addStretch(1)
        prow.addWidget(b_img)
        prow.addWidget(b_rand)
        right = QVBoxLayout()
        right.setSpacing(10)
        right.addLayout(prow)
        right.addWidget(self.preview, 1)

        top = QHBoxLayout()
        top.setSpacing(16)
        top.addWidget(scroll, 0)
        top.addLayout(right, 1)

        self.summary = QLabel()
        self.summary.setObjectName("summary")
        self.bar = QProgressBar()
        self.bar.setVisible(False)
        self.b_gen = QPushButton("產生版本")
        self.b_gen.setObjectName("primary")
        self.b_gen.setDefault(True)
        self.b_gen.setCursor(Qt.PointingHandCursor)
        self.b_gen.clicked.connect(self.generate)
        self.b_close = QPushButton("關閉")
        self.b_close.clicked.connect(self.close)
        bottom = QHBoxLayout()
        bottom.addWidget(self.summary, 1)
        bottom.addWidget(self.bar, 1)
        bottom.addWidget(self.b_close)
        bottom.addWidget(self.b_gen)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 14, 16, 14)
        lay.setSpacing(12)
        lay.addLayout(top, 1)
        lay.addLayout(bottom)

        self._connect_all(form_widget)
        resize_on = self.cmb_resize.currentData() != "none"
        self.sp_rw.setEnabled(resize_on)
        self.sp_rh.setEnabled(resize_on)
        self.next_preview_image()
        self._update_summary()

    # ------------------------------------------------------------------ groups
    def _split_group(self, cfg):
        g = QGroupBox("1. 切分 Train / Valid / Test")
        f = QFormLayout(g)
        self.sp_train, self.sp_valid, self.sp_test = (self._spin(0, 100, v) for v in (cfg.train, cfg.valid, cfg.test))
        row = QHBoxLayout()
        for lab, sp in (("Train", self.sp_train), ("Valid", self.sp_valid), ("Test", self.sp_test)):
            row.addWidget(QLabel(lab))
            row.addWidget(sp)
        f.addRow("比例 %", row)
        self.sp_seed = self._spin(0, 999999, cfg.seed)
        f.addRow("亂數種子", self.sp_seed)
        self.cb_skip = QCheckBox("略過沒有標註的圖片")
        self.cb_skip.setChecked(cfg.skip_unannotated)
        f.addRow(self.cb_skip)
        f.addRow(_hint("擴增只會套用在 Train；Valid / Test 只做前處理。"))
        return g

    def _pre_group(self, cfg):
        g = QGroupBox("2. 前處理（所有切分）")
        f = QFormLayout(g)
        self.cmb_resize = QComboBox()
        for key, text in (("none", "不調整大小"), ("stretch", "拉伸至"), ("fit", "等比例縮放 + 黑邊補齊至")):
            self.cmb_resize.addItem(text, key)
        self.cmb_resize.setCurrentIndex(max(0, self.cmb_resize.findData(cfg.resize_mode)))
        self.sp_rw, self.sp_rh = self._spin(16, 8192, cfg.resize_w), self._spin(16, 8192, cfg.resize_h)
        row = QHBoxLayout()
        row.addWidget(self.cmb_resize)
        row.addWidget(self.sp_rw)
        row.addWidget(QLabel("×"))
        row.addWidget(self.sp_rh)
        f.addRow("Resize", row)
        self.cb_gray = QCheckBox("轉灰階")
        self.cb_gray.setChecked(cfg.grayscale)
        f.addRow(self.cb_gray)
        f.addRow(_hint("EXIF 方向一律自動校正 (auto-orient)。"))
        return g

    def _aug_group(self, title, flags, params, aug: AugConfig):
        g = QGroupBox(title)
        grid = QGridLayout(g)
        r = 0
        for key, text in flags:
            cb = QCheckBox(text)
            cb.setChecked(bool(getattr(aug, key)))
            grid.addWidget(cb, r, 0, 1, 3)
            self.aug_rows[key] = (cb, None)
            r += 1
        for key, text, unit, mx, default, dec in params:
            cur = float(getattr(aug, key))
            cb = QCheckBox(text)
            cb.setChecked(cur > 0)
            sp = QDoubleSpinBox()
            sp.setDecimals(dec)
            sp.setRange(0.1 if dec else 1, mx)
            sp.setSingleStep(0.5 if dec else 1)
            sp.setValue(cur if cur > 0 else default)
            sp.setEnabled(cur > 0)
            cb.toggled.connect(sp.setEnabled)
            grid.addWidget(cb, r, 0)
            grid.addWidget(sp, r, 1)
            grid.addWidget(QLabel(unit), r, 2)
            self.aug_rows[key] = (cb, sp)
            r += 1
        return g

    def _output_group(self, cfg):
        g = QGroupBox("5. 輸出")
        f = QFormLayout(g)
        self.sp_mult = self._spin(1, 50, cfg.multiplier)
        f.addRow("每張 Train 圖輸出數量（含原圖）", self.sp_mult)
        self.sp_vis = self._spin(0, 100, int(cfg.min_visibility))
        f.addRow("框保留門檻（擴增後至少保留 % 面積）", self.sp_vis)
        self.fmt_boxes = {}
        row = QVBoxLayout()
        for key, text in FORMAT_LABELS.items():
            cb = QCheckBox(text)
            cb.setChecked(key in cfg.formats)
            self.fmt_boxes[key] = cb
            row.addWidget(cb)
        f.addRow("匯出格式", row)
        f.addRow(_hint("YOLO(bbox) 會把旋轉框轉成外接矩形；YOLO-OBB / DOTA 會把 bbox 視為 0° 旋轉框。"))
        return g

    @staticmethod
    def _spin(lo, hi, val):
        s = QSpinBox()
        s.setRange(lo, hi)
        s.setValue(int(val))
        return s

    def _connect_all(self, root: QWidget):
        for w in root.findChildren(QSpinBox) + root.findChildren(QDoubleSpinBox):
            w.valueChanged.connect(self._changed)
        for w in root.findChildren(QCheckBox):
            w.toggled.connect(self._changed)
        for w in root.findChildren(QComboBox):
            w.currentIndexChanged.connect(self._changed)

    # ------------------------------------------------------------------ config
    def config(self) -> VersionConfig:
        aug = AugConfig()
        for key, (cb, sp) in self.aug_rows.items():
            if sp is None:
                setattr(aug, key, cb.isChecked())
            else:
                setattr(aug, key, float(sp.value()) if cb.isChecked() else 0.0)
        return VersionConfig(
            train=self.sp_train.value(), valid=self.sp_valid.value(), test=self.sp_test.value(),
            seed=self.sp_seed.value(), skip_unannotated=self.cb_skip.isChecked(),
            resize_mode=self.cmb_resize.currentData(), resize_w=self.sp_rw.value(), resize_h=self.sp_rh.value(),
            grayscale=self.cb_gray.isChecked(), multiplier=self.sp_mult.value(),
            min_visibility=self.sp_vis.value(),
            formats=[k for k, cb in self.fmt_boxes.items() if cb.isChecked()], aug=aug)

    def _changed(self, *_):
        resize_on = self.cmb_resize.currentData() != "none"
        self.sp_rw.setEnabled(resize_on)
        self.sp_rh.setEnabled(resize_on)
        self._update_summary()
        self.render_preview()

    def _update_summary(self):
        cfg = self.config()
        names = [n for n in self.project.images if not cfg.skip_unannotated or self.project.box_count(n)]
        sp = split_items(names, cfg)
        mult = cfg.multiplier if cfg.aug.any_enabled() else 1
        self.summary.setText(
            f"來源 {len(names)} 張 → Train {len(sp['train'])}×{mult} = {len(sp['train']) * mult}，"
            f"Valid {len(sp['valid'])}，Test {len(sp['test'])}")

    # ------------------------------------------------------------------ preview
    def next_preview_image(self):
        names = [n for n in self.project.images if self.project.box_count(n)] or self.project.images
        if not names:
            return
        rng = np.random.default_rng()
        self.preview_name = names[int(rng.integers(len(names)))]
        self._preview_img = imageio.imread(self.project.image_path(self.preview_name))
        self.render_preview()

    def reroll(self):
        self.preview_seed += 1
        self.render_preview()

    def render_preview(self):
        if self.preview_name is None or getattr(self, "_preview_img", None) is None:
            return
        cfg = self.config()
        anns = self.project.get(self.preview_name).annotations
        colors = [c.color for c in self.project.classes]
        names = self.project.class_names()
        rng = np.random.default_rng(self.preview_seed)
        tiles = []
        for k in range(4):
            img, boxes = make_sample(self._preview_img, anns, cfg, rng, augmented=k > 0 and cfg.aug.any_enabled())
            tiles.append(draw_annotations(img, boxes, colors, names))
        tile = 480
        hexc = theme.current["canvas"]
        bg = [int(hexc[i:i + 2], 16) for i in (5, 3, 1)]  # BGR
        canvas = np.full((tile * 2 + 6, tile * 2 + 6, 3), bg, np.uint8)
        for i, t in enumerate(tiles):
            h, w = t.shape[:2]
            r = tile / max(h, w)
            t = cv2.resize(t, (max(1, int(w * r)), max(1, int(h * r))), interpolation=cv2.INTER_AREA)
            y, x = (i // 2) * (tile + 6), (i % 2) * (tile + 6)
            canvas[y:y + t.shape[0], x:x + t.shape[1]] = t
        pm = to_qpixmap(canvas)
        self._preview_pm = pm
        self._fit_preview()

    def _fit_preview(self):
        pm = getattr(self, "_preview_pm", None)
        if pm is not None:
            self.preview.setPixmap(pm.scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._fit_preview()

    # ------------------------------------------------------------------ generate
    def generate(self):
        cfg = self.config()
        if not cfg.formats:
            QMessageBox.warning(self, "產生版本", "請至少選擇一種匯出格式")
            return
        if cfg.train + cfg.valid + cfg.test <= 0:
            QMessageBox.warning(self, "產生版本", "切分比例總和不可為 0")
            return
        self.project.settings["version_config"] = cfg.to_dict()
        self.project.save_meta()
        out = next_version_dir(self.project)
        self.worker = Worker(self.project, cfg, out)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished_ok.connect(self._on_done)
        self.worker.failed.connect(self._on_failed)
        self.bar.setVisible(True)
        self.bar.setValue(0)
        self.b_gen.setEnabled(False)
        self.b_close.setText("取消")
        self.b_close.clicked.disconnect()
        self.b_close.clicked.connect(self.worker.cancel)
        self.worker.start()

    def _reset_buttons(self):
        self.b_gen.setEnabled(True)
        self.b_close.setText("關閉")
        self.b_close.clicked.disconnect()
        self.b_close.clicked.connect(self.close)

    def _on_progress(self, done, total, name):
        self.bar.setMaximum(max(total, 1))
        self.bar.setValue(done)
        self.bar.setFormat(f"{done}/{total}  {name}")

    def _on_done(self, stats, out):
        self._reset_buttons()
        self.bar.setVisible(False)
        if stats.get("cancelled"):
            QMessageBox.information(self, "產生版本", f"已取消（部分檔案留在 {out}）")
            return
        r = QMessageBox.question(
            self, "產生版本完成",
            f"Train {stats['train']} 張、Valid {stats['valid']} 張、Test {stats['test']} 張，共 {stats['boxes']} 個框。\n\n"
            f"輸出位置：\n{out}\n\n要開啟資料夾嗎？")
        if r == QMessageBox.Yes:
            os.startfile(out)

    def _on_failed(self, msg):
        self._reset_buttons()
        self.bar.setVisible(False)
        QMessageBox.critical(self, "產生版本失敗", msg)

    def closeEvent(self, e):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait()
        self.project.settings["version_config"] = self.config().to_dict()
        self.project.save_meta()
        super().closeEvent(e)
