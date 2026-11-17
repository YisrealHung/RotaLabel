"""Roboflow-style "Generate version" dialog: split, preprocess, augment, preview, export."""
from __future__ import annotations

import os

import cv2
import numpy as np
from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFormLayout, QGridLayout,
                               QGroupBox, QHBoxLayout, QLabel, QMessageBox, QProgressBar, QPushButton,
                               QScrollArea, QSizePolicy, QSpinBox, QTabWidget, QVBoxLayout, QWidget)

from .. import imageio
from ..augment.mosaic import MosaicConfig, build_pools, generate_mosaic, make_mosaic, mosaic_splits
from ..augment.pipeline import AugConfig
from ..augment.version import (MAX_MULTIPLIER, TEST_TARGET, VersionConfig, draw_annotations,
                               generate_version, make_sample, next_version_dir, split_copies,
                               split_items)
from ..exporters import FORMAT_LABELS
from . import theme

# key, label, unit, max, default-when-enabled, decimals
GEOMETRIC = [
    ("rotate", "任意角度旋轉", "± 度", 180, 15, 0),
    ("shear", "剪切 Shear", "± 度", 45, 10, 0),
    ("scale", "縮放", "± %", 50, 20, 0),
    ("crop", "隨機裁切 (放大)", "最多 %", 80, 20, 0),
    ("translate_x", "左右平移", "± px", 4096, 50, 0),
    ("translate_y", "上下平移", "± px", 4096, 50, 0),
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
    """Runs ``job(progress=..., cancelled=...)`` (generate_version / generate_mosaic) off the UI thread."""
    progress = Signal(int, int, str)
    finished_ok = Signal(dict, str)
    failed = Signal(str)

    def __init__(self, job, out_dir):
        super().__init__()
        self.job, self.out_dir = job, out_dir
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            stats = self.job(progress=lambda a, b, n: self.progress.emit(a, b, n),
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
        self.mosaic_aug_rows: dict[str, tuple[QCheckBox, QDoubleSpinBox | None]] = {}
        self._pool_cache: tuple[int, list] | None = None   # (padding, patches) for the mosaic preview

        cfg = VersionConfig.from_dict(project.settings.get("version_config", {}))
        mcfg = MosaicConfig.from_dict(project.settings.get("mosaic_config", {}))

        form_widget = QWidget()
        form = QVBoxLayout(form_widget)
        form.setContentsMargins(0, 0, 10, 0)
        form.setSpacing(4)
        form.addWidget(self._split_group(cfg))
        form.addWidget(self._pre_group(cfg))
        form.addWidget(self._aug_group("3. 幾何擴增 · 框會跟著轉換", FLAGS, GEOMETRIC, cfg.aug,
                                       tail=self._translate_options))
        form.addWidget(self._aug_group("4. 色彩 / 像素擴增", [], PHOTOMETRIC, cfg.aug))
        form.addWidget(self._output_group(cfg))
        form.addStretch(1)

        mosaic_widget = QWidget()
        mform = QVBoxLayout(mosaic_widget)
        mform.setContentsMargins(0, 0, 10, 0)
        mform.setSpacing(4)
        mform.addWidget(self._mosaic_layout_group(mcfg))
        mform.addWidget(self._mosaic_geo_group(mcfg))
        mform.addWidget(self._aug_group("3. 色彩 / 像素擴增", [], PHOTOMETRIC, mcfg.aug,
                                        rows=self.mosaic_aug_rows, tail=self._mosaic_scope_option(mcfg)))
        mform.addWidget(_hint("切分比例、亂數種子、灰階與匯出格式沿用「一般擴增」頁的設定；Resize 不套用，"
                              "輸出大小就是背景範圍。"))
        mform.addStretch(1)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._scroll(form_widget), "一般擴增")
        self.tabs.addTab(self._scroll(mosaic_widget), "拼接擴增")
        self.tabs.setMinimumWidth(620)
        self.tabs.setCurrentIndex(1 if project.settings.get("version_tab") == "mosaic" else 0)

        # preview
        title = QLabel("擴增預覽")
        title.setObjectName("summary")
        hint = QLabel()
        hint.setObjectName("hint")
        self.preview_hint = hint
        self.preview = QLabel("預覽")
        self.preview.setObjectName("preview")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        b_img = QPushButton("換一張圖片")
        b_img.clicked.connect(self.next_preview_image)
        self.b_img = b_img
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
        top.addWidget(self.tabs, 0)
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
        self._connect_all(mosaic_widget)
        self.tabs.currentChanged.connect(self._changed)
        resize_on = self.cmb_resize.currentData() != "none"
        self.sp_rw.setEnabled(resize_on)
        self.sp_rh.setEnabled(resize_on)
        self._sync_mode()
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
        f.addRow(_hint(f"有啟用擴增時：Train / Valid 依倍數擴增；Test 固定輸出 {TEST_TARGET} 張"
                       f"（不足時以擴增補足，超過時只取前 {TEST_TARGET} 張）。"))
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

    def _aug_group(self, title, flags, params, aug, tail=None, rows=None):
        rows = self.aug_rows if rows is None else rows
        g = QGroupBox(title)
        grid = QGridLayout(g)
        r = 0
        for key, text in flags:
            cb = QCheckBox(text)
            cb.setChecked(bool(getattr(aug, key)))
            grid.addWidget(cb, r, 0, 1, 3)
            rows[key] = (cb, None)
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
            rows[key] = (cb, sp)
            r += 1
        if tail:
            tail(grid, r, aug)
        return g

    # ------------------------------------------------------------------ mosaic page
    def _mosaic_layout_group(self, m: MosaicConfig):
        g = QGroupBox("1. 背景範圍與張數")
        f = QFormLayout(g)
        self.sp_cw, self.sp_ch = self._spin(64, 8192, m.canvas_w), self._spin(64, 8192, m.canvas_h)
        row = QHBoxLayout()
        row.addWidget(self.sp_cw)
        row.addWidget(QLabel("×"))
        row.addWidget(self.sp_ch)
        row.addWidget(QLabel("px"))
        row.addStretch(1)
        f.addRow("背景範圍 寬 × 高", row)
        self.sp_cnt = {s: self._spin(0, 100000, v) for s, v in m.counts().items()}
        row = QHBoxLayout()
        for lab, s in (("Train", "train"), ("Valid", "valid"), ("Test", "test")):
            self.sp_cnt[s].setMaximumWidth(90)
            row.addWidget(QLabel(lab))
            row.addWidget(self.sp_cnt[s])
        row.addStretch(1)
        f.addRow("產生張數", row)
        self.sp_np, self.sp_nj = self._spin(1, 500, m.patches), self._spin(0, 500, m.patches_jitter)
        row = QHBoxLayout()
        row.addWidget(self.sp_np)
        row.addWidget(QLabel("±"))
        row.addWidget(self.sp_nj)
        self.lbl_range = QLabel()
        row.addWidget(self.lbl_range)
        row.addStretch(1)
        f.addRow("每張拼接張數", row)
        f.addRow(_hint("原圖不夠時會重複使用同一張，但每份複本的角度都不同（至少差 30°）。"))
        self.cb_overlap = QCheckBox("允許圖塊背景互相重疊（標註框不會被蓋住）")
        self.cb_overlap.setChecked(bool(m.allow_overlap))
        f.addRow(self.cb_overlap)
        self.sp_minfit = self._spin(1, 100, int(m.min_fit_scale))
        row = QHBoxLayout()
        row.addWidget(self.sp_minfit)
        row.addWidget(QLabel("%（放不下時最多縮到這麼小；100 = 不縮小）"))
        row.addStretch(1)
        f.addRow("最小縮放", row)
        self.sp_pad = self._spin(0, 1000, m.padding)
        f.addRow("裁切外擴 px", self.sp_pad)
        self.sp_gap = self._spin(0, 1000, m.gap)
        f.addRow("間距 px", self.sp_gap)
        f.addRow(_hint("間距：允許重疊時是標註框周圍保留的距離，不重疊時是圖塊之間的距離。"))
        f.addRow(_hint("每張原圖裁出「容納所有標註框的方框」當圖塊，隨機貼到背景上。找不到空位時先逐步縮小再試，"
                       "縮到最小仍放不下才少放。比背景大的圖塊會自動縮小。空白處填入所用圖塊邊緣的平均色。"))
        return g

    def _mosaic_geo_group(self, m: MosaicConfig):
        g = QGroupBox("2. 圖塊幾何變化 · 框會跟著轉換")
        grid = QGridLayout(g)
        self.cb_mflip_h, self.cb_mflip_v, self.cb_mrot90 = (QCheckBox(t) for t in
                                                            ("水平翻轉", "垂直翻轉", "90° 旋轉（隨機 0/90/180/270）"))
        for r, (cb, val) in enumerate(((self.cb_mflip_h, m.flip_h), (self.cb_mflip_v, m.flip_v),
                                       (self.cb_mrot90, m.rot90))):
            cb.setChecked(bool(val))
            grid.addWidget(cb, r, 0, 1, 3)
        self.m_geo = {}
        for r, (key, text, unit, mx, default) in enumerate((("rotate", "任意角度旋轉", "± 度", 180, 15),
                                                            ("scale", "縮放", "± %", 90, 20)), start=3):
            cur = float(getattr(m, key))
            cb = QCheckBox(text)
            cb.setChecked(cur > 0)
            sp = QDoubleSpinBox()
            sp.setDecimals(0)
            sp.setRange(1, mx)
            sp.setValue(cur if cur > 0 else default)
            sp.setEnabled(cur > 0)
            cb.toggled.connect(sp.setEnabled)
            grid.addWidget(cb, r, 0)
            grid.addWidget(sp, r, 1)
            grid.addWidget(QLabel(unit), r, 2)
            self.m_geo[key] = (cb, sp)
        grid.addWidget(_hint("任意角度旋轉後，圖塊四角露出的部分會用背景色。"), 5, 0, 1, 3)
        return g

    def _mosaic_scope_option(self, m: MosaicConfig):
        def tail(grid: QGridLayout, r: int, _aug):
            self.cmb_mscope = QComboBox()
            self.cmb_mscope.addItem("每個圖塊各自隨機", "patch")
            self.cmb_mscope.addItem("整張拼接圖一起", "canvas")
            self.cmb_mscope.setCurrentIndex(max(0, self.cmb_mscope.findData(m.photometric_scope)))
            grid.addWidget(QLabel("套用方式"), r, 0)
            grid.addWidget(self.cmb_mscope, r, 1, 1, 2)
        return tail

    @staticmethod
    def _scroll(w: QWidget) -> QScrollArea:
        s = QScrollArea()
        s.setWidgetResizable(True)
        s.setWidget(w)
        s.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        return s

    def is_mosaic(self) -> bool:
        return self.tabs.currentIndex() == 1

    def mosaic_config(self) -> MosaicConfig:
        aug = AugConfig()
        for key, (cb, sp) in self.mosaic_aug_rows.items():
            setattr(aug, key, float(sp.value()) if cb.isChecked() else 0.0)
        geo = {k: (float(sp.value()) if cb.isChecked() else 0.0) for k, (cb, sp) in self.m_geo.items()}
        return MosaicConfig(
            canvas_w=self.sp_cw.value(), canvas_h=self.sp_ch.value(),
            count_train=self.sp_cnt["train"].value(), count_valid=self.sp_cnt["valid"].value(),
            count_test=self.sp_cnt["test"].value(),
            patches=self.sp_np.value(), patches_jitter=self.sp_nj.value(),
            allow_overlap=self.cb_overlap.isChecked(), min_fit_scale=self.sp_minfit.value(),
            padding=self.sp_pad.value(), gap=self.sp_gap.value(),
            flip_h=self.cb_mflip_h.isChecked(), flip_v=self.cb_mflip_v.isChecked(), rot90=self.cb_mrot90.isChecked(),
            rotate=geo["rotate"], scale=geo["scale"],
            photometric_scope=self.cmb_mscope.currentData(), aug=aug)

    def _sync_mode(self):
        mosaic = self.is_mosaic()
        self.b_img.setVisible(not mosaic)
        self.preview_hint.setText("隨機拼接樣本（使用所有已標註圖片的圖塊）" if mosaic
                                  else "左上為前處理後原圖，其餘為隨機擴增樣本")

    def _preview_pool(self, padding: int) -> list:
        """Patches of (up to 60) annotated images, cached per padding."""
        if self._pool_cache and self._pool_cache[0] == padding:
            return self._pool_cache[1]
        names = [n for n in self.project.images if self.project.box_count(n)]
        if len(names) > 60:
            idx = np.random.default_rng(0).choice(len(names), 60, replace=False)
            names = [names[i] for i in sorted(idx)]
        pool = build_pools(self.project, {"train": names}, padding)["train"]
        self._pool_cache = (padding, pool)
        return pool

    def _translate_options(self, grid: QGridLayout, r: int, aug: AugConfig):
        cb = QCheckBox("允許框超出邊界（超出部分裁掉，依「框保留門檻」決定是否保留）")
        cb.setChecked(bool(aug.translate_allow_out))
        grid.addWidget(cb, r, 0, 1, 3)
        grid.addWidget(_hint("不勾選時會自動限制平移量，讓所有框完整留在圖內。平移以前處理後的圖片像素計算。"),
                       r + 1, 0, 1, 3)
        self.aug_rows["translate_allow_out"] = (cb, None)
        tx, ty = self.aug_rows["translate_x"][0], self.aug_rows["translate_y"][0]

        def sync(*_):
            cb.setEnabled(tx.isChecked() or ty.isChecked())
        tx.toggled.connect(sync)
        ty.toggled.connect(sync)
        sync()

    def _output_group(self, cfg):
        g = QGroupBox("5. 輸出")
        f = QFormLayout(g)
        self.sp_mult = self._spin(1, MAX_MULTIPLIER, min(cfg.multiplier, MAX_MULTIPLIER))
        f.addRow("每張 Train / Valid 圖輸出數量（含原圖）", self.sp_mult)
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
        f.addRow(_hint("YOLO(bbox) 會把旋轉框轉成外接矩形；YOLO-OBB / DOTA / COCO-OBB 會把 bbox 視為 0° 旋轉框。"))
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
        self._sync_mode()
        self._update_summary()
        self.render_preview()

    def _update_summary(self):
        cfg = self.config()
        m = self.mosaic_config()
        lo, hi = m.patch_range()
        self.lbl_range.setText(f"→ 每張 {lo} 張" if lo == hi else f"→ 每張 {lo} ~ {hi} 張隨機")
        if self.is_mosaic():
            sp = mosaic_splits(self.project, cfg)
            n ={s: (m.counts()[s] if sp[s] else 0) for s in ("train", "valid", "test")}
            self.summary.setText(
                f"拼接 {m.canvas_w}×{m.canvas_h} → Train {n['train']}、Valid {n['valid']}、Test {n['test']} 張"
                f"（圖塊來源：Train {len(sp['train'])}、Valid {len(sp['valid'])}、Test {len(sp['test'])} 張原圖）")
            return
        names = [n for n in self.project.images if not cfg.skip_unannotated or self.project.box_count(n)]
        sp = split_items(names, cfg)
        cp = split_copies(sp, cfg)
        mult = cfg.multiplier if cfg.aug.any_enabled() else 1
        self.summary.setText(
            f"來源 {len(names)} 張 → Train {len(sp['train'])}×{mult} = {sum(cp['train'])}，"
            f"Valid {len(sp['valid'])}×{mult} = {sum(cp['valid'])}，"
            f"Test {len(sp['test'])} → {sum(cp['test'])}")

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
        if self.is_mosaic():
            self._render_mosaic_preview()
            return
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

    def _render_mosaic_preview(self):
        m = self.mosaic_config()
        pool = self._preview_pool(m.padding)
        if not pool:
            self._preview_pm = None
            self.preview.setPixmap(QPixmap())
            self.preview.setText("沒有已標註的圖片可以拼接")
            return
        img, boxes = make_mosaic(pool, m, np.random.default_rng(self.preview_seed), grayscale=self.cb_gray.isChecked())
        img = draw_annotations(img, boxes, [c.color for c in self.project.classes], self.project.class_names())
        self._preview_pm = to_qpixmap(img)
        self._fit_preview()

    def _fit_preview(self):
        pm = getattr(self, "_preview_pm", None)
        if pm is not None and not pm.isNull() and self.preview.width() > 0 and self.preview.height() > 0:
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
        self._save_settings()
        out = next_version_dir(self.project)
        if self.is_mosaic():
            m = self.mosaic_config()
            sp = mosaic_splits(self.project, cfg)
            if not any(sp[s] and m.counts()[s] for s in ("train", "valid", "test")):
                QMessageBox.warning(self, "產生版本", "沒有可拼接的切分：請確認有已標註的圖片，且張數不全為 0")
                return

            def job(**kw):
                return generate_mosaic(self.project, cfg, m, out, **kw)
        else:
            def job(**kw):
                return generate_version(self.project, cfg, out, **kw)
        self.worker = Worker(job, out)
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
            try:
                os.startfile(out)
            except OSError:
                pass

    def _on_failed(self, msg):
        self._reset_buttons()
        self.bar.setVisible(False)
        QMessageBox.critical(self, "產生版本失敗", msg)

    def _stop_worker(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait()

    def reject(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            return
        super().reject()

    def closeEvent(self, e):
        self._stop_worker()
        self._save_settings()
        super().closeEvent(e)

    def _save_settings(self):
        self.project.settings["version_config"] = self.config().to_dict()
        self.project.settings["mosaic_config"] = self.mosaic_config().to_dict()
        self.project.settings["version_tab"] = "mosaic" if self.is_mosaic() else "normal"
        self.project.save_meta()
