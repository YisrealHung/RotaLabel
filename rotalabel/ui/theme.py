"""Dark / light themes (QSS) and self-drawn vector icons (no extra dependencies)."""
from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QApplication

PALETTES = {
    "dark": {
        "bg": "#1e1f24", "panel": "#25262c", "panel2": "#2d2f36", "input": "#1a1b20",
        "border": "#383a43", "text": "#e6e7eb", "muted": "#9095a3", "accent": "#7c5cff",
        "accent_hover": "#9277ff", "accent_text": "#ffffff", "canvas": "#16171b",
        "selection": "#3b3270", "ok": "#3ecf8e", "warn": "#f5a524",
    },
    "light": {
        "bg": "#f4f5f8", "panel": "#ffffff", "panel2": "#eef0f4", "input": "#ffffff",
        "border": "#dcdfe6", "text": "#1f2329", "muted": "#6b7280", "accent": "#6a4cf5",
        "accent_hover": "#5a3de0", "accent_text": "#ffffff", "canvas": "#e3e5ea",
        "selection": "#e6e0ff", "ok": "#16a34a", "warn": "#d97706",
    },
}

current: dict = dict(PALETTES["dark"])
current_name = "dark"


def qss(p: dict) -> str:
    return f"""
* {{ font-family: "Microsoft JhengHei UI", "Segoe UI", sans-serif; font-size: 10pt; }}
QMainWindow, QDialog {{ background: {p['bg']}; color: {p['text']}; }}
QWidget {{ color: {p['text']}; }}
QToolTip {{ background: {p['panel2']}; color: {p['text']}; border: 1px solid {p['border']}; padding: 4px 6px; }}

QMenuBar {{ background: {p['bg']}; padding: 2px 4px; }}
QMenuBar::item {{ padding: 4px 10px; border-radius: 6px; background: transparent; }}
QMenuBar::item:selected {{ background: {p['panel2']}; }}
QMenu {{ background: {p['panel']}; border: 1px solid {p['border']}; border-radius: 8px; padding: 6px; }}
QMenu::item {{ padding: 6px 28px 6px 14px; border-radius: 6px; }}
QMenu::item:selected {{ background: {p['selection']}; }}
QMenu::separator {{ height: 1px; background: {p['border']}; margin: 4px 8px; }}

QToolBar {{ background: {p['bg']}; border: none; border-bottom: 1px solid {p['border']}; padding: 4px 6px; spacing: 2px; }}
QToolBar::separator {{ width: 1px; background: {p['border']}; margin: 6px 6px; }}
QToolButton {{ background: transparent; border: 1px solid transparent; border-radius: 7px; padding: 5px 8px; }}
QToolButton:hover {{ background: {p['panel2']}; }}
QToolButton:pressed {{ background: {p['border']}; }}
QToolButton:checked {{ background: {p['selection']}; border: 1px solid {p['accent']}; }}
QToolButton:disabled {{ color: {p['muted']}; }}
QToolButton#primary {{ background: {p['accent']}; color: {p['accent_text']}; font-weight: bold; }}
QToolButton#primary:hover {{ background: {p['accent_hover']}; }}

QDockWidget {{ titlebar-close-icon: none; titlebar-normal-icon: none; }}
QDockWidget::title {{ background: {p['bg']}; padding: 8px 10px 4px 10px; text-align: left;
                      color: {p['muted']}; font-weight: bold; }}
QDockWidget > QWidget {{ background: {p['panel']}; }}
QMainWindow::separator {{ background: {p['bg']}; width: 4px; height: 4px; }}

QListWidget {{ background: {p['panel']}; border: none; outline: none; padding: 4px; }}
QListWidget::item {{ border-radius: 6px; padding: 2px; }}
QListWidget::item:hover {{ background: {p['panel2']}; }}
QListWidget::item:selected {{ background: {p['selection']}; color: {p['text']}; }}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background: {p['input']}; border: 1px solid {p['border']}; border-radius: 6px; padding: 5px 8px;
    selection-background-color: {p['accent']}; }}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{ border: 1px solid {p['accent']}; }}
QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {{ color: {p['muted']}; background: {p['panel2']}; }}
QSpinBox::up-button, QDoubleSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::down-button {{
    width: 16px; border: none; background: transparent; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{ background: {p['panel']}; border: 1px solid {p['border']};
    selection-background-color: {p['selection']}; outline: none; }}

QPushButton {{ background: {p['panel2']}; border: 1px solid {p['border']}; border-radius: 7px; padding: 6px 14px; }}
QPushButton:hover {{ border: 1px solid {p['accent']}; }}
QPushButton:pressed {{ background: {p['border']}; }}
QPushButton:disabled {{ color: {p['muted']}; }}
QPushButton#primary {{ background: {p['accent']}; color: {p['accent_text']}; border: none; font-weight: bold; padding: 7px 20px; }}
QPushButton#primary:hover {{ background: {p['accent_hover']}; }}
QPushButton#primary:disabled {{ background: {p['border']}; color: {p['muted']}; }}
QPushButton#flat {{ background: transparent; border: none; padding: 4px 8px; color: {p['muted']}; }}
QPushButton#flat:hover {{ color: {p['text']}; background: {p['panel2']}; }}

QCheckBox {{ spacing: 8px; }}
QCheckBox::indicator {{ width: 16px; height: 16px; border-radius: 4px; border: 1px solid {p['border']}; background: {p['input']}; }}
QCheckBox::indicator:hover {{ border: 1px solid {p['accent']}; }}
QCheckBox::indicator:checked {{ background: {p['accent']}; border: 1px solid {p['accent']};
    image: url(:/rotalabel/check.png); }}

QGroupBox {{ background: {p['panel']}; border: 1px solid {p['border']}; border-radius: 10px;
             margin-top: 22px; padding: 12px 10px 10px 10px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 4px; top: 0px; padding: 0 4px; color: {p['text']}; font-weight: bold; }}

QScrollArea {{ border: none; background: transparent; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle {{ background: {p['border']}; border-radius: 3px; min-height: 24px; min-width: 24px; }}
QScrollBar::handle:hover {{ background: {p['muted']}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
QAbstractScrollArea::corner {{ background: transparent; border: none; }}
QGraphicsView {{ border: none; }}

QProgressBar {{ background: {p['panel2']}; border: none; border-radius: 6px; height: 12px; text-align: center; color: {p['text']}; }}
QProgressBar::chunk {{ background: {p['accent']}; border-radius: 6px; }}

QStatusBar {{ background: {p['bg']}; border-top: 1px solid {p['border']}; color: {p['muted']}; }}
QStatusBar QLabel {{ color: {p['muted']}; padding: 0 8px; }}
QStatusBar::item {{ border: none; }}
QLabel#muted {{ color: {p['muted']}; }}
QLabel#hint {{ color: {p['muted']}; font-size: 9pt; }}
QLabel#preview {{ background: {p['canvas']}; border-radius: 10px; color: {p['muted']}; }}
QLabel#summary {{ color: {p['text']}; font-weight: bold; }}
QMessageBox {{ background: {p['panel']}; }}
"""


# ---------------------------------------------------------------------- icons
def _path(name: str) -> tuple[QPainterPath, bool]:
    """Return (path, filled) on a 24x24 grid."""
    p = QPainterPath()
    filled = False
    if name == "folder":
        p.moveTo(3, 7); p.lineTo(3, 18); p.quadTo(3, 19, 4, 19); p.lineTo(20, 19); p.quadTo(21, 19, 21, 18)
        p.lineTo(21, 9); p.quadTo(21, 8, 20, 8); p.lineTo(11, 8); p.lineTo(9, 5); p.lineTo(4, 5)
        p.quadTo(3, 5, 3, 6); p.closeSubpath()
    elif name == "save":
        p.addRoundedRect(QRectF(4, 4, 16, 16), 2, 2)
        p.addRect(QRectF(8, 4, 8, 5))
        p.addRoundedRect(QRectF(8, 13, 8, 7), 1, 1)
    elif name == "prev":
        p.moveTo(15, 5); p.lineTo(8, 12); p.lineTo(15, 19)
    elif name == "next":
        p.moveTo(9, 5); p.lineTo(16, 12); p.lineTo(9, 19)
    elif name == "select":
        p.moveTo(6, 3); p.lineTo(6, 19); p.lineTo(10, 15); p.lineTo(13, 21); p.lineTo(15.5, 20)
        p.lineTo(12.5, 14); p.lineTo(18, 14); p.closeSubpath()
    elif name == "bbox":
        p.addRect(QRectF(5, 6, 14, 12))
        for x, y in ((5, 6), (19, 6), (5, 18), (19, 18)):
            p.addRect(QRectF(x - 1.5, y - 1.5, 3, 3))
    elif name == "obb":
        c, a = QPointF(12, 12), math.radians(-25)
        pts = []
        for dx, dy in ((-7, -5), (7, -5), (7, 5), (-7, 5)):
            pts.append(QPointF(c.x() + dx * math.cos(a) - dy * math.sin(a), c.y() + dx * math.sin(a) + dy * math.cos(a)))
        p.moveTo(pts[0])
        for q in pts[1:]:
            p.lineTo(q)
        p.closeSubpath()
        p.moveTo(17, 3); p.arcTo(QRectF(13, 1, 9, 9), 120, -150)
    elif name == "undo":
        p.moveTo(8, 6); p.lineTo(4, 10); p.lineTo(8, 14)
        p.moveTo(4, 10); p.lineTo(14, 10); p.cubicTo(22, 10, 22, 20, 14, 20); p.lineTo(10, 20)
    elif name == "redo":
        p.moveTo(16, 6); p.lineTo(20, 10); p.lineTo(16, 14)
        p.moveTo(20, 10); p.lineTo(10, 10); p.cubicTo(2, 10, 2, 20, 10, 20); p.lineTo(14, 20)
    elif name == "fit":
        for (x, y, dx, dy) in ((4, 4, 1, 1), (20, 4, -1, 1), (4, 20, 1, -1), (20, 20, -1, -1)):
            p.moveTo(x, y + 5 * dy); p.lineTo(x, y); p.lineTo(x + 5 * dx, y)
        p.addRect(QRectF(9, 9, 6, 6))
    elif name == "layers":
        p.moveTo(12, 3); p.lineTo(21, 8); p.lineTo(12, 13); p.lineTo(3, 8); p.closeSubpath()
        p.moveTo(3, 12); p.lineTo(12, 17); p.lineTo(21, 12)
        p.moveTo(3, 16); p.lineTo(12, 21); p.lineTo(21, 16)
    elif name == "import":
        p.moveTo(12, 3); p.lineTo(12, 15); p.moveTo(7, 10); p.lineTo(12, 15); p.lineTo(17, 10)
        p.moveTo(4, 15); p.lineTo(4, 20); p.lineTo(20, 20); p.lineTo(20, 15)
    elif name == "help":
        p.addEllipse(QRectF(3, 3, 18, 18))
        p.moveTo(9.5, 9.5); p.cubicTo(9.5, 6, 14.5, 6, 14.5, 9.5); p.cubicTo(14.5, 12, 12, 11.5, 12, 14.5)
        p.addEllipse(QRectF(11.2, 16.6, 1.6, 1.6))
    elif name == "theme":
        p.addEllipse(QRectF(4, 4, 16, 16))
        p.moveTo(12, 4); p.arcTo(QRectF(4, 4, 16, 16), 90, -180); p.closeSubpath()
    elif name == "tag":
        p.moveTo(3, 5); p.lineTo(3, 12); p.lineTo(12, 21); p.lineTo(21, 12); p.lineTo(12, 3); p.lineTo(5, 3)
        p.quadTo(3, 3, 3, 5)
        p.addEllipse(QRectF(6.5, 6.5, 3, 3))
    elif name == "check":
        p.moveTo(5, 12.5); p.lineTo(10, 17); p.lineTo(19, 7)
    elif name == "chev_up":
        p.moveTo(6, 15); p.lineTo(12, 9); p.lineTo(18, 15)
    elif name == "chev_down":
        p.moveTo(6, 9); p.lineTo(12, 15); p.lineTo(18, 9)
    return p, filled


def make_icon(name: str, color: str | None = None, size: int = 48) -> QIcon:
    col = QColor(color or current["text"])
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    painter = QPainter(pm)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.scale(size / 24, size / 24)
    path, filled = _path(name)
    pen = QPen(col, 1.8, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(col if filled else Qt.NoBrush)
    painter.drawPath(path)
    painter.end()
    icon = QIcon(pm)
    # disabled state: muted color
    dis = QPixmap(size, size)
    dis.fill(Qt.transparent)
    painter = QPainter(dis)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.scale(size / 24, size / 24)
    painter.setPen(QPen(QColor(current["muted"]), 1.8, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
    painter.drawPath(path)
    painter.end()
    icon.addPixmap(dis, QIcon.Disabled)
    return icon


def _image_file(name: str, color: str) -> str:
    """QSS can only reference image files; render the icon into the temp dir and return its URL path."""
    import os
    import tempfile
    path = os.path.join(tempfile.gettempdir(), f"rotalabel_{name}_{color.lstrip('#')}.png")
    if not os.path.exists(path):
        make_icon(name, color, 32).pixmap(32, 32).save(path)
    return path.replace("\\", "/")


def apply_theme(app: QApplication, name: str = "dark") -> None:
    global current_name
    current_name = name if name in PALETTES else "dark"
    current.clear()
    current.update(PALETTES[current_name])
    app.setStyle("Fusion")
    p = current
    up, down = _image_file("chev_up", p["muted"]), _image_file("chev_down", p["muted"])
    sheet = qss(p).replace(":/rotalabel/check.png", _image_file("check", "#ffffff"))
    sheet += f"""
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{ image: url({up}); width: 10px; height: 10px; }}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{ image: url({down}); width: 10px; height: 10px; }}
QSpinBox::up-arrow:disabled, QDoubleSpinBox::up-arrow:disabled,
QSpinBox::down-arrow:disabled, QDoubleSpinBox::down-arrow:disabled {{ image: none; }}
QComboBox::down-arrow {{ image: url({down}); width: 12px; height: 12px; }}
"""
    app.setStyleSheet(sheet)


def app_font() -> QFont:
    f = QFont("Microsoft JhengHei UI")
    f.setPointSizeF(10)
    return f
