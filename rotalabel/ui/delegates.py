"""Custom painted list rows: image list, class list, annotation list."""
from __future__ import annotations

from PySide6.QtCore import QRectF, QSize, Qt
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPen
from PySide6.QtWidgets import QStyle, QStyledItemDelegate

from . import theme

DATA = Qt.UserRole + 1


def _background(p: QPainter, option, rect: QRectF):
    pal = theme.current
    if option.state & QStyle.State_Selected:
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(pal["selection"]))
        p.drawRoundedRect(rect, 6, 6)
        bar = QRectF(rect.left(), rect.top() + 6, 3, rect.height() - 12)
        p.setBrush(QColor(pal["accent"]))
        p.drawRoundedRect(bar, 1.5, 1.5)
    elif option.state & QStyle.State_MouseOver:
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(pal["panel2"]))
        p.drawRoundedRect(rect, 6, 6)


def _pill(p: QPainter, right: float, cy: float, text: str, bg: QColor, fg: QColor, font: QFont) -> float:
    """Draw a rounded pill ending at x=right; returns its left x."""
    fm = QFontMetrics(font)
    w = fm.horizontalAdvance(text) + 12
    h = fm.height() + 2
    r = QRectF(right - w, cy - h / 2, w, h)
    p.setPen(Qt.NoPen)
    p.setBrush(bg)
    p.drawRoundedRect(r, h / 2, h / 2)
    p.setPen(fg)
    p.setFont(font)
    p.drawText(r, Qt.AlignCenter, text)
    return r.left()


def _small(font: QFont) -> QFont:
    f = QFont(font)
    f.setPointSizeF(max(7.5, font.pointSizeF() - 1.5))
    return f


class _Base(QStyledItemDelegate):
    height = 30

    def sizeHint(self, option, index):
        return QSize(option.rect.width(), self.height)


class ImageDelegate(_Base):
    """DATA = number of boxes on the image."""

    def paint(self, p: QPainter, option, index):
        pal = theme.current
        p.save()
        p.setRenderHint(QPainter.Antialiasing)
        rect = QRectF(option.rect).adjusted(2, 1, -2, -1)
        _background(p, option, rect)
        n = index.data(DATA) or 0
        cy = rect.center().y()
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(pal["ok"] if n else pal["border"]))
        p.drawEllipse(QRectF(rect.left() + 10, cy - 4, 8, 8))
        right = rect.right() - 8
        if n:
            bg = QColor(pal["accent"])
            bg.setAlpha(60)
            right = _pill(p, right, cy, str(n), bg, QColor(pal["text"]), _small(option.font)) - 6
        p.setPen(QColor(pal["text"] if n else pal["muted"]))
        p.setFont(option.font)
        tr = QRectF(rect.left() + 26, rect.top(), right - rect.left() - 26, rect.height())
        name = QFontMetrics(option.font).elidedText(index.data(Qt.DisplayRole) or "", Qt.ElideMiddle, int(tr.width()))
        p.drawText(tr, Qt.AlignVCenter | Qt.AlignLeft, name)
        p.restore()


class ClassDelegate(_Base):
    """DATA = (color, shortcut text)."""

    height = 32

    def paint(self, p: QPainter, option, index):
        pal = theme.current
        p.save()
        p.setRenderHint(QPainter.Antialiasing)
        rect = QRectF(option.rect).adjusted(2, 1, -2, -1)
        _background(p, option, rect)
        color, key = index.data(DATA) or ("#888888", "")
        cy = rect.center().y()
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(color))
        p.drawRoundedRect(QRectF(rect.left() + 10, cy - 7, 14, 14), 4, 4)
        right = rect.right() - 8
        if key:
            right = _pill(p, right, cy, key, QColor(pal["panel2"]), QColor(pal["muted"]), _small(option.font)) - 6
        p.setPen(QColor(pal["text"]))
        p.setFont(option.font)
        p.drawText(QRectF(rect.left() + 34, rect.top(), right - rect.left() - 34, rect.height()),
                   Qt.AlignVCenter | Qt.AlignLeft, index.data(Qt.DisplayRole) or "")
        p.restore()


class AnnotationDelegate(_Base):
    """DATA = dict(color, name, kind, angle, w, h, n)."""

    height = 34

    def paint(self, p: QPainter, option, index):
        pal = theme.current
        d = index.data(DATA)
        if not d:
            return super().paint(p, option, index)
        p.save()
        p.setRenderHint(QPainter.Antialiasing)
        rect = QRectF(option.rect).adjusted(2, 1, -2, -1)
        _background(p, option, rect)
        cy = rect.center().y()
        small = _small(option.font)
        p.setPen(QColor(pal["muted"]))
        p.setFont(small)
        p.drawText(QRectF(rect.left() + 6, rect.top(), 20, rect.height()), Qt.AlignCenter, str(d["n"]))
        col = QColor(d["color"])
        p.setPen(QPen(col, 2))
        p.setBrush(Qt.NoBrush)
        if d["kind"] == "obb":
            p.save()
            p.translate(rect.left() + 36, cy)
            p.rotate(-20)
            p.drawRect(QRectF(-6, -4.5, 12, 9))
            p.restore()
        else:
            p.drawRect(QRectF(rect.left() + 30, cy - 4.5, 12, 9))
        right = rect.right() - 8
        if d["kind"] == "obb":
            tag, bg = f"旋轉 {d['angle']:.1f}°", QColor(pal["accent"])
        else:
            tag, bg = "BBox", QColor(pal["muted"])
        bg.setAlpha(55)
        right = _pill(p, right, cy, tag, bg, QColor(pal["text"]), small) - 6
        p.setPen(QColor(pal["muted"]))
        p.setFont(small)
        size = f"{d['w']:.0f}×{d['h']:.0f}"
        sw = QFontMetrics(small).horizontalAdvance(size)
        p.drawText(QRectF(right - sw, rect.top(), sw, rect.height()), Qt.AlignVCenter, size)
        right -= sw + 8
        p.setPen(QColor(pal["text"]))
        p.setFont(option.font)
        p.drawText(QRectF(rect.left() + 52, rect.top(), right - rect.left() - 52, rect.height()),
                   Qt.AlignVCenter | Qt.AlignLeft, d["name"])
        p.restore()
