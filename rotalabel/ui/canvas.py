"""Annotation canvas: zoom/pan, drawing bbox / rotated boxes, move, resize, rotate, undo."""
from __future__ import annotations

import copy
import math

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (QBrush, QColor, QFont, QFontMetricsF, QImageReader, QPainter, QPen,
                           QPixmap, QUndoCommand, QUndoStack)
from PySide6.QtWidgets import QGraphicsItem, QGraphicsPixmapItem, QGraphicsScene, QGraphicsView

from ..model import BBOX, OBB, Annotation

HANDLE_PX = 9          # handle size in screen pixels
ROT_OFFSET_PX = 28     # distance of the rotation handle above the top edge
MIN_BOX = 2.0

# handle id -> local sign (sx, sy)
HANDLES = {
    "tl": (-1, -1), "t": (0, -1), "tr": (1, -1), "r": (1, 0),
    "br": (1, 1), "b": (0, 1), "bl": (-1, 1), "l": (-1, 0),
}
CURSORS = {
    "tl": Qt.SizeFDiagCursor, "br": Qt.SizeFDiagCursor, "tr": Qt.SizeBDiagCursor, "bl": Qt.SizeBDiagCursor,
    "t": Qt.SizeVerCursor, "b": Qt.SizeVerCursor, "l": Qt.SizeHorCursor, "r": Qt.SizeHorCursor,
}


class BoxItem(QGraphicsItem):
    def __init__(self, ann: Annotation, canvas: "Canvas"):
        super().__init__()
        self.ann = ann
        self.canvas = canvas
        self.setZValue(1)
        self.setAcceptHoverEvents(False)
        self.sync()

    def sync(self):
        self.prepareGeometryChange()
        self.setPos(self.ann.cx, self.ann.cy)
        self.setRotation(self.ann.angle)
        self.update()

    def px(self) -> float:
        """Scene units per screen pixel."""
        return 1.0 / max(self.canvas.transform().m11(), 1e-6)

    def boundingRect(self) -> QRectF:
        m = (HANDLE_PX + ROT_OFFSET_PX + 24) * self.px()
        w, h = self.ann.w, self.ann.h
        return QRectF(-w / 2 - m, -h / 2 - m, w + 2 * m, h + 2 * m)

    def rect(self) -> QRectF:
        return QRectF(-self.ann.w / 2, -self.ann.h / 2, self.ann.w, self.ann.h)

    def rot_handle_pos(self) -> QPointF:
        return QPointF(0, -self.ann.h / 2 - ROT_OFFSET_PX * self.px())

    def paint(self, p: QPainter, option, widget=None):
        color = self.canvas.class_color(self.ann.cls)
        selected = self.canvas.selected_item is self
        hovered = self.canvas.hover_item is self
        s = self.px()
        r = self.rect()

        pen = QPen(color, 3 if selected else 2)
        pen.setCosmetic(True)
        if self.ann.kind == OBB and not selected:
            pen.setStyle(Qt.SolidLine)
        p.setPen(pen)
        fill = QColor(color)
        fill.setAlpha(70 if selected else (45 if hovered else 18))
        p.setBrush(QBrush(fill))
        p.drawRect(r)

        if self.ann.kind == OBB:
            # small tick on the top edge so the orientation is visible
            tick = QPen(color, 2)
            tick.setCosmetic(True)
            p.setPen(tick)
            p.drawLine(QPointF(0, r.top()), QPointF(0, r.top() + min(r.height() / 2, 10 * s)))

        # label (constant screen size)
        if self.canvas.show_labels:
            name = self.canvas.class_name(self.ann.cls)
            if self.ann.kind == OBB:
                name += f"  {self.ann.angle:.1f}°"
            p.save()
            p.translate(r.topLeft())
            p.scale(s, s)
            font = QFont()
            font.setPointSizeF(9)
            p.setFont(font)
            fm = QFontMetricsF(font)
            tw, th = fm.horizontalAdvance(name) + 8, fm.height() + 2
            bg = QColor(color)
            bg.setAlpha(210)
            p.setPen(Qt.NoPen)
            p.setBrush(bg)
            p.drawRect(QRectF(0, -th, tw, th))
            p.setPen(Qt.black if bg.lightness() > 140 else Qt.white)
            p.drawText(QRectF(4, -th, tw, th), Qt.AlignVCenter, name)
            p.restore()

        if selected:
            hs = HANDLE_PX * s
            hp = QPen(Qt.white, 1)
            hp.setCosmetic(True)
            if self.ann.kind == OBB:
                lp = QPen(color, 1)
                lp.setCosmetic(True)
                p.setPen(lp)
                rh = self.rot_handle_pos()
                p.drawLine(QPointF(0, r.top()), rh)
                p.setPen(hp)
                p.setBrush(QColor("#ffcc00"))
                p.drawEllipse(rh, hs * 0.7, hs * 0.7)
            p.setPen(hp)
            p.setBrush(color)
            for sx, sy in HANDLES.values():
                c = QPointF(sx * r.width() / 2, sy * r.height() / 2)
                p.drawRect(QRectF(c.x() - hs / 2, c.y() - hs / 2, hs, hs))


class SnapshotCommand(QUndoCommand):
    """Undo by restoring the whole annotation list of the current image."""

    def __init__(self, canvas: "Canvas", before: list, after: list, text: str):
        super().__init__(text)
        self.canvas, self.before, self.after = canvas, before, after
        self.first = True

    def redo(self):
        if self.first:
            self.first = False
            return
        self.canvas._restore(self.after)

    def undo(self):
        self.canvas._restore(self.before)


class Canvas(QGraphicsView):
    changed = Signal()                # annotations modified
    selection_changed = Signal(int)   # index or -1
    cursor_moved = Signal(float, float)
    zoom_changed = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setBackgroundBrush(QColor("#16171b"))
        self.setFrameShape(QGraphicsView.NoFrame)

        self.classes = []                 # list[ClassDef] (shared with project)
        self.current_class = 0
        self.mode = BBOX                  # "select" | "bbox" | "obb"
        self.show_labels = True
        self.undo_stack = QUndoStack(self)

        self.pixmap_item: QGraphicsPixmapItem | None = None
        self.img_w = self.img_h = 0
        self.items: list[BoxItem] = []
        self.selected_item: BoxItem | None = None
        self.hover_item: BoxItem | None = None
        self.clipboard: list[Annotation] = []

        self._drag = None
        self._space = False
        self._cursor_scene: QPointF | None = None

    # ---------------------------------------------------------------- helpers
    def class_color(self, idx: int) -> QColor:
        if 0 <= idx < len(self.classes):
            return QColor(self.classes[idx].color)
        return QColor("#ff00ff")

    def class_name(self, idx: int) -> str:
        return self.classes[idx].name if 0 <= idx < len(self.classes) else f"#{idx}"

    def annotations(self) -> list[Annotation]:
        return [it.ann for it in self.items]

    def _snapshot(self) -> list[Annotation]:
        return copy.deepcopy(self.annotations())

    def _commit(self, before: list[Annotation], text: str):
        self.undo_stack.push(SnapshotCommand(self, before, self._snapshot(), text))
        self.changed.emit()

    def _restore(self, anns: list[Annotation]):
        self._set_items(copy.deepcopy(anns))
        self.select(None)
        self.changed.emit()

    def _set_items(self, anns: list[Annotation]):
        for it in self.items:
            self.scene().removeItem(it)
        self.items = []
        self.selected_item = self.hover_item = None
        for a in anns:
            self._add_item(a)

    def _add_item(self, ann: Annotation) -> BoxItem:
        it = BoxItem(ann, self)
        self.scene().addItem(it)
        self.items.append(it)
        return it

    def selected_index(self) -> int:
        return self.items.index(self.selected_item) if self.selected_item in self.items else -1

    def select(self, item: BoxItem | None):
        if item is self.selected_item:
            return
        old = self.selected_item
        self.selected_item = item
        for it in (old, item):
            if it is not None:
                it.update()
        self.selection_changed.emit(self.selected_index())

    def select_index(self, idx: int):
        self.select(self.items[idx] if 0 <= idx < len(self.items) else None)

    # ---------------------------------------------------------------- image
    def load_image(self, path: str, anns: list[Annotation]) -> tuple[int, int]:
        reader = QImageReader(path)
        reader.setAutoTransform(True)
        img = reader.read()
        self.scene().clear()
        self.items = []
        self.selected_item = self.hover_item = None
        self.undo_stack.clear()
        if img.isNull():
            self.pixmap_item = None
            self.img_w = self.img_h = 0
            return 0, 0
        self.pixmap_item = self.scene().addPixmap(QPixmap.fromImage(img))
        self.pixmap_item.setZValue(0)
        self.pixmap_item.setTransformationMode(Qt.SmoothTransformation)
        self.img_w, self.img_h = img.width(), img.height()
        margin = max(self.img_w, self.img_h)
        self.setSceneRect(QRectF(-margin, -margin, self.img_w + 2 * margin, self.img_h + 2 * margin))
        self._set_items(copy.deepcopy(anns))
        self.fit_view()
        self.selection_changed.emit(-1)
        return self.img_w, self.img_h

    def fit_view(self):
        """Fit the image to the view; keeps re-fitting on window resize until the user zooms/pans."""
        if not self.pixmap_item:
            return
        self._auto_fit = True
        self.fitInView(QRectF(0, 0, self.img_w, self.img_h), Qt.KeepAspectRatio)
        self._on_zoom()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if getattr(self, "_auto_fit", False) and self.pixmap_item:
            self.fitInView(QRectF(0, 0, self.img_w, self.img_h), Qt.KeepAspectRatio)
            self._on_zoom()

    def showEvent(self, e):
        super().showEvent(e)
        if getattr(self, "_auto_fit", False):
            self.fit_view()

    def _on_zoom(self):
        for it in self.items:
            it.sync()
        self.zoom_changed.emit(self.transform().m11())

    def set_mode(self, mode: str):
        self.mode = mode
        self._update_cursor(None)
        self.viewport().update()

    def refresh(self):
        for it in self.items:
            it.update()

    # ---------------------------------------------------------------- edits
    def delete_selected(self):
        it = self.selected_item
        if not it:
            return
        before = self._snapshot()
        idx = self.items.index(it)
        self.scene().removeItem(it)
        self.items.remove(it)
        self.selected_item = None
        self.hover_item = None
        self._commit(before, "刪除")
        if self.items:
            self.select(self.items[min(idx, len(self.items) - 1)])
        else:
            self.selection_changed.emit(-1)

    def rotate_selected(self, delta: float):
        it = self.selected_item
        if not it:
            return
        before = self._snapshot()
        if it.ann.kind == BBOX:
            it.ann.kind = OBB
        it.ann.angle = _norm180(it.ann.angle + delta)
        it.sync()
        self._commit(before, "旋轉")

    def set_selected_class(self, cls: int):
        it = self.selected_item
        if not it or it.ann.cls == cls:
            return
        before = self._snapshot()
        it.ann.cls = cls
        it.update()
        self._commit(before, "變更類別")

    def toggle_selected_kind(self):
        it = self.selected_item
        if not it:
            return
        before = self._snapshot()
        new = it.ann.to_bbox() if it.ann.kind == OBB else it.ann.to_obb()
        it.ann.__dict__.update(new.__dict__)
        it.sync()
        self._commit(before, "切換框類型")

    def nudge_selected(self, dx: float, dy: float):
        it = self.selected_item
        if not it:
            return
        before = self._snapshot()
        it.ann.cx += dx
        it.ann.cy += dy
        it.sync()
        self._commit(before, "移動")

    def copy_selected(self):
        if self.selected_item:
            self.clipboard = [copy.deepcopy(self.selected_item.ann)]

    def paste(self):
        if not self.clipboard or not self.pixmap_item:
            return
        before = self._snapshot()
        it = None
        for a in self.clipboard:
            a = copy.deepcopy(a)
            if not 0 <= a.cls < len(self.classes):
                a.cls = self.current_class
            it = self._add_item(a)
        self._commit(before, "貼上")
        self.select(it)

    # ---------------------------------------------------------------- hit testing
    def _handle_at(self, it: BoxItem, sp: QPointF) -> str | None:
        lp = it.mapFromScene(sp)
        tol = HANDLE_PX * it.px()
        if it.ann.kind == OBB:
            rh = it.rot_handle_pos()
            if math.hypot(lp.x() - rh.x(), lp.y() - rh.y()) <= tol:
                return "rot"
        w2, h2 = it.ann.w / 2, it.ann.h / 2
        for name, (sx, sy) in HANDLES.items():
            if abs(lp.x() - sx * w2) <= tol * 0.75 and abs(lp.y() - sy * h2) <= tol * 0.75:
                return name
        return None

    def _item_at(self, sp: QPointF) -> BoxItem | None:
        """Smallest box containing the point (makes nested boxes selectable)."""
        best, best_area = None, None
        for it in self.items:
            lp = it.mapFromScene(sp)
            if abs(lp.x()) <= it.ann.w / 2 and abs(lp.y()) <= it.ann.h / 2:
                area = it.ann.w * it.ann.h
                if best is None or area < best_area:
                    best, best_area = it, area
        return best

    def _clamp(self, p: QPointF) -> QPointF:
        return QPointF(min(max(p.x(), 0), self.img_w), min(max(p.y(), 0), self.img_h))

    def _update_cursor(self, sp: QPointF | None):
        if self._space or (self._drag and self._drag["type"] == "pan"):
            self.viewport().setCursor(Qt.ClosedHandCursor if self._drag else Qt.OpenHandCursor)
            return
        if sp is not None and self.selected_item:
            h = self._handle_at(self.selected_item, sp)
            if h == "rot":
                self.viewport().setCursor(Qt.PointingHandCursor)
                return
            if h:
                # pick the cursor matching the handle's on-screen direction
                self.viewport().setCursor(_rotated_cursor(h, self.selected_item.ann.angle))
                return
        if self.mode == "select":
            over = sp is not None and self._item_at(sp) is not None
            self.viewport().setCursor(Qt.SizeAllCursor if over else Qt.ArrowCursor)
        else:
            self.viewport().setCursor(Qt.CrossCursor)

    # ---------------------------------------------------------------- mouse
    def mousePressEvent(self, e):
        sp = self.mapToScene(e.position().toPoint())
        if e.button() == Qt.MiddleButton or (e.button() == Qt.LeftButton and self._space):
            self._auto_fit = False
            self._drag = {"type": "pan", "start": e.position(), "h": self.horizontalScrollBar().value(),
                          "v": self.verticalScrollBar().value()}
            self._update_cursor(sp)
            return
        if not self.pixmap_item:
            return
        if e.button() == Qt.RightButton:
            self.select(self._item_at(sp))
            return
        if e.button() != Qt.LeftButton:
            return

        sel = self.selected_item
        if sel:
            h = self._handle_at(sel, sp)
            if h:
                self._drag = {"type": "rotate" if h == "rot" else "resize", "handle": h, "item": sel,
                              "orig": copy.deepcopy(sel.ann), "before": self._snapshot()}
                return

        if self.mode == "select" or e.modifiers() & Qt.ControlModifier:
            it = self._item_at(sp)
            self.select(it)
            if it:
                self._drag = {"type": "move", "item": it, "start": sp, "orig": copy.deepcopy(it.ann),
                              "before": self._snapshot()}
            return

        # draw (bbox and obb are drawn the same way)
        p = self._clamp(sp)
        self.select(None)
        self._drag = {"type": "draw", "start": p, "end": p}

    def mouseMoveEvent(self, e):
        sp = self.mapToScene(e.position().toPoint())
        self._cursor_scene = sp
        self.cursor_moved.emit(sp.x(), sp.y())
        d = self._drag
        if d is None:
            if self.mode == "select":
                hov = self._item_at(sp) if self.pixmap_item else None
                if hov is not self.hover_item:
                    old, self.hover_item = self.hover_item, hov
                    for it in (old, hov):
                        if it:
                            it.update()
            self._update_cursor(sp)
            if self.mode != "select":
                self.viewport().update()
            return

        t = d["type"]
        if t == "pan":
            delta = e.position() - d["start"]
            self.horizontalScrollBar().setValue(int(d["h"] - delta.x()))
            self.verticalScrollBar().setValue(int(d["v"] - delta.y()))
        elif t == "draw":
            d["end"] = self._clamp(sp)
            self.viewport().update()
        elif t == "move":
            o, a = d["orig"], d["item"].ann
            dx, dy = sp.x() - d["start"].x(), sp.y() - d["start"].y()
            a.cx, a.cy = o.cx + dx, o.cy + dy
            if a.kind == BBOX:  # keep plain boxes inside the image
                a.cx = min(max(a.cx, a.w / 2), self.img_w - a.w / 2) if a.w <= self.img_w else a.cx
                a.cy = min(max(a.cy, a.h / 2), self.img_h - a.h / 2) if a.h <= self.img_h else a.cy
            d["item"].sync()
        elif t == "resize":
            self._resize(d, sp, e.modifiers())
        elif t == "rotate":
            a = d["item"].ann
            ang = math.degrees(math.atan2(sp.y() - a.cy, sp.x() - a.cx)) + 90
            if e.modifiers() & Qt.ShiftModifier:
                ang = round(ang / 15) * 15
            a.angle = _norm180(ang)
            d["item"].sync()
        self.selection_changed.emit(self.selected_index()) if t in ("move", "resize", "rotate") else None

    def _resize(self, d, sp: QPointF, mods):
        """Resize in the box's own (rotated) frame; the opposite handle stays fixed."""
        o, a = d["orig"], d["item"].ann
        sx, sy = HANDLES[d["handle"]]
        t = math.radians(o.angle)
        c, s = math.cos(t), math.sin(t)
        vx, vy = sp.x() - o.cx, sp.y() - o.cy
        qx, qy = vx * c + vy * s, -vx * s + vy * c          # point in original local frame
        ax, ay = -sx * o.w / 2, -sy * o.h / 2               # fixed anchor (local)
        if o.kind == BBOX:
            # keep the dragged edge inside the image
            qx = min(max(qx, -o.cx), self.img_w - o.cx)
            qy = min(max(qy, -o.cy), self.img_h - o.cy)
        if sx:
            w, lcx = max(abs(qx - ax), MIN_BOX), (qx + ax) / 2
        else:
            w, lcx = o.w, 0.0
        if sy:
            h, lcy = max(abs(qy - ay), MIN_BOX), (qy + ay) / 2
        else:
            h, lcy = o.h, 0.0
        a.w, a.h = w, h
        a.cx = o.cx + lcx * c - lcy * s
        a.cy = o.cy + lcx * s + lcy * c
        d["item"].sync()

    def mouseReleaseEvent(self, e):
        d = self._drag
        self._drag = None
        if d is None:
            return
        t = d["type"]
        if t == "draw":
            p0, p1 = d["start"], d["end"]
            w, h = abs(p1.x() - p0.x()), abs(p1.y() - p0.y())
            if w >= MIN_BOX and h >= MIN_BOX:
                before = self._snapshot()
                ann = Annotation(self.current_class, (p0.x() + p1.x()) / 2, (p0.y() + p1.y()) / 2, w, h, 0.0,
                                 OBB if self.mode == OBB else BBOX)
                it = self._add_item(ann)
                self._commit(before, "新增框")
                self.select(it)
            self.viewport().update()
        elif t in ("move", "resize", "rotate"):
            if d["item"].ann != d["orig"]:
                self._commit(d["before"], {"move": "移動", "resize": "縮放", "rotate": "旋轉"}[t])
        self._update_cursor(self.mapToScene(e.position().toPoint()))

    def mouseDoubleClickEvent(self, e):
        # double click on a box in draw mode selects it (quick edit without switching tools)
        sp = self.mapToScene(e.position().toPoint())
        it = self._item_at(sp)
        if it:
            self.select(it)
        else:
            super().mouseDoubleClickEvent(e)

    def wheelEvent(self, e):
        mods = e.modifiers()
        if not mods & Qt.ControlModifier:
            # wheel: scroll vertically; shift + wheel: scroll horizontally (touchpads also send x)
            self._auto_fit = False
            dx, dy = e.angleDelta().x(), e.angleDelta().y()
            if mods & Qt.ShiftModifier:
                dx, dy = dx or dy, 0
            hb, vb = self.horizontalScrollBar(), self.verticalScrollBar()
            hb.setValue(hb.value() - dx)
            vb.setValue(vb.value() - dy)
            return
        # ctrl + wheel: zoom
        factor = 1.0015 ** e.angleDelta().y()
        cur = self.transform().m11()
        factor = min(max(factor, 0.02 / cur), 60 / cur)
        self._auto_fit = False
        # keep the scene point under the cursor fixed (Qt's AnchorUnderMouse relies on the
        # base mouseMoveEvent, which we override, so anchor manually)
        vp = e.position()
        before = self.mapToScene(vp.toPoint())
        anchor = self.transformationAnchor()
        self.setTransformationAnchor(QGraphicsView.NoAnchor)
        self.scale(factor, factor)
        self.setTransformationAnchor(anchor)
        delta = self.mapFromScene(before) - vp.toPoint()
        hb, vb = self.horizontalScrollBar(), self.verticalScrollBar()
        hb.setValue(hb.value() + delta.x())
        vb.setValue(vb.value() + delta.y())
        self._on_zoom()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Space and not e.isAutoRepeat():
            self._space = True
            self._update_cursor(None)
            return
        step = 10 if e.modifiers() & Qt.ShiftModifier else 1
        moves = {Qt.Key_Left: (-step, 0), Qt.Key_Right: (step, 0), Qt.Key_Up: (0, -step), Qt.Key_Down: (0, step)}
        if e.key() in moves and self.selected_item:
            self.nudge_selected(*moves[e.key()])
            return
        if e.key() == Qt.Key_Escape:
            if self._drag and self._drag["type"] == "draw":
                self._drag = None
                self.viewport().update()
            else:
                self.select(None)
            return
        super().keyPressEvent(e)

    def keyReleaseEvent(self, e):
        if e.key() == Qt.Key_Space and not e.isAutoRepeat():
            self._space = False
            self._update_cursor(self._cursor_scene)
            return
        super().keyReleaseEvent(e)

    def leaveEvent(self, e):
        self._cursor_scene = None
        self.viewport().update()
        super().leaveEvent(e)

    # ---------------------------------------------------------------- overlay
    def drawForeground(self, p: QPainter, rect: QRectF):
        if not self.pixmap_item:
            _draw_empty_state(self, p)
            return
        # image border, so dark images stay distinguishable from the background
        border = QPen(QColor(150, 150, 150), 1)
        border.setCosmetic(True)
        p.setPen(border)
        p.setBrush(Qt.NoBrush)
        p.drawRect(QRectF(0, 0, self.img_w, self.img_h))
        d = self._drag
        if d and d["type"] == "draw":
            p0, p1 = d["start"], d["end"]
            pen = QPen(self.class_color(self.current_class), 2, Qt.DashLine)
            pen.setCosmetic(True)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            p.drawRect(QRectF(p0, p1).normalized())
        if self.mode != "select" and self._cursor_scene is not None and not (d and d["type"] == "pan"):
            c = self._cursor_scene
            if 0 <= c.x() <= self.img_w and 0 <= c.y() <= self.img_h:
                pen = QPen(QColor(255, 255, 255, 160), 1, Qt.DashLine)
                pen.setCosmetic(True)
                p.setPen(pen)
                p.drawLine(QPointF(0, c.y()), QPointF(self.img_w, c.y()))
                p.drawLine(QPointF(c.x(), 0), QPointF(c.x(), self.img_h))


def _draw_empty_state(view: QGraphicsView, p: QPainter):
    from . import theme
    pal = theme.current
    p.save()
    p.resetTransform()
    r = QRectF(view.viewport().rect())
    box = QRectF(r.center().x() - 170, r.center().y() - 90, 340, 180)
    pen = QPen(QColor(pal["border"]), 2, Qt.DashLine)
    p.setPen(pen)
    p.setBrush(Qt.NoBrush)
    p.drawRoundedRect(box, 14, 14)
    f = QFont()
    f.setPointSizeF(15)
    f.setBold(True)
    p.setFont(f)
    p.setPen(QColor(pal["text"]))
    p.drawText(QRectF(box.left(), box.top() + 38, box.width(), 34), Qt.AlignCenter, "開始標註")
    f.setPointSizeF(10)
    f.setBold(False)
    p.setFont(f)
    p.setPen(QColor(pal["muted"]))
    p.drawText(QRectF(box.left(), box.top() + 84, box.width(), 60), Qt.AlignCenter,
               "按 Ctrl+O 開啟圖片資料夾\nW 畫 BBox · E 畫旋轉框 · F1 快捷鍵")
    p.restore()


def _norm180(a: float) -> float:
    """Keep angles in (-180, 180] for display/editing."""
    a = (a + 180.0) % 360.0 - 180.0
    return 180.0 if a == -180.0 else a


def _rotated_cursor(handle: str, angle: float):
    sx, sy = HANDLES[handle]
    base = math.degrees(math.atan2(sy, sx)) + angle
    a = base % 180
    if a < 22.5 or a >= 157.5:
        return Qt.SizeHorCursor
    if a < 67.5:
        return Qt.SizeFDiagCursor
    if a < 112.5:
        return Qt.SizeVerCursor
    return Qt.SizeBDiagCursor
