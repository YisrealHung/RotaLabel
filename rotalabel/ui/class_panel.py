from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QColorDialog, QHBoxLayout, QInputDialog, QListWidget, QListWidgetItem,
                               QMessageBox, QPushButton, QVBoxLayout, QWidget)

from .delegates import DATA, ClassDelegate


class ClassPanel(QWidget):
    class_selected = Signal(int)          # user picked a class (also re-classes the selected box)
    current_class_changed = Signal(int)   # current row changed programmatically (drawing class only)
    classes_changed = Signal()
    about_to_delete = Signal()
    class_deleted = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project = None
        self.list = QListWidget()
        self.list.setItemDelegate(ClassDelegate(self.list))
        self.list.setMouseTracking(True)
        self.list.currentRowChanged.connect(self._on_row)
        self.list.itemDoubleClicked.connect(lambda _: self.rename())
        btns = QHBoxLayout()
        btns.setSpacing(2)
        for text, fn, tip in (("＋ 新增", self.add, "新增類別"), ("改名", self.rename, "重新命名（也可雙擊）"),
                              ("顏色", self.recolor, "變更顏色"), ("刪除", self.delete, "刪除類別與其所有框")):
            b = QPushButton(text)
            b.setObjectName("flat")
            b.setToolTip(tip)
            b.setCursor(Qt.PointingHandCursor)
            b.clicked.connect(fn)
            btns.addWidget(b)
        btns.addStretch(1)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 2, 6, 6)
        lay.setSpacing(4)
        lay.addWidget(self.list)
        lay.addLayout(btns)

    def set_project(self, project):
        self.project = project
        self.reload()
        if project and project.classes:
            self._set_row_silently(0)

    def reload(self):
        row = self.list.currentRow()
        self.list.blockSignals(True)
        self.list.clear()
        if self.project:
            for i, c in enumerate(self.project.classes):
                it = QListWidgetItem(c.name)
                it.setData(DATA, (c.color, str(i + 1) if i < 9 else ""))
                it.setToolTip(f"{c.name}" + (f"（快捷鍵 {i + 1}）" if i < 9 else ""))
                self.list.addItem(it)
        self.list.blockSignals(False)
        if self.list.count():
            self._set_row_silently(min(max(row, 0), self.list.count() - 1))

    def _set_row_silently(self, row: int):
        """Change the current row without re-classing the selected box (class_selected is not emitted)."""
        self.list.blockSignals(True)
        self.list.setCurrentRow(row)
        self.list.blockSignals(False)
        self.current_class_changed.emit(row)

    def current(self) -> int:
        return max(self.list.currentRow(), 0)

    def set_current(self, idx: int):
        if 0 <= idx < self.list.count():
            self.list.setCurrentRow(idx)

    def _on_row(self, row):
        if row >= 0:
            self.class_selected.emit(row)

    def add(self):
        if not self.project:
            return
        name, ok = QInputDialog.getText(self, "新增類別", "類別名稱：")
        name = name.strip()
        if ok and name:
            if name in self.project.class_names():
                QMessageBox.warning(self, "新增類別", "類別名稱已存在")
                return
            idx = self.project.add_class(name)
            self.reload()
            self._set_row_silently(idx)
            self.classes_changed.emit()

    def rename(self):
        if not self.project or self.list.currentRow() < 0:
            return
        c = self.project.classes[self.current()]
        name, ok = QInputDialog.getText(self, "改名", "類別名稱：", text=c.name)
        name = name.strip()
        if ok and name and name != c.name:
            if name in self.project.class_names():
                QMessageBox.warning(self, "改名", "類別名稱已存在")
                return
            c.name = name
            self.project.save_meta()
            self.reload()
            self.classes_changed.emit()

    def recolor(self):
        if not self.project or self.list.currentRow() < 0:
            return
        c = self.project.classes[self.current()]
        col = QColorDialog.getColor(QColor(c.color), self, "選擇顏色")
        if col.isValid():
            c.color = col.name()
            self.project.save_meta()
            self.reload()
            self.classes_changed.emit()

    def delete(self):
        if not self.project or self.list.currentRow() < 0:
            return
        if len(self.project.classes) <= 1:
            QMessageBox.information(self, "刪除類別", "至少需保留一個類別")
            return
        idx = self.current()
        name = self.project.classes[idx].name
        r = QMessageBox.question(self, "刪除類別",
                                 f"刪除類別「{name}」會一併刪除所有圖片中此類別的框，且無法復原。確定嗎？")
        if r != QMessageBox.Yes:
            return
        self.about_to_delete.emit()  # main window flushes the open image first
        n = self.project.delete_class(idx)
        self.reload()
        self.class_deleted.emit()
        QMessageBox.information(self, "刪除類別", f"已刪除 {n} 個框")
