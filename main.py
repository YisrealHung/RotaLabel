import sys

import rotalabel.ui  # noqa: F401  (must run before PySide6 is imported: fixes Qt DLL loading in conda)
from PySide6.QtWidgets import QApplication

from rotalabel.ui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("RotaLabel")
    app.setStyle("Fusion")
    win = MainWindow()
    if len(sys.argv) > 1:
        win.open_project(sys.argv[1])
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
