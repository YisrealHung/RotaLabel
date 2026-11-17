import multiprocessing
import sys


def main():
    # Qt is imported here, not at module level: dataset generation starts worker processes, and on
    # Windows each one re-imports this file; they must not load Qt.
    import rotalabel.ui  # noqa: F401  (must run before PySide6 is imported: fixes Qt DLL loading in conda)
    from PySide6.QtWidgets import QApplication

    from rotalabel.ui.main_window import MainWindow

    def _excepthook(exc_type, exc_value, exc_tb):
        import traceback
        if sys.stderr is not None:
            traceback.print_exception(exc_type, exc_value, exc_tb)

    sys.excepthook = _excepthook
    app = QApplication(sys.argv)
    app.setOrganizationName("RotaLabel")
    app.setApplicationName("RotaLabel")
    app.setStyle("Fusion")
    win = MainWindow()
    if len(sys.argv) > 1:
        win.open_project(sys.argv[1])
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    multiprocessing.freeze_support()   # needed when packaged as an .exe
    main()
