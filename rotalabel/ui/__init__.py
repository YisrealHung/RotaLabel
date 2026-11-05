"""Importing this package prepares Qt to load inside conda environments on Windows.

Qt 6 on Windows links against the system ``icuuc.dll`` (unversioned ICU symbols such as
``ucnv_open``).  Conda environments ship their own ``Library\\bin\\icuuc.dll`` with
versioned symbols (``ucnv_open_73``), and conda puts that folder on the DLL search path,
so ``from PySide6 import QtCore`` fails with "The specified procedure could not be found".
Loading the system copy first makes the loader reuse it for Qt.
"""
import os
import sys


def _preload_system_icu() -> None:
    if sys.platform != "win32":
        return
    path = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "icuuc.dll")
    if os.path.isfile(path):
        import ctypes
        try:
            ctypes.WinDLL(path)
        except OSError:
            pass


_preload_system_icu()
