"""Run independent per-image jobs on several processes.

Each process has its own GIL, so OpenCV / NumPy work spreads over every core. Results come back in
task order, so output file names and label files are identical whatever the number of workers.
Task functions must be module-level (picklable) and must not touch Qt.
"""
from __future__ import annotations

import os
import pickle
import tempfile
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Iterable

import cv2

_init_fn: Callable | None = None


def default_workers() -> int:
    """About one worker per physical core. Image work gains nothing from the second hyper-thread of a
    core (measured on an 8-core / 16-thread CPU: 8 workers 5.3 s, 15 workers 6.3 s for 130 mosaics)."""
    n = os.cpu_count() or 1
    return max(1, n // 2 if n >= 4 else n)


def resolve_workers(workers: int, n_tasks: int) -> int:
    """0 (or less) = automatic. Never more workers than tasks."""
    w = default_workers() if not workers or workers <= 0 else int(workers)
    return max(1, min(w, n_tasks))


def _worker_init(payload_path: str):
    cv2.setNumThreads(1)   # one OpenCV thread per process; the processes are the parallelism
    with open(payload_path, "rb") as f:
        init, initargs = pickle.load(f)
    if init is not None:
        init(*initargs)


def run_tasks(fn: Callable, tasks: list, workers: int = 0, init: Callable | None = None, initargs: tuple = (),
              cancelled: Callable[[], bool] | None = None) -> Iterable:
    """Yield ``fn(task)`` for every task, in order. ``init(*initargs)`` runs once per process before any
    task (and once in this process when running serially). Stops early when ``cancelled()`` is true."""
    n = resolve_workers(workers, len(tasks))
    if n <= 1:
        if init is not None:
            init(*initargs)
        for t in tasks:
            if cancelled and cancelled():
                return
            yield fn(t)
        return
    chunk = max(1, min(8, len(tasks) // (n * 4)))
    # The init data (e.g. all mosaic patches, several MB) goes through a temp file, not the process
    # start-up pipe: on Windows a large start-up payload makes every worker wait for the previous one
    # to start (~0.4 s each), so 15 workers took ~6 s just to come up.
    fd, payload = tempfile.mkstemp(prefix="rotalabel_", suffix=".pkl")
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump((init, initargs), f, protocol=pickle.HIGHEST_PROTOCOL)
        ex = ProcessPoolExecutor(max_workers=n, initializer=_worker_init, initargs=(payload,))
        try:
            for r in ex.map(fn, tasks, chunksize=chunk):
                if cancelled and cancelled():
                    return
                yield r
        finally:
            ex.shutdown(wait=True, cancel_futures=True)
    finally:
        try:
            os.remove(payload)
        except OSError:
            pass
