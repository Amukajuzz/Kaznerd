"""Shared helpers: resource monitoring, JSON I/O, directory setup."""
import json
import os
import threading
import time
from pathlib import Path

import psutil

import config


def ensure_dirs():
    """Create every output directory used by the project."""
    for d in (config.DATA_DIR, config.RAW_DIR, config.UD_DIR, config.PROCESSED_DIR,
              config.MODELS_DIR, config.FIGURES_DIR, config.RESULTS_DIR, config.DOCS_DIR):
        Path(d).mkdir(parents=True, exist_ok=True)


class ResourceMonitor:
    """Context manager measuring wall time and peak RSS growth of the process.

    A background thread samples the resident set size, so memory allocated by
    C extensions (crfsuite, numpy, pyarrow) is counted too — unlike tracemalloc.
    """

    def __init__(self, interval=0.05):
        self.interval = interval
        self.seconds = 0.0
        self.peak_mb = 0.0
        self._proc = psutil.Process(os.getpid())
        self._stop = threading.Event()

    def _sample(self):
        while not self._stop.is_set():
            rss = self._proc.memory_info().rss
            self._peak = max(self._peak, rss)
            time.sleep(self.interval)

    def __enter__(self):
        self._start_rss = self._proc.memory_info().rss
        self._peak = self._start_rss
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self._t0
        self._stop.set()
        self._thread.join()
        self._peak = max(self._peak, self._proc.memory_info().rss)
        self.peak_mb = (self._peak - self._start_rss) / 2 ** 20
        return False


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=float)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def print_header(text):
    print("\n" + "=" * 70 + f"\n{text}\n" + "=" * 70)
