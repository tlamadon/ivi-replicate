"""Shared pynvml-based GPU utilization sampler.

Usage:
    from gpu_util import GpuUtilSampler

    sampler = GpuUtilSampler().reset().start()
    # ... run training / gradient step ...
    sampler.stop()
    snap = sampler.snapshot()      # dict with mean/max/p90 for gpu + memBW %

Falls back to null-fields if `pynvml` is not importable at module load
(the training script still runs; util fields land as None in the output).
"""
from __future__ import annotations

import threading
import time

import numpy as np


try:
    import pynvml  # type: ignore
    pynvml.nvmlInit()
    _NVML_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
    _NVML_OK = True
except Exception:
    _NVML_HANDLE = None
    _NVML_OK = False


class GpuUtilSampler:
    """Background thread that polls NVML for GPU compute + memory-bandwidth
    utilization at ~50 Hz by default (20 ms poll interval)."""

    def __init__(self, poll_interval_s: float = 0.02):
        self.poll_interval_s = poll_interval_s
        self.gpu_pct: list[int] = []
        self.mem_pct: list[int] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> "GpuUtilSampler":
        if not _NVML_OK:
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> "GpuUtilSampler":
        if self._thread is None:
            return self
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._thread = None
        return self

    def reset(self) -> "GpuUtilSampler":
        self.gpu_pct = []
        self.mem_pct = []
        return self

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                u = pynvml.nvmlDeviceGetUtilizationRates(_NVML_HANDLE)
                self.gpu_pct.append(int(u.gpu))
                self.mem_pct.append(int(u.memory))
            except Exception:
                pass
            time.sleep(self.poll_interval_s)

    def snapshot(self) -> dict:
        if not _NVML_OK or not self.gpu_pct:
            return {
                "n_samples":    0,
                "gpu_pct_mean": None,
                "gpu_pct_max":  None,
                "gpu_pct_p90":  None,
                "mem_pct_mean": None,
                "mem_pct_max":  None,
            }
        g = np.asarray(self.gpu_pct, dtype=np.float64)
        m = np.asarray(self.mem_pct, dtype=np.float64)
        return {
            "n_samples":    int(g.size),
            "gpu_pct_mean": float(g.mean()),
            "gpu_pct_max":  float(g.max()),
            "gpu_pct_p90":  float(np.quantile(g, 0.9)),
            "mem_pct_mean": float(m.mean()),
            "mem_pct_max":  float(m.max()),
        }
