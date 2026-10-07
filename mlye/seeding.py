"""Reproducible RNG setup shared by every replication entry point.

All randomness in `mlye` and `experiments/` is drawn either from the global
torch / numpy RNGs or from generators explicitly seeded from a cell seed, so
seeding the global state at the top of each script fixes every draw.

`seed_everything` additionally asks PyTorch for deterministic kernels. With
the same seed, GPU model, driver and torch build, reruns are bitwise
identical; across GPU models (e.g. L40S vs H100) results agree only up to
floating-point reduction order.
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch

# cuBLAS needs a fixed workspace size for deterministic GEMMs. It must be set
# before the first CUDA context is created, so do it at import time.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def _runtime_info(seed: int) -> dict:
    """Software / hardware actually used by this process (for provenance)."""
    import platform
    import sys
    info = {
        "seed": seed,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": (torch.backends.cudnn.version()
                  if torch.backends.cudnn.is_available() else None),
        "cuda_available": torch.cuda.is_available(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }
    if torch.cuda.is_available():
        dev = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(dev)
        info["gpu"] = {
            "name": props.name,
            "capability": f"{props.major}.{props.minor}",
            "total_memory_mb": round(props.total_memory / 2**20),
            "count_visible": torch.cuda.device_count(),
            "peak_allocated_mb": round(torch.cuda.max_memory_allocated(dev) / 2**20, 1),
            "peak_reserved_mb": round(torch.cuda.max_memory_reserved(dev) / 2**20, 1),
        }
        try:
            import pynvml
            pynvml.nvmlInit()
            v = pynvml.nvmlSystemGetDriverVersion()
            info["gpu"]["driver"] = v.decode() if isinstance(v, bytes) else v
            pynvml.nvmlShutdown()
        except Exception:
            pass
    return info


def _register_runtime_report(seed: int) -> None:
    """If $MLYE_RUNTIME_INFO names a file, write `_runtime_info` there at exit.

    replicate.py sets it so each task's manifest records the GPU, library
    versions and peak memory of the process that actually ran.
    """
    path = os.environ.get("MLYE_RUNTIME_INFO")
    if not path:
        return
    import atexit
    import json

    def _write():
        try:
            with open(path, "w") as f:
                json.dump(_runtime_info(seed), f, indent=1)
        except Exception:
            pass

    atexit.register(_write)


def seed_everything(seed: int, deterministic: bool = True) -> int:
    """Seed python, numpy and torch (CPU + all CUDA devices).

    Returns the seed so callers can record it in their output JSON.
    """
    seed = int(seed)
    _register_runtime_report(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # warn_only: a few ops (e.g. index_add on CUDA) have no deterministic
        # implementation; warn instead of crashing so long runs complete.
        torch.use_deterministic_algorithms(True, warn_only=True)
    return seed


def env_seed(name: str = "SEED", default: int = 11) -> int:
    """Read an integer seed from the environment (default if unset/empty)."""
    v = os.environ.get(name, "")
    return int(v) if v.strip() else int(default)
