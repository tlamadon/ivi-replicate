"""Assemble the time-to-convergence bundle (VI vs SMC, hockey-stick DGP).

Reference copy: bundles/time-to-convergence-hockey-bundle.json. The original
bundle was assembled by hand from the 16 cell JSONs written by
`_time_to_convergence_hockey.py`; this script reproduces that layout.

Inputs:
  output/bundles/cells/_time_to_convergence_hockey/conv_hockey_{vi,smc}_N{1k,30k}_T{6,40}_{h32,h64,K50,K200}.json

Output:
  $OUT_JSON (default output/bundles/time-to-convergence-hockey-bundle.json)

Optional provenance env vars: RUN_ID, BACKEND (copied into metadata).
"""
import datetime
import json
import os

SOURCE_DIR = os.environ.get(
    "SOURCE_DIR", os.path.join("output", "bundles", "cells", "_time_to_convergence_hockey"))
OUT_JSON = os.environ.get(
    "OUT_JSON", os.path.join("output", "bundles", "time-to-convergence-hockey-bundle.json"))

CORNERS = [
    {"name": "small", "N": 1000,  "T": 6},
    {"name": "wide",  "N": 1000,  "T": 40},
    {"name": "tall",  "N": 30000, "T": 6},
    {"name": "big",   "N": 30000, "T": 40},
]
CONFIGS = {
    "vi":  [{"H": 32}, {"H": 64}],
    "smc": [{"K": 50, "resample": "systematic"},
            {"K": 200, "resample": "systematic"}],
}
CRITERION_DEFINITION = (
    "At each snapshot (every LOG_EVERY epochs), compute the L2 distance "
    "between the mean 12-D free-theta vector over the last CONV_WINDOW "
    "snapshots and the mean over the CONV_WINDOW snapshots before those. "
    "Convergence declared at the first snapshot where this drift stays "
    "below CONV_TAU for CONV_HOLD consecutive checks.")
CELL_KEYS = ("converged_at", "converged_wall_s", "total_wall_s",
             "l2_truth_final", "final_theta", "peak_gpu_mem_mb", "gpu_util",
             "snapshots")


def cell_path(method, N, T, param):
    n_tag = f"{N // 1000}k" if N >= 1000 and N % 1000 == 0 else str(N)
    p_tag = f"h{param['H']}" if method == "vi" else f"K{param['K']}"
    return os.path.join(SOURCE_DIR, f"conv_hockey_{method}_N{n_tag}_T{T}_{p_tag}.json")


def main():
    cells, first = [], None
    for corner in CORNERS:
        for method in ("vi", "smc"):
            for param in CONFIGS[method]:
                p = cell_path(method, corner["N"], corner["T"], param)
                with open(p) as f:
                    doc = json.load(f)
                first = first or doc
                res = doc["result"]
                cells.append({
                    "corner": corner["name"], "N": corner["N"],
                    "T": corner["T"], "method": method, "param": dict(param),
                    **{k: res[k] for k in CELL_KEYS},
                })

    cfg = first["config"]
    bundle = {
        "metadata": {
            "title": "Time-to-convergence, VI vs SMC on the hockey-stick DGP",
            "produced_by": ("scripts/_time_to_convergence_hockey.py + "
                            "scripts/_export_time_to_convergence_hockey_bundle.py"),
            "git_hash": cfg.get("git_hash"),
            "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "run_id": os.environ.get("RUN_ID"),
            "backend": os.environ.get("BACKEND"),
            "n_cells": len(cells),
        },
        "dgp": cfg["dgp"],
        "spec": "specs/compute-simulation-hockeystick.md",
        "truth": first["truth"],
        "convergence_criterion": {
            "definition": CRITERION_DEFINITION,
            **{k: cfg[k] for k in ("conv_tau", "conv_window", "conv_hold",
                                   "log_every", "max_epochs", "lr", "clip",
                                   "obs_seed", "vi_seed")},
        },
        "corners": CORNERS,
        "configs": CONFIGS,
        "cells": cells,
    }
    os.makedirs(os.path.dirname(OUT_JSON) or ".", exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(bundle, f, indent=1)
    print(f"wrote {OUT_JSON}  ({len(cells)} cells)")


if __name__ == "__main__":
    main()
