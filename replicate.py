#!/usr/bin/env python3
"""Replication pipeline: data -> bundles -> figures, all written under output/.

This file is the single source of truth for *what* is run: every task, its
layer, exact settings (copied from the cluster workflows that produced the
published bundles), seeds, dependencies and resources. It is stdlib-only.

    output/data/       layer 1  data matrix (not tracked) + sample statistics
    output/bundles/    layer 2  the bundles (+ cells/ intermediates)
    output/figures/    layer 3  LaTeX fragments, PDFs, figures.pdf
    output/manifests/, output/logs/   provenance and logs

The published outputs are tracked at those same paths; regenerating
overwrites them, so `git diff` (or `replicate.py compare`) shows what changed.

    python replicate.py list    [--block B] [--with-bootstrap]
    python replicate.py run     [--layer 1,2,3] [--block B] [--only GLOB] [--with-bootstrap]
                                [--smoke] [--workdir DIR] [--force] [--dry-run]
    python replicate.py hut                                    # write workflows/hut.replication_*.json
    python replicate.py exec    TASK_ID [--smoke]              # run one task (used by the workflows)
    python replicate.py verify-data
    python replicate.py collect [--workdir DIR] [--out DIR]    # PROVENANCE.json (+ copies with --out)
    python replicate.py compare [--workdir DIR] [--ref REF]    # regenerated vs published (git REF)
    python replicate.py summary [--workdir DIR] [--readme]     # dependency graph + compute used

Blocks (independent; each becomes one scripthut workflow):
    ar1n     AR(1)+Normal simulation: 13-cell sweep, mu1 profile, binding
             function, quiver grid  -> ar1n_* bundles
    hetero   hetero-scale simulation (rho=0.30), 6 cells
             -> hetero_scale_parameter_summary.json
    hockey   hockey-stick simulation, T=6 (11 cells) and T=40 (1 cell), plus
             the past-2-period contour -> hockeystick_parameter_summary*.json,
             contour_bundle.json
    timing   VI vs SMC time-to-convergence, 16 cells
             -> time-to-convergence-hockey-bundle.json
    bpp      PSID/BPP data build + empirical VI/IVI fits (+ optional
             200-replicate bootstrap) -> bundle_psid_heteroscale.json,
             2026-07-13-bpp-hetero-mean-indep-bundle.json

Seeds: every stochastic script calls `mlye.seeding.seed_everything` on start
and derives all draws from explicit seeds (OBS_SEED=11, NOISE_SEED=12345,
VI_SEED=11007, BPP_*_SEED=11, ...). Every cell below passes its seeds
explicitly, so nothing depends on script defaults.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field

REPO = os.path.dirname(os.path.abspath(__file__))

# Everything the pipeline writes lives under output/, organised by layer.
# The published outputs (bundles, sample statistics, bootstrap aggregates)
# are tracked in git at these same paths: regenerating overwrites them, so
# `git diff` shows exactly what changed.
OUT = "output"
DATA_DIR = f"{OUT}/data"                  # layer 1
BUNDLES_DIR = f"{OUT}/bundles"            # layer 2: the bundles ...
CELLS = f"{BUNDLES_DIR}/cells"            # ... and their intermediate per-cell results
FIG_DIR = f"{OUT}/figures"                # layer 3
MANIFEST_DIR = f"{OUT}/manifests"
LOG_DIR = f"{OUT}/logs"
MATRIX = f"{DATA_DIR}/bpp_y_matrix.npy"
WORKFLOW_DIR = "workflows"
STACK = "mlye-replication"
BPP_SHA256 = "c9cda89925b3c35ae67ebaac8a3dfb748ac55d76750d609c0879d61555a9ffae"

# Environment applied to every task (determinism + import path).
BASE_ENV = {
    "PYTHONHASHSEED": "0",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "PYTHONUNBUFFERED": "1",
}


@dataclass
class Task:
    id: str
    block: str
    script: str                       # path under scripts/
    env: dict = field(default_factory=dict)
    outputs: list = field(default_factory=list)
    deps: list = field(default_factory=list)
    smoke_env: dict = field(default_factory=dict)
    inputs: list = field(default_factory=list)  # files read beyond deps' outputs ("?path" = optional)
    args: list = field(default_factory=list)    # command-line arguments for the script
    layer: int = 2                              # 1 data, 2 bundles, 3 figures
    gpu: bool = True
    partition: str = "standard_l40s"
    time: str = "01:00:00"
    mem: str = "8G"
    note: str = ""                    # provenance (workflow file @ commit)


# ----------------------------------------------------------------------
# Smoke overrides: tiny epochs / panels so the whole DAG runs in minutes on
# one small GPU. They change the numbers, never the code path.
# ----------------------------------------------------------------------
SMOKE_SIM = {"N": "1000", "N_EPOCHS_VI_OBS": "40", "N_EPOCHS_INNER": "30",
             "N_ITERS_OUTER": "2", "N_EPOCHS_ENCODER": "30",
             "N_EPOCHS_MLE": "40", "L_IWAE": "5", "K_SMC": "50",
             "TJN_WARM_DECODER_FREEZE_EPOCHS": "10", "LOG_EVERY": "10"}
SMOKE_BPP_SCALE = {"BPP_HETERO_SCALE_N_EPOCHS": "40",
                   "BPP_HETERO_SCALE_N_EPOCHS_VI_OBS": "40",
                   "BPP_HETERO_SCALE_N_EPOCHS_INNER": "30",
                   "BPP_HETERO_SCALE_N_ITERS_OUTER": "2",
                   "BPP_HETERO_SCALE_N_EPOCHS_ENCODER": "30",
                   "BPP_HETERO_SCALE_LOG_EVERY": "10"}
SMOKE_BPP_MEAN = {"BPP_HETERO_MEAN_N_EPOCHS": "40",
                  "BPP_HETERO_MEAN_N_EPOCHS_VI_OBS": "40",
                  "BPP_HETERO_MEAN_N_EPOCHS_INNER": "30",
                  "BPP_HETERO_MEAN_N_ITERS_OUTER": "2",
                  "BPP_HETERO_MEAN_N_EPOCHS_ENCODER": "30",
                  "BPP_HETERO_MEAN_LOG_EVERY": "10"}


# ----------------------------------------------------------------------
# Block: ar1n  (bundles: ar1n_parameter_summary, ar1n_quiver_bundle,
#               ar1n_binding_mu1_bundle, ar1n_mu1_profile_bundle)
# ----------------------------------------------------------------------
AR1N_DIR = f"{CELLS}/_simulation_ar1n_sweep"
AR1N_UPSTREAM = f"{AR1N_DIR}/picard_meanfield_alpha0.60_ep8000_lr1e-2_crn.json"
AR1N_COMMON = {"MU1": "0.9", "SIGMA0": "-1.48", "SIGMA_Z1": "-0.733",
               "LOG_EVERY": "100", "OBS_SEED": "11", "NOISE_SEED": "12345",
               "VI_SEED": "11007"}


def _ar1n_cells():
    """The 13 cells of ar1n_parameter_summary.json.

    lr=1e-2 cells: hut.simulation_ar1n_sweep.json @ 8cf934c and
    hut.simulation_ar1n_ivi_comparison.json @ 8cf934c (log sigma_eps=-1.470).
    lr=1e-3 cells: hut.simulation_ar1n.json @ afeed8d, which used the earlier
    calibration log sigma_eps=-1.833 (as recorded in the bundle's
    truth_tagged_in_source).
    """
    vi = {"METHOD": "vi_only", "FIX_NOISE": "1", "N_EPOCHS_VI_OBS": "10000"}
    ivi = {"FIX_NOISE": "1", "N_EPOCHS_INNER": "8000", "N_EPOCHS_VI_OBS": "10000",
           "N_ITERS_OUTER": "10", "ENCODER": "mean_field"}
    lr2 = {"LR": "0.01", "LR_TAG": "1e-2", "LOG_SIGMA_EPS": "-1.470"}
    lr3 = {"LR": "0.001", "LR_TAG": "1e-3", "LOG_SIGMA_EPS": "-1.833"}
    src2 = "hut.simulation_ar1n_sweep/_ivi_comparison.json @ 8cf934c"
    src3 = "hut.simulation_ar1n.json @ afeed8d"
    cells = [
        ("vi_meanfield_h32_ep10000_lr1e-2_crn", {**vi, **lr2, "ENCODER": "mean_field"}, src2, "01:00:00"),
        ("vi_jointnormal_h32_ep10000_lr1e-2_crn", {**vi, **lr2, "ENCODER": "jn"}, src2, "01:00:00"),
        ("vi_struct_markov_h32_ep10000_lr1e-2_crn", {**vi, **lr2, "ENCODER": "struct_markov"}, src2, "01:00:00"),
        ("vi_tridiag_h32_ep10000_lr1e-2_crn", {**vi, **lr2, "ENCODER": "tridiag"}, src2, "01:00:00"),
        ("picard_meanfield_alpha0.60_ep8000_lr1e-2_crn", {**ivi, **lr2, "METHOD": "picard", "PICARD_ALPHA": "0.6"}, src2, "02:00:00"),
        ("picard_meanfield_alpha1.00_ep8000_lr1e-2_crn", {**ivi, **lr2, "METHOD": "picard", "PICARD_ALPHA": "1.0"}, src2, "02:00:00"),
        ("anderson_meanfield_alpha0.60_m4_ep8000_lr1e-2_crn", {**ivi, **lr2, "METHOD": "anderson", "PICARD_ALPHA": "0.6", "M_MEM": "4"}, src2, "02:00:00"),
        ("mle_noerror_lbfgs", {"METHOD": "mle_noerror", "N": "30000", "LOG_SIGMA_EPS": "-1.470"}, src2, "00:30:00"),
        ("vi_meanfield_h32_ep10000_lr1e-3_crn", {**vi, **lr3, "ENCODER": "mean_field"}, src3, "01:00:00"),
        ("vi_jointnormal_h32_ep10000_lr1e-3_crn", {**vi, **lr3, "ENCODER": "jn"}, src3, "01:00:00"),
        ("vi_struct_markov_h32_ep10000_lr1e-3_crn", {**vi, **lr3, "ENCODER": "struct_markov"}, src3, "01:00:00"),
        ("vi_tjn_h32_ep10000_lr1e-3_crn", {**vi, **lr3, "ENCODER": "tjn", "TJN_WARM_DECODER_FREEZE_EPOCHS": "500"}, src3, "01:30:00"),
        ("anderson_meanfield_alpha0.60_m4_ep8000_lr1e-3_crn", {**ivi, **lr3, "METHOD": "anderson", "PICARD_ALPHA": "0.6", "M_MEM": "4"}, src3, "02:00:00"),
    ]
    return cells


def block_ar1n(smoke=False):
    T = []
    sweep_ids = []
    for name, env, src, tl in _ar1n_cells():
        tid = "ar1n-sweep-" + name.replace(".", "")
        sweep_ids.append(tid)
        T.append(Task(tid, "ar1n", "_simulation_ar1n_sweep.py",
                      env={**AR1N_COMMON, **env},
                      outputs=[f"{AR1N_DIR}/{name}.json"],
                      smoke_env=SMOKE_SIM, time=tl, note=src))
    up = "ar1n-sweep-picard_meanfield_alpha060_ep8000_lr1e-2_crn"
    T.append(Task("ar1n-export-summary", "ar1n", "_export_ar1n_parameter_summary.py",
                  outputs=[f"{BUNDLES_DIR}/ar1n_parameter_summary.json"],
                  deps=sweep_ids, gpu=False, time="00:10:00"))

    # mu1 profile: 20 cells + aggregate (hut.simulation_ar1n_mu1_profile.json)
    n = 2 if smoke else 20
    prof = {"N_GRID": str(n), "N_EPOCHS": "10000", "LR": "1e-2",
            "N_EPOCHS_NONAMORT": "2000", "LR_NONAMORT": "1e-3", "FIX_NOISE": "1",
            "NOISE_SEED": "12345", "MU1_MIN": "0.5", "MU1_MAX": "1.0"}
    prof_smoke = {"N_EPOCHS": "30", "N_EPOCHS_NONAMORT": "20"}
    D = f"{CELLS}/_simulation_ar1n_mu1_profile"
    ids = []
    for i in range(n):
        ids.append(f"ar1n-profile-{i}")
        T.append(Task(ids[-1], "ar1n", "_simulation_ar1n_mu1_profile.py",
                      env={**prof, "CELL_I": str(i)}, outputs=[f"{D}/profile_cell_{i}.json"],
                      deps=[up], smoke_env=prof_smoke, time="00:30:00"))
    T.append(Task("ar1n-profile-aggregate", "ar1n", "_simulation_ar1n_mu1_profile.py",
                  env={**prof, "MODE": "aggregate"}, outputs=[f"{D}/profile.json"],
                  deps=ids + [up], smoke_env=prof_smoke, gpu=False, time="00:10:00"))
    T.append(Task("ar1n-export-profile", "ar1n", "_export_ar1n_mu1_profile_bundle.py",
                  outputs=[f"{BUNDLES_DIR}/ar1n_mu1_profile_bundle.json"],
                  deps=["ar1n-profile-aggregate"], gpu=False, time="00:10:00"))

    # binding function in mu1: 15 rows + b* + aggregate
    n = 2 if smoke else 15
    bind = {"N_GRID": str(n), "N_EPOCHS_VI": "8000", "LR": "1e-2", "FIX_NOISE": "1",
            "NOISE_SEED": "12345"}
    D = f"{CELLS}/_simulation_ar1n_binding_mu1"
    ids = []
    for r in range(n):
        ids.append(f"ar1n-binding-{r}")
        T.append(Task(ids[-1], "ar1n", "_simulation_ar1n_binding_mu1.py",
                      env={**bind, "ROW": str(r)}, outputs=[f"{D}/binding_mu1_row_{r}.json"],
                      deps=[up], smoke_env={"N_EPOCHS_VI": "30"}, time="00:30:00"))
    ids.append("ar1n-binding-bstar")
    T.append(Task("ar1n-binding-bstar", "ar1n", "_simulation_ar1n_binding_mu1.py",
                  env={**bind, "MODE": "bstar"}, outputs=[f"{D}/binding_mu1_bstar.json"],
                  deps=[up], smoke_env={"N_EPOCHS_VI": "30"}, time="00:30:00"))
    T.append(Task("ar1n-binding-aggregate", "ar1n", "_simulation_ar1n_binding_mu1.py",
                  env={**bind, "MODE": "aggregate"}, outputs=[f"{D}/binding_mu1.json"],
                  deps=ids + [up], gpu=False, time="00:10:00"))
    T.append(Task("ar1n-export-binding", "ar1n", "_export_ar1n_binding_mu1_bundle.py",
                  outputs=[f"{BUNDLES_DIR}/ar1n_binding_mu1_bundle.json"],
                  deps=["ar1n-binding-aggregate"], gpu=False, time="00:10:00"))

    # quiver: 10x10 grid + b* + aggregate
    n = 2 if smoke else 10
    quiv = {"N_GRID": str(n), "N_EPOCHS_VI": "8000", "LR": "1e-2", "FIX_NOISE": "1",
            "NOISE_SEED": "12345", "GRID_PAD": "0.30"}
    D = f"{CELLS}/_simulation_ar1n_quiver"
    ids = []
    for i in range(n):
        for j in range(n):
            ids.append(f"ar1n-quiver-{i}-{j}")
            T.append(Task(ids[-1], "ar1n", "_simulation_ar1n_quiver.py",
                          env={**quiv, "CELL_I": str(i), "CELL_J": str(j)},
                          outputs=[f"{D}/quiver_cell_{i}_{j}.json"], deps=[up],
                          smoke_env={"N_EPOCHS_VI": "30"}, time="00:30:00"))
    ids.append("ar1n-quiver-bstar")
    T.append(Task("ar1n-quiver-bstar", "ar1n", "_simulation_ar1n_quiver.py",
                  env={**quiv, "MODE": "bstar"}, outputs=[f"{D}/quiver_bstar.json"],
                  deps=[up], smoke_env={"N_EPOCHS_VI": "30"}, time="00:30:00"))
    T.append(Task("ar1n-quiver-aggregate", "ar1n", "_simulation_ar1n_quiver.py",
                  env={**quiv, "MODE": "aggregate"}, outputs=[f"{D}/quiver.json"],
                  deps=ids + [up], time="00:15:00"))
    T.append(Task("ar1n-export-quiver", "ar1n", "_export_ar1n_quiver_bundle.py",
                  outputs=[f"{BUNDLES_DIR}/ar1n_quiver_bundle.json"],
                  deps=["ar1n-quiver-aggregate"], gpu=False, time="00:10:00"))
    return T


# ----------------------------------------------------------------------
# Block: hetero  (bundle: hetero_scale_parameter_summary)
# hut.hetero_scale_ivi_rho0-30.json @ 3cdeadf (N_ITERS_OUTER=10, as recorded
# in the bundle; later revisions raised it to 25).
# ----------------------------------------------------------------------
def block_hetero(smoke=False):
    D = f"{CELLS}/_simulation_hetero_scale_ivi_sweep"
    base = {"RHO": "0.30", "LR": "0.01", "LR_TAG": "1e-2", "LOG_EVERY": "200",
            "NOISE_SEED": "12345", "OBS_SEED": "11"}
    ivi = {"FIX_NOISE": "1", "N_EPOCHS_INNER": "16000", "N_EPOCHS_VI_OBS": "20000",
           "N_ITERS_OUTER": "10", "PICARD_ALPHA": "0.6"}
    vi = {"METHOD": "vi_only", "FIX_NOISE": "1", "N_EPOCHS_VI_OBS": "20000"}
    cells = [
        ("picard_cold_alpha0.60_ep16000_lr1e-2_crn_rho0.30", {**ivi, "METHOD": "picard_cold"}, "05:00:00"),
        ("anderson_alpha0.60_m4_ep16000_lr1e-2_crn_rho0.30", {**ivi, "METHOD": "anderson", "M_MEM": "4"}, "05:00:00"),
        ("vi_only_meanfield_h64_ep20000_lr1e-2_crn_rho0.30", {**vi, "ENCODER": "mean_field"}, "01:00:00"),
        ("vi_only_struct_markov_h64_ep20000_lr1e-2_crn_rho0.30", {**vi, "ENCODER": "struct_markov"}, "01:00:00"),
        ("vi_only_tjn_h64_ep20000_lr1e-2_crn_rho0.30", {**vi, "ENCODER": "tjn", "TJN_WARM_DECODER_FREEZE_EPOCHS": "1000"}, "01:30:00"),
        ("mle_direct_no_me_ep20000_lr1e-2_rho0.30", {"METHOD": "mle_direct", "EMISSION": "degenerate", "N_EPOCHS_MLE": "20000"}, "00:30:00"),
    ]
    T, ids = [], []
    for name, env, tl in cells:
        ids.append("hetero-" + name.replace(".", ""))
        T.append(Task(ids[-1], "hetero", "_simulation_hetero_scale_ivi_sweep.py",
                      env={**base, **env}, outputs=[f"{D}/{name}.json"],
                      smoke_env=SMOKE_SIM, time=tl,
                      note="hut.hetero_scale_ivi_rho0-30.json @ 3cdeadf"))
    T.append(Task("hetero-export-summary", "hetero", "_export_hetero_scale_parameter_summary.py",
                  outputs=[f"{BUNDLES_DIR}/hetero_scale_parameter_summary.json"], deps=ids,
                  gpu=False, time="00:10:00"))
    return T


# ----------------------------------------------------------------------
# Block: hockey  (bundles: hockeystick_parameter_summary(_longt), contour)
# ----------------------------------------------------------------------
def block_hockey(smoke=False):
    D = f"{CELLS}/_simulation_hockeystick_ivi_sweep"
    D40 = f"{CELLS}/_simulation_hockeystick_t40_ivi_sweep"
    base = {"SIGMA0": "-1.8", "SIGMA2": "0.35", "LR": "0.01", "LR_TAG": "1e-2",
            "LOG_EVERY": "200", "NOISE_SEED": "12345", "OBS_SEED": "11"}
    ivi = {"FIX_NOISE": "1", "N_EPOCHS_INNER": "16000", "N_EPOCHS_VI_OBS": "20000",
           "PICARD_ALPHA": "0.6"}
    vi = {"METHOD": "vi_only", "FIX_NOISE": "1", "N_EPOCHS_VI_OBS": "20000"}
    sfx = "ep20000_lr1e-2_crn_sig0--1.80"
    cells = [
        # shipping cells (hut.hockeystick_ivi_sig0-18.json @ 084ab10)
        ("picard_cold_alpha0.60_ep16000_lr1e-2_crn_sig0--1.80", {**ivi, "METHOD": "picard_cold", "N_ITERS_OUTER": "25"}, "04:00:00", "@ 084ab10"),
        ("mle_direct_no_me_ep20000_lr1e-2_sig0--1.80", {"METHOD": "mle_direct", "EMISSION": "degenerate", "N_EPOCHS_MLE": "20000"}, "00:30:00", "@ 084ab10"),
        (f"vi_only_meanfield_h64_{sfx}", {**vi, "ENCODER": "mean_field"}, "00:45:00", "@ 084ab10"),
        (f"vi_only_struct_markov_h64_{sfx}", {**vi, "ENCODER": "struct_markov"}, "00:45:00", "@ 084ab10"),
        (f"vi_only_tridiag_jn_h64_{sfx}", {**vi, "ENCODER": "tridiag_jn"}, "00:45:00", "@ 084ab10"),
        # historical (non-shipping) cells still listed in the bundle
        ("anderson_alpha0.60_m4_ep16000_lr1e-2_crn_sig0--1.80", {**ivi, "METHOD": "anderson", "M_MEM": "4", "N_ITERS_OUTER": "10"}, "02:00:00", "@ 56b5d49"),
        ("anderson_meanfield_h64_alpha0.60_m4_ep16000_lr1e-2_crn_sig0--1.80", {**ivi, "METHOD": "anderson", "M_MEM": "4", "N_ITERS_OUTER": "10", "ENCODER": "mean_field"}, "02:00:00", "@ 56b5d49"),
        ("anderson_struct_markov_h64_alpha0.60_m4_ep16000_lr1e-2_crn_sig0--1.80", {**ivi, "METHOD": "anderson", "M_MEM": "4", "N_ITERS_OUTER": "10", "ENCODER": "struct_markov"}, "02:00:00", "@ 56b5d49"),
        # written by an earlier revision under METHOD=vi_only; OUT_NAME keeps
        # the historical filename the bundle refers to
        (f"vi_meanfield_h64_{sfx}", {**vi, "ENCODER": "mean_field", "OUT_NAME": f"vi_meanfield_h64_{sfx}.json"}, "01:00:00", "@ e9ef170"),
        (f"vi_struct_markov_h64_{sfx}", {**vi, "ENCODER": "struct_markov", "OUT_NAME": f"vi_struct_markov_h64_{sfx}.json"}, "01:00:00", "@ e9ef170"),
        (f"vi_tjn_h64_{sfx}", {**vi, "ENCODER": "tjn", "TJN_WARM_DECODER_FREEZE_EPOCHS": "1000", "OUT_NAME": f"vi_tjn_h64_{sfx}.json"}, "01:30:00", "@ e9ef170"),
    ]
    T, ids = [], []
    for name, env, tl, at in cells:
        ids.append("hockey-" + name.replace(".", ""))
        T.append(Task(ids[-1], "hockey", "_simulation_hockeystick_ivi_sweep.py",
                      env={**base, **env}, outputs=[f"{D}/{name}.json"],
                      smoke_env=SMOKE_SIM, time=tl,
                      note=f"hut.hockeystick_ivi_sig0-18.json {at}"))
    T.append(Task("hockey-export-summary", "hockey", "_export_hockeystick_parameter_summary.py",
                  outputs=[f"{BUNDLES_DIR}/hockeystick_parameter_summary.json"], deps=ids,
                  gpu=False, time="00:10:00"))
    # T=40 (hut.hockeystick_ivi_t40.json @ 834debd)
    t40 = "picard_cold_alpha0.60_ep16000_lr1e-2_crn_sig0--1.80_T40"
    T.append(Task("hockey-t40-picard_cold", "hockey", "_simulation_hockeystick_ivi_sweep.py",
                  env={**base, **ivi, "METHOD": "picard_cold", "N_ITERS_OUTER": "50",
                       "T": "40", "OUT_DIR": D40},
                  outputs=[f"{D40}/{t40}.json"], smoke_env=SMOKE_SIM,
                  partition="long_hopper", time="1-12:00:00", mem="16G",
                  note="hut.hockeystick_ivi_t40.json @ 834debd"))
    T.append(Task("hockey-t40-export-summary", "hockey", "_export_hockeystick_parameter_summary.py",
                  env={"SOURCE_DIR": D40,
                       "OUT_JSON": f"{BUNDLES_DIR}/hockeystick_parameter_summary_longt.json"},
                  outputs=[f"{BUNDLES_DIR}/hockeystick_parameter_summary_longt.json"],
                  deps=["hockey-t40-picard_cold"], gpu=False, time="00:10:00"))
    # Contour (run by hand originally; script defaults). Reads only `truth`
    # from the T=6 picard cell.
    C = f"{CELLS}/_contour_past_2period_imputed"
    T.append(Task("hockey-contour", "hockey", "_contour_past_2period_imputed.py",
                  env={"N_GRID": "140", "K": "6000", "N_TRAIN": "8000", "N_EPOCHS": "8000",
                       "LR": "1e-3", "HIDDEN_DIM": "64", "SMC_K": "2000", "SMC_DRAWS": "8",
                       "ADAPT_SEED": "11", "TRAIN_SEED": "42", "TAIL_SEED": "11"},
                  outputs=[f"{BUNDLES_DIR}/contour_bundle.json"], deps=[ids[0]],
                  smoke_env={"N_GRID": "8", "K": "200", "N_TRAIN": "500", "N_EPOCHS": "30",
                             "SMC_K": "100", "SMC_DRAWS": "2"},
                  time="03:00:00", mem="16G", note="no workflow; script defaults"))
    return T


# ----------------------------------------------------------------------
# Block: timing  (bundle: time-to-convergence-hockey-bundle)
# hut.time_to_convergence_hockey.json (run 65c46b48, pythia-nb standard_l40s).
# Wall-clock numbers are hardware-specific: run on L40S to compare.
# ----------------------------------------------------------------------
def block_timing(smoke=False):
    D = f"{CELLS}/_time_to_convergence_hockey"
    corners = [("small", 1000, 6), ("wide", 1000, 40), ("tall", 30000, 6), ("big", 30000, 40)]
    tl = {("small", "vi"): "00:20:00", ("small", "smc"): "00:20:00",
          ("wide", "vi"): "00:45:00", ("wide", "smc"): "00:45:00",
          ("tall", "vi"): "00:30:00", ("tall", "smc"): "00:30:00",
          ("big", "vi"): "00:45:00", ("big", "smc"): "02:00:00"}
    mem = {"small": "4G", "wide": "4G", "tall": "8G", "big": "16G"}
    base = {"LR": "1e-2", "LOG_EVERY": "100", "CONV_TAU": "0.02", "CONV_WINDOW": "5",
            "CONV_HOLD": "2", "MAX_EPOCHS": "25000", "CLIP": "5.0",
            "OBS_SEED": "11", "VI_SEED": "11007"}
    T, ids = [], []
    for cname, N, Tn in corners:
        ntag = f"{N // 1000}k"
        for method, params in (("vi", [("H", 32), ("H", 64)]),
                               ("smc", [("K", 50), ("K", 200)])):
            for key, val in params:
                ptag = f"h{val}" if key == "H" else f"K{val}"
                env = {**base, "METHOD": method, "N": str(N), "T": str(Tn), key: str(val)}
                if method == "smc":
                    env["RESAMPLE"] = "systematic"
                ids.append(f"timing-{method}-{cname}-{ptag}")
                T.append(Task(ids[-1], "timing", "_time_to_convergence_hockey.py",
                              env=env, outputs=[f"{D}/conv_hockey_{method}_N{ntag}_T{Tn}_{ptag}.json"],
                              smoke_env={"MAX_EPOCHS": "300"}, time=tl[(cname, method)],
                              mem=mem[cname]))
    T.append(Task("timing-export-bundle", "timing", "_export_time_to_convergence_hockey_bundle.py",
                  outputs=[f"{BUNDLES_DIR}/time-to-convergence-hockey-bundle.json"], deps=ids,
                  gpu=False, time="00:10:00"))
    return T


# ----------------------------------------------------------------------
# Block: bpp  (bundles: bundle_psid_heteroscale, bpp hetero-mean-indep)
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Block: data  (layer 1, local; bundle: bpp_selection_table.json)
# Sample statistics behind the PSID data table. Reads the ICPSR files
# directly, so it runs where they are (not on the cluster).
# ----------------------------------------------------------------------
def block_data(smoke=False):
    return [Task("bpp-sample-stats", "data", "psid_bpp_selection_table.py", layer=1, gpu=False,
                 outputs=[f"{DATA_DIR}/bpp_selection_table.json"],
                 inputs=[f"$BPP_EXT/{f}" for f in ("data.dta", "tax9192.dta", "natpr.dta")],
                 time="00:20:00", note="BPP Table 1 / Appendix A vs our selection vs balanced panel")]


BOOT_COND = f"{CELLS}/_bpp_hetero_scale_ivi_bootstrap_k40"
BOOT_UNCOND = f"{CELLS}/_bpp_hetero_scale_ivi_bootstrap_k40_uncond"


def block_bpp(smoke=False, with_bootstrap=False):
    T = [Task("bpp-data", "bpp", "psid_bpp_build.py", layer=1,
              outputs=[MATRIX], gpu=False, time="00:20:00",
              inputs=[f"$BPP_EXT/{f}" for f in ("data.dta", "tax9192.dta", "natpr.dta")],
              mem="8G", note=f"ICPSR 210782 -> {MATRIX}")]
    hs = {"BPP_HETERO_SCALE_SEED": "11"}
    # hetero-scale VI fits, K=40 ELBO draws (hut.bpp_hetero_scale_vi_k40.json)
    vi_ids = []
    for enc in ("mf", "jn", "sm"):
        vi_ids.append(f"bpp-hs-vi-{enc}")
        T.append(Task(vi_ids[-1], "bpp", f"bpp_hetero_scale_vi_{enc}.py",
                      env={**hs, "BPP_HETERO_SCALE_NDRAWS": "40", "BPP_HETERO_SCALE_N_EPOCHS": "20000"},
                      outputs=[f"{CELLS}/employment-bpp-hetero-scale-vi-{enc}/bpp_hetero_scale_vi_{enc}.json"],
                      deps=["bpp-data"], smoke_env=SMOKE_BPP_SCALE,
                      partition="standard_hopper", time="04:00:00", mem="32G",
                      note="hut.bpp_hetero_scale_vi_k40.json"))
    ivi = {**hs, "BPP_HETERO_SCALE_NOISE_SEED": "12345", "BPP_HETERO_SCALE_METHOD": "picard",
           "BPP_HETERO_SCALE_PICARD_ALPHA": "0.6", "BPP_HETERO_SCALE_NDRAWS": "40"}
    it50 = f"{CELLS}/employment-bpp-hetero-scale-ivi-jn-k40-picard-a06-it50"
    T.append(Task("bpp-hs-ivi-it50", "bpp", "bpp_hetero_scale_ivi_jn.py",
                  env={**ivi, "BPP_HETERO_SCALE_N_ITERS_OUTER": "50",
                       "BPP_HETERO_SCALE_OUT_DIR": it50,
                       "BPP_HETERO_SCALE_OUT_JSON": f"{it50}/bpp_hetero_scale_ivi_jn.json"},
                  outputs=[f"{it50}/bpp_hetero_scale_ivi_jn.json"], deps=["bpp-data"],
                  smoke_env=SMOKE_BPP_SCALE, partition="standard_hopper",
                  time="08:00:00", mem="32G", note="hut.bpp_hetero_scale_ivi_k40_it50.json"))
    # hetero-mean-independent VI + IVI (hut.bpp_hetero_mean_indep.json; defaults)
    hm = {"BPP_HETERO_MEAN_SEED": "11", "BPP_HETERO_MEAN_NDRAWS": "40"}
    T.append(Task("bpp-hm-vi", "bpp", "bpp_hetero_mean_indep_vi_jn.py",
                  env={**hm, "BPP_HETERO_MEAN_N_EPOCHS": "20000"},
                  outputs=[f"{CELLS}/employment-bpp-hetero-mean-indep-vi-jn/bpp_hetero_mean_indep_vi_jn.json"],
                  deps=["bpp-data"], smoke_env=SMOKE_BPP_MEAN,
                  partition="standard_hopper", time="04:00:00", mem="32G",
                  note="hut.bpp_hetero_mean_indep.json"))
    T.append(Task("bpp-hm-ivi", "bpp", "bpp_hetero_mean_indep_ivi_jn.py",
                  env={**hm, "BPP_HETERO_MEAN_NOISE_SEED": "12345", "BPP_HETERO_MEAN_METHOD": "picard",
                       "BPP_HETERO_MEAN_PICARD_ALPHA": "0.6", "BPP_HETERO_MEAN_N_ITERS_OUTER": "10"},
                  outputs=[f"{CELLS}/employment-bpp-hetero-mean-indep-ivi-jn/bpp_hetero_mean_indep_ivi_jn.json"],
                  deps=["bpp-data"], smoke_env=SMOKE_BPP_MEAN,
                  partition="standard_hopper", time="08:00:00", mem="32G",
                  note="hut.bpp_hetero_mean_indep.json"))

    boot_deps = []
    if with_bootstrap:
        # Full-IVI nonparametric bootstrap, B=100 each:
        # conditional  (hut.bpp_hetero_scale_ivi_bootstrap_k40[_fill].json):
        #     household resample seed 31000+b, shared simulation seeds;
        # unconditional (hut.bpp_hetero_scale_ivi_bootstrap_k40_uncond[_r64].json):
        #     additionally SEED=5000+b, NOISE_SEED=900000+b.
        B = 2 if smoke else 100
        rep = {**ivi, "BPP_HETERO_SCALE_N_ITERS_OUTER": "20", "BPP_HETERO_SCALE_SKIP_PHASE3": "1"}
        for kind, D in (("cond", BOOT_COND), ("uncond", BOOT_UNCOND)):
            rids = []
            for b in range(B):
                env = {**rep, "BPP_HETERO_SCALE_RESAMPLE_SEED": str(31000 + b),
                       "BPP_HETERO_SCALE_OUT_DIR": D,
                       "BPP_HETERO_SCALE_OUT_JSON": f"{D}/rep_{b}.json"}
                if kind == "uncond":
                    env.update({"BPP_HETERO_SCALE_SEED": str(5000 + b),
                                "BPP_HETERO_SCALE_NOISE_SEED": str(900000 + b)})
                rids.append(f"bpp-boot-{kind}-{b}")
                T.append(Task(rids[-1], "bpp", "bpp_hetero_scale_ivi_jn.py", env=env,
                              outputs=[f"{D}/rep_{b}.json"], deps=["bpp-data"],
                              smoke_env=SMOKE_BPP_SCALE, partition="standard_hopper",
                              time="08:00:00", mem="32G"))
            agg_env = {"BPP_BOOT_AGG_DIR": D, "BPP_BOOT_N_ITERS": "20"}
            agg_deps = list(rids)
            if kind == "uncond":
                agg_env["BPP_BOOT_AGG_PAIR"] = BOOT_COND
                agg_deps += [f"bpp-boot-cond-{b}" for b in range(B)]
            boot_deps.append(f"bpp-boot-{kind}-aggregate")
            T.append(Task(boot_deps[-1], "bpp", "_bpp_hetero_scale_ivi_bootstrap_aggregate.py",
                          env=agg_env, outputs=[f"{D}/bootstrap_theta.json"],
                          deps=agg_deps, gpu=False, time="00:10:00",
                          smoke_env={"BPP_BOOT_N_ITERS": SMOKE_BPP_SCALE["BPP_HETERO_SCALE_N_ITERS_OUTER"]}))

    T.append(Task("bpp-export-heteroscale", "bpp", "bpp_hetero_scale_bundle.py",
                  outputs=[f"{BUNDLES_DIR}/bundle_psid_heteroscale.json"],
                  deps=vi_ids + ["bpp-hs-ivi-it50"] + boot_deps, gpu=False, time="00:10:00",
                  inputs=[f"{BOOT_COND}/bootstrap_theta.json", f"{BOOT_UNCOND}/bootstrap_theta.json"]))
    T.append(Task("bpp-export-meanindep", "bpp", "_export_bpp_hetero_mean_indep_bundle.py",
                  outputs=[f"{BUNDLES_DIR}/2026-07-13-bpp-hetero-mean-indep-bundle.json"],
                  deps=["bpp-hm-vi", "bpp-hm-ivi", "bpp-hs-ivi-it50"] + boot_deps[:1],
                  gpu=False, time="00:10:00",
                  inputs=[f"{BOOT_COND}/bootstrap_theta.json", MATRIX]))
    return T


# ----------------------------------------------------------------------
# Layer 3: figures and tables (block "figures"; runs locally, needs Tectonic)
# ----------------------------------------------------------------------
# Order = order in the combined document (the paper's order). `number` is the
# paper numbering (fig<N> / tab<N>, appendix tables tabB1, tabE1, ...);
# `online` marks web-appendix items (no EPS export). `bundle` is the primary
# input (published name; None for fixed-content items), passed as {bundle};
# `reads` lists further bundles the generator loads from the same directory.
FIGURES = [
    {"name": "sample_vs_population", "number": "fig1", "bundle": None},
    {"name": "ivi_binding_function", "number": "fig2", "bundle": None},
    {"name": "ar1n_parameter_table", "number": "tab1", "bundle": "ar1n_parameter_summary.json"},
    {"name": "ar1n_trajectory_fixed_point", "number": "fig3", "bundle": "ar1n_quiver_bundle.json"},
    {"name": "ar1n_likelihood_elbo", "number": "fig4", "bundle": "ar1n_mu1_profile_bundle.json",
     "args": ["--json", "{bundle}", "--out", "{out}"]},
    {"name": "ar1n_binding_function", "number": "fig5", "bundle": "ar1n_binding_mu1_bundle.json"},
    {"name": "hockeystick_parameter_table", "number": "tab2", "bundle": "hockeystick_parameter_summary.json",
     "reads": ["hockeystick_parameter_summary_longt.json"]},
    {"name": "hockeystick_mu_sigma_z1_e", "number": "fig7", "bundle": "hockeystick_parameter_summary.json"},
    {"name": "hockeystick_posteriors", "number": "fig6", "bundle": "contour_bundle.json"},
    {"name": "hockeystick_resource_table", "number": "tabE1", "online": True,
     "bundle": "time-to-convergence-hockey-bundle.json"},
    {"name": "hetero_scale_parameter_table", "number": "tab3", "bundle": "hetero_scale_parameter_summary.json",
     "args": ["--summary", "{bundle}", "--output", "{out}"]},
    {"name": "hetero_scale_a_sigma_z1_e", "number": "fig8", "bundle": "hetero_scale_parameter_summary.json"},
    {"name": "psid_parameter_table", "number": "tab4", "bundle": "bundle_psid_heteroscale.json"},
    {"name": "psid_a_sigma_z1_eps", "number": "fig9", "bundle": "bundle_psid_heteroscale.json"},
    {"name": "psid_persistence_residual", "number": "fig10", "bundle": "bundle_psid_heteroscale.json"},
    {"name": "psid_mixture_density", "number": "fig11", "bundle": "bundle_psid_heteroscale.json"},
    {"name": "psid_ce_risk_premium", "number": "fig12", "bundle": "bundle_psid_heteroscale.json"},
    {"name": "psid_hetro_comp_parameter_table", "number": "tabH1", "online": True,
     "bundle": "2026-07-13-bpp-hetero-mean-indep-bundle.json", "reads": ["bundle_psid_heteroscale.json"]},
    {"name": "psid_sample_table", "number": "tabB1", "bundle": "bpp_selection_table.json"},
    {"name": "computational_settings_table", "number": "tabF1", "online": True, "bundle": None},
]


def block_figures(smoke=False):
    """One generator task per artifact, then the compile task.

    Generators read the bundles in place (output/bundles/, output/data/):
    the published versions, or the regenerated ones once layer 2 has run.
    """
    T, frag_ids = [], []
    for fig in FIGURES:
        out = f"{FIG_DIR}/tex/{fig['name']}.tex"
        args = fig.get("args", ["{bundle}", "-o", "{out}"] if fig["bundle"] else ["-o", "{out}"])
        args = [a.replace("{out}", out) for a in args]
        if fig["bundle"]:
            args = [a.replace("{bundle}", BUNDLE_PATH[fig["bundle"]]) for a in args]
        frag_ids.append(f"fig-{fig['name']}")
        T.append(Task(frag_ids[-1], "figures", f"fig_{fig['name']}.py", layer=3, gpu=False,
                      args=args, outputs=[out], time="00:05:00",
                      inputs=[BUNDLE_PATH[b] for b in fig.get("reads", [])]))
    T.append(Task("fig-compile", "figures", "fig_compile.py", layer=3, gpu=False,
                  outputs=[f"{FIG_DIR}/pdf/{f['name']}.pdf" for f in FIGURES] + [f"{FIG_DIR}/figures.pdf"],
                  inputs=["scripts/fig_preamble.tex"], deps=frag_ids, time="00:30:00",
                  note="Tectonic (mise.toml)"))
    # journal EPS files (paper items only; needs Ghostscript), independent of fig-compile
    T.append(Task("fig-eps", "figures", "fig_eps.py", layer=3, gpu=False,
                  outputs=[f"{FIG_DIR}/eps/{f['number']}_{f['name']}.eps"
                           for f in FIGURES if not f.get("online")],
                  inputs=["scripts/fig_preamble.tex"], deps=frag_ids, time="00:30:00",
                  note="Tectonic + Ghostscript eps2write"))
    return T


BLOCKS = ["ar1n", "hetero", "hockey", "timing", "bpp"]   # layers 1-2 (cluster)
ALL_BLOCKS = ["data"] + BLOCKS + ["figures"]             # + local layer-1 data stats, layer 3

# Where each published bundle lives (path -> published name). The 11
# estimation bundles are layer-2 outputs; the sample statistics are layer 1.
BUNDLE_NAMES = [
    "ar1n_parameter_summary.json", "ar1n_mu1_profile_bundle.json",
    "ar1n_binding_mu1_bundle.json", "ar1n_quiver_bundle.json",
    "hetero_scale_parameter_summary.json", "hockeystick_parameter_summary.json",
    "hockeystick_parameter_summary_longt.json", "contour_bundle.json",
    "time-to-convergence-hockey-bundle.json", "bundle_psid_heteroscale.json",
    "2026-07-13-bpp-hetero-mean-indep-bundle.json",
]
BUNDLE_MAP = {f"{BUNDLES_DIR}/{n}": n for n in BUNDLE_NAMES}
BUNDLE_MAP[f"{DATA_DIR}/bpp_selection_table.json"] = "bpp_selection_table.json"
BUNDLE_PATH = {n: p for p, n in BUNDLE_MAP.items()}


def all_tasks(smoke=False, with_bootstrap=False, blocks=None):
    builders = {"ar1n": block_ar1n, "hetero": block_hetero, "hockey": block_hockey,
                "timing": block_timing,
                "bpp": lambda smoke: block_bpp(smoke, with_bootstrap),
                "data": block_data,
                "figures": block_figures}
    tasks = []
    for b in blocks or ALL_BLOCKS:
        tasks += builders[b](smoke=smoke)
    ids = [t.id for t in tasks]
    assert len(ids) == len(set(ids)), "duplicate task ids"
    known = set(ids)
    for t in tasks:
        missing = [d for d in t.deps if d not in known]
        assert not missing, f"{t.id}: unknown deps {missing}"
    return tasks


def toposort(tasks):
    by_id = {t.id: t for t in tasks}
    out, seen = [], set()

    def visit(t, stack=()):
        if t.id in seen:
            return
        if t.id in stack:
            raise ValueError(f"dependency cycle at {t.id}")
        for d in t.deps:
            visit(by_id[d], stack + (t.id,))
        seen.add(t.id)
        out.append(t)

    for t in tasks:
        visit(t)
    return out


# ----------------------------------------------------------------------
# Execution
# ----------------------------------------------------------------------
def git_hash():
    try:
        r = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"],
                           capture_output=True, text=True, timeout=5)
        return r.stdout.strip() or None
    except Exception:
        return None


SWEEP_SCRIPTS = {"_simulation_ar1n_sweep.py", "_simulation_hetero_scale_ivi_sweep.py",
                 "_simulation_hockeystick_ivi_sweep.py"}


def task_env(t, smoke):
    env = dict(os.environ)
    env.update(BASE_ENV)
    env["PYTHONPATH"] = os.pathsep.join(
        [REPO, os.path.join(REPO, "scripts")] +
        ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    env.update(t.env)
    if smoke:
        env.update(t.smoke_env)
        if t.script in SWEEP_SCRIPTS and t.outputs:
            # these scripts encode epoch counts in the filename
            env.setdefault("OUT_NAME", os.path.basename(t.outputs[0]))
    if any(p.startswith("$BPP_EXT/") for p in t.inputs) and "BPP_EXT" not in env:
        local = os.path.join(REPO, "ext", "bpp")
        env["BPP_EXT"] = local if os.path.isdir(local) else os.path.expanduser("~/data/bpp_ext")
    return env


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare_workdir(workdir, with_bootstrap=False):
    """A scratch tree laid out like a fresh checkout's output/.

    Scripts read/write paths relative to cwd (output/...), so a run in
    `workdir` never touches the repo's output/. The tracked published files
    are copied in (so layer 3 and the bpp bundles work without recomputing
    everything); the ICPSR inputs and the data matrix are linked.
    """
    os.makedirs(workdir, exist_ok=True)
    ext = os.path.join(workdir, "ext")
    if os.path.isdir(os.path.join(REPO, "ext")) and not os.path.lexists(ext):
        os.symlink(os.path.join(REPO, "ext"), ext)
    for name in ("bpp_y_matrix.npy", "bpp_y_matrix.meta.json"):
        src, dst = os.path.join(REPO, DATA_DIR, name), os.path.join(workdir, DATA_DIR, name)
        if os.path.exists(src) and not os.path.lexists(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.symlink(src, dst)
    tracked = subprocess.run(["git", "-C", REPO, "ls-files", OUT], capture_output=True,
                             text=True).stdout.split()
    for rel in tracked:
        if with_bootstrap and "_bpp_hetero_scale_ivi_bootstrap_k40" in rel:
            continue  # the bootstrap stage recomputes these
        src, dst = os.path.join(REPO, rel), os.path.join(workdir, rel)
        if not os.path.exists(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)


def _git_state():
    def git(*a):
        try:
            r = subprocess.run(["git", "-C", REPO, *a], capture_output=True, text=True, timeout=10)
            return r.stdout.strip() if r.returncode == 0 else None
        except Exception:
            return None
    st = git("status", "--porcelain", "--untracked-files=no")
    return {"commit": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": None if st is None else bool(st)}


def _file_record(path):
    return {"sha256": sha256(path), "bytes": os.path.getsize(path)}


def task_inputs(t, by_id, env):
    """Files a task reads: its deps' outputs plus declared extra inputs.

    Figure generators read only the bundles named in their arguments.
    """
    if t.block == "figures" and t.args:
        paths = [a for a in t.args if a in BUNDLE_MAP]   # + t.inputs below (secondary reads)
    else:
        paths = [o for d in t.deps for o in by_id[d].outputs]
    for p in t.inputs:
        if p.startswith("?"):          # optional: recorded only if present
            p = p[1:]
        if p.startswith("$BPP_EXT/"):
            p = os.path.join(env.get("BPP_EXT", ""), p[len("$BPP_EXT/"):])
        paths.append(p)
    return list(dict.fromkeys(paths))


def _utc():
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _read_runtime(path, attempts=5):
    """Read the runtime report the script wrote at exit.

    Provenance must never fail a finished task: on network filesystems
    (pythia's /project) a just-written file can briefly fail to open even
    after os.path.exists() succeeds, so retry, then record why it is missing.
    """
    err = None
    for i in range(attempts):
        try:
            with open(path) as f:
                info = json.load(f)
            try:
                os.remove(path)
            except OSError:
                pass
            return info
        except FileNotFoundError as e:
            err = e
        except (OSError, ValueError) as e:  # partial write / transient NFS error
            err = e
        time.sleep(1 + i)
    return {"unavailable": f"{type(err).__name__}: {err}"}


def run_task(t, cwd, smoke=False, log_dir=None, by_id=None):
    """Run one task in `cwd` and write its provenance manifest.

    manifests/<task>.json records: code version (git commit, dirty flag,
    sha256 of the script and uv.lock), the exact env and seeds, sha256 of
    every input and output, start/end time and wall clock, the Slurm job,
    and the runtime the script itself reports (GPU model/driver/memory,
    CUDA/cuDNN/torch versions, peak GPU memory; see mlye.seeding).
    """
    by_id = by_id or {}
    env = task_env(t, smoke)
    os.makedirs(os.path.join(cwd, MANIFEST_DIR), exist_ok=True)
    runtime_path = os.path.join(cwd, MANIFEST_DIR, f".{t.id}.runtime.json")
    if os.path.exists(runtime_path):
        os.remove(runtime_path)
    env["MLYE_RUNTIME_INFO"] = runtime_path
    script = os.path.join(REPO, "scripts", t.script)
    cmd = [sys.executable, script, *t.args]
    log_dir = log_dir or os.path.join(cwd, LOG_DIR)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{t.id}.log")

    in_paths = task_inputs(t, by_id, env)
    optional = {p[1:] for p in t.inputs if p.startswith("?")}
    inputs = {}
    for p in in_paths:
        full = p if os.path.isabs(p) else os.path.join(REPO if p.startswith("scripts/") else cwd, p)
        if os.path.exists(full):
            inputs[p] = _file_record(full)
        elif p not in optional:
            inputs[p] = None

    for o in t.outputs:  # scripts may assume their output folders exist
        os.makedirs(os.path.dirname(os.path.join(cwd, o)), exist_ok=True)
    started = _utc()
    t0 = time.time()
    with open(log_path, "w") as log:
        log.write(f"# {t.id}\n# cmd: {' '.join(cmd)}\n# env: {json.dumps({**t.env, **(t.smoke_env if smoke else {})})}\n")
        log.flush()
        rc = subprocess.run(cmd, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    wall = time.time() - t0
    finished = _utc()
    missing = [o for o in t.outputs if not os.path.exists(os.path.join(cwd, o))]
    if t.id == "bpp-data" and rc == 0 and not missing:
        got = sha256(os.path.join(cwd, MATRIX))
        if got != BPP_SHA256:
            print(f"  WARNING {MATRIX} sha256 {got} != expected {BPP_SHA256}")

    with open(script) as f:
        reports_runtime = "seed_everything(" in f.read()
    runtime = _read_runtime(runtime_path) if reports_runtime else None
    run_env = {**t.env, **(t.smoke_env if smoke else {})}
    if "OUT_NAME" in env and "OUT_NAME" not in run_env:
        run_env["OUT_NAME"] = env["OUT_NAME"]
    manifest = {
        "task": t.id, "block": t.block, "note": t.note, "smoke": smoke,
        "returncode": rc,
        "timing": {"started_utc": started, "finished_utc": finished, "wall_s": round(wall, 1)},
        "code": {**_git_state(),
                 "script": f"scripts/{t.script}",
                 "script_sha256": sha256(script),
                 "uv_lock_sha256": (sha256(os.path.join(REPO, "uv.lock"))
                                    if os.path.exists(os.path.join(REPO, "uv.lock")) else None)},
        "env": {**BASE_ENV, **run_env},
        "seeds": {k: v for k, v in run_env.items() if "SEED" in k},
        "host": {"node": os.uname().nodename,
                 **{k.lower(): os.environ[k] for k in
                    ("SLURM_JOB_ID", "SLURM_JOB_PARTITION", "SLURM_JOB_NODELIST",
                     "SLURM_CLUSTER_NAME") if k in os.environ}},
        "runtime": runtime,
        "deps": list(t.deps),
        "inputs": inputs,
        "outputs": {o: _file_record(os.path.join(cwd, o)) for o in t.outputs
                    if os.path.exists(os.path.join(cwd, o))},
    }
    # Kept out of the output folders: several exporters glob *.json there.
    with open(manifest_path(t, cwd), "w") as f:
        json.dump(manifest, f, indent=1)
    return rc, missing, wall, log_path


def record_preexisting(t, cwd, smoke):
    """Manifest for outputs that were already present and not produced here."""
    os.makedirs(os.path.join(cwd, MANIFEST_DIR), exist_ok=True)
    with open(manifest_path(t, cwd), "w") as f:
        json.dump({
            "task": t.id, "block": t.block, "smoke": smoke, "returncode": None,
            "mode": "pre-existing: outputs were present and not produced in this run",
            "timing": {"wall_s": 0.0}, "code": _git_state(),
            "host": {"node": os.uname().nodename}, "deps": [], "inputs": {},
            "outputs": {o: _file_record(os.path.join(cwd, o)) for o in t.outputs},
        }, f, indent=1)


def manifest_path(t, cwd):
    return os.path.join(cwd, MANIFEST_DIR, f"{t.id}.json")


def outputs_exist(t, cwd):
    return all(os.path.exists(os.path.join(cwd, o)) for o in t.outputs)


def select(tasks, block=None, only=None):
    if block:
        tasks = [t for t in tasks if t.block in block.split(",")]
    if only:
        pats = only.split(",")
        tasks = [t for t in tasks if any(fnmatch.fnmatch(t.id, p) for p in pats)]
    return tasks


def cmd_run(args):
    cwd = os.path.abspath(args.workdir) if args.workdir else REPO
    if args.workdir:
        prepare_workdir(cwd, args.with_bootstrap)
    blocks = args.block.split(",") if args.block else None
    tasks = toposort(all_tasks(args.smoke, args.with_bootstrap, blocks))
    layers = {int(x) for x in args.layer.split(",")} if args.layer else {1, 2, 3}
    chosen = {t.id for t in select(tasks, None, args.only) if t.layer in layers}
    by_id = {t.id: t for t in tasks}
    failed, written = set(), set()
    for t in tasks:
        if t.id not in chosen:
            continue
        # make-like: rerun when an input was rewritten earlier in this run, even
        # if the outputs exist (they may be the tracked published versions)
        stale = written & set(task_inputs(t, by_id, task_env(t, args.smoke)))
        if outputs_exist(t, cwd) and not stale and not args.force:
            print(f"[skip] {t.id} (outputs exist)")
            if not os.path.exists(manifest_path(t, cwd)):
                record_preexisting(t, cwd, args.smoke)
            continue
        bad = [d for d in t.deps if d in failed]
        if bad:
            print(f"[blocked] {t.id} (failed deps: {bad})")
            failed.add(t.id)
            continue
        if args.dry_run:
            print(f"[dry] {t.id}: {t.script} {t.env}")
            written.update(t.outputs)
            continue
        print(f"[run] {t.id} ...", flush=True)
        rc, missing, wall, log = run_task(t, cwd, args.smoke, by_id=by_id)
        if rc != 0 or missing:
            failed.add(t.id)
            print(f"[FAIL] {t.id} rc={rc} missing={missing} ({wall:.0f}s) log: {log}", flush=True)
        else:
            written.update(t.outputs)
            print(f"[ok] {t.id} ({wall:.0f}s)", flush=True)
    if failed:
        print(f"\n{len(failed)} task(s) failed or blocked: {sorted(failed)}")
        sys.exit(1)


def _copy_to_hut_outputs(t, cwd):
    out_dir = os.environ.get("SCRIPTHUT_OUTPUT_DIR")
    if not out_dir:
        return
    for o in t.outputs:
        p = os.path.join(cwd, o)
        if os.path.exists(p):
            shutil.copy2(p, out_dir)
    if os.path.exists(manifest_path(t, cwd)):
        shutil.copy2(manifest_path(t, cwd), os.path.join(out_dir, f"{t.id}.manifest.json"))


def cmd_exec(args):
    """Run a single task in the current directory (scripthut entry point)."""
    tasks = {t.id: t for t in all_tasks(args.smoke, with_bootstrap=args.with_bootstrap)}
    t = tasks[args.task_id]
    cwd = os.getcwd()
    if t.id == "bpp-data":
        # Prefer a prebuilt matrix with the right checksum (e.g. synced to the
        # cluster) over rebuilding from the ICPSR files.
        pre = os.path.join(os.environ.get("MLYE_DATA_DIR", os.path.expanduser("~/data")), "bpp_y_matrix.npy")
        dst = os.path.join(cwd, MATRIX)
        ext = task_env(t, args.smoke)["BPP_EXT"]
        if not os.path.exists(os.path.join(ext, "data.dta")) and os.path.exists(pre) \
                and sha256(pre) == BPP_SHA256:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            started = _utc()
            shutil.copy2(pre, dst)
            os.makedirs(os.path.join(cwd, MANIFEST_DIR), exist_ok=True)
            with open(manifest_path(t, cwd), "w") as f:
                json.dump({
                    "task": t.id, "block": t.block, "smoke": args.smoke, "returncode": 0,
                    "mode": "copied prebuilt matrix (ICPSR files not on this host)",
                    "timing": {"started_utc": started, "finished_utc": _utc(), "wall_s": 0.0},
                    "code": _git_state(), "host": {"node": os.uname().nodename},
                    "deps": [], "inputs": {pre: _file_record(pre)},
                    "outputs": {t.outputs[0]: _file_record(dst)},
                }, f, indent=1)
            print(f"copied prebuilt {pre} (sha256 ok)")
            _copy_to_hut_outputs(t, cwd)
            return
    missing_deps = [o for d in t.deps for o in tasks[d].outputs
                    if not os.path.exists(os.path.join(cwd, o))]
    if missing_deps:
        sys.exit(f"{t.id}: missing upstream outputs {missing_deps}")
    rc, missing, wall, log = run_task(t, cwd, args.smoke, by_id=tasks)
    with open(log) as f:
        sys.stdout.write(f.read())
    _copy_to_hut_outputs(t, cwd)
    if rc != 0 or missing:
        sys.exit(f"{t.id} failed: rc={rc} missing={missing}")


def _workflow(title, desc, tasks, flag, with_bootstrap=False):
    wf = {"title": title, "description": desc, "tasks": []}
    for t in tasks:
        task = {
            "id": t.id, "name": t.id,
            "command": f"python3 replicate.py exec {t.id}{flag}"
                       + (" --with-bootstrap" if with_bootstrap else ""),
            "partition": t.partition, "cpus": 2, "memory": t.mem,
            "time_limit": t.time,
            "env": [{"stacks": [STACK]}],
        }
        if t.gpu:
            task["gres"] = "gpu:1"
        if t.deps:
            task["dependencies"] = list(t.deps)
        wf["tasks"].append(task)
    # Final task: ship bundles + manifests + PROVENANCE.json with the run.
    block = tasks[0].block
    wf["tasks"].append({
        "id": f"{block}-collect", "name": f"{block}-collect",
        "command": f'python3 replicate.py collect --block {block} --out "$SCRIPTHUT_OUTPUT_DIR"',
        "partition": "standard_l40s", "cpus": 1, "memory": "2G", "time_limit": "00:10:00",
        "env": [{"stacks": [STACK]}],
        "dependencies": [t.id for t in tasks],
    })
    return wf


def cmd_hut(args):
    """Write workflows/hut.replication_<block>[_smoke].json.

    One workflow per block, plus bpp with the bootstrap stage; each in a full
    and a smoke variant. scripthut discovers them through the source's
    `workflows_glob` (e.g. `**/hut.*.json`).
    """
    os.makedirs(os.path.join(REPO, WORKFLOW_DIR), exist_ok=True)
    variants = [(b, b, False) for b in BLOCKS] + [("bpp_with_bootstrap", "bpp", True)]
    for smoke in (False, True):
        flag = " --smoke" if smoke else ""
        desc = ("Generated by `python replicate.py hut`. Do not edit; "
                "edit replicate.py and regenerate.")
        for name, block, boot in variants:
            tasks = all_tasks(smoke, boot, [block])
            wf = _workflow(f"replication: {name}{' (smoke)' if smoke else ''}", desc, tasks, flag, boot)
            fname = f"hut.replication_{name}{'_smoke' if smoke else ''}.json"
            with open(os.path.join(REPO, WORKFLOW_DIR, fname), "w") as f:
                json.dump(wf, f, indent=1)
            print(f"wrote {WORKFLOW_DIR}/{fname}  ({len(tasks)} tasks)")


def cmd_list(args):
    blocks = args.block.split(",") if args.block else None
    tasks = all_tasks(args.smoke, args.with_bootstrap, blocks)
    for b in blocks or ALL_BLOCKS:
        bt = [t for t in tasks if t.block == b]
        print(f"== {b}: {len(bt)} tasks  (layer {'/'.join(sorted({str(t.layer) for t in bt}))})")
        for t in bt:
            print(f"  {t.id:60s} {t.script:45s} {t.partition:15s} {t.time}"
                  + (f"  deps={len(t.deps)}" if t.deps else ""))


def cmd_verify_data(args):
    p = os.path.join(REPO, MATRIX)
    if not os.path.exists(p):
        sys.exit(f"{MATRIX} missing; run: python replicate.py run --layer 1")
    got = sha256(p)
    print(f"{p}: {got}  {'OK' if got == BPP_SHA256 else 'MISMATCH (expected ' + BPP_SHA256 + ')'}")
    sys.exit(0 if got == BPP_SHA256 else 1)


def build_provenance(cwd, blocks=None):
    """Trace every regenerated bundle back through the task manifests.

    For each bundle: its sha256, the producing task, every upstream task
    (transitively, via the deps recorded in the manifests) with its GPU,
    wall clock and code version, and the external inputs (data, raw ICPSR
    files, reference aggregates). The chain is checked: each recorded input
    hash must equal the output hash recorded by the task that produced it,
    and each output must still hash to what its manifest says.
    """
    mdir = os.path.join(cwd, MANIFEST_DIR)
    manifests = {}
    if os.path.isdir(mdir):
        for f in sorted(os.listdir(mdir)):
            if f.endswith(".json") and not f.startswith("."):
                with open(os.path.join(mdir, f)) as fh:
                    m = json.load(fh)
                manifests[m["task"]] = m
    producer = {o: tid for tid, m in manifests.items() for o in m.get("outputs", {})}
    report = {"generated_utc": _utc(), "workdir": cwd, "code_now": _git_state(),
              "bundles": {}, "figures": {}}
    artifacts = [(src, name, "bundles") for src, name in BUNDLE_MAP.items()]
    artifacts += [(f"{FIG_DIR}/figures.pdf", "figures.pdf", "figures")]
    artifacts += [(f"{FIG_DIR}/pdf/{f['name']}.pdf", f"{f['name']}.pdf", "figures") for f in FIGURES]
    for src, name, kind in artifacts:
        p = os.path.join(cwd, src)
        tid = producer.get(src)
        if not os.path.exists(p) or tid is None:
            continue
        if kind == "figures" and name != "figures.pdf":
            # a single figure: its own fragment (+ preamble), not every fragment
            roots = [f"{FIG_DIR}/tex/{name[:-4]}.tex", "scripts/fig_preamble.tex"]
        else:
            roots = [src]
        if blocks and manifests[tid].get("block") not in blocks:
            continue
        # File-level lineage: from each file to the task that wrote it, then to
        # the inputs that task read. A bundle no task wrote in this tree is the
        # published version: an external input, recorded with its hash.
        closure, seen_files, stack = [], set(), list(roots)
        while stack:
            f = stack.pop()
            if f in seen_files:
                continue
            seen_files.add(f)
            x = producer.get(f)
            if x is None:
                continue
            if x not in closure:
                closure.append(x)
            stack.extend((manifests[x].get("inputs") or {}).keys())
        problems, external = [], {}
        missing_manifests = sorted({d for x in closure for d in manifests[x].get("deps", [])
                                    if d not in manifests})
        # A dep that did not run here is fine if every file it would produce
        # was hashed as an input (e.g. the shipped bootstrap aggregates when
        # the bootstrap stage is off); then that file is an external input.
        known = {t.id: t for t in all_tasks(with_bootstrap=True)}
        hashed = {ip for x in closure for ip, rec in (manifests[x].get("inputs") or {}).items() if rec}
        for d in missing_manifests:
            outs = known[d].outputs if d in known else []
            if not outs or not all(o in hashed for o in outs):
                problems.append(f"no manifest for upstream task {d}")
        for x in closure:
            m = manifests[x]
            if m.get("returncode") not in (0, None):
                problems.append(f"{x}: returncode {m['returncode']}")
            for o, rec in m.get("outputs", {}).items():
                fp = os.path.join(cwd, o)
                if os.path.exists(fp) and sha256(fp) != rec["sha256"]:
                    problems.append(f"{o}: changed since {x} wrote it")
            for ip, rec in (m.get("inputs") or {}).items():
                if rec is None:
                    problems.append(f"{x}: input {ip} was missing")
                    continue
                up = producer.get(ip)
                if up is None:
                    external[ip] = rec["sha256"]
                elif manifests[up]["outputs"][ip]["sha256"] != rec["sha256"]:
                    problems.append(f"{x}: input {ip} does not match the output of {up}")
        gpu_s, commits = {}, set()
        for x in closure:
            m = manifests[x]
            gname = ((m.get("runtime") or {}).get("gpu") or {}).get("name", "cpu")
            gpu_s[gname] = round(gpu_s.get(gname, 0.0) + m.get("timing", {}).get("wall_s", 0.0), 1)
            c = m.get("code") or {}
            commits.add(f"{(c.get('commit') or '?')[:12]}{'+dirty' if c.get('dirty') else ''}")
        report[kind][name] = {
            "path": src, "sha256": sha256(p), "producer": tid,
            "n_tasks": len(closure), "wall_s_by_device": gpu_s,
            "code_commits": sorted(commits),
            "smoke": any(manifests[x].get("smoke") for x in closure),
            "preexisting": sorted(x for x in closure
                                  if str(manifests[x].get("mode", "")).startswith("pre-existing")),
            "external_inputs": external,
            "chain_ok": not problems, "problems": problems,
            "tasks": {x: {"wall_s": manifests[x].get("timing", {}).get("wall_s"),
                          "gpu": ((manifests[x].get("runtime") or {}).get("gpu") or {}).get("name"),
                          "node": (manifests[x].get("host") or {}).get("node"),
                          "outputs": {o: r["sha256"] for o, r in manifests[x].get("outputs", {}).items()}}
                      for x in sorted(closure)},
        }
    return report


def cmd_collect(args):
    """Copy bundles, figures, manifests and PROVENANCE.json to an output dir."""
    cwd = os.path.abspath(args.workdir) if args.workdir else REPO
    dst = os.path.abspath(args.out) if args.out else os.path.join(cwd, OUT)
    blocks = args.block.split(",") if args.block else None
    report = build_provenance(cwd, blocks)
    entries = [(kind, name, b) for kind in ("bundles", "figures") for name, b in report[kind].items()]
    in_place = os.path.abspath(dst) == os.path.abspath(os.path.join(cwd, OUT))
    for kind, name, b in entries:
        if not in_place:  # e.g. --out "$SCRIPTHUT_OUTPUT_DIR": ship copies with the run
            target = os.path.join(dst, kind, name)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            shutil.copy2(os.path.join(cwd, b["path"]), target)
        print(f"  {kind[:-1]:6s} {name:48s} tasks={b['n_tasks']:3d} "
              f"chain={'ok' if b['chain_ok'] else 'BROKEN'} devices={b['wall_s_by_device']}")
        for pr in b["problems"][:5]:
            print(f"      ! {pr}")
    mdir = os.path.join(cwd, MANIFEST_DIR)
    if os.path.isdir(mdir) and not in_place:
        os.makedirs(os.path.join(dst, "manifests"), exist_ok=True)
        keep = {x for _, _, b in entries for x in b["tasks"]}
        for f in os.listdir(mdir):
            if f.endswith(".json") and not f.startswith(".") and (not blocks or f[:-5] in keep):
                shutil.copy2(os.path.join(mdir, f), os.path.join(dst, "manifests", f))
    with open(os.path.join(dst, "PROVENANCE.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(f"wrote {os.path.join(dst, 'PROVENANCE.json')}  "
          f"({len(report['bundles'])} bundles, {len(report['figures'])} figure artifacts)")
    if args.strict and not all(b["chain_ok"] for _, _, b in entries):
        sys.exit(1)


# ----------------------------------------------------------------------
# Run summary: dependency graph + compute used, for the README
# ----------------------------------------------------------------------
SUMMARY_BEGIN = "<!-- run-summary:begin (generated by `python replicate.py summary --readme`) -->"
SUMMARY_END = "<!-- run-summary:end -->"
# Not in the default run; measured on the original cluster runs (Jul 2026):
# ~63 min per replicate on an H100, 2 x 100 replicates.
BOOTSTRAP_H100_HOURS_PER_REP = 63 / 60
BOOTSTRAP_REPS = 200


def artifact_labels():
    """Paper numbering of the layer-3 artifacts ("Figure 3", "Table B1", ...)."""
    return {f["name"]: ("Figure " if f["number"].startswith("fig") else "Table ")
            + f["number"][3:] + (" (web appendix)" if f.get("online") else "")
            for f in FIGURES}


def figure_bundles(fig):
    return ([fig["bundle"]] if fig["bundle"] else []) + list(fig.get("reads", []))


def _lineage(manifests, start):
    """Task ids upstream of file `start` (file-level walk over manifests)."""
    producer = {o: t for t, m in manifests.items() for o in m.get("outputs", {})}
    seen, tasks, stack = set(), set(), [start]
    while stack:
        f = stack.pop()
        if f in seen:
            continue
        seen.add(f)
        t = producer.get(f)
        if t is None or t in tasks:
            continue
        tasks.add(t)
        stack.extend((manifests[t].get("inputs") or {}).keys())
    return tasks


def _hours(manifests, tasks):
    by_dev = {}
    for t in tasks:
        m = manifests[t]
        dev = ((m.get("runtime") or {}).get("gpu") or {}).get("name")
        dev = dev.replace("NVIDIA ", "").replace(" 80GB HBM3", "") if dev else "CPU"
        by_dev[dev] = by_dev.get(dev, 0.0) + (m.get("timing") or {}).get("wall_s", 0.0) / 3600
    return by_dev


def _fmt_hours(by_dev):
    gpu = {d: h for d, h in by_dev.items() if d != "CPU"}
    if not gpu:
        return "CPU only"
    return ", ".join(f"{h:.1f} h {d}" for d, h in sorted(gpu.items(), key=lambda x: -x[1]))


def render_summary(manifests):
    labels = artifact_labels()
    block_of = {}
    for t in all_tasks(with_bootstrap=True):
        for o in t.outputs:
            block_of[o] = t.block
    used_by = {}
    for f in FIGURES:
        for b in figure_bundles(f):
            used_by.setdefault(b, []).append(f["name"])
    commits = sorted({(m.get("code") or {}).get("commit", "")[:7] for m in manifests.values()} - {""})
    starts = [m["timing"]["started_utc"] for m in manifests.values() if (m.get("timing") or {}).get("started_utc")]
    ends = [m["timing"]["finished_utc"] for m in manifests.values() if (m.get("timing") or {}).get("finished_utc")]

    L = [SUMMARY_BEGIN, "", "### Run summary", ""]
    L.append(f"Measured on the replication run of {min(starts)[:10]} "
             f"(commit `{'`, `'.join(commits)}`, {len(manifests)} tasks with manifests). "
             "GPU-hours are task wall-clock time on the GPU type the task actually ran on.")
    L += ["", "#### Dependency graph", "", "```mermaid", "flowchart LR"]
    L.append('  icpsr[("ICPSR 210782<br/>data.dta, tax9192.dta, natpr.dta")]')
    L.append(f'  matrix["{MATRIX}"]')
    L.append('  icpsr --> matrix')
    blocks = ["data", "ar1n", "hetero", "hockey", "timing", "bpp"]
    nid = lambda s: "".join(ch if ch.isalnum() else "_" for ch in s)
    for b in blocks:
        tasks = [t for t, m in manifests.items() if m.get("block") == b]
        L.append(f'  blk_{b}{{{{"{b}<br/>{len(tasks)} task{"s" if len(tasks) != 1 else ""} · {_fmt_hours(_hours(manifests, tasks)).replace(", ", "<br/>")}"}}}}')
    L.append('  boot{{"bootstrap 2×100 IVI<br/>NOT regenerated<br/>~%.0f h H100 est."}}'
             % (BOOTSTRAP_REPS * BOOTSTRAP_H100_HOURS_PER_REP))
    L.append('  icpsr --> blk_data')
    L.append('  matrix --> blk_bpp')
    L.append('  matrix -.-> boot')
    for src, name in BUNDLE_MAP.items():
        blk = block_of.get(src, "?")
        L.append(f'  b_{nid(name)}[/"{name}"/]')
        L.append(f'  blk_{blk} --> b_{nid(name)}')
    L.append(f'  boot -. "bootstrap SEs" .-> b_{nid("bundle_psid_heteroscale.json")}')
    L.append('  fixed(["fixed content<br/>(diagrams, settings table)"])')
    for f in FIGURES:
        L.append(f'  f_{f["name"]}["{labels[f["name"]]}<br/>{f["name"]}"]')
        for b in figure_bundles(f):
            L.append(f'  b_{nid(b)} --> f_{f["name"]}')
        if not figure_bundles(f):
            L.append(f'  fixed --> f_{f["name"]}')
    L += ["  classDef bundle fill:#eef,stroke:#669", "  classDef todo stroke-dasharray: 5 5",
          "  class " + ",".join(f"b_{nid(n)}" for n in BUNDLE_MAP.values()) + " bundle",
          "  class boot todo", "```", ""]

    L += ["#### Compute by block", "",
          "| Block | Tasks | GPU-hours | Longest task | Elapsed (first start → last finish) |",
          "|---|---:|---|---|---|"]
    for b in blocks:
        tasks = [t for t, m in manifests.items() if m.get("block") == b]
        if not tasks:
            continue
        lt = max(tasks, key=lambda t: manifests[t]["timing"].get("wall_s", 0))
        st = min(manifests[t]["timing"]["started_utc"] for t in tasks)
        en = max(manifests[t]["timing"]["finished_utc"] for t in tasks)
        import datetime as _dt
        el = (_dt.datetime.fromisoformat(en) - _dt.datetime.fromisoformat(st)).total_seconds() / 3600
        L.append(f"| {b} | {len(tasks)} | {_fmt_hours(_hours(manifests, tasks))} | "
                 f"`{lt}` ({manifests[lt]['timing']['wall_s'] / 3600:.1f} h) | {el:.1f} h |")
    allh = _hours(manifests, list(manifests))
    L.append(f"| **total** | **{len(manifests)}** | **{_fmt_hours(allh)}** "
             f"(= {sum(h for d, h in allh.items() if d != 'CPU'):.0f} GPU-hours) | | |")
    L.append(f"| bootstrap (not regenerated) | {BOOTSTRAP_REPS} + 2 | "
             f"~{BOOTSTRAP_REPS * BOOTSTRAP_H100_HOURS_PER_REP:.0f} h H100 (estimate) | | |")
    L += ["", "The data matrix (`bpp-data`, in the bpp block) and the sample statistics (`data` "
          "block) run on a CPU in minutes; so does layer 3 (figures).", ""]

    L += ["#### Bundles and where they are used", "",
          "| Bundle | Block | Tasks in lineage | GPU-hours in lineage | Used by |", "|---|---|---:|---|---|"]
    for src, name in BUNDLE_MAP.items():
        lin = _lineage(manifests, src)
        users = ", ".join(f"{labels[n]} (`{n}`)" for n in used_by.get(name, [])) or "—"
        hrs = _fmt_hours(_hours(manifests, lin)) if lin else "not in this run"
        L.append(f"| `{name}` | {block_of.get(src, '?')} | {len(lin) if lin else '—'} | {hrs} | {users} |")
    L += ["", "Lineages overlap (e.g. the ar1n Picard cell feeds four bundles), so the "
          "per-bundle hours do not add up to the block totals.", ""]

    L += ["#### Figures and tables", "", "| Artifact | Generator | Bundles | Bootstrap |", "|---|---|---|---|"]
    boot_use = {"psid_parameter_table": "IVI-row standard errors (100 replicates)",
                "psid_a_sigma_z1_eps": "95% band around the IVI curves (100 replicates)"}
    for f in FIGURES:
        L.append(f"| {labels[f['name']]} | `scripts/fig_{f['name']}.py` | "
                 + (", ".join(f"`{b}`" for b in figure_bundles(f)) or "fixed content")
                 + f" | {boot_use.get(f['name'], '—')} |")
    L += ["", "The bootstrap quantities come from the shipped "
          f"`{BOOT_UNCOND}/bootstrap_theta.json` until the "
          "bootstrap stage is rerun (`hut.replication_bpp_with_bootstrap.json`).", "", SUMMARY_END]
    return "\n".join(L) + "\n"


# Paths recorded by runs made before the output/ reorganisation (e.g. the
# 2026-10-05 pythia run), so `summary` still reads their manifests.
_LEGACY_BUNDLE_PATHS = {
    "results/_simulation_ar1n_sweep/ar1n_parameter_summary.json": "ar1n_parameter_summary.json",
    "results/_simulation_ar1n_mu1_profile/ar1n_mu1_profile_bundle.json": "ar1n_mu1_profile_bundle.json",
    "results/_simulation_ar1n_binding_mu1/ar1n_binding_mu1_bundle.json": "ar1n_binding_mu1_bundle.json",
    "results/_simulation_ar1n_quiver/ar1n_quiver_bundle.json": "ar1n_quiver_bundle.json",
    "results/_simulation_hetero_scale_ivi_sweep/hetero_scale_parameter_summary.json": "hetero_scale_parameter_summary.json",
    "results/_simulation_hockeystick_ivi_sweep/hockeystick_parameter_summary.json": "hockeystick_parameter_summary.json",
    "results/_simulation_hockeystick_t40_ivi_sweep/hockeystick_parameter_summary.json": "hockeystick_parameter_summary_longt.json",
    "results/_contour_past_2period_imputed/contour_bundle.json": "contour_bundle.json",
    "results/_time_to_convergence_hockey/time-to-convergence-hockey-bundle.json": "time-to-convergence-hockey-bundle.json",
    "results/employment-bpp-hetero-scale-bundle/bundle_psid_heteroscale.json": "bundle_psid_heteroscale.json",
    "results/employment-bpp-hetero-mean-indep-bundle/bpp_hetero_mean_indep_bundle.json": "2026-07-13-bpp-hetero-mean-indep-bundle.json",
    "results/employment-bpp-selection-table/bpp_selection_table.json": "bpp_selection_table.json",
}


def _modern_path(p):
    if p in _LEGACY_BUNDLE_PATHS:
        return BUNDLE_PATH[_LEGACY_BUNDLE_PATHS[p]]
    if p == "data/bpp_y_matrix.npy":
        return MATRIX
    return CELLS + p[len("results"):] if p.startswith("results/") else p


def _modernise(m):
    for k in ("inputs", "outputs"):
        if m.get(k):
            m[k] = {_modern_path(p): v for p, v in m[k].items()}
    return m


def cmd_summary(args):
    mdir = os.path.join(os.path.abspath(args.workdir) if args.workdir else REPO, MANIFEST_DIR)
    manifests = {}
    for f in sorted(os.listdir(mdir)):
        if f.endswith(".json") and not f.startswith("."):
            with open(os.path.join(mdir, f)) as fh:
                m = json.load(fh)
            if m.get("block") in BLOCKS + ["data"] and m.get("timing", {}).get("started_utc"):
                manifests[m["task"]] = _modernise(m)
    if not manifests:
        sys.exit(f"no layer-1/2 task manifests with timing found in {mdir}")
    md = render_summary(manifests)
    if args.readme:
        readme = os.path.join(REPO, "README.md")
        with open(readme) as f:
            text = f.read()
        if SUMMARY_BEGIN in text:
            i, j = text.index(SUMMARY_BEGIN), text.index(SUMMARY_END) + len(SUMMARY_END)
            text = text[:i] + md.rstrip("\n") + text[j:]
        else:
            anchor = "## 4. Reproducibility and pipeline management"
            text = text.replace(anchor, md + "\n" + anchor, 1)
        with open(readme, "w") as f:
            f.write(text)
        print(f"updated {readme}")
    else:
        sys.stdout.write(md)


# Fields that legitimately differ between runs (time stamps, wall clocks,
# provenance strings).
VOLATILE = {"generated_at", "created_utc", "git_hash", "run_id", "backend",
            "produced_by", "wall_time", "wall_time_s", "wall_s", "elapsed_s",
            "total_wall_s", "converged_wall_s", "timestamp", "peak_gpu_mem_mb",
            "gpu_util", "device", "host"}


def _compare(a, b, path, stats, rtol, atol):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in set(a) | set(b):
            if k in VOLATILE or k.startswith("wall") or "_wall" in k:
                continue  # timings are hardware/load dependent
            if k not in a or k not in b:
                if a.get(k) is None and b.get(k) is None:
                    continue  # absent == null
                stats["structure"].append(f"{path}.{k} only in {'reference' if k in a else 'new'}")
                continue
            _compare(a[k], b[k], f"{path}.{k}", stats, rtol, atol)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            stats["structure"].append(f"{path}: length {len(a)} vs {len(b)}")
            return
        for i, (x, y) in enumerate(zip(a, b)):
            _compare(x, y, f"{path}[{i}]", stats, rtol, atol)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) \
            and not isinstance(a, bool) and not isinstance(b, bool):
        stats["n"] += 1
        if (a != a) and (b != b):
            return
        if not math.isclose(a, b, rel_tol=rtol, abs_tol=atol):
            stats["off"].append((path, a, b))
    elif a != b:
        stats["text"] += 1


def cmd_compare(args):
    """Regenerated bundles vs the published ones (as committed at git HEAD)."""
    cwd = os.path.abspath(args.workdir) if args.workdir else REPO
    bad = 0
    for src, name in BUNDLE_MAP.items():
        new_p = os.path.join(cwd, src)
        r = subprocess.run(["git", "-C", REPO, "show", f"{args.ref}:{src}"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(f"{name:48s} no published version at {args.ref}:{src}")
            continue
        if not os.path.exists(new_p):
            print(f"{name:48s} not regenerated")
            continue
        ref, new = json.loads(r.stdout), json.load(open(new_p))
        if ref == new:
            print(f"{name:48s} IDENTICAL to {args.ref}")
            continue
        stats = {"n": 0, "off": [], "structure": [], "text": 0}
        _compare(ref, new, "$", stats, args.rtol, args.atol)
        ok = not stats["off"] and not stats["structure"]
        bad += not ok
        print(f"{name:48s} {'OK  ' if ok else 'DIFF'} numbers={stats['n']} "
              f"outside_tol={len(stats['off'])} structure={len(stats['structure'])} "
              f"text_diffs={stats['text']}")
        if args.verbose:
            for s in stats["structure"][:10]:
                print(f"    structure: {s}")
            for p, x, y in stats["off"][:args.verbose]:
                print(f"    {p}: ref={x!r} new={y!r}")
    print(f"\nrtol={args.rtol} atol={args.atol}. Seeds are fixed, so differences come "
          "from GPU model / library versions (see README.md).")
    sys.exit(1 if bad and args.strict else 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, bootstrap=True):
        p.add_argument("--smoke", action="store_true",
                       help="tiny epochs/grids: exercises every code path in minutes")
        if bootstrap:
            p.add_argument("--with-bootstrap", action="store_true",
                           help="include the 2x100-replicate BPP bootstrap (~200 H100-hours)")

    p = sub.add_parser("list"); common(p); p.add_argument("--block")
    p = sub.add_parser("run"); common(p)
    p.add_argument("--block"); p.add_argument("--only", help="comma-separated task-id globs")
    p.add_argument("--layer", help="comma-separated layers: 1 data, 2 bundles, 3 figures (default all)")
    p.add_argument("--workdir", help="run in a scratch tree instead of the repo")
    p.add_argument("--force", action="store_true", help="rerun even if outputs exist")
    p.add_argument("--dry-run", action="store_true")
    sub.add_parser("hut")
    p = sub.add_parser("exec"); common(p); p.add_argument("task_id")
    sub.add_parser("verify-data")
    p = sub.add_parser("collect", help="write output/PROVENANCE.json; with --out also copy bundles, figures, manifests")
    p.add_argument("--workdir"); p.add_argument("--out"); p.add_argument("--block")
    p.add_argument("--strict", action="store_true", help="exit 1 if any provenance chain is broken")
    p = sub.add_parser("summary", help="dependency graph + compute per block/bundle from manifests")
    p.add_argument("--workdir", help="tree whose manifests/ to read (default: repo)")
    p.add_argument("--readme", action="store_true", help="write it into README.md")
    p = sub.add_parser("compare", help="regenerated bundles vs the published ones at git HEAD")
    p.add_argument("--workdir"); p.add_argument("--ref", default="HEAD", help="git ref of the published bundles")
    p.add_argument("--rtol", type=float, default=0.05)
    p.add_argument("--atol", type=float, default=0.02)
    p.add_argument("--verbose", type=int, default=0, metavar="N",
                   help="print up to N out-of-tolerance values per bundle")
    p.add_argument("--strict", action="store_true", help="exit 1 on any difference")
    args = ap.parse_args()
    {"list": cmd_list, "run": cmd_run, "hut": cmd_hut, "exec": cmd_exec, "summary": cmd_summary,
     "verify-data": cmd_verify_data, "collect": cmd_collect,
     "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    main()
