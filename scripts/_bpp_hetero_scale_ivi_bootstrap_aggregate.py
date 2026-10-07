r"""Aggregate the full-IVI bootstrap replicates (rep_*.json) into a
single artefact with the replicated parameter estimates and the summary
inference used in the journal / report.

Consumes: output/bundles/cells/_bpp_hetero_scale_ivi_bootstrap_k40/rep_<b>.json
  (one per replicate; resample seed 31000+b; K=40, Picard 0.6, 20
  outer iterations; theta^(b) = last iterate theta_20)
Produces: output/bundles/cells/_bpp_hetero_scale_ivi_bootstrap_k40/bootstrap_theta.json
  - 'replicates': B x 15 matrix of theta_K draws (row order = sorted
    replicate index; use with 'param_names' and 'rep_indices') — the
    raw material for delta-method-free inference on any transformation
    of the parameters (evaluate the transform on each row, take
    percentiles).
  - 'point_estimate': theta_K of the 50-iter K=40 Picard fit on the
    actual data (the reportable point estimate).
  - 'se', 'ci95': per-parameter bootstrap SD and percentile CI.
  - 'final_g_test': E[g]=0 check at the final iterate across
    replicates (per-component t stats + joint sum-of-t^2).

Rerunnable: safe to re-invoke as stragglers land; it aggregates
whatever rep_*.json files exist.

Env knobs:
    BPP_BOOT_AGG_DIR    replicate folder (default the conditional run,
        output/bundles/cells/_bpp_hetero_scale_ivi_bootstrap_k40; point at
        .../_bpp_hetero_scale_ivi_bootstrap_k40_uncond for the
        fresh-seed run)
    BPP_BOOT_AGG_PAIR   optional second replicate folder; when set, a
        'paired_decomposition' block is added: per-parameter SD of the
        per-replicate difference (the simulation-noise component, since
        household resamples are shared) and the mean difference with
        its t stat (the realized offset of the common CRN seed used by
        the folder given in BPP_BOOT_AGG_PAIR).
"""
import os
import sys
import json
import glob

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bpp_hetero_scale_ivi_jn import PARAM_NAMES  # noqa: E402

REP_DIR = os.environ.get("BPP_BOOT_AGG_DIR",
                         "output/bundles/cells/_bpp_hetero_scale_ivi_bootstrap_k40")
PAIR_DIR = os.environ.get("BPP_BOOT_AGG_PAIR", "")
POINT_JSON = ("output/bundles/cells/employment-bpp-hetero-scale-ivi-jn-k40-picard-a06-it50/"
              "bpp_hetero_scale_ivi_jn.json")
OUT_JSON = os.path.join(REP_DIR, "bootstrap_theta.json")
# Outer iterations each replicate must have completed (BPP_HETERO_SCALE_N_ITERS_OUTER
# of the replicate runs; 20 for the published bootstrap).
N_ITERS = int(os.environ.get("BPP_BOOT_N_ITERS", 20))


def main():
    reps, TH, G = [], [], []
    for f in sorted(glob.glob(os.path.join(REP_DIR, "rep_*.json"))):
        idx = int(os.path.basename(f).split('rep_')[1].split('.')[0])
        d = json.load(open(f))
        assert d['data']['resample_seed'] == 31000 + idx, f
        traj = d['estimates']['ivi']['trajectory']
        assert traj[-1]['iter'] == N_ITERS, f
        reps.append(idx)
        TH.append([d['estimates']['ivi']['theta_K'][p] for p in PARAM_NAMES])
        G.append([traj[-1]['g'][p] for p in PARAM_NAMES])
    TH, G = np.array(TH), np.array(G)
    B = len(reps)

    point = json.load(open(POINT_JSON))['estimates']['ivi']['theta_K']

    tstat = G.mean(0) / (G.std(0, ddof=1) / np.sqrt(B))
    out = {
        'meta': {
            'design': 'full-IVI nonparametric bootstrap, household '
                      'resample (seed 31000+b) of the N=741 BPP panel',
            'estimator': 'hetero-scale joint-normal IVI, K=40 CRN ELBO '
                         'draws, Picard alpha=0.6 (binding-equation '
                         f'form), {N_ITERS} outer iterations, theta^(b) = '
                         f'theta_{N_ITERS} (last iterate)',
            'runs': ['823d93c3 (68 reps)', '53ef9b9d (fill-in)'],
            'point_estimate_source': POINT_JSON,
            'B': B,
            'param_names': PARAM_NAMES,
        },
        'rep_indices': reps,
        'replicates': TH.tolist(),
        'point_estimate': {p: float(point[p]) for p in PARAM_NAMES},
        'boot_mean': {p: float(v) for p, v in zip(PARAM_NAMES, TH.mean(0))},
        'se': {p: float(v) for p, v in zip(PARAM_NAMES,
                                           TH.std(0, ddof=1))},
        'ci95': {p: [float(lo), float(hi)] for p, lo, hi in zip(
            PARAM_NAMES, np.percentile(TH, 2.5, axis=0),
            np.percentile(TH, 97.5, axis=0))},
        'final_g_test': {
            'per_component_t': {p: float(t)
                                for p, t in zip(PARAM_NAMES, tstat)},
            'sum_t2': float((tstat ** 2).sum()),
            'chi2_15_crit_95': 25.0,
            'mean_g_norm': float(np.linalg.norm(G.mean(0))),
            'mean_norm_g_per_rep': float(
                np.linalg.norm(G, axis=1).mean()),
        },
    }
    if PAIR_DIR:
        TH2, reps2 = [], []
        for f in sorted(glob.glob(os.path.join(PAIR_DIR, "rep_*.json"))):
            idx = int(os.path.basename(f).split('rep_')[1].split('.')[0])
            d = json.load(open(f))
            reps2.append(idx)
            TH2.append([d['estimates']['ivi']['theta_K'][p]
                        for p in PARAM_NAMES])
        common = sorted(set(reps) & set(reps2))
        A = np.array([TH[reps.index(i)] for i in common])
        Bm = np.array([TH2[reps2.index(i)] for i in common])
        D = A - Bm      # this-folder minus pair-folder, same resample
        tD = D.mean(0) / (D.std(0, ddof=1) / np.sqrt(len(common)))
        out['paired_decomposition'] = {
            'pair_dir': PAIR_DIR,
            'n_pairs': len(common),
            'sim_component_sd': {p: float(v) for p, v in zip(
                PARAM_NAMES, D.std(0, ddof=1))},
            'mean_diff': {p: float(v) for p, v in zip(
                PARAM_NAMES, D.mean(0))},
            'mean_diff_t': {p: float(v) for p, v in zip(PARAM_NAMES, tD)},
        }

    with open(OUT_JSON, 'w') as f:
        json.dump(out, f, indent=1)
    print(f"wrote {OUT_JSON}  (B={B})")
    print(f"{'param':17s} {'point':>9s} {'SE':>8s} {'CI2.5':>9s} {'CI97.5':>9s}")
    for p in PARAM_NAMES:
        lo, hi = out['ci95'][p]
        print(f"{p:17s} {out['point_estimate'][p]:9.4f} "
              f"{out['se'][p]:8.4f} {lo:9.4f} {hi:9.4f}")
    print(f"E[g]=0 joint: sum t^2 = {out['final_g_test']['sum_t2']:.1f} "
          f"(chi2_15 crit 25.0)")


if __name__ == "__main__":
    main()
