"""Wall time to convergence: VI and SMC-EM on the simulation-hockeystick DGP.

Sibling of `_time_to_convergence.py` (AR(1)n), targeting the hockey-stick
DGP from `specs/compute-simulation-hockeystick.md`. Differences:

- DGP: softplus-kink poly-2 mu + sinh-arcsinh z1 + sinh-arcsinh emission.
- Truth calibration: alpha_0 = -0.25, alpha_1 = 0.1, mu_1 = 0.9,
  sigma_0 = -1.8 (spec-tightened), sigma_2 = 0.35, gamma_1 = 0.34,
  gamma_2 = 0.89, psi_1 = 0.033, psi_2 = 0.47.
- Free-theta count: 12 (13-dim vector minus prior.z1_skew, which is
  pinned at 0 with requires_grad=False per the spec).
- Prior class: `MarkovNormalConditionalPolyPrior(law_model='softplus',
  z1_distr='sinh')`.
- Decoder: `MASinhEmission(theta=0, fix_theta=True, fix_beta=False)`
  (log_sigma and log_beta both trainable).

Otherwise identical to the AR(1)n script — cold-start Adam at lr=1e-2,
per-100-epoch snapshots, windowed-average parameter drift stopping rule
with CONV_TAU=0.02 / CONV_WINDOW=5 / CONV_HOLD=2, MAX_EPOCHS=15000.

Env vars: METHOD, N, T, H, K, LR, LOG_EVERY, CONV_TAU, CONV_WINDOW,
CONV_HOLD, MAX_EPOCHS, CLIP, OBS_SEED, VI_SEED, RESAMPLE. Same defaults
as `_time_to_convergence.py`.

Output: output/bundles/cells/_time_to_convergence_hockey/conv_hockey_<method>_N<n_tag>_T<T>_<param>.json
"""

import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import softplus_sinh_posterior_comparison as M
from smc_truth_profile import _Bundle
from smc_smoothers import select_smc_smoother
from gpu_util import GpuUtilSampler
from mlye.models.full_model import FullModel


OUT_DIR = Path("output/bundles/cells/_time_to_convergence_hockey")

# 12 free parameters — PARAM_NAMES minus 'z1_skew' (pinned at 0 per spec).
FREE_NAMES = [k for k in M.PARAM_NAMES if k != "z1_skew"]


def _env_int(name, default):
    return int(os.environ.get(name, default))

def _env_float(name, default):
    return float(os.environ.get(name, default))


# Hockey-stick truth calibration (specs/compute-simulation-hockeystick.md).
# Overrides M.TRUTH (which defaults to the paper old sigma_0 = -1.55) so
# simulate_from_truth uses the spec calibration.
HOCKEY_TRUTH = dict(M.TRUTH)
HOCKEY_TRUTH["alpha0"]        = -0.25
HOCKEY_TRUTH["log_alpha1"]    = float(np.log(0.10))
HOCKEY_TRUTH["mu0"]           =  0.0
HOCKEY_TRUTH["mu1"]           =  0.9
HOCKEY_TRUTH["mu2"]           =  0.0
HOCKEY_TRUTH["sigma0"]        = -1.8   # spec-tightened from paper -1.35
HOCKEY_TRUTH["sigma1"]        =  0.0
HOCKEY_TRUTH["sigma2"]        =  0.35
HOCKEY_TRUTH["z1_log_std"]    = float(np.log(np.exp(0.34) - 1.0))  # gamma_1 = 0.34
HOCKEY_TRUTH["z1_skew"]       =  0.0
HOCKEY_TRUTH["z1_log_tail"]   = float(-np.log(0.89))               # gamma_2 = 0.89
HOCKEY_TRUTH["log_sigma_eps"] = float(np.log(0.033))               # psi_1
HOCKEY_TRUTH["log_beta"]      = float(-np.log(0.47))               # psi_2
M.TRUTH = HOCKEY_TRUTH


CFG = {
    "method":       os.environ.get("METHOD"),
    "N":            _env_int("N", 1000),
    "T":            _env_int("T", 6),
    "H":            _env_int("H", 32),
    "K":            _env_int("K", 50),
    "lr":           _env_float("LR", 1e-2),
    "log_every":    _env_int("LOG_EVERY", 100),
    "conv_tau":     _env_float("CONV_TAU", 0.02),
    "conv_window":  _env_int("CONV_WINDOW", 5),
    "conv_hold":    _env_int("CONV_HOLD", 2),
    "max_epochs":   _env_int("MAX_EPOCHS", 15000),
    "clip":         _env_float("CLIP", 5.0),
    "obs_seed":     _env_int("OBS_SEED", 11),
    "vi_seed":      _env_int("VI_SEED", 11007),
    "resample":     os.environ.get("RESAMPLE", "systematic"),
    "dgp":          "hockeystick",
}
if CFG["method"] not in ("vi", "smc"):
    raise SystemExit(f"METHOD must be 'vi' or 'smc' (got {CFG['method']!r})")


def _git_hash():
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"],
                           capture_output=True, check=False, timeout=2)
        return r.stdout.decode().strip() or None
    except Exception:
        return None


def _peak_gpu_mem_mb(device):
    if device.type != "cuda":
        return None
    return float(torch.cuda.max_memory_allocated() / (1024 ** 2))


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def _pin_z1_skew(prior):
    """Pin `prior.z1_skew = 0` (requires_grad=False) per the hockey-stick spec."""
    with torch.no_grad():
        prior.z1_skew.fill_(0.0)
    prior.z1_skew.requires_grad_(False)


def _extract_free_vec(prior, decoder):
    from smc_truth_profile import _Bundle
    theta = M.extract_params(_Bundle(prior, decoder))
    return np.array([theta[k] for k in FREE_NAMES], dtype=np.float64), theta


def _windowed_avg_drift(snapshot_vecs, window):
    if len(snapshot_vecs) < 2 * window:
        return None
    recent = np.stack(snapshot_vecs[-window:], axis=0).mean(axis=0)
    older  = np.stack(snapshot_vecs[-2*window:-window], axis=0).mean(axis=0)
    return float(np.linalg.norm(recent - older))


# ----------------------------------------------------------------------
# VI
# ----------------------------------------------------------------------

def _train_vi(y_obs, cfg, device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(cfg["vi_seed"])
    np.random.seed(cfg["vi_seed"])

    encoder = M.build_encoder("joint_normal", cfg["T"], hidden_dim=cfg["H"]).to(device)
    prior, decoder = M.build_prior_decoder(cfg["T"], device)
    model = FullModel(encoder=encoder, decoder=decoder, prior=prior).to(device)
    _pin_z1_skew(model.prior)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"])

    truth_vec = np.array([HOCKEY_TRUTH[k] for k in FREE_NAMES], dtype=np.float64)
    init_vec, init_theta = _extract_free_vec(prior, decoder)

    snapshots = []
    snapshot_vecs = []
    hold_count = 0
    prev_snap_vec = init_vec.copy()
    sampler = GpuUtilSampler().reset().start()
    _sync(device); t0 = time.perf_counter()
    converged_at = None
    converged_wall = None
    for epoch in range(cfg["max_epochs"]):
        opt.zero_grad()
        loss = -model.elbo(y_obs, ndraws=1)
        if not torch.isfinite(loss):
            print(f"  [vi H={cfg['H']}] NaN at epoch {epoch}, aborting cell", flush=True)
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, cfg["clip"])
        opt.step()

        if (epoch + 1) % cfg["log_every"] == 0:
            _sync(device)
            wall = time.perf_counter() - t0
            snap_vec, snap_theta = _extract_free_vec(prior, decoder)
            d = float(np.linalg.norm(snap_vec - prev_snap_vec))
            snapshot_vecs.append(snap_vec.copy())
            drift = _windowed_avg_drift(snapshot_vecs, cfg["conv_window"])
            l2_truth = float(np.linalg.norm(snap_vec - truth_vec))
            snapshots.append({
                "epoch":     int(epoch + 1),
                "wall_s":    float(wall),
                "elbo":      float(-loss.item()),
                "theta":     {k: float(v) for k, v in zip(FREE_NAMES, snap_vec)},
                "delta_l2":  d,
                "drift_l2":  drift,
                "l2_truth":  l2_truth,
            })
            if len(snapshots) % 10 == 0 or epoch < 200:
                drift_str = f"{drift:.4f}" if drift is not None else "n/a"
                print(f"  [vi H={cfg['H']}] ep {epoch+1:6d}  ELBO={-loss.item():+.3f}  "
                      f"delta={d:.4f}  drift={drift_str}  L2_truth={l2_truth:.3f}  wall={wall:.1f}s",
                      flush=True)
            prev_snap_vec = snap_vec
            if drift is not None and drift < cfg["conv_tau"]:
                hold_count += 1
                if hold_count >= cfg["conv_hold"]:
                    converged_at = epoch + 1
                    converged_wall = float(wall)
                    break
            else:
                hold_count = 0

    _sync(device)
    total_wall = float(time.perf_counter() - t0)
    sampler.stop()
    peak_mem = _peak_gpu_mem_mb(device)

    final_vec, final_theta = _extract_free_vec(prior, decoder)
    return {
        "H":               cfg["H"],
        "init_theta":      {k: init_theta[k] for k in FREE_NAMES},
        "final_theta":     {k: final_theta[k] for k in FREE_NAMES},
        "final_theta_vec": final_vec.tolist(),
        "l2_truth_final":  float(np.linalg.norm(final_vec - truth_vec)),
        "snapshots":       snapshots,
        "converged_at":    converged_at,
        "converged_wall_s": converged_wall,
        "total_wall_s":    total_wall,
        "peak_gpu_mem_mb": peak_mem,
        "gpu_util":        sampler.snapshot(),
    }


# ----------------------------------------------------------------------
# SMC-EM
# ----------------------------------------------------------------------

def _train_smc(y_obs, cfg, device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(cfg["vi_seed"])
    np.random.seed(cfg["vi_seed"])

    prior, decoder = M.build_prior_decoder(cfg["T"], device)
    _pin_z1_skew(prior)

    params = list(prior.parameters()) + list(decoder.parameters())
    params = [p for p in params if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"])

    smc_sampler = select_smc_smoother(cfg["resample"])

    truth_vec = np.array([HOCKEY_TRUTH[k] for k in FREE_NAMES], dtype=np.float64)
    init_vec, init_theta = _extract_free_vec(prior, decoder)

    snapshots = []
    snapshot_vecs = []
    hold_count = 0
    prev_snap_vec = init_vec.copy()
    sampler = GpuUtilSampler().reset().start()
    _sync(device); t0 = time.perf_counter()
    converged_at = None
    converged_wall = None
    for epoch in range(cfg["max_epochs"]):
        opt.zero_grad()
        with torch.no_grad():
            z_sample = smc_sampler(prior, decoder, y_obs, cfg["K"])
        log_pz   = prior.log_prob(z_sample)
        log_py_z = decoder.log_likelihood(y_obs, z_sample)
        loss = -(log_py_z + log_pz).mean()
        if not torch.isfinite(loss):
            print(f"  [smc K={cfg['K']}] NaN at epoch {epoch}, aborting cell", flush=True)
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, cfg["clip"])
        opt.step()

        if (epoch + 1) % cfg["log_every"] == 0:
            _sync(device)
            wall = time.perf_counter() - t0
            snap_vec, snap_theta = _extract_free_vec(prior, decoder)
            d = float(np.linalg.norm(snap_vec - prev_snap_vec))
            snapshot_vecs.append(snap_vec.copy())
            drift = _windowed_avg_drift(snapshot_vecs, cfg["conv_window"])
            l2_truth = float(np.linalg.norm(snap_vec - truth_vec))
            snapshots.append({
                "epoch":     int(epoch + 1),
                "wall_s":    float(wall),
                "m_obj":     float(-loss.item()),
                "theta":     {k: float(v) for k, v in zip(FREE_NAMES, snap_vec)},
                "delta_l2":  d,
                "drift_l2":  drift,
                "l2_truth":  l2_truth,
            })
            if len(snapshots) % 10 == 0 or epoch < 200:
                drift_str = f"{drift:.4f}" if drift is not None else "n/a"
                print(f"  [smc K={cfg['K']}] ep {epoch+1:6d}  M-obj={-loss.item():+.3f}  "
                      f"delta={d:.4f}  drift={drift_str}  L2_truth={l2_truth:.3f}  wall={wall:.1f}s",
                      flush=True)
            prev_snap_vec = snap_vec
            if drift is not None and drift < cfg["conv_tau"]:
                hold_count += 1
                if hold_count >= cfg["conv_hold"]:
                    converged_at = epoch + 1
                    converged_wall = float(wall)
                    break
            else:
                hold_count = 0

    _sync(device)
    total_wall = float(time.perf_counter() - t0)
    sampler.stop()
    peak_mem = _peak_gpu_mem_mb(device)

    final_vec, final_theta = _extract_free_vec(prior, decoder)
    return {
        "K":               cfg["K"],
        "init_theta":      {k: init_theta[k] for k in FREE_NAMES},
        "final_theta":     {k: final_theta[k] for k in FREE_NAMES},
        "final_theta_vec": final_vec.tolist(),
        "l2_truth_final":  float(np.linalg.norm(final_vec - truth_vec)),
        "snapshots":       snapshots,
        "converged_at":    converged_at,
        "converged_wall_s": converged_wall,
        "total_wall_s":    total_wall,
        "peak_gpu_mem_mb": peak_mem,
        "gpu_util":        sampler.snapshot(),
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    print(f"config={CFG}", flush=True)

    y_obs = M.simulate_from_truth(
        CFG["N"], CFG["T"], HOCKEY_TRUTH, CFG["obs_seed"], device)
    print(f"y_obs: shape={tuple(y_obs.shape)}, std={float(y_obs.std()):.4f}",
          flush=True)

    if CFG["method"] == "vi":
        result = _train_vi(y_obs, CFG, device)
        param_tag = f"h{CFG['H']}"
    else:
        result = _train_smc(y_obs, CFG, device)
        param_tag = f"K{CFG['K']}"

    cw = result.get("converged_wall_s")
    cw_str = f"{cw:.1f}s" if cw is not None else "did-not-converge"
    print(f"\n[{CFG['method']} {param_tag}] converged_at={result['converged_at']}  "
          f"converged_wall={cw_str}  "
          f"total_wall={result['total_wall_s']:.1f}s  "
          f"L2_truth={result['l2_truth_final']:.3f}  "
          f"peak_mem={result['peak_gpu_mem_mb']:.1f}MB", flush=True)

    bundle = {
        "config": {
            **CFG,
            "produced_by": "scripts/_time_to_convergence_hockey.py",
            "git_hash":    _git_hash(),
            "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        },
        "truth":     {k: float(v) for k, v in HOCKEY_TRUTH.items()},
        "free_names": FREE_NAMES,
        "device":    str(device),
        "result":    result,
    }

    n_tag = (f"{CFG['N']//1000}k" if CFG["N"] >= 1000 and CFG["N"] % 1000 == 0
             else str(CFG["N"]))
    out_path = OUT_DIR / f"conv_hockey_{CFG['method']}_N{n_tag}_T{CFG['T']}_{param_tag}.json"
    tmp = out_path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(bundle, f, indent=2)
    os.replace(tmp, out_path)
    print(f"\nwrote {out_path}", flush=True)


if __name__ == "__main__":
    # Fix every RNG (python / numpy / torch CPU+CUDA) and request deterministic
    # kernels before any draw; the script's own explicit seeds still apply.
    from mlye.seeding import seed_everything, env_seed
    seed_everything(env_seed("OBS_SEED", 11))
    main()
