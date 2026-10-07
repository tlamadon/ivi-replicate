"""Posterior-contour bundle for the past-2-period-imputed observation.

Implements `specs/compute-contour_past_2period_imputed.md`. Reads the
`truth` block from a simulation-hockeystick cell JSON, pins the
(prior, decoder) at HOCKEY_TRUTH, fits two amortised encoders
(joint-normal + mean-field) on a fresh DGP draw, computes the true
posterior p(z_1, z_2 | y_{1:T}) on a 2D grid via IS-marginalisation
of z_{3:T}, and writes a self-describing JSON bundle. Also renders
a quick-check matplotlib PDF/PNG (the deliberate compute-scope
exception documented in the spec).

Output:
  output/bundles/cells/_contour_past_2period_imputed/contour_bundle.json
  output/bundles/cells/_contour_past_2period_imputed/contour_past_2period_imputed.{pdf,png}
  output/bundles/cells/_contour_past_2period_imputed/encoder_{joint_normal,normal_diagonal}_h64_n8000_e8000.pt
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
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.distributions import Normal

# scripts/ is on PYTHONPATH per the spec's invocation.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from softplus_sinh_posterior_comparison import (
    PARAM_NAMES, build_prior_decoder, set_prior_decoder_to_truth,
    reparam_simulate,
)
from smc_truth_profile import _Bundle
from climb_per_step import smc_ffbs_sample

from mlye.models.encoders import JointNormalConfig
from mlye.models.full_model import FullModel


OUT_DIR = Path("output/bundles/cells/_contour_past_2period_imputed")
DEFAULT_UPSTREAM = (
    "output/bundles/cells/_simulation_hockeystick_ivi_sweep/"
    "picard_cold_alpha0.60_ep16000_lr1e-2_crn_sig0--1.80.json"
)
TAG = "past_2period_imputed"
Y_OBS_VALUES = [-0.1, 0.1, 0.1, 0.1, 0.1, 0.1]


# ----------------------------------------------------------------------
# Config from env (with spec defaults)
# ----------------------------------------------------------------------

def _env_int(name, default):
    return int(os.environ.get(name, default))

def _env_float(name, default):
    return float(os.environ.get(name, default))


CFG = {
    "n_grid":       _env_int("N_GRID", 140),
    "K_tail":       _env_int("K", 6000),
    "n_train":      _env_int("N_TRAIN", 8000),
    "n_epochs":     _env_int("N_EPOCHS", 8000),
    "lr":           _env_float("LR", 1e-3),
    "hidden_dim":   _env_int("HIDDEN_DIM", 64),
    "smc_K":        _env_int("SMC_K", 2000),
    "smc_draws":    _env_int("SMC_DRAWS", 8),
    "adapt_seed":   _env_int("ADAPT_SEED", 11),
    "train_seed":   _env_int("TRAIN_SEED", 42),
    "tail_seed":    _env_int("TAIL_SEED", 11),
    "pad_floor_sigma_eps_mult": 6,
    "pad_floor_smc_sd_mult":    4,
    "pad_floor_abs":            0.25,
}
UPSTREAM_JSON = os.environ.get("UPSTREAM_JSON", DEFAULT_UPSTREAM)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _git_hash():
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, check=False,
            timeout=2,
        )
        return out.stdout.decode().strip() or None
    except Exception:
        return None


def load_truth(path):
    with open(path) as f:
        d = json.load(f)
    if d.get("config", {}).get("T") != 6:
        raise AssertionError(
            f"{path}: expected config.T == 6, got {d.get('config', {}).get('T')!r}")
    truth = d["truth"]
    missing = [k for k in PARAM_NAMES if k not in truth]
    if missing:
        raise AssertionError(
            f"{path}: truth block missing keys {missing}")
    # Cast to plain floats so we can JSON-roundtrip.
    return {k: float(truth[k]) for k in PARAM_NAMES}


def install_truth(prior, decoder, truth):
    set_prior_decoder_to_truth(_Bundle(prior, decoder), truth)
    # Pin z1_skew at 0 per the simulation-hockeystick contract.
    prior.z1_skew.requires_grad_(False)


def smc_smoother_summary(prior, decoder, y, K, n_draws, seed):
    """(z_mean, z_std) of shape (N, T) averaging n_draws FFBS samples."""
    torch.manual_seed(seed)
    draws = []
    for _ in range(n_draws):
        draws.append(smc_ffbs_sample(prior, decoder, y, K))
    draws = torch.stack(draws, dim=0)
    return draws.mean(0), draws.std(0)


def simulate_training_batch(prior, decoder, truth, n_train, T, seed, device):
    """Fresh DGP draw via reparam_simulate. Returns y (n_train, T)."""
    torch.manual_seed(seed)
    z_noise = torch.randn(n_train, T, device=device)
    y_noise = torch.randn(n_train, T, device=device)
    theta = torch.tensor([truth[k] for k in PARAM_NAMES], device=device,
                          dtype=torch.float32)
    with torch.no_grad():
        return reparam_simulate(theta, z_noise, y_noise)


def fit_or_load_encoder(ptype, prior, decoder, y_train, ckpt_path, cfg, device):
    """Fit (or load cached) amortised encoder. Returns (encoder, elbo, from_cache)."""
    encoder = JointNormalConfig(
        dim=y_train.shape[1], type=ptype, regularize=1e-3,
        hidden_dim=cfg["hidden_dim"],
    ).build().to(device)
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device)
        encoder.load_state_dict(state["encoder"])
        print(f"  [{ptype}] loaded cache: {ckpt_path.name}", flush=True)
        return encoder, state.get("elbo"), True
    # Train.
    for p in prior.parameters():   p.requires_grad_(False)
    for p in decoder.parameters(): p.requires_grad_(False)
    model = FullModel(encoder=encoder, decoder=decoder, prior=prior).to(device)
    opt = torch.optim.AdamW(encoder.parameters(), lr=cfg["lr"])
    print(f"  [{ptype}] training {cfg['n_epochs']} epochs at lr={cfg['lr']}",
          flush=True)
    t0 = time.time()
    last_elbo = None
    log_every = max(cfg["n_epochs"] // 16, 100)
    for epoch in range(cfg["n_epochs"]):
        opt.zero_grad()
        loss = -model.elbo(y_train, ndraws=1)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5.0)
        opt.step()
        if epoch % log_every == 0 or epoch == cfg["n_epochs"] - 1:
            last_elbo = -float(loss.item())
            print(f"    epoch {epoch:5d}  ELBO={last_elbo:+.4f}  "
                  f"({time.time()-t0:.1f}s)", flush=True)
    torch.save({"encoder": encoder.state_dict(), "elbo": last_elbo}, ckpt_path)
    print(f"  [{ptype}] saved {ckpt_path.name}", flush=True)
    return encoder, last_elbo, False


@torch.no_grad()
def encoder_gauss2d(encoder, y_obs):
    """Return mu (2,) and Sigma (2, 2) for the (z_1, z_2) marginal."""
    y_batch = y_obs.unsqueeze(0)
    mu, L = encoder.forward(y_batch)
    mu = mu.squeeze(0)
    L = L.squeeze(0)
    Sigma = L @ L.T
    return mu[:2], Sigma[:2, :2]


def gaussian_logp_on_grid(mu2, Sigma22, z1_grid, z2_grid):
    """Closed-form 2D Gaussian log density on the (z1, z2) grid."""
    Z1, Z2 = torch.meshgrid(z1_grid, z2_grid, indexing="ij")
    x = torch.stack([Z1, Z2], dim=-1)
    d = x - mu2
    inv = torch.linalg.inv(Sigma22)
    _, logdet = torch.linalg.slogdet(2 * np.pi * Sigma22)
    quad = torch.einsum("...i,ij,...j->...", d, inv, d)
    return -0.5 * (quad + logdet)


@torch.no_grad()
def log_tail_marginal(prior, decoder, y_tail, z2_grid, K, seed):
    """log L(z2) = log E_{z3..zT | z2}[ prod_t p(y_t | z_t) ] via IS with
    the conditional prior chain as proposal.

    Variance reduction:
      - CRN across z2 grid columns (shared eps per t).
      - Antithetic pairing of K base draws.
    """
    if K % 2 != 0:
        raise ValueError("K must be even for antithetic pairing")
    torch.manual_seed(seed)
    n_z2 = z2_grid.shape[0]
    device = z2_grid.device
    T_tail = y_tail.shape[0]
    dist_e = decoder.get_distribution()
    half = K // 2

    eps_per_t = []
    for _ in range(T_tail):
        base = torch.randn(1, half, device=device)
        eps_per_t.append(torch.cat([base, -base], dim=1))

    z_prev = z2_grid.unsqueeze(-1).expand(n_z2, K).contiguous()
    log_p_tail = torch.zeros(n_z2, K, device=device)
    for t in range(T_tail):
        mu_t, sig_t = prior.get_mu_sigma(z_prev.unsqueeze(-1))
        mu_t = mu_t.squeeze(-1); sig_t = sig_t.squeeze(-1)
        eps = eps_per_t[t].expand(n_z2, K)
        z_t = mu_t + sig_t * eps
        log_p_tail = log_p_tail + dist_e.log_prob(y_tail[t] - z_t)
        z_prev = z_t

    log_marg = torch.logsumexp(log_p_tail, dim=-1) - float(np.log(K))
    log_w = log_p_tail - log_p_tail.max(dim=-1, keepdim=True).values
    w = log_w.exp()
    ess = (w.sum(dim=-1) ** 2) / (w.pow(2).sum(dim=-1))
    return log_marg, ess


@torch.no_grad()
def compute_log_p_true(prior, decoder, y_obs, z1_grid, z2_grid, K, seed):
    """log p(z1, z2 | y_{1:T}) up to additive const, on the (z1, z2) grid."""
    eta0 = prior.get_eta0()
    dist_e = decoder.get_distribution()

    log_pz1 = eta0.log_prob(z1_grid.unsqueeze(-1)).reshape(-1)         # (nz1,)
    log_py1 = dist_e.log_prob(y_obs[0] - z1_grid).reshape(-1)          # (nz1,)
    log_py2 = dist_e.log_prob(y_obs[1] - z2_grid).reshape(-1)          # (nz2,)

    mu_2, sig_2 = prior.get_mu_sigma(z1_grid.unsqueeze(-1))
    mu_2 = mu_2.squeeze(-1); sig_2 = sig_2.squeeze(-1)
    log_pz2_z1 = Normal(mu_2.unsqueeze(-1), sig_2.unsqueeze(-1)).log_prob(
        z2_grid.unsqueeze(0))                                          # (nz1, nz2)

    log_L, ess = log_tail_marginal(prior, decoder, y_obs[2:], z2_grid,
                                    K=K, seed=seed)                    # (nz2,)
    log_p = (
        log_pz1.unsqueeze(-1)
        + log_py1.unsqueeze(-1)
        + log_pz2_z1
        + log_py2.unsqueeze(0)
        + log_L.unsqueeze(0)
    )
    return log_p, ess


# ----------------------------------------------------------------------
# Quick-check matplotlib rendering (the documented exception)
# ----------------------------------------------------------------------

def plot_panel(z1_grid, z2_grid, panels, y_marker, smc_marker, out_pdf, out_png):
    """3 side-by-side square axes sharing the same range and contour levels."""
    n = len(panels)
    levels = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
    lo = float(min(z1_grid.min(), z2_grid.min()))
    hi = float(max(z1_grid.max(), z2_grid.max()))

    box_in = 4.5
    left_in, right_in, top_in, bot_in, gap_in = 0.6, 0.15, 0.55, 0.55, 0.45
    fig_w = left_in + n * box_in + (n - 1) * gap_in + right_in
    fig_h = bot_in + box_in + top_in
    fig = plt.figure(figsize=(fig_w, fig_h))

    axes = []
    for i, (subtitle, log_p) in enumerate(panels):
        left = (left_in + i * (box_in + gap_in)) / fig_w
        bottom = bot_in / fig_h
        width = box_in / fig_w
        height = box_in / fig_h
        ax = fig.add_axes([left, bottom, width, height])
        axes.append(ax)
        P = np.exp(log_p - log_p.max())
        cs = ax.contour(z1_grid, z2_grid, P.T, levels=levels,
                        cmap="viridis", linewidths=0.9)
        ax.clabel(cs, inline=True, fontsize=7, fmt="%.2f")
        ax.plot([smc_marker[0]], [smc_marker[1]], "rx", ms=9, mew=1.8,
                label=r"SMC $E[z_{1,2}|y]$")
        ax.plot([y_marker[0]], [y_marker[1]], "k+", ms=9, mew=1.4,
                label=r"$(y_1, y_2)$")
        ax.axhline(0, color="k", lw=0.3, alpha=0.4)
        ax.axvline(0, color="k", lw=0.3, alpha=0.4)
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.3, alpha=0.4)
        ax.set_xlabel(r"$z_1$")
        ax.set_title(subtitle, fontsize=10)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        if i > 0:
            ax.set_yticklabels([])
    axes[0].set_ylabel(r"$z_2$")
    axes[0].legend(loc="upper left", fontsize=8, frameon=False)
    suptitle = (
        rf"$p(z_1, z_2 \mid y_{{1:6}})$ — past_2period_imputed "
        f"— y = [{', '.join(f'{v:+.2f}' for v in Y_OBS_VALUES)}]"
    )
    fig.suptitle(suptitle, fontsize=10, y=1 - 0.02 / fig_h * 4)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out_png, dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  upstream={UPSTREAM_JSON}", flush=True)
    print(f"config={CFG}", flush=True)

    truth = load_truth(UPSTREAM_JSON)
    T = 6
    print(f"loaded truth ({len(truth)} keys); "
          f"sigma_eps={np.exp(truth['log_sigma_eps']):.4f}, "
          f"beta={np.exp(truth['log_beta']):.3f}", flush=True)

    prior, decoder = build_prior_decoder(T, device)
    install_truth(prior, decoder, truth)

    y_obs = torch.tensor(Y_OBS_VALUES, device=device, dtype=torch.float32)

    # SMC smoother summary on this single observation.
    z_mean, z_std = smc_smoother_summary(
        prior, decoder, y_obs.unsqueeze(0),
        K=CFG["smc_K"], n_draws=CFG["smc_draws"], seed=CFG["adapt_seed"])
    zm = z_mean[0].cpu().numpy(); zs = z_std[0].cpu().numpy()
    print(f"SMC E[z|y]   = {zm}", flush=True)
    print(f"SMC sd[z|y]  = {zs}", flush=True)

    # Adaptive grid.
    sigma_eps = float(np.exp(truth["log_sigma_eps"]))
    pad = max(
        CFG["pad_floor_sigma_eps_mult"] * sigma_eps,
        CFG["pad_floor_smc_sd_mult"] * float(max(zs[0], zs[1])),
        CFG["pad_floor_abs"],
    )
    lo = float(min(zm[0], zm[1])) - pad
    hi = float(max(zm[0], zm[1])) + pad
    z1_grid = torch.linspace(lo, hi, CFG["n_grid"], device=device)
    z2_grid = torch.linspace(lo, hi, CFG["n_grid"], device=device)
    print(f"grid [{lo:.3f}, {hi:.3f}]^2, n_grid={CFG['n_grid']}, "
          f"K_tail={CFG['K_tail']}, pad={pad:.3f}", flush=True)

    # Encoders: simulate training batch at truth, fit/load each encoder.
    print(f"\n=== Simulating N={CFG['n_train']} training batch from truth ===",
          flush=True)
    y_train = simulate_training_batch(
        prior, decoder, truth, n_train=CFG["n_train"], T=T,
        seed=CFG["train_seed"], device=device)
    print(f"y_train std per t: {y_train.std(dim=0).cpu().numpy()}", flush=True)

    encoders = {}
    encoder_meta = {}
    for ptype in ("joint_normal", "normal_diagonal"):
        ckpt_name = (f"encoder_{ptype}_h{CFG['hidden_dim']}"
                      f"_n{CFG['n_train']}_e{CFG['n_epochs']}.pt")
        ckpt_path = OUT_DIR / ckpt_name
        enc, elbo, from_cache = fit_or_load_encoder(
            ptype, prior, decoder, y_train, ckpt_path, CFG, device)
        encoders[ptype] = enc
        encoder_meta[ptype] = {
            "final_elbo": (float(elbo) if elbo is not None else None),
            "loaded_from_cache": bool(from_cache),
            "checkpoint_filename": ckpt_name,
        }

    # 2D encoder Gaussians on the grid.
    for ptype, enc in encoders.items():
        mu2, Sigma22 = encoder_gauss2d(enc, y_obs)
        sd1 = float(torch.sqrt(Sigma22[0, 0])); sd2 = float(torch.sqrt(Sigma22[1, 1]))
        corr = float(Sigma22[0, 1] / (sd1 * sd2 + 1e-30))
        encoder_meta[ptype].update({
            "mu":    [float(mu2[0]), float(mu2[1])],
            "sd":    [sd1, sd2],
            "corr":  corr,
            "Sigma": [[float(Sigma22[0, 0]), float(Sigma22[0, 1])],
                       [float(Sigma22[1, 0]), float(Sigma22[1, 1])]],
        })
        if ptype == "joint_normal":
            log_p_jn = gaussian_logp_on_grid(
                mu2, Sigma22, z1_grid, z2_grid).cpu().numpy()
        else:
            log_p_mf = gaussian_logp_on_grid(
                mu2, Sigma22, z1_grid, z2_grid).cpu().numpy()
        print(f"  q[{ptype}]: mu=({float(mu2[0]):+.3f},{float(mu2[1]):+.3f}) "
              f"sd=({sd1:.3f},{sd2:.3f}) corr={corr:+.3f}", flush=True)

    # True posterior on the grid.
    t0 = time.time()
    log_p_true_t, ess = compute_log_p_true(
        prior, decoder, y_obs, z1_grid, z2_grid,
        K=CFG["K_tail"], seed=CFG["tail_seed"])
    log_p_true = log_p_true_t.cpu().numpy()
    ess_np = ess.cpu().numpy()
    print(f"true posterior grid done in {time.time()-t0:.1f}s; "
          f"ess (min/median/max)/K = "
          f"{ess_np.min():.1f}/{np.median(ess_np):.1f}/{ess_np.max():.1f} "
          f"of {CFG['K_tail']}", flush=True)

    # Normalise log-densities to max=0.
    log_p_true -= log_p_true.max()
    log_p_jn   -= log_p_jn.max()
    log_p_mf   -= log_p_mf.max()

    # ----- Bundle -----
    derived = {
        "sigma_eps": float(np.exp(truth["log_sigma_eps"])),
        "beta":      float(np.exp(truth["log_beta"])),
        "sigma_z1":  float(np.log1p(np.exp(truth["z1_log_std"]))),   # softplus
        "alpha1":    float(np.exp(truth["log_alpha1"])),
    }
    dgp_block = dict(truth)
    dgp_block.update(derived)

    bundle = {
        "metadata": {
            "name":           "contour_past_2period_imputed",
            "produced_by":    "scripts/_contour_past_2period_imputed.py",
            "git_hash":       _git_hash(),
            "created_utc":    datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "upstream_json":  UPSTREAM_JSON,
            "tag":            TAG,
            "T":              T,
            "device":         str(device),
            "quick_check_rendered": True,
        },
        "config": {
            **CFG,
            "pad_used": pad,
        },
        "dgp": dgp_block,
        "observation": {
            "tag":         TAG,
            "y_obs":       [float(v) for v in Y_OBS_VALUES],
            "description": ("y_1=-0.1, y_2=+0.1 sign-flip; "
                             "y_{3:T} imputed at y_2"),
        },
        "smc_smoother": {
            "z_mean": [float(v) for v in zm],
            "z_std":  [float(v) for v in zs],
        },
        "grid": {
            "z1_grid": [float(v) for v in z1_grid.cpu().numpy()],
            "z2_grid": [float(v) for v in z2_grid.cpu().numpy()],
            "lo":      lo,
            "hi":      hi,
        },
        "log_p_true": log_p_true.tolist(),
        "log_p_jn":   log_p_jn.tolist(),
        "log_p_mf":   log_p_mf.tolist(),
        "encoders":   encoder_meta,
        "ess_tail": {
            "min":         float(ess_np.min()),
            "median":      float(np.median(ess_np)),
            "max":         float(ess_np.max()),
            "ess_per_z2":  [float(v) for v in ess_np],
        },
        "markers": {
            "y_marker":      [float(Y_OBS_VALUES[0]), float(Y_OBS_VALUES[1])],
            "smc_marker":    [float(zm[0]), float(zm[1])],
            "diagonal_line": [[lo, lo], [hi, hi]],
        },
        "ai_prompt": (
            "This bundle ships three log-density grids on a square "
            f"[lo, hi]^2 grid (lo={lo:.3f}, hi={hi:.3f}, n_grid="
            f"{CFG['n_grid']}) for the single observation "
            "y_obs = past_2period_imputed = "
            f"[{', '.join(f'{v:+.2f}' for v in Y_OBS_VALUES)}] under "
            "the simulation-hockeystick DGP at HOCKEY_TRUTH (softplus-kink "
            "law of motion + sinh-arcsinh z_1 + sinh-arcsinh emission, "
            "theta=0). The three grids — `log_p_true`, `log_p_jn`, "
            "`log_p_mf` — are normalised so `max = 0`; convert to a "
            "density via P = exp(log_p) and pick contour levels in "
            "P-space (e.g. {0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, "
            "0.99}). Indexing convention: `log_p_*[i][j]` = log p(z_1 "
            "= z1_grid[i], z_2 = z2_grid[j] | y). The y_marker is "
            "(y_1, y_2); smc_marker is the SMC FFBS posterior mean of "
            "(z_1, z_2); diagonal_line is the z_1 = z_2 reference. "
            "The truth anchor (HOCKEY_TRUTH) is in `dgp`; the IVI cell "
            "JSON in `metadata.upstream_json` is the canonical source."
        ),
    }

    bundle_path = Path("output/bundles/contour_bundle.json")
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = bundle_path.with_suffix(".json.tmp")
    with open(tmp_path, "w") as f:
        json.dump(bundle, f, indent=2)
    os.replace(tmp_path, bundle_path)
    print(f"wrote {bundle_path}", flush=True)

    # ----- Quick-check matplotlib rendering -----
    panels = [
        ("true posterior",        log_p_true),
        (r"joint-normal $q$",     log_p_jn),
        (r"mean-field $q$",       log_p_mf),
    ]
    pdf_path = OUT_DIR / f"contour_{TAG}.pdf"
    png_path = OUT_DIR / f"contour_{TAG}.png"
    plot_panel(
        z1_grid.cpu().numpy(), z2_grid.cpu().numpy(), panels,
        y_marker=[Y_OBS_VALUES[0], Y_OBS_VALUES[1]],
        smc_marker=[float(zm[0]), float(zm[1])],
        out_pdf=str(pdf_path), out_png=str(png_path),
    )
    print(f"wrote {pdf_path}", flush=True)
    print(f"wrote {png_path}", flush=True)


if __name__ == "__main__":
    # Fix every RNG (python / numpy / torch CPU+CUDA) and request deterministic
    # kernels before any draw; the script's own explicit seeds still apply.
    from mlye.seeding import seed_everything, env_seed
    seed_everything(env_seed("ADAPT_SEED", 11))
    main()
