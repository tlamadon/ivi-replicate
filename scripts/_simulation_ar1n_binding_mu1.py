r"""Picard binding function for mu_1 on the simulation-ar1n DGP with the
mean-field encoder.

1-D slice analogue of `_analysis_hockeystick_ivi_quiver.py`. At each
grid point mu_1^(0) on a 1-D grid:

  - Build theta_0 with mu_1 = mu_1^(0) and all other free parameters
    at truth.
  - Simulate y_sim(theta_0) via M.reparam_simulate using strong CRN
    (z_noise, y_noise pre-drawn at obs_seed and reused everywhere).
  - Fit a mean-field h=32 VI on y_sim with ALL EIGHT free parameters
    trainable (mu_0, mu_1, mu_2, sigma_0, sigma_1, sigma_2,
    z1_log_std, log_sigma_eps) --- the same parameter set as the
    upstream Layer 1a mean-field VI fit. The four shape parameters
    (log_beta, theta_MA, z1_skew, z1_log_tail) remain pinned at zero
    via construction-time options + M._apply_pinning_masks. mu_1 is
    initialised at mu_1^(0); the other seven free parameters at
    truth. Read off the converged mu_1 = b_{mu1}(mu_1^(0)).

After the grid, also fit a single fixed-point reference b_{mu1}^*:
the same all-eight-free-params mean-field VI applied to y_obs (not
y_sim). Under strong CRN at mu_1^(0) = mu_1^truth this equals
b_{mu1}(mu_1^truth) on the binding curve, and because the upstream
Layer 1a VI fit on y_obs trains exactly the same eight free
parameters, b^* equals mu_1^VI by construction --- i.e. the binding
function passes through the upstream VI estimate when evaluated at
the truth.

Upstream input: the Picard-mean-field cell of simulation-ar1n. The
upstream JSON's `truth`, `theta_VI_obs`, and `final_theta` provide
figure markers; the grid is centred on the spec calibration truth
(mu_1^truth = 0.9) with half-width 0.35.

Restartable: if `binding_mu1.json` already exists and its config
matches the current (n_grid, n_epochs_vi, lr, grid_min, grid_max),
completed cells are reused.

Env vars (with defaults):
  N_GRID         grid points along mu_1                (15)
  N_EPOCHS_VI    inner-VI epochs per cell              (8000)
  LR             inner-VI lr                           (1e-2)
  FIX_NOISE      tie encoder reparam noise (CRN)       (1)
  NOISE_SEED     encoder-noise seed                    (12345)

Output:
  output/bundles/cells/_simulation_ar1n_binding_mu1/binding_mu1.json
"""
import json
import os
import sys
import time

import numpy as np
import torch

torch.distributions.Distribution.set_default_validate_args(False)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _ar1n_posterior_comparison as M
from mlye.models.encoders.base import JointNormalConfig
from mlye.models.priors.markov import MarkovNormalConditionalPolyPrior
from mlye.models.decoders.ma import MASinhEmission
from mlye.models.full_model import FullModel


UPSTREAM_PATH = (
    "output/bundles/cells/_simulation_ar1n_sweep/"
    "picard_meanfield_alpha0.60_ep8000_lr1e-2_crn.json"
)
OUT_DIR = "output/bundles/cells/_simulation_ar1n_binding_mu1"
OUT_PATH = os.path.join(OUT_DIR, "binding_mu1.json")


def row_path(i):
    """Per-cell output written by a parallel ROW=<i> task."""
    return os.path.join(OUT_DIR, f"binding_mu1_row_{i}.json")


def bstar_path():
    """Output written by the MODE=bstar task."""
    return os.path.join(OUT_DIR, "binding_mu1_bstar.json")

HIDDEN_DIM = 32

# Grid half-width around mu_1^truth, per spec Layer 1d.
GRID_HALF_WIDTH = 0.35


def build_meanfield_model(T, device, truth):
    """Build the AR(1)+Normal well-specified prior + decoder + mean-field
    h=32 encoder, with all prior + decoder leaf parameters initialised
    at `truth` and the four shape parameters (log_beta, theta_MA,
    z1_skew, z1_log_tail) pinned at zero by `M._apply_pinning_masks`.
    """
    encoder = JointNormalConfig(
        dim=T, type='normal_diagonal',
        regularize=1e-3, hidden_dim=HIDDEN_DIM).build().to(device)
    prior, decoder = M.build_prior_decoder(T, device)
    model = FullModel(encoder=encoder, decoder=decoder, prior=prior).to(device)
    M._apply_pinning_masks(model, device)
    M.set_prior_decoder_to_truth(model, truth)
    return model


def vi_fit(y, T, device, mu1_init, truth, n_epochs, lr,
           eps_fixed, n_retries=10, base_seed=11_007):
    """Fit mean-field h=32 VI with all eight free parameters trainable.

    Initialises mu_1 at `mu1_init` and the other seven free
    parameters (mu_0, mu_2, sigma_0, sigma_1, sigma_2, z1_log_std,
    log_sigma_eps) at truth. The four shape parameters (log_beta,
    theta_MA, z1_skew, z1_log_tail) remain pinned via
    construction-time options + M._apply_pinning_masks. Matches the
    parameter set of the upstream Layer 1a mean-field VI fit.
    """
    last_err = None
    for s in M._retry_seeds(base_seed, n_retries):
        try:
            torch.manual_seed(s); np.random.seed(s)
            model = build_meanfield_model(T, device, truth)
            with torch.no_grad():
                model.prior.net_mu.coeffs.data[1] = float(mu1_init)
            train_params = [p for p in model.parameters() if p.requires_grad]
            opt = torch.optim.AdamW(train_params, lr=lr)
            nan_streak = 0
            for ep in range(n_epochs):
                opt.zero_grad()
                loss = -model.elbo(y, ndraws=1, eps_all=eps_fixed)
                if not torch.isfinite(loss):
                    nan_streak += 1
                    if ep < 200 and nan_streak >= 10:
                        raise M.NaNFailure(f"NaN streak at ep {ep}")
                    continue
                nan_streak = 0
                loss.backward()
                torch.nn.utils.clip_grad_norm_(train_params, 5.0)
                opt.step()
            mu1_hat = float(model.prior.net_mu.coeffs[1].item())
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return mu1_hat
        except M.NaNFailure as e:
            print(f"    VI NaN bail (seed={s}): {e}; retrying",
                  flush=True)
            last_err = e
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    print(f"    VI failed after {n_retries} retries "
          f"(last err: {last_err})", flush=True)
    return None


def load_upstream():
    if not os.path.exists(UPSTREAM_PATH):
        sys.exit(f"upstream JSON missing: {UPSTREAM_PATH}\n"
                  f"Run the simulation-ar1n picard-mf cell first.")
    upstream = json.load(open(UPSTREAM_PATH))
    truth = upstream['truth']
    vi_mu1 = float(upstream['theta_VI_obs']['mu1'])
    ivi_mu1 = float(upstream['final_theta']['mu1'])
    return truth, vi_mu1, ivi_mu1


def cfg_compatible(cached_cfg, cfg):
    """A cached binding_mu1.json is reusable iff these knobs match.
    Differences in other knobs are non-fatal (we keep the cell
    estimates anyway)."""
    keys = ['n_grid', 'n_epochs_vi', 'lr', 'grid_min', 'grid_max']
    for k in keys:
        if abs(float(cached_cfg.get(k, float('nan')))
               - float(cfg.get(k, float('nan')))) > 1e-9:
            return False
    return True


def aggregate_rows(n_grid, mu1_grid, truth_mu1, vi_mu1, ivi_mu1, cfg):
    """Merge per-row + b_star JSONs into binding_mu1.json."""
    b_mu1      = [None] * n_grid
    b_mu1_star = None
    for i in range(n_grid):
        rp = row_path(i)
        if not os.path.exists(rp):
            print(f"  [aggregate] missing {rp}", flush=True)
            continue
        with open(rp) as f:
            row = json.load(f)
        b_mu1[i] = row.get('b')
    sp = bstar_path()
    if os.path.exists(sp):
        with open(sp) as f:
            star = json.load(f)
        b_mu1_star = star.get('b_star')
    else:
        print(f"  [aggregate] missing {sp}", flush=True)
    out = {
        'config':      cfg,
        'mu1_grid':    mu1_grid.tolist(),
        'truth_mu1':   truth_mu1,
        'vi_mu1':      vi_mu1,
        'ivi_mu1':     ivi_mu1,
        'b_mu1':       [(None if v is None else float(v)) for v in b_mu1],
        'b_mu1_star':  (None if b_mu1_star is None else float(b_mu1_star)),
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_PATH, 'w') as f:
        json.dump(out, f, indent=2)
    n_done = sum(1 for v in b_mu1 if v is not None)
    print(f"\nAggregated {n_done}/{n_grid} grid cells + "
          f"b_star={out['b_mu1_star']}. Wrote {OUT_PATH}", flush=True)


def main():
    n_grid       = int(os.environ.get('N_GRID', 15))
    n_epochs_vi  = int(os.environ.get('N_EPOCHS_VI', 8000))
    lr           = float(os.environ.get('LR', 1e-2))
    fix_noise    = int(os.environ.get('FIX_NOISE', 1)) != 0
    noise_seed   = int(os.environ.get('NOISE_SEED', 12345))
    # Parallel mode: ROW=<i> computes one grid cell; MODE=bstar computes the
    # y_obs reference fit; MODE=aggregate merges per-row JSONs into
    # binding_mu1.json. Default: sequential full sweep.
    ROW  = os.environ.get('ROW', None)
    MODE = os.environ.get('MODE', None)

    truth, vi_mu1, ivi_mu1 = load_upstream()
    truth_mu1 = float(truth['mu1'])
    grid_min = truth_mu1 - GRID_HALF_WIDTH
    grid_max = truth_mu1 + GRID_HALF_WIDTH
    mu1_grid = np.linspace(grid_min, grid_max, n_grid)

    cfg = {
        'dgp':          'simulation-ar1n',
        'n_grid':       n_grid,
        'n_epochs_vi':  n_epochs_vi,
        'lr':           lr,
        'fix_noise':    fix_noise,
        'noise_seed':   noise_seed,
        'grid_min':     float(grid_min),
        'grid_max':     float(grid_max),
        'hidden_dim':   HIDDEN_DIM,
        'N':            30000,
        'T':            6,
        'obs_seed':     11,
        'upstream':     UPSTREAM_PATH,
    }

    # --- MODE=aggregate: merge row + bstar JSONs, no GPU work ----------
    if MODE == 'aggregate':
        os.makedirs(OUT_DIR, exist_ok=True)
        aggregate_rows(n_grid, mu1_grid, truth_mu1, vi_mu1, ivi_mu1, cfg)
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device={device}", flush=True)
    print(f"DGP: simulation-ar1n (AR(1) + Normal).", flush=True)
    print(f"hyperparams: n_grid={n_grid} n_epochs_vi={n_epochs_vi} "
          f"lr={lr} fix_noise={fix_noise} noise_seed={noise_seed}",
          flush=True)
    print(f"mu_1 grid: [{grid_min:+.4f}, {grid_max:+.4f}] "
          f"({n_grid} points)", flush=True)
    print(f"truth_mu1={truth_mu1:+.4f}  vi_mu1={vi_mu1:+.4f}  "
          f"ivi_mu1={ivi_mu1:+.4f}", flush=True)

    N, T = cfg['N'], cfg['T']
    OBS_SEED = cfg['obs_seed']

    # Pre-draw strong-CRN tensors for the DGP simulator and the encoder.
    torch.manual_seed(OBS_SEED)
    z_noise = torch.randn(N, T, device=device)
    y_noise = torch.randn(N, T, device=device)
    if fix_noise:
        torch.manual_seed(noise_seed)
        eps_fixed = torch.randn((1, N, T),
                                  dtype=torch.float32, device=device)
        print(f"[crn] eps_fixed pre-drawn at noise_seed={noise_seed} "
              f"shape={tuple(eps_fixed.shape)}", flush=True)
    else:
        eps_fixed = None

    # Simulate y_obs (used by b*).
    y_obs = M.simulate_from_truth(N, T, truth, OBS_SEED, device)
    print(f"y_obs: std={y_obs.std().item():+.4f}  "
          f"std(diff)={torch.diff(y_obs, dim=1).std().item():+.4f}",
          flush=True)

    PARAM_NAMES = M.PARAM_NAMES
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- MODE=bstar: only run the y_obs reference fit -----------------
    if MODE == 'bstar':
        print(f"\n=== b_star: all-eight-free-params VI on y_obs ===",
              flush=True)
        t0 = time.time()
        mu1_star = vi_fit(
            y_obs, T, device, truth_mu1, truth,
            n_epochs=n_epochs_vi, lr=lr, eps_fixed=eps_fixed)
        wall_s = time.time() - t0
        print(f"  b_star: mu1_hat={mu1_star if mu1_star is None else f'{mu1_star:+.4f}'}"
              f"  ({wall_s:.1f}s)", flush=True)
        out = {
            'config':  cfg,
            'b_star':  (None if mu1_star is None else float(mu1_star)),
            'wall_s':  wall_s,
        }
        with open(bstar_path(), 'w') as f:
            json.dump(out, f, indent=2)
        print(f"\nDone. Wrote {bstar_path()}", flush=True)
        return

    # --- ROW=<i>: compute one grid cell -------------------------------
    if ROW is not None:
        i = int(ROW)
        if not (0 <= i < n_grid):
            raise ValueError(f"ROW={i} out of range [0, {n_grid})")
        mu1_0 = float(mu1_grid[i])
        print(f"\n=== ROW={i}/{n_grid}: mu1_0={mu1_0:+.4f} ===", flush=True)
        theta0 = np.array([truth[k] for k in PARAM_NAMES], dtype=np.float64)
        theta0[PARAM_NAMES.index('mu1')] = mu1_0
        theta0_t = torch.tensor(theta0, device=device, dtype=torch.float32)
        with torch.no_grad():
            y_sim = M.reparam_simulate(theta0_t, z_noise, y_noise)
        t0 = time.time()
        mu1_hat = vi_fit(
            y_sim, T, device, mu1_0, truth,
            n_epochs=n_epochs_vi, lr=lr, eps_fixed=eps_fixed)
        wall_s = time.time() - t0
        print(f"  VI: mu1_hat={mu1_hat if mu1_hat is None else f'{mu1_hat:+.4f}'}"
              f"  ({wall_s:.1f}s)", flush=True)
        out = {
            'config':  cfg,
            'i':       i,
            'mu1_0':   mu1_0,
            'b':       (None if mu1_hat is None else float(mu1_hat)),
            'wall_s':  wall_s,
        }
        with open(row_path(i), 'w') as f:
            json.dump(out, f, indent=2)
        print(f"\nDone. Wrote {row_path(i)}", flush=True)
        return

    # --- Default: sequential full sweep -------------------------------
    # Restartable cache: reuse cells from a prior compatible run.
    b_mu1      = [None] * n_grid
    b_mu1_star = None
    if os.path.exists(OUT_PATH):
        try:
            cached = json.load(open(OUT_PATH))
            if 'b_mu1' not in cached:
                print(f"[cache] stale schema (no `b_mu1` key, likely from the "
                      f"pre-2026-06-23 constrained+full layout), ignoring "
                      f"{OUT_PATH}", flush=True)
            elif cfg_compatible(cached.get('config', {}), cfg):
                cached_b    = cached.get('b_mu1', [])
                cached_star = cached.get('b_mu1_star', None)
                for i in range(min(n_grid, len(cached_b))):
                    if cached_b[i] is not None:
                        b_mu1[i] = float(cached_b[i])
                if cached_star is not None:
                    b_mu1_star = float(cached_star)
                n_done = sum(1 for v in b_mu1 if v is not None)
                print(f"[cache] reusing {n_done}/{n_grid} cells "
                      f"and b_star={b_mu1_star}", flush=True)
            else:
                print(f"[cache] incompatible config, ignoring "
                      f"{OUT_PATH}", flush=True)
        except Exception as e:
            print(f"[cache] failed to load {OUT_PATH}: {e}", flush=True)

    def write_partial():
        out = {
            'config':     cfg,
            'mu1_grid':   mu1_grid.tolist(),
            'truth_mu1':  truth_mu1,
            'vi_mu1':     vi_mu1,
            'ivi_mu1':    ivi_mu1,
            'b_mu1':      [(None if v is None else float(v)) for v in b_mu1],
            'b_mu1_star': (None if b_mu1_star is None
                              else float(b_mu1_star)),
        }
        with open(OUT_PATH, 'w') as f:
            json.dump(out, f, indent=2)

    for i, mu1_0 in enumerate(mu1_grid):
        if b_mu1[i] is not None:
            print(f"[grid {i+1}/{n_grid}] mu1_0={float(mu1_0):+.4f} "
                  f"cached, skipping", flush=True)
            continue
        print(f"\n=== grid {i+1}/{n_grid}: mu1_0={float(mu1_0):+.4f} ===",
              flush=True)
        # Build theta_0 = truth with mu_1 replaced by mu1_0.
        theta0 = np.array([truth[k] for k in PARAM_NAMES], dtype=np.float64)
        theta0[PARAM_NAMES.index('mu1')] = float(mu1_0)
        theta0_t = torch.tensor(theta0, device=device, dtype=torch.float32)
        with torch.no_grad():
            y_sim = M.reparam_simulate(theta0_t, z_noise, y_noise)
        t0 = time.time()
        mu1_hat = vi_fit(
            y_sim, T, device, float(mu1_0), truth,
            n_epochs=n_epochs_vi, lr=lr, eps_fixed=eps_fixed)
        print(f"  VI: mu1_hat={mu1_hat if mu1_hat is None else f'{mu1_hat:+.4f}'}"
              f"  ({time.time()-t0:.1f}s)", flush=True)
        b_mu1[i] = mu1_hat
        write_partial()

    # b* reference: all-eight-free-params VI on y_obs (not y_sim),
    # initialised at mu_1^truth. Equals mu_1^VI by construction under
    # strong CRN on this well-specified DGP.
    if b_mu1_star is None:
        print(f"\n=== b_star: all-eight-free-params VI on y_obs ===",
              flush=True)
        t0 = time.time()
        mu1_star = vi_fit(
            y_obs, T, device, truth_mu1, truth,
            n_epochs=n_epochs_vi, lr=lr, eps_fixed=eps_fixed)
        print(f"  b_star: mu1_hat={mu1_star if mu1_star is None else f'{mu1_star:+.4f}'}"
              f"  ({time.time()-t0:.1f}s)", flush=True)
        b_mu1_star = mu1_star
        write_partial()

    print(f"\nDone. Wrote {OUT_PATH}", flush=True)


if __name__ == "__main__":
    # Fix every RNG (python / numpy / torch CPU+CUDA) and request deterministic
    # kernels before any draw; the script's own explicit seeds still apply.
    from mlye.seeding import seed_everything, env_seed
    seed_everything(env_seed("OBS_SEED", 11))
    main()
