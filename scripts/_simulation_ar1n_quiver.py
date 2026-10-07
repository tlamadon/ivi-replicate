r"""Picard binding-function quiver in (mu_1, log_sigma_eps) for the
simulation-ar1n DGP with the mean-field (normal_diagonal) encoder.

Layer 1c of compute-simulation-ar1n.md. At each (m1_0, lse_0) grid cell
we:

  - simulate y_sim(theta_0) via M.reparam_simulate using strong CRN
    (z_noise, y_noise pre-drawn at obs_seed and reused everywhere);
  - fit a **full** mean-field h=32 VI on y_sim with **all eight free
    parameters** trainable (mu_0, mu_1, mu_2, sigma_0, sigma_1,
    sigma_2, z1_log_std, log_sigma_eps); only the four pinned shape
    entries (log_beta, theta_MA, z1_skew, z1_log_tail) stay at the
    construction-time defaults. The eight free params' converged
    values are recorded; the binding-map output projects onto the
    plotted slice by reading off (m1_hat, lse_hat); the other six
    components are auxiliaries;
  - compute the **closed-form** marginal log-likelihood contour
    log p(y_obs | theta) at the grid point's theta, with the
    non-target parameters (mu_0, mu_2, sigma_0, sigma_1, sigma_2,
    z1_log_std) at truth, using the helper `M.ar1n_marginal_logp`.
    Because mu_0 = mu_2 = sigma_1 = sigma_2 = 0 at truth (the AR(1)
    slice), the closed form applies at every grid cell --- no BPF
    fallback is needed at the contour stage. (The converged
    binding-map theta may drift off the slice; that is not used by
    the contour evaluator.)

The script's b* reference is the same full mean-field VI map applied
to y_obs (not y_sim).

Invocation modes:
  1. Per-cell mode (CELL_I=<i> CELL_J=<j>): compute only cell (i, j)
     and write `quiver_cell_<i>_<j>.json` with the per-cell contour
     value, binding-map values, and converged auxiliaries.
  2. b* mode (MODE=bstar): single full-VI fit on y_obs; write
     `quiver_bstar.json`.
  3. Aggregate mode (MODE=aggregate): read every
     `quiver_cell_*.json` + `quiver_bstar.json` and merge into the
     canonical `quiver.json`.
  4. Default (no CELL_I/CELL_J, no MODE): sequential full sweep
     (debugging / legacy path); emits the canonical `quiver.json`.

Upstream input: the Picard mean-field cell of simulation-ar1n. See
specs/compute-simulation-ar1n.md (Layer 1c) for the contract.

Env vars (with defaults):
  N_GRID         grid points per axis                  (10)
  N_EPOCHS_VI    full-VI epochs per cell               (8000)
  LR             full-VI lr                            (1e-2)
  FIX_NOISE      tie encoder reparam noise (CRN)       (1)
  NOISE_SEED     encoder-noise seed                    (12345)
  GRID_PAD       padding around {truth, VI, IVI} hull  (0.30)
  CELL_I, CELL_J per-cell mode (0..N_GRID-1 each)      (unset)
  MODE           'bstar' or 'aggregate'                (unset)

Output:
  output/bundles/cells/_simulation_ar1n_quiver/quiver_cell_<i>_<j>.json  (per-cell)
  output/bundles/cells/_simulation_ar1n_quiver/quiver_bstar.json         (b*)
  output/bundles/cells/_simulation_ar1n_quiver/quiver.json               (aggregate /
                                                              default)
"""
import copy
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
OUT_DIR = "output/bundles/cells/_simulation_ar1n_quiver"
OUT_PATH = os.path.join(OUT_DIR, "quiver.json")

HIDDEN_DIM = 32


def cell_path(i, j):
    return os.path.join(OUT_DIR, f"quiver_cell_{i}_{j}.json")


def bstar_path():
    return os.path.join(OUT_DIR, "quiver_bstar.json")


def build_meanfield_model(T, device, truth, mu1, log_sigma_eps):
    """Build the AR(1)+Normal well-specified prior + decoder + mean-field
    h=32 encoder, with prior + decoder leaf parameters initialised at
    truth for (mu_0, mu_2, sigma_0, sigma_1, sigma_2, z1_log_std) and
    at (mu1, log_sigma_eps) for the target coordinates. The four
    pinned shape entries (log_beta, theta_MA, z1_skew, z1_log_tail)
    stay at construction-time defaults via M._apply_pinning_masks.

    All eight free parameters (mu_0, mu_1, mu_2, sigma_0, sigma_1,
    sigma_2, z1_log_std, log_sigma_eps) remain trainable. The
    decoder's log_beta is non-trainable (fix_beta=True).
    """
    encoder = JointNormalConfig(
        dim=T, type='normal_diagonal',
        regularize=1e-3, hidden_dim=HIDDEN_DIM).build().to(device)
    prior, decoder = M.build_prior_decoder(T, device)
    model = FullModel(encoder=encoder, decoder=decoder, prior=prior).to(device)
    M._apply_pinning_masks(model, device)
    M.set_prior_decoder_to_truth(model, truth)
    with torch.no_grad():
        # Initialise the target coordinate at the grid point; mu_0 /
        # mu_2 stay at truth (= 0) but remain trainable.
        model.prior.net_mu.coeffs.data[1] = float(mu1)
        model.decoder.log_sigma.fill_(float(log_sigma_eps))
    return model


def full_vi_fit(y, T, device, mu1_init, log_sigma_init, truth,
                  n_epochs, lr, eps_fixed, n_retries=5,
                  base_seed=11_007):
    """Fit a full mean-field h=32 VI on y with all eight free parameters
    trainable. Initialises target coordinates at
    (mu1_init, log_sigma_init); other free params at truth.

    Returns a dict mapping each of `M.FREE_NAMES` to its converged float
    value (eight keys: mu0, mu1, mu2, sigma0, sigma1, sigma2,
    z1_log_std, log_sigma_eps), or None on repeated NaN bail.
    """
    last_err = None
    for s in M._retry_seeds(base_seed, n_retries):
        try:
            torch.manual_seed(s); np.random.seed(s)
            model = build_meanfield_model(T, device, truth,
                                            mu1_init, log_sigma_init)
            train = [p for p in model.parameters() if p.requires_grad]
            opt = torch.optim.AdamW(train, lr=lr)
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
                torch.nn.utils.clip_grad_norm_(train, 5.0)
                opt.step()
            p = M.extract_params(model)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return {k: float(p[k]) for k in M.FREE_NAMES}
        except M.NaNFailure as e:
            print(f"    full VI NaN bail (seed={s}): {e}; retrying",
                  flush=True)
            last_err = e
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    print(f"    full VI failed after {n_retries} retries "
          f"(last err: {last_err})", flush=True)
    return None


def load_problem(grid_pad, n_grid):
    """Read upstream JSON, return everything needed to define the grid +
    fit cells. Deterministic given (grid_pad, n_grid)."""
    if not os.path.exists(UPSTREAM_PATH):
        sys.exit(f"upstream JSON missing: {UPSTREAM_PATH}\n"
                 f"Run simulation-ar1n (Picard mean-field cell) first.")
    upstream = json.load(open(UPSTREAM_PATH))
    truth = upstream['truth']
    theta_vi = upstream['theta_VI_obs']
    theta_ivi = upstream['final_theta']
    trajectory = upstream['trajectory']

    m1_truth, lse_truth = float(truth['mu1']), float(truth['log_sigma_eps'])
    m1_vi,    lse_vi    = float(theta_vi['mu1']), float(theta_vi['log_sigma_eps'])
    m1_ivi,   lse_ivi   = float(theta_ivi['mu1']), float(theta_ivi['log_sigma_eps'])
    traj_m1  = [float(t['theta_k']['mu1']) for t in trajectory]
    traj_lse = [float(t['theta_k']['log_sigma_eps']) for t in trajectory]

    N = int(upstream['config']['N'])
    T = int(upstream['config']['T'])
    obs_seed = int(upstream['config']['obs_seed'])

    m1_lo  = min(m1_truth,  m1_vi,  m1_ivi)  - grid_pad
    m1_hi  = max(m1_truth,  m1_vi,  m1_ivi)  + grid_pad
    lse_lo = min(lse_truth, lse_vi, lse_ivi) - grid_pad
    lse_hi = max(lse_truth, lse_vi, lse_ivi) + grid_pad
    m1_grid  = np.linspace(m1_lo,  m1_hi,  n_grid)
    lse_grid = np.linspace(lse_lo, lse_hi, n_grid)

    return {
        'truth': truth, 'N': N, 'T': T, 'obs_seed': obs_seed,
        'm1_grid': m1_grid, 'lse_grid': lse_grid,
        'markers': {'truth': (m1_truth, lse_truth),
                    'vi':    (m1_vi,    lse_vi),
                    'ivi':   (m1_ivi,   lse_ivi)},
        'trajectory_m1': traj_m1, 'trajectory_lse': traj_lse,
    }


def maybe_pre_draw_eps(fix_noise, T, N, noise_seed, device):
    """Pre-draw the encoder eps tensor for CRN. None when fix_noise=False."""
    if not fix_noise:
        return None
    # mean-field normal_diagonal: get_seed_dim() == T.
    torch.manual_seed(noise_seed)
    eps_fixed = torch.randn((1, N, T), dtype=torch.float32, device=device)
    print(f"[crn] eps_fixed pre-drawn at seed={noise_seed} "
          f"shape={tuple(eps_fixed.shape)}", flush=True)
    return eps_fixed


def pre_draw_strong_crn(N, T, obs_seed, device):
    """Pre-draw z_noise, y_noise for reparam_simulate (strong CRN)."""
    torch.manual_seed(obs_seed)
    z_noise = torch.randn(N, T, device=device)
    y_noise = torch.randn(N, T, device=device)
    return z_noise, y_noise


def run_cell(i, j, m1, lse, prob, device, n_epochs_vi, lr, eps_fixed,
              z_noise, y_noise):
    """Compute (logp contour, full-VI binding-map output) for cell (i, j)."""
    truth = prob['truth']; T = prob['T']
    PARAM_NAMES = M.PARAM_NAMES

    # y_obs (used for the contour log-p).
    y_obs = M.simulate_from_truth(prob['N'], T, truth, prob['obs_seed'],
                                    device)

    # --- Closed-form log-p contour for y_obs at theta = (m1, lse, truth) ---
    logp_val = float(M.ar1n_marginal_logp(
        y_obs,
        mu1=m1,
        sigma0=truth['sigma0'],
        z1_log_std=truth['z1_log_std'],
        log_sigma_eps=lse))

    # --- Simulate y_sim(theta_0) ---
    theta0 = np.array([truth[k] for k in PARAM_NAMES], dtype=np.float64)
    theta0[PARAM_NAMES.index('mu1')] = float(m1)
    theta0[PARAM_NAMES.index('log_sigma_eps')] = float(lse)
    theta0_t = torch.tensor(theta0, device=device, dtype=torch.float32)
    with torch.no_grad():
        y_sim = M.reparam_simulate(theta0_t, z_noise, y_noise)

    # --- Full mean-field VI binding-map fit on y_sim (8 free params) ---
    t0 = time.time()
    fit = full_vi_fit(
        y_sim, T, device, float(m1), float(lse), truth,
        n_epochs=n_epochs_vi, lr=lr, eps_fixed=eps_fixed)
    wall_vi = time.time() - t0

    if fit is None:
        b_m1 = b_lse = None
        b_aux = {f"b_{k}": None for k in M.FREE_NAMES
                   if k not in ('mu1', 'log_sigma_eps')}
    else:
        b_m1  = float(fit['mu1'])
        b_lse = float(fit['log_sigma_eps'])
        # All non-target free params recorded as auxiliaries (6 keys).
        b_aux = {f"b_{k}": float(fit[k]) for k in M.FREE_NAMES
                   if k not in ('mu1', 'log_sigma_eps')}

    return {
        'i': i, 'j': j,
        'm1_0':              float(m1),
        'lse_0':             float(lse),
        'logp_exact':        logp_val,
        'b_m1':              b_m1,
        'b_lse':             b_lse,
        **b_aux,
        'wall_vi_s':         wall_vi,
    }


def run_bstar(prob, device, n_epochs_vi, lr, eps_fixed):
    """Full mean-field VI on y_obs (the fixed-point reference b*).

    Reports the converged target-coord projection (b_star_m1, b_star_lse)
    plus all six non-target free params as `b_star_<name>` auxiliaries.
    """
    truth = prob['truth']; T = prob['T']
    m1_truth, lse_truth = prob['markers']['truth']
    y_obs = M.simulate_from_truth(prob['N'], T, truth, prob['obs_seed'],
                                    device)
    t0 = time.time()
    fit = full_vi_fit(
        y_obs, T, device, m1_truth, lse_truth, truth,
        n_epochs=n_epochs_vi, lr=lr, eps_fixed=eps_fixed)
    if fit is None:
        b_star_m1 = b_star_lse = None
        aux = {f"b_star_{k}": None for k in M.FREE_NAMES
                 if k not in ('mu1', 'log_sigma_eps')}
    else:
        b_star_m1  = float(fit['mu1'])
        b_star_lse = float(fit['log_sigma_eps'])
        aux = {f"b_star_{k}": float(fit[k]) for k in M.FREE_NAMES
                 if k not in ('mu1', 'log_sigma_eps')}
    return {
        'b_star_m1':           b_star_m1,
        'b_star_lse':          b_star_lse,
        **aux,
        'wall_s':              time.time() - t0,
    }


def aggregate(prob, cfg):
    n_grid = cfg['n_grid']
    m1_grid = prob['m1_grid']; lse_grid = prob['lse_grid']
    logp_grid = np.full((n_grid, n_grid), np.nan)
    b_m1  = np.full((n_grid, n_grid), np.nan)
    b_lse = np.full((n_grid, n_grid), np.nan)
    n_found = 0
    for i in range(n_grid):
        for j in range(n_grid):
            p = cell_path(i, j)
            if not os.path.exists(p):
                continue
            with open(p) as f:
                entry = json.load(f)
            cell = entry.get('cell')
            if cell is None:
                continue
            # Cell layout convention: i = lse row, j = m1 col.
            # The schema fields `m1_0`, `lse_0` carry the grid values so
            # the consumer is decoupled from the indexing.
            logp_grid[i, j] = cell['logp_exact']
            if cell['b_m1'] is not None:
                b_m1[i, j] = cell['b_m1']
            if cell['b_lse'] is not None:
                b_lse[i, j] = cell['b_lse']
            n_found += 1
    # b_star
    b_star_m1 = None; b_star_lse = None
    bp = bstar_path()
    if os.path.exists(bp):
        with open(bp) as f:
            star = json.load(f)
        b_star_m1  = star.get('bstar', {}).get('b_star_m1')
        b_star_lse = star.get('bstar', {}).get('b_star_lse')
    else:
        print(f"  [aggregate] missing {bp}", flush=True)
    m1_truth, lse_truth = prob['markers']['truth']
    m1_vi,    lse_vi    = prob['markers']['vi']
    m1_ivi,   lse_ivi   = prob['markers']['ivi']
    out = {
        'config': {**cfg, 'upstream': UPSTREAM_PATH},
        'm1_grid':         m1_grid.tolist(),
        'lse_grid':        lse_grid.tolist(),
        'logp_grid':       logp_grid.tolist(),
        'b_m1':            b_m1.tolist(),
        'b_lse':           b_lse.tolist(),
        'b_star_m1':       b_star_m1,
        'b_star_lse':      b_star_lse,
        'truth':           {'mu1': m1_truth, 'log_sigma_eps': lse_truth},
        'vi':              {'mu1': m1_vi,    'log_sigma_eps': lse_vi},
        'ivi':             {'mu1': m1_ivi,   'log_sigma_eps': lse_ivi},
        'trajectory_m1':   prob['trajectory_m1'],
        'trajectory_lse':  prob['trajectory_lse'],
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_PATH, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nAggregated {n_found}/{n_grid*n_grid} cells "
          f"(+bstar={b_star_m1 is not None}) -> {OUT_PATH}", flush=True)


def main():
    n_grid = int(os.environ.get('N_GRID', 10))
    n_epochs_vi = int(os.environ.get('N_EPOCHS_VI', 8000))
    lr = float(os.environ.get('LR', 1e-2))
    fix_noise = int(os.environ.get('FIX_NOISE', 1)) != 0
    noise_seed = int(os.environ.get('NOISE_SEED', 12345))
    grid_pad = float(os.environ.get('GRID_PAD', 0.30))
    cell_i_env = os.environ.get('CELL_I', None)
    cell_j_env = os.environ.get('CELL_J', None)
    mode = os.environ.get('MODE', '').lower()

    cfg = {
        'dgp': 'simulation-ar1n', 'n_grid': n_grid,
        'n_epochs_vi': n_epochs_vi, 'lr': lr, 'fix_noise': fix_noise,
        'noise_seed': noise_seed, 'grid_pad': grid_pad,
        'hidden_dim': HIDDEN_DIM,
    }
    prob = load_problem(grid_pad, n_grid)
    cfg.update({'N': prob['N'], 'T': prob['T'], 'obs_seed': prob['obs_seed']})

    os.makedirs(OUT_DIR, exist_ok=True)

    # --- MODE=aggregate: merge cell + bstar JSONs, no GPU work ---------
    if mode == 'aggregate':
        print(f"=== aggregate {n_grid}x{n_grid} cell files -> "
              f"quiver.json ===", flush=True)
        aggregate(prob, cfg)
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    N, T = prob['N'], prob['T']
    print(f"device={device}", flush=True)
    print(f"N={N} T={T} obs_seed={prob['obs_seed']}", flush=True)
    print(f"hyperparams: n_epochs_vi={n_epochs_vi} lr={lr} "
          f"fix_noise={fix_noise} noise_seed={noise_seed}", flush=True)
    m1_grid = prob['m1_grid']; lse_grid = prob['lse_grid']
    print(f"Grid: mu_1 in [{m1_grid[0]:+.3f}, {m1_grid[-1]:+.3f}]  "
          f"log_sigma_eps in [{lse_grid[0]:+.3f}, {lse_grid[-1]:+.3f}]  "
          f"{n_grid}x{n_grid}={n_grid**2} cells", flush=True)

    eps_fixed = maybe_pre_draw_eps(fix_noise, T, N, noise_seed, device)

    # --- MODE=bstar: only run the y_obs reference fit ------------------
    if mode == 'bstar':
        print("=== MODE=bstar: full VI on y_obs ===", flush=True)
        res = run_bstar(prob, device, n_epochs_vi, lr, eps_fixed)
        print(f"  b*: m1={res['b_star_m1']}  lse={res['b_star_lse']}  "
              f"({res['wall_s']:.0f}s)", flush=True)
        out = {'config': {**cfg, 'upstream': UPSTREAM_PATH}, 'bstar': res}
        with open(bstar_path(), 'w') as f:
            json.dump(out, f, indent=2)
        print(f"\nDone. Wrote {bstar_path()}", flush=True)
        return

    # Strong-CRN tensors for simulating y_sim(theta_0).
    z_noise, y_noise = pre_draw_strong_crn(N, T, prob['obs_seed'], device)

    # --- CELL_I=<i> CELL_J=<j>: per-cell mode --------------------------
    if cell_i_env is not None or cell_j_env is not None:
        if cell_i_env is None or cell_j_env is None:
            sys.exit("Need both CELL_I and CELL_J for per-cell mode")
        i = int(cell_i_env); j = int(cell_j_env)
        if not (0 <= i < n_grid and 0 <= j < n_grid):
            sys.exit(f"CELL_I,CELL_J=({i},{j}) out of range [0,{n_grid})")
        lse = float(lse_grid[i])
        m1  = float(m1_grid[j])
        print(f"=== CELL=({i},{j}): m1={m1:+.4f}  lse={lse:+.4f} ===",
              flush=True)
        cell = run_cell(i, j, m1, lse, prob, device, n_epochs_vi, lr,
                          eps_fixed, z_noise, y_noise)
        print(f"  logp_exact={cell['logp_exact']:+.4f}  "
              f"b_m1={cell['b_m1']}  b_lse={cell['b_lse']}  "
              f"({cell['wall_vi_s']:.0f}s)", flush=True)
        out = {'config': {**cfg, 'upstream': UPSTREAM_PATH}, 'cell': cell}
        with open(cell_path(i, j), 'w') as f:
            json.dump(out, f, indent=2)
        print(f"\nDone. Wrote {cell_path(i, j)}", flush=True)
        return

    # --- Default: sequential full sweep --------------------------------
    print(f"=== full sequential sweep ({n_grid**2} cells) ===", flush=True)
    t_start = time.time()
    for i in range(n_grid):
        for j in range(n_grid):
            cp = cell_path(i, j)
            if os.path.exists(cp):
                print(f"  cell ({i},{j}) cached, skipping", flush=True)
                continue
            lse = float(lse_grid[i])
            m1  = float(m1_grid[j])
            print(f"--- cell ({i},{j}): m1={m1:+.4f}  lse={lse:+.4f} ---",
                  flush=True)
            cell = run_cell(i, j, m1, lse, prob, device, n_epochs_vi, lr,
                              eps_fixed, z_noise, y_noise)
            with open(cp, 'w') as f:
                json.dump({'config': {**cfg, 'upstream': UPSTREAM_PATH},
                           'cell': cell}, f, indent=2)
    # b*
    if not os.path.exists(bstar_path()):
        print("--- b*: full VI on y_obs ---", flush=True)
        res = run_bstar(prob, device, n_epochs_vi, lr, eps_fixed)
        with open(bstar_path(), 'w') as f:
            json.dump({'config': {**cfg, 'upstream': UPSTREAM_PATH},
                       'bstar': res}, f, indent=2)
    aggregate(prob, cfg)
    print(f"\nDone in {(time.time()-t_start)/60:.1f} min.", flush=True)


if __name__ == "__main__":
    # Fix every RNG (python / numpy / torch CPU+CUDA) and request deterministic
    # kernels before any draw; the script's own explicit seeds still apply.
    from mlye.seeding import seed_everything, env_seed
    seed_everything(env_seed("OBS_SEED", 11))
    main()
