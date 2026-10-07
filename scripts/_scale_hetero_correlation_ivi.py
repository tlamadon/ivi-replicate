r"""Anderson IVI on the scale-heterogeneity DGP — try to correct the
kurtosis-variance bias on mu_alpha, log_beta that joint VI exhibits.

Companion to `_scale_hetero_correlation_sweep.py` (which establishes
the joint-VI bias baseline at the same hockey-stick Phase-1
hyperparameters). Here we run a few Anderson IVI outer iters starting
from each cell's theta_VI_obs, simulating y_sim from theta_0 at each
outer step and re-fitting full joint VI on the simulated data.

Reuses the sweep script's:
  - TRUTH_BASE, MU_ALPHA_MARG, SIGMA_ALPHA_MARG
  - calibrate_extra_hetero, build_model, set_truth, simulate_panel
  - extract_params

Adds:
  - PARAM_NAMES (16-vec ordering)
  - reparam_simulate (differentiable in the 16-vec; needed by the outer
    loop to push gradients through y_sim)
  - fit_vi_inner (re-uses the sweep script's loop)
  - anderson_step (taken from `_simulation_hockeystick_ivi_sweep.py`)
  - outer Anderson loop

Output: output/bundles/cells/_scale_hetero_correlation_ivi/anderson_rho<RHO>.json
"""
import os
import sys
import time
import json

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _scale_hetero_correlation_sweep import (
    TRUTH_BASE, MU_ALPHA_MARG, SIGMA_ALPHA_MARG,
    calibrate_extra_hetero, build_model, set_truth, simulate_panel,
    extract_params,
)


# Same ordering everywhere — 13 base + 3 hetero = 16 params.
PARAM_NAMES = [
    'mu0', 'mu1', 'mu2',
    'sigma0', 'sigma1', 'sigma2',
    'z1_log_std', 'z1_skew', 'z1_log_tail',
    'log_beta',
    'beta_a0', 'beta_a1', 'log_sigma_a_cond',
]
DIM = len(PARAM_NAMES)  # 13 for this hetero-scale model (no log_sigma_eps)


def theta_to_vec(theta):
    return np.array([float(theta[k]) for k in PARAM_NAMES], dtype=np.float64)


def vec_to_theta(vec):
    return {k: float(v) for k, v in zip(PARAM_NAMES, vec)}


def L2_dist(theta_vec, truth_vec):
    return float(np.linalg.norm(theta_vec - truth_vec))


# ---- Differentiable simulator wrt the 13-vec theta_0 --------------------

def reparam_simulate(theta_vec, u_z, u_alpha, u_y):
    """Differentiable simulate (y, z, alpha) under the scale-hetero DGP.

    theta_vec: (13,) tensor in PARAM_NAMES order.
    u_z: (N, T) — N(0,1) noise driving z_t (z_1 from u_z[:,0]; transitions
           from u_z[:,1:])
    u_alpha: (N,) — N(0,1) noise driving alpha
    u_y:   (N, T) — N(0,1) noise driving y_t
    Returns y (N, T).
    """
    mu0          = theta_vec[0]
    mu1          = theta_vec[1]
    mu2          = theta_vec[2]
    sigma0       = theta_vec[3]
    sigma1       = theta_vec[4]
    sigma2       = theta_vec[5]
    z1_log_std   = theta_vec[6]
    z1_skew      = theta_vec[7]
    z1_log_tail  = theta_vec[8]
    log_beta     = theta_vec[9]
    beta_a0      = theta_vec[10]
    beta_a1      = theta_vec[11]
    log_sigma_a_cond = theta_vec[12]

    z1_scale = F.softplus(z1_log_std)
    z1_tail  = torch.exp(z1_log_tail)
    sigma_a  = torch.exp(log_sigma_a_cond)
    beta     = torch.exp(log_beta)

    N, T = u_z.shape
    z = torch.zeros(N, T, device=u_z.device, dtype=u_z.dtype)
    # z_1 ~ sinh-arcsinh
    z[:, 0] = z1_scale * torch.sinh(
        (torch.asinh(u_z[:, 0]) + z1_skew) * z1_tail)
    # z_t | z_{t-1}: poly-2 mu, softplus(poly-2) sigma — no softplus kink
    for t in range(1, T):
        zlag = z[:, t - 1].clone()
        mu_t = (mu0 + mu1 * zlag + mu2 * zlag ** 2).clamp(-10.0, 10.0)
        sig_raw = sigma0 + sigma1 * zlag + sigma2 * zlag ** 2
        sig_t = F.softplus(sig_raw).clamp(0.0, 10.0) + 1e-3
        z[:, t] = mu_t + sig_t * u_z[:, t]
    # alpha | z_1: linear
    alpha = beta_a0 + beta_a1 * z[:, 0] + sigma_a * u_alpha
    # y_t = z_t + exp(alpha) * sinh-arcsinh(0, 1, 0, beta) noise
    eps_y = torch.sinh(torch.asinh(u_y) * beta)
    y = z + torch.exp(alpha).unsqueeze(1) * eps_y
    return y


# ---- Inner VI fit (full joint: encoder + prior + decoder) ---------------

def fit_vi_inner(y, T, n_epochs, lr, vi_seed, fix_noise, noise_seed,
                  hidden_dim, device, init_at_truth=False, truth_dict=None,
                  log_every=None, label=''):
    """Train a fresh model (or one initialised at truth) on y for n_epochs
    epochs of AdamW. Returns the converged theta dict."""
    torch.manual_seed(vi_seed); np.random.seed(vi_seed)
    model = build_model(T, device, hidden_dim=hidden_dim)
    if init_at_truth and truth_dict is not None:
        # Set prior+decoder at truth, leave encoder fresh.
        set_truth(model, TRUTH_BASE,
                    truth_dict['beta_a0'], truth_dict['beta_a1'],
                    truth_dict['log_sigma_a_cond'])
    # Some experiments use warm start; here, fresh init is the IVI default.
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    if fix_noise:
        torch.manual_seed(noise_seed)
        seed_dim = model.encoder.get_seed_dim()
        u_crn = torch.randn(1, y.shape[0], seed_dim, device=device,
                              dtype=torch.float32)
    t0 = time.time()
    if log_every is None:
        log_every = max(1, n_epochs // 5)
    nan_streak = 0
    trace = []
    for epoch in range(n_epochs):
        opt.zero_grad()
        if fix_noise:
            loss = -model.elbo(y, ndraws=1, eps_all=u_crn)
        else:
            loss = -model.elbo(y, ndraws=1)
        if not torch.isfinite(loss):
            nan_streak += 1
            if nan_streak > 50 and epoch < 500:
                raise RuntimeError(f"NaN at epoch {epoch}")
            continue
        nan_streak = 0
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if (epoch + 1) % log_every == 0:
            p = extract_params(model)
            print(f"    [{label}] ep {epoch+1:5d}: ELBO={-loss.item():+.3f}  "
                  f"mu_a={p['_mu_alpha_marg_implied']:+.3f}  "
                  f"sd_a={p['_sigma_alpha_marg_implied']:.3f}  "
                  f"log_b={p['log_beta']:+.3f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
            trace.append({'epoch': epoch + 1, 'elbo': float(-loss.item())})
    p = extract_params(model)
    p['_wall_s'] = float(time.time() - t0)
    # build the canonical theta dict (PARAM_NAMES) from the fitted model
    theta = {k: p[k] for k in PARAM_NAMES}
    return theta, trace, model


# ---- Anderson (type-II) update -----------------------------------------

def anderson_step(x_k, g_k, history, m, beta, reg=1e-8):
    """Anderson type-II: x_{k+1} = x_k + beta * g_k - (DeltaX + beta DeltaG) gamma."""
    if not history:
        return x_k + beta * g_k, np.zeros(0)
    Xs = np.array([h['x'] for h in history[-m:]])
    Gs = np.array([h['g'] for h in history[-m:]])
    dX = Xs - x_k
    dG = Gs - g_k
    # gamma = argmin || g_k - dG.T gamma ||
    A = dG.T  # shape (D, m)
    try:
        # Normal equations (m x m) — fit gamma in R^m, not R^D.
        gamma, *_ = np.linalg.lstsq(A.T @ A + reg * np.eye(A.shape[1]),
                                       A.T @ g_k, rcond=None)
    except np.linalg.LinAlgError:
        gamma = np.zeros(A.shape[1])
    x_next = x_k + beta * g_k - (dX.T + beta * dG.T) @ gamma
    return x_next, gamma


# ---- Phase 1: theta_VI_obs (a re-fit identical to the sweep) ------------

def fit_phase1(rho, N, T, n_epochs, lr, hidden_dim, vi_seed, sim_seed,
                fix_noise, noise_seed, device):
    print(f"\n--- Phase 1: theta_VI_obs at rho={rho} ---", flush=True)
    # 1. set truth and simulate y_obs
    beta_a0, beta_a1, log_sigma_a_cond = calibrate_extra_hetero(
        rho, MU_ALPHA_MARG, SIGMA_ALPHA_MARG, TRUTH_BASE['z1_log_std'])
    truth_dict = dict(TRUTH_BASE)
    truth_dict['beta_a0']          = beta_a0
    truth_dict['beta_a1']          = beta_a1
    truth_dict['log_sigma_a_cond'] = log_sigma_a_cond
    truth_dict['rho_target']       = rho

    torch.manual_seed(sim_seed); np.random.seed(sim_seed)
    truth_model = build_model(T, device, hidden_dim=hidden_dim)
    set_truth(truth_model, TRUTH_BASE, beta_a0, beta_a1, log_sigma_a_cond)
    y_obs, _, alpha_true = simulate_panel(truth_model, N, T, sim_seed, device)
    del truth_model
    print(f"  y range: [{float(y_obs.min()):.2f}, {float(y_obs.max()):.2f}]  "
          f"alpha emp mean={float(alpha_true.mean()):.3f}  "
          f"sd={float(alpha_true.std()):.3f}", flush=True)

    # 2. fit JN h=64 joint VI from random init
    theta_VI_obs, trace, _ = fit_vi_inner(
        y_obs, T, n_epochs, lr, vi_seed, fix_noise, noise_seed, hidden_dim,
        device, init_at_truth=False, label=f'VI_obs')
    return y_obs, theta_VI_obs, truth_dict, trace


# ---- Main ---------------------------------------------------------------

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)

    N        = int(os.environ.get('N', 30000))
    T        = int(os.environ.get('T', 6))
    HIDDEN   = int(os.environ.get('HIDDEN_DIM', 64))
    N_EP_VI_OBS = int(os.environ.get('N_EPOCHS_VI_OBS', 20000))
    N_EP_INNER  = int(os.environ.get('N_EPOCHS_INNER',  16000))
    LR       = float(os.environ.get('LR', 1e-2))
    FIX_NOISE= int(os.environ.get('FIX_NOISE', 1))
    NOISE_SEED=int(os.environ.get('NOISE_SEED', 12345))
    SIM_SEED = int(os.environ.get('SIM_SEED', 11))
    VI_SEED  = int(os.environ.get('VI_SEED', 11007))
    N_ITERS  = int(os.environ.get('N_ITERS', 5))
    M_MEM    = int(os.environ.get('M_MEM', 4))
    BETA     = float(os.environ.get('BETA', 0.6))
    RESTART_FACTOR = float(os.environ.get('RESTART_FACTOR', 1.5))
    RHO      = float(os.environ.get('RHO', 0.0))

    print(f"Config: N={N} T={T} HIDDEN={HIDDEN} N_EP_VI_OBS={N_EP_VI_OBS} "
          f"N_EP_INNER={N_EP_INNER} LR={LR} N_ITERS={N_ITERS} M_MEM={M_MEM} "
          f"BETA={BETA} RHO={RHO}", flush=True)

    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "output/bundles/cells/_scale_hetero_correlation_ivi")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir,
                              f"anderson_rho{RHO:.2f}.json".replace('.', '_'))
    out_path = out_path.replace('_json', '.json')

    # --- Phase 1 ---
    y_obs, theta_VI_obs, truth_dict, trace_vi_obs = fit_phase1(
        RHO, N, T, N_EP_VI_OBS, LR, HIDDEN, VI_SEED, SIM_SEED, FIX_NOISE,
        NOISE_SEED, device)
    theta_VI_obs_vec = theta_to_vec(theta_VI_obs)
    truth_vec = theta_to_vec(truth_dict)
    L2_VI = L2_dist(theta_VI_obs_vec, truth_vec)
    print(f"\ntheta_VI_obs:  mu_a={theta_VI_obs['beta_a0']:+.4f}  "
          f"log_b={theta_VI_obs['log_beta']:+.4f}  "
          f"L2(VI, truth)={L2_VI:.4f}", flush=True)

    payload = {
        'config': {'N': N, 'T': T, 'hidden_dim': HIDDEN, 'lr': LR,
                    'n_epochs_vi_obs': N_EP_VI_OBS,
                    'n_epochs_inner': N_EP_INNER,
                    'n_iters': N_ITERS, 'm_mem': M_MEM, 'beta': BETA,
                    'restart_factor': RESTART_FACTOR,
                    'fix_noise': bool(FIX_NOISE), 'noise_seed': NOISE_SEED,
                    'sim_seed': SIM_SEED, 'vi_seed': VI_SEED,
                    'rho': RHO,
                    'param_names': PARAM_NAMES,
                    'truth_base': TRUTH_BASE,
                    'mu_alpha_marg': MU_ALPHA_MARG,
                    'sigma_alpha_marg': SIGMA_ALPHA_MARG},
        'truth':         truth_dict,
        'theta_VI_obs':  theta_VI_obs,
        'L2_VI_obs_truth': L2_VI,
        'trajectory':    [{'iter': 0,
                            'theta_k': theta_VI_obs,
                            'L2_truth': L2_VI}],
    }
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"  wrote {out_path}", flush=True)

    # --- Phase 2: Anderson outer loop ---
    torch.manual_seed(SIM_SEED * 1000)
    u_z     = torch.randn(N, T, device=device, dtype=torch.float32)
    u_alpha = torch.randn(N,    device=device, dtype=torch.float32)
    u_y     = torch.randn(N, T, device=device, dtype=torch.float32)

    theta_k = dict(theta_VI_obs)
    history = []
    prev_g_norm = None
    print(f"\n=== Phase 2: IVI outer loop (Anderson m={M_MEM} beta={BETA}, "
          f"{N_ITERS} iters) ===", flush=True)
    for k in range(1, N_ITERS + 1):
        iter_t0 = time.time()
        theta_t = torch.tensor([theta_k[name] for name in PARAM_NAMES],
                                  dtype=torch.float32, device=device)
        with torch.no_grad():
            y_sim = reparam_simulate(theta_t, u_z, u_alpha, u_y)

        print(f"\n[iter {k}] inner VI on y_sim ({N_EP_INNER} ep) ...",
              flush=True)
        theta_VI, inner_trace, _ = fit_vi_inner(
            y_sim, T, N_EP_INNER, LR, VI_SEED, FIX_NOISE, NOISE_SEED, HIDDEN,
            device, init_at_truth=False, label=f'iter{k}')
        iter_wall = time.time() - iter_t0
        theta_VI_vec = theta_to_vec(theta_VI)
        x_k = theta_to_vec(theta_k)

        # residual & Anderson update
        g_k = theta_VI_obs_vec - theta_VI_vec
        g_norm = float(np.linalg.norm(g_k))
        restart = False
        if prev_g_norm is not None and g_norm > RESTART_FACTOR * prev_g_norm:
            print(f"    [restart] ||g||={g_norm:.4f} "
                  f"(was {prev_g_norm:.4f})", flush=True)
            history = []
            restart = True
        x_next, gamma = anderson_step(x_k, g_k, history, m=M_MEM, beta=BETA)
        history.append({'x': x_k.copy(), 'g': g_k.copy()})
        prev_g_norm = g_norm

        theta_next = vec_to_theta(x_next)
        L2 = L2_dist(x_next, truth_vec)
        print(f"  inner VI final: mu_a={theta_VI['beta_a0']:+.4f}  "
              f"log_b={theta_VI['log_beta']:+.4f}  "
              f"sigma0={theta_VI['sigma0']:+.4f}", flush=True)
        print(f"  theta_{k}: mu_a={theta_next['beta_a0']:+.4f}  "
              f"log_b={theta_next['log_beta']:+.4f}  "
              f"L2={L2:.4f}  ||g||={g_norm:.4f}  iter_wall={iter_wall:.1f}s",
              flush=True)

        payload['trajectory'].append({
            'iter':         k,
            'iter_wall_s':  iter_wall,
            'theta_k':      theta_next,
            'theta_VI':     theta_VI,
            'L2_truth':     L2,
            'g_norm':       g_norm,
            'restart':      bool(restart),
            'gamma':        gamma.tolist() if len(gamma) else None,
            'inner_trace':  inner_trace,
        })
        with open(out_path, 'w') as f:
            json.dump(payload, f, indent=2)
        print(f"  [cache write: {out_path}]", flush=True)
        theta_k = theta_next

    payload['final_theta'] = theta_k
    payload['final_L2']    = L2_dist(theta_to_vec(theta_k), truth_vec)
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"\nDone. final L2={payload['final_L2']:.4f}", flush=True)


if __name__ == "__main__":
    main()
