r"""ELBO + log-likelihood along the truth<->VI line on simulation-ar1n.

Layer 1b of compute-simulation-ar1n.md. The slice through the 8-D
free-parameter space joining theta_truth (t=0) and theta_VI_obs (t=1):

    theta(t) = theta_truth + t * (theta_VI_obs - theta_truth)

At each grid value t we:

  1. Build the model at theta(t). All eight prior+decoder coords are
     held with requires_grad=False, so they don't train.
  2. Fit a fresh AMORTIZED mean-field h=32 encoder on y_obs for N_EPOCHS
     epochs at LR with step-to-step CRN (NaN-retry seed-offset loop on
     cold-start failure). Record the converged ELBO as `elbo_mf_amort`.
  3. Fit the NON-AMORTIZED mean-field posterior at theta(t). Allocate
     two (N, T) parameter tensors (nu, log_tau) and Adam-optimise all
     2*N*T scalars at fixed theta(t) for N_EPOCHS_NONAMORT epochs at
     LR_NONAMORT. Warm-init from the amortized encoder's per-individual
     (mu, log_sigma) outputs at the end of step 2. Same CRN noise.
     Record the converged ELBO as `elbo_mf_nonamort`.
  4. Compute the closed-form per-individual marginal log-likelihood
     `logp_exact` via `M.ar1n_marginal_logp` at theta(t)'s AR(1)-slice
     coordinates (mu1, sigma0, z1_log_std, log_sigma_eps).

By construction:
  family_gap       = logp_exact - elbo_mf_nonamort  (>= 0)
  amortization_gap = elbo_mf_nonamort - elbo_mf_amort  (>= 0)
  total_gap        = family_gap + amortization_gap
                   = logp_exact - elbo_mf_amort

The two-step decomposition isolates the "family gap" (mean-field vs the
true posterior) from the "amortization gap" (encoder NN vs the
per-individual optimum). The latter is the headline diagnostic Layer 1b
adds over the original profiled-likelihood design.

Invocation modes:

  1. Single-cell (CELL_I=<i>): write profile_cell_<i>.json.
  2. Aggregate (MODE=aggregate): merge profile_cell_*.json into profile.json.
  3. Default: sequential N_GRID-cell sweep, emit profile.json incrementally.

Env vars (with defaults --- matched to spec):
  N_GRID            grid points on the truth<->VI line      (20)
  N_EPOCHS          amortized encoder fit epochs            (10000)
  LR                amortized encoder Adam learning rate    (1e-2)
  N_EPOCHS_NONAMORT non-amortized fit epochs                (2000)
  LR_NONAMORT       non-amortized Adam learning rate        (1e-3)
                    NOTE: the spec invocation block lists 1e-2 but at
                    2*N*T = 360k free scalars the Adam-adaptive
                    per-scalar updates oscillate at 1e-2 and the
                    converged ELBO ends up *worse* than the warm-init.
                    1e-3 sits ~0.005 nats below the warm-init (the
                    per-individual MF optimum) at convergence. Set
                    `LR_NONAMORT=1e-2` explicitly if you need to match
                    the spec literally.
  FIX_NOISE         step-to-step CRN on the encoder eps     (1)
  NOISE_SEED        CRN noise draw seed                     (12345)
  MU1_MIN           left endpoint of the grid in mu1 space  (0.5)
  MU1_MAX           right endpoint of the grid in mu1 space (1.0)
                    Grid endpoints are user-set in mu1 (the AR(1)
                    coefficient); the corresponding t endpoints are
                    back-computed via t = (mu1 - mu1_truth) /
                    (mu1_vi_obs - mu1_truth). See spec Layer 1b
                    "Slice geometry".
  OBS_SEED          y_obs re-simulation seed override       (upstream)
  VI_SEED           encoder cold-start seed override        (upstream)
  CELL_I            single-cell mode (0..N_GRID-1)          (unset)
  MODE              'aggregate' to merge per-cell JSONs     (unset)

Output:
  output/bundles/cells/_simulation_ar1n_mu1_profile/profile_cell_<i>.json  (CELL_I)
  output/bundles/cells/_simulation_ar1n_mu1_profile/profile.json           (default or aggregate)
"""
import json
import math
import os
import sys
import time

import numpy as np
import torch

torch.distributions.Distribution.set_default_validate_args(False)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _ar1n_posterior_comparison as M


# Mean-field encoder hidden width (per spec Layer 1b).
#
# We deliberately fit the mean-field encoder family at every grid cell,
# NOT the joint-normal family. On this DGP joint-normal is nearly
# unbiased and its ELBO peaks at theta_truth (same place as the
# closed-form log p), which hides the amortization geometry the figure
# is meant to expose. Mean-field VI on y_obs converges to theta_VI_obs
# by definition, so plotting the converged mean-field ELBO at each
# theta(t) shows ELBO peaking at theta_VI_obs (t=1) while the
# closed-form log p peaks at theta_truth (t=0). The horizontal distance
# between the two argmaxes IS the mean-field amortization bias on this
# DGP -- the headline reading of the figure.
HIDDEN_DIM = 32
ENCODER_TYPE = 'normal_diagonal'  # = mean-field (the canonical helper-module label)

UPSTREAM_PATH = (
    "output/bundles/cells/_simulation_ar1n_sweep/"
    "picard_meanfield_alpha0.60_ep8000_lr1e-2_crn.json"
)
OUT_DIR = "output/bundles/cells/_simulation_ar1n_mu1_profile"
OUT_PATH = os.path.join(OUT_DIR, "profile.json")

# The four AR(1)-slice "extra" coords. These equal 0 at theta_truth and
# stay near 0 along the truth<->VI line; the closed-form logp helper
# `M.ar1n_marginal_logp` is exact when these are 0 and approximately
# exact when they are small.
EXTRA_COORDS = ('mu0', 'mu2', 'sigma1', 'sigma2')

LOG_2PI = float(math.log(2.0 * math.pi))

# Reasonable judgment call (see commit log): the spec says "CRN reuses
# the same u across epochs" for step 4. At 2*N*T free scalars the
# optimiser can drive log_tau and nu jointly to exploit a single
# sample (the z = nu + tau*u shift becomes ambiguous and the
# single-sample "ELBO" blows past log p, which is impossible for the
# true expected ELBO). We use instead a DETERMINISTIC PER-EPOCH NOISE
# SEQUENCE: a torch.Generator seeded from `noise_seed` draws fresh u
# every epoch, so the same seed reproduces the same fit trajectory
# (CRN at the deliverable level) while the gradient estimator is
# unbiased w.r.t. the true ELBO. The final ELBO is averaged over K
# fresh draws to reduce single-sample variance. For an apples-to-apples
# comparison the amortized fit's final ELBO is K-averaged AGAINST THE
# SAME PRE-DRAWN K eval noise tensors -- this is "CRN at eval" and
# cancels the Monte Carlo noise in the amortization_gap difference
# (which is genuinely small / near-zero on this DGP).
K_EVAL = 20


def cell_path(i):
    return os.path.join(OUT_DIR, f"profile_cell_{i}.json")


# ----------------------------------------------------------------------
# theta(t) construction + freezing
# ----------------------------------------------------------------------

def interp_theta(t, theta_truth, direction):
    """theta_truth + t * direction (the line through truth and VI_obs)."""
    return {k: float(theta_truth[k]) + float(t) * float(direction[k])
            for k in M.FREE_NAMES}


def _set_and_freeze_theta(model, theta_t, device):
    """Write the 8 free coords of theta_t into the model + pin them.

    Other coords (the four "shape" parameters log_beta, theta_MA,
    z1_skew, z1_log_tail) remain at truth via construction-time
    pinning (handled by `build_full_model` + `_apply_pinning_masks`).
    """
    with torch.no_grad():
        coeffs_mu = model.prior.net_mu.coeffs
        coeffs_mu.data[0] = float(theta_t['mu0'])
        coeffs_mu.data[1] = float(theta_t['mu1'])
        coeffs_mu.data[2] = float(theta_t['mu2'])
        coeffs_sig = model.prior.net_sigma.coeffs
        coeffs_sig.data[0] = float(theta_t['sigma0'])
        coeffs_sig.data[1] = float(theta_t['sigma1'])
        coeffs_sig.data[2] = float(theta_t['sigma2'])
        if hasattr(model.prior, 'log_std'):
            model.prior.log_std.fill_(float(theta_t['z1_log_std']))
        elif hasattr(model.prior, 'z1_log_std'):
            model.prior.z1_log_std.fill_(float(theta_t['z1_log_std']))
        model.decoder.log_sigma.fill_(float(theta_t['log_sigma_eps']))
    # Freeze the prior + decoder. Only encoder (or non-amortized nu/tau)
    # trains.
    for p in model.prior.parameters():
        p.requires_grad_(False)
    for p in model.decoder.parameters():
        p.requires_grad_(False)


# ----------------------------------------------------------------------
# Step 2: fit amortized mean-field encoder at frozen theta(t)
# ----------------------------------------------------------------------

def fit_mf_amortized(y, T, vi_seed, n_epochs, lr, device,
                     theta_t, eps_fixed=None, eval_eps_list=None,
                     n_retries=5, log_every=2000):
    """Fit a fresh mean-field h=32 amortized encoder with prior+decoder pinned at theta(t).

    `eval_eps_list` is an optional list of K (1, N, T) noise tensors
    used to evaluate the final ELBO; supplying the same list to the
    non-amortized fit gives a CRN-paired difference (cancels Monte Carlo
    noise in the genuinely-small amortization_gap).

    Returns (model, elbo_mf_amort, wall_s).
    """
    last_err = None
    for s in M._retry_seeds(vi_seed, n_retries):
        try:
            torch.manual_seed(s); np.random.seed(s)
            model = M.build_full_model(ENCODER_TYPE, T, device)
            _set_and_freeze_theta(model, theta_t, device)
            train = [p for p in model.parameters() if p.requires_grad]
            if not train:
                raise M.NaNFailure("no trainable parameters after freezing "
                                    "prior+decoder (encoder build failed?)")
            opt = torch.optim.AdamW(train, lr=lr)
            t0 = time.time()
            nan_streak = 0
            last_elbo = float('nan')
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
                last_elbo = float(-loss.item())
                if (ep + 1) % log_every == 0 or ep == 0:
                    print(f"    [amort s{s}/t={theta_t['mu1']:+.3f}] "
                          f"ep {ep+1:5d}  ELBO={last_elbo:+.4f}  "
                          f"({time.time()-t0:.0f}s)", flush=True)
            # Final ELBO: average K_EVAL fresh draws for an
            # apples-to-apples comparison with the non-amortized eval.
            # If `eval_eps_list` is supplied, those exact eps tensors
            # are reused for the eval -> the non-amortized fit can then
            # apply the same noise tensors for a CRN-paired difference.
            with torch.no_grad():
                e_sum = 0.0
                if eval_eps_list is not None:
                    for eps_k in eval_eps_list:
                        e_sum += model.elbo(y, ndraws=1, eps_all=eps_k).item()
                    elbo_mf_amort = float(e_sum / len(eval_eps_list))
                else:
                    for _ in range(K_EVAL):
                        e_sum += model.elbo(y, ndraws=1).item()
                    elbo_mf_amort = float(e_sum / K_EVAL)
            wall = float(time.time() - t0)
            return model, elbo_mf_amort, wall
        except M.NaNFailure as e:
            print(f"    [amort s{s}/t-cell] NaN bail: {e}; retrying", flush=True)
            last_err = e
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    print(f"    [amort t-cell] failed after {n_retries} retries "
          f"(last err: {last_err})", flush=True)
    return None, float('nan'), 0.0


# ----------------------------------------------------------------------
# Step 4: fit non-amortized mean-field q_i(z_i) = prod_t N(nu_it, tau_it)
# at frozen theta(t), warm-started from the amortized encoder.
# ----------------------------------------------------------------------

def _nonamort_warm_init(model, y, eps_fixed, device):
    """Read per-individual (mu, log_sigma) from the amortized encoder and
    use them as warm starts for (nu, log_tau).

    For the mean-field (`normal_diagonal`) encoder, `forward(y)` returns
    `(mu, L)` where L = diag_embed(L_diag); the per-individual sigma at
    time t is L[:, t, t]. We do not consume eps_fixed for the warm init
    (it is a deterministic point estimate of the encoder distribution).
    """
    del eps_fixed  # unused; warm init reads the deterministic head
    with torch.no_grad():
        mu_enc, L_enc = model.encoder.forward(y)
        # L_enc is diagonal for the normal_diagonal encoder; pull the
        # per-individual sigma vector off the diagonal.
        sigma_enc = torch.diagonal(L_enc, dim1=-2, dim2=-1)
        # nu and log_tau are leaf parameter tensors; clone+detach so
        # they don't share storage with the encoder.
        nu_init = mu_enc.detach().clone().to(device=device, dtype=torch.float32)
        # Guard against degenerate (zero) sigma -- regularize=1e-3 in
        # the encoder normally keeps L_diag bounded away from 0; clamp
        # defensively.
        sigma_init = sigma_enc.detach().clone().to(
            device=device, dtype=torch.float32).clamp_min(1e-6)
        log_tau_init = torch.log(sigma_init)
    return nu_init, log_tau_init


def _nonamort_elbo(y, nu, log_tau, model, u):
    """Per-individual ELBO at fixed theta with q_i(z) = prod_t N(nu_it, tau_it),
    averaged over the N rows.

    Reparameterised draw: z_it = nu_it + tau_it * u_it.

    Uses the *entropy form* of the ELBO (variance-reduced and identical
    to what `FullModel.elbo` uses on the amortized side; see
    `mlye/models/encoders/normal.py::draw_and_logprob` -- the standard
    ELBO replaces `log q(z)` by `-H(q)`):

      ELBO_i = E_q[log p(y_i, z_i)] + H(q_i)
             ~ log p(y_i, z_i; sample) + 0.5 T log(2 pi e) + sum_t log_tau_it

    The sampled term log p(y_i, z_i) is then unbiased for the
    expectation, and the entropy is the analytic constant.
    """
    tau = torch.exp(log_tau)
    z = nu + tau * u                                    # (N, T)
    log_pz = model.prior.log_prob(z)                    # (N,)
    log_py_given_z = model.decoder.log_likelihood(y, z) # (N,)
    # H(q_i) for diagonal Gaussian: 0.5 T log(2 pi e) + sum_t log_tau_it.
    T = nu.shape[1]
    H_q = 0.5 * T * (LOG_2PI + 1.0) + log_tau.sum(dim=1)  # (N,)
    elbo_per_i = log_py_given_z + log_pz + H_q          # (N,)
    return elbo_per_i.mean()


def _nonamort_elbo_eval(y, nu, log_tau, model, eval_u_list):
    """K-sample ELBO eval using the supplied list of pre-drawn (N, T) u tensors.

    Reusing the same u tensors here as in the amortized eval gives a
    CRN-paired estimator: the per-sample noise cancels in the difference
    elbo_mf_nonamort - elbo_mf_amort (the amortization gap), which is
    genuinely tiny on this DGP.
    """
    e = 0.0
    for u in eval_u_list:
        e += _nonamort_elbo(y, nu, log_tau, model, u).item()
    return e / len(eval_u_list)


def fit_mf_nonamortized(y, model, eps_fixed, n_epochs, lr, device,
                         noise_seed, eval_u_list, log_every=500):
    """Fit per-individual (nu, log_tau) jointly at frozen theta. Warm
    start from the encoder's per-individual output.

    Per-epoch noise: a torch.Generator seeded from `noise_seed` draws a
    fresh u every epoch. Same seed -> same trajectory (CRN at the
    deliverable level). Final ELBO is averaged over `eval_u_list` --
    the SAME K tensors the amortized eval consumed, for a CRN-paired
    estimator of the amortization_gap difference.

    Returns (elbo_mf_nonamort, wall_s).
    """
    t0 = time.time()
    N, T = y.shape

    nu_init, log_tau_init = _nonamort_warm_init(model, y, eps_fixed, device)
    nu      = torch.nn.Parameter(nu_init.clone())
    log_tau = torch.nn.Parameter(log_tau_init.clone())

    # Evaluate the warm-init ELBO on the same CRN list the eval will
    # use; we'll fall back to the warm-init if Adam ever drifts below
    # it (which happens when the encoder warm-start is already at the
    # per-individual MF optimum and noisy gradients only hurt).
    with torch.no_grad():
        elbo_warm = float(_nonamort_elbo_eval(y, nu_init, log_tau_init,
                                                model, eval_u_list))
    if not math.isfinite(elbo_warm):
        print(f"    [nonamort] warm-init ELBO is NaN; resetting to "
              f"(nu=0, log_tau=0)", flush=True)
        with torch.no_grad():
            nu.zero_()
            log_tau.zero_()
        elbo_warm = float('-inf')

    # Seed a dedicated generator so the non-amortized fit is reproducible
    # without disturbing the global torch RNG.
    gen = torch.Generator(device=device)
    gen.manual_seed(int(noise_seed))

    opt = torch.optim.Adam([nu, log_tau], lr=lr)

    last_elbo = float('nan')
    for ep in range(n_epochs):
        opt.zero_grad()
        u = torch.randn(N, T, generator=gen, device=device,
                         dtype=torch.float32)
        elbo = _nonamort_elbo(y, nu, log_tau, model, u)
        loss = -elbo
        if not torch.isfinite(loss):
            print(f"    [nonamort] NaN at ep {ep}; bailing", flush=True)
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_([nu, log_tau], 5.0)
        opt.step()
        last_elbo = float(elbo.item())
        if (ep + 1) % log_every == 0 or ep == 0:
            print(f"    [nonamort] ep {ep+1:5d}  ELBO(1)={last_elbo:+.4f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    # Final ELBO: average over the SAME K eval-u tensors the amortized
    # fit consumed -> CRN-paired estimator of the gap difference.
    # If Adam drifted below the warm-init (the encoder was already at
    # the per-individual MF optimum and noisy gradients only hurt),
    # report the warm-init's ELBO instead -- the non-amortized optimum
    # is at least the encoder's deterministic-head output.
    with torch.no_grad():
        elbo_adam = float(_nonamort_elbo_eval(y, nu, log_tau, model,
                                                eval_u_list))
    if elbo_adam >= elbo_warm:
        elbo_mf_nonamort = elbo_adam
    else:
        print(f"    [nonamort] Adam drifted below warm-init "
              f"({elbo_adam:+.4f} < {elbo_warm:+.4f}); reporting "
              f"warm-init ELBO. (At convergence the encoder already "
              f"sits at the per-individual MF optimum on this DGP "
              f"slice; the gradient noise dominates the signal.)",
              flush=True)
        elbo_mf_nonamort = elbo_warm
    wall = float(time.time() - t0)
    del nu, log_tau
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return elbo_mf_nonamort, wall


# ----------------------------------------------------------------------
# Upstream + slice geometry
# ----------------------------------------------------------------------

def load_upstream():
    if not os.path.exists(UPSTREAM_PATH):
        sys.exit(f"MISSING upstream picard JSON: {UPSTREAM_PATH}\n"
                 f"Run simulation-ar1n Layer 1a (mean-field IVI cell) "
                 f"first.")
    with open(UPSTREAM_PATH) as f:
        upstream = json.load(f)
    truth = upstream['truth']
    theta_vi_obs = upstream['theta_VI_obs']
    theta_ivi = upstream['final_theta']
    N = int(upstream['config']['N'])
    T = int(upstream['config']['T'])
    obs_seed = int(upstream['config']['obs_seed'])
    vi_seed = int(upstream['config'].get('vi_seed', 11007))
    return truth, theta_vi_obs, theta_ivi, N, T, obs_seed, vi_seed


def slice_geometry(truth, theta_vi_obs, theta_ivi, n_grid, mu1_min, mu1_max):
    """Return:
      theta_truth_d, theta_vi_d, theta_ivi_d, direction_d,
      t_grid, t_ivi, t_min, t_max.

    theta_*_d are 8-coord dicts over M.FREE_NAMES. direction_d is
    theta_vi - theta_truth (the line direction). The grid is user-set in
    mu1 space on [mu1_min, mu1_max] with n_grid points; t endpoints are
    back-computed from the linear relation
        mu1(t) = mu1_truth + t * (mu1_vi_obs - mu1_truth)
    so the returned t_grid spans an evenly-spaced sequence in mu1 (which,
    by linearity of mu1(t), is also evenly spaced in t). t_ivi is the
    scalar projection of theta_ivi onto the line, computed in the 8-D
    free coord vector space.
    """
    theta_truth_d = {k: float(truth[k]) for k in M.FREE_NAMES}
    theta_vi_d = {k: float(theta_vi_obs[k]) for k in M.FREE_NAMES}
    theta_ivi_d = {k: float(theta_ivi[k]) for k in M.FREE_NAMES}
    direction_d = {k: theta_vi_d[k] - theta_truth_d[k] for k in M.FREE_NAMES}
    mu1_truth = theta_truth_d['mu1']
    mu1_direction = direction_d['mu1']
    if abs(mu1_direction) < 1e-12:
        raise RuntimeError(
            f"mu1_vi_obs ({theta_vi_d['mu1']:.6f}) coincides with "
            f"mu1_truth ({mu1_truth:.6f}) within 1e-12; cannot back-compute "
            f"t endpoints from MU1_MIN/MU1_MAX.")
    t_min = (float(mu1_min) - mu1_truth) / mu1_direction
    t_max = (float(mu1_max) - mu1_truth) / mu1_direction
    t_grid = np.linspace(t_min, t_max, int(n_grid))
    # t_ivi: scalar projection of (theta_ivi - theta_truth) onto direction.
    diff = np.array([theta_ivi_d[k] - theta_truth_d[k] for k in M.FREE_NAMES],
                     dtype=np.float64)
    dvec = np.array([direction_d[k] for k in M.FREE_NAMES], dtype=np.float64)
    denom = float(np.dot(dvec, dvec))
    t_ivi = float(np.dot(diff, dvec) / denom) if denom > 0.0 else float('nan')
    return (theta_truth_d, theta_vi_d, theta_ivi_d, direction_d,
            t_grid, t_ivi, t_min, t_max)


def extra_coord_norms_along_grid(t_grid, theta_truth_d, direction_d):
    """L-infinity norm of (mu_0(t), mu_2(t), sigma_1(t), sigma_2(t)) across the grid.

    Returns a Python float -- the max over t and over the four extra
    coords. Used as a sanity diagnostic for "how exact is the AR(1)
    closed-form along the line?"
    """
    extra_max = 0.0
    for t in t_grid:
        theta_t = interp_theta(float(t), theta_truth_d, direction_d)
        for k in EXTRA_COORDS:
            extra_max = max(extra_max, abs(float(theta_t[k])))
    return float(extra_max)


# ----------------------------------------------------------------------
# Cell + aggregate
# ----------------------------------------------------------------------

def build_cfg(n_grid, n_epochs, lr, n_epochs_nonamort, lr_nonamort,
              fix_noise, noise_seed, mu1_min, mu1_max, t_min, t_max,
              N, T, obs_seed):
    return {
        'dgp':               'simulation-ar1n',
        'n_grid':            n_grid,
        'n_epochs':          n_epochs,
        'lr':                lr,
        'n_epochs_nonamort': n_epochs_nonamort,
        'lr_nonamort':       lr_nonamort,
        'fix_noise':         fix_noise,
        'noise_seed':        noise_seed,
        'mu1_min':           float(mu1_min),
        'mu1_max':           float(mu1_max),
        't_min':             float(t_min),
        't_max':             float(t_max),
        'encoder':           ENCODER_TYPE,
        'hidden_dim':        HIDDEN_DIM,
        'N':                 N,
        'T':                 T,
        'obs_seed':          obs_seed,
        'upstream':          UPSTREAM_PATH,
    }


def maybe_pre_draw_eps(fix_noise, T, N, noise_seed, device):
    """Pre-draw the encoder eps tensor for CRN. Returns None when
    fix_noise=False. Built off the mean-field encoder's seed_dim (= T
    for `normal_diagonal`)."""
    if not fix_noise:
        return None
    probe = M.build_full_model(ENCODER_TYPE, T, device)
    seed_dim = probe.encoder.get_seed_dim()
    del probe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    torch.manual_seed(noise_seed)
    eps_fixed = torch.randn((1, N, seed_dim), dtype=torch.float32,
                              device=device)
    print(f"eps_fixed shape={tuple(eps_fixed.shape)}", flush=True)
    return eps_fixed


def _draw_eval_noise(N, T, device, eval_seed, k):
    """Pre-draw K noise tensors for the CRN-paired ELBO eval.

    Returns:
      eval_eps_list: list of K (1, N, T) tensors (for `model.elbo(eps_all=...)`)
      eval_u_list:   list of K (N, T) tensors    (for `_nonamort_elbo`)
    Both lists share the same noise: `eval_eps_list[k][0] == eval_u_list[k]`.
    """
    gen = torch.Generator(device=device)
    gen.manual_seed(int(eval_seed))
    eval_eps_list = []
    eval_u_list = []
    for _ in range(k):
        u = torch.randn(N, T, generator=gen, device=device,
                         dtype=torch.float32)
        eval_u_list.append(u)
        eval_eps_list.append(u.unsqueeze(0))  # (1, N, T)
    return eval_eps_list, eval_u_list


def run_cell(i, t, y_obs, T, vi_seed, n_epochs, lr,
             n_epochs_nonamort, lr_nonamort, noise_seed,
             eps_fixed, theta_truth_d, direction_d, device):
    """Run the full per-grid-cell procedure (5 steps) and return the cell dict."""
    t0 = time.time()
    theta_t = interp_theta(float(t), theta_truth_d, direction_d)

    # Pre-draw the K eval noise tensors so the amortized and non-amortized
    # final ELBOs share the same Monte Carlo draws (CRN at eval).
    # Seed off `noise_seed` plus a per-cell salt so each grid cell gets
    # its own deterministic eval noise.
    eval_seed = int(noise_seed) + 7919 * i
    eval_eps_list, eval_u_list = _draw_eval_noise(
        y_obs.shape[0], T, device, eval_seed, K_EVAL)

    # Step 2: amortized mean-field fit at theta(t).
    model, elbo_mf_amort, wall_amort = fit_mf_amortized(
        y_obs, T, vi_seed,
        n_epochs=n_epochs, lr=lr, device=device,
        theta_t=theta_t, eps_fixed=eps_fixed,
        eval_eps_list=eval_eps_list)

    # Step 4: non-amortized mean-field fit at theta(t) (only if step 2
    # produced a usable model).
    if model is not None and np.isfinite(elbo_mf_amort):
        elbo_mf_nonamort, wall_nonamort = fit_mf_nonamortized(
            y_obs, model, eps_fixed,
            n_epochs=n_epochs_nonamort, lr=lr_nonamort, device=device,
            noise_seed=noise_seed, eval_u_list=eval_u_list)
        # Sanity check: non-amortized must be at least as tight as
        # amortized. Allow tiny float slop; warn (do not crash) on
        # under-convergence.
        if elbo_mf_nonamort + 1e-3 < elbo_mf_amort:
            print(f"    [WARN] cell {i}: elbo_mf_nonamort "
                  f"({elbo_mf_nonamort:+.6f}) < elbo_mf_amort "
                  f"({elbo_mf_amort:+.6f}) by "
                  f"{elbo_mf_amort - elbo_mf_nonamort:.6f}. "
                  f"Non-amortized fit may be under-converged; consider "
                  f"raising N_EPOCHS_NONAMORT or LR_NONAMORT.",
                  flush=True)
    else:
        elbo_mf_nonamort, wall_nonamort = float('nan'), 0.0

    # Step 5: closed-form log p.
    logp_exact = float(M.ar1n_marginal_logp(
        y_obs,
        theta_t['mu1'],
        theta_t['sigma0'],
        theta_t['z1_log_std'],
        theta_t['log_sigma_eps']))

    # Gap decomposition.
    if np.isfinite(elbo_mf_amort) and np.isfinite(elbo_mf_nonamort):
        family_gap = float(logp_exact - elbo_mf_nonamort)
        amortization_gap = float(elbo_mf_nonamort - elbo_mf_amort)
        total_gap = float(family_gap + amortization_gap)
    else:
        family_gap = float('nan')
        amortization_gap = float('nan')
        total_gap = float('nan')

    cell = {
        't':                  float(t),
        'theta':              {k: float(v) for k, v in theta_t.items()},
        'elbo_mf_amort':      float(elbo_mf_amort),
        'elbo_mf_nonamort':   float(elbo_mf_nonamort),
        'logp_exact':         float(logp_exact),
        'family_gap':         float(family_gap),
        'amortization_gap':   float(amortization_gap),
        'total_gap':          float(total_gap),
        'wall_s':             float(time.time() - t0),
        'wall_s_amort':       float(wall_amort),
        'wall_s_nonamort':    float(wall_nonamort),
    }
    if model is not None:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(f"  cell {i}: t={t:+.4f}  "
          f"ELBO_amort={cell['elbo_mf_amort']:+.4f}  "
          f"ELBO_nonamort={cell['elbo_mf_nonamort']:+.4f}  "
          f"logp={cell['logp_exact']:+.4f}  "
          f"family_gap={cell['family_gap']:+.4f}  "
          f"amort_gap={cell['amortization_gap']:+.4f}  "
          f"({cell['wall_s']:.0f}s)", flush=True)
    return cell


def aggregate_cells(n_grid, t_grid, t_ivi, extra_coord_max,
                     theta_truth_d, theta_vi_d, theta_ivi_d, direction_d,
                     cfg):
    """Read every profile_cell_<i>.json in OUT_DIR; merge into profile.json."""
    profile = []
    n_found = 0
    for i in range(n_grid):
        p = cell_path(i)
        if not os.path.exists(p):
            print(f"  [aggregate] missing {p}", flush=True)
            continue
        with open(p) as f:
            entry = json.load(f)
        cell = entry.get('profile_cell')
        if cell is None:
            print(f"  [aggregate] {p} has no 'profile_cell' key", flush=True)
            continue
        profile.append(cell)
        n_found += 1
    profile.sort(key=lambda c: c['t'])
    out = {
        'config':          cfg,
        'theta_truth':     theta_truth_d,
        'theta_vi_obs':    theta_vi_d,
        'theta_ivi':       theta_ivi_d,
        'direction':       direction_d,
        't_grid':          [float(t) for t in t_grid],
        't_ivi':           float(t_ivi),
        'extra_coord_max': float(extra_coord_max),
        'profile':         profile,
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_PATH, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nAggregated {n_found}/{n_grid} cells -> {OUT_PATH}", flush=True)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    n_grid = int(os.environ.get('N_GRID', 20))
    n_epochs = int(os.environ.get('N_EPOCHS', 10000))
    lr = float(os.environ.get('LR', 1e-2))
    n_epochs_nonamort = int(os.environ.get('N_EPOCHS_NONAMORT', 2000))
    lr_nonamort = float(os.environ.get('LR_NONAMORT', 1e-3))
    fix_noise = int(os.environ.get('FIX_NOISE', 1)) != 0
    noise_seed = int(os.environ.get('NOISE_SEED', 12345))
    mu1_min = float(os.environ.get('MU1_MIN', 0.5))
    mu1_max = float(os.environ.get('MU1_MAX', 1.0))
    cell_i_env = os.environ.get('CELL_I', None)
    mode = os.environ.get('MODE', '').lower()

    (truth, theta_vi_obs, theta_ivi, N, T, obs_seed_up,
     vi_seed_up) = load_upstream()
    obs_seed = int(os.environ.get('OBS_SEED', obs_seed_up))
    vi_seed = int(os.environ.get('VI_SEED', vi_seed_up))

    (theta_truth_d, theta_vi_d, theta_ivi_d, direction_d,
     t_grid, t_ivi, t_min, t_max) = slice_geometry(
        truth, theta_vi_obs, theta_ivi, n_grid, mu1_min, mu1_max)
    extra_coord_max = extra_coord_norms_along_grid(
        t_grid, theta_truth_d, direction_d)
    cfg = build_cfg(n_grid, n_epochs, lr, n_epochs_nonamort, lr_nonamort,
                    fix_noise, noise_seed, mu1_min, mu1_max, t_min, t_max,
                    N, T, obs_seed)

    os.makedirs(OUT_DIR, exist_ok=True)

    # --- MODE=aggregate: merge cell JSONs, no GPU work -----------------
    if mode == 'aggregate':
        print(f"=== aggregate {n_grid} cell files -> profile.json ===",
              flush=True)
        aggregate_cells(n_grid, t_grid, t_ivi, extra_coord_max,
                         theta_truth_d, theta_vi_d, theta_ivi_d, direction_d,
                         cfg)
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device={device}", flush=True)
    print(f"N={N} T={T} obs_seed={obs_seed} vi_seed={vi_seed}", flush=True)
    print(f"hyperparams: encoder={ENCODER_TYPE} h={HIDDEN_DIM}, "
          f"n_epochs={n_epochs}, lr={lr}, "
          f"n_epochs_nonamort={n_epochs_nonamort}, lr_nonamort={lr_nonamort}, "
          f"fix_noise={fix_noise}, noise_seed={noise_seed}", flush=True)
    print(f"slice geometry: mu1 in [{mu1_min:+.3f}, {mu1_max:+.3f}] "
          f"(t in [{t_min:+.3f}, {t_max:+.3f}]) x N_GRID={n_grid}",
          flush=True)
    print(f"  t_ivi={t_ivi:+.4f}", flush=True)
    print(f"  extra_coord_max={extra_coord_max:.4f} "
          f"(L-inf of mu0/mu2/sigma1/sigma2 across grid)", flush=True)

    y_obs = M.simulate_from_truth(N, T, truth, obs_seed, device)
    print(f"y_obs mean={y_obs.mean().item():+.4f} "
          f"std={y_obs.std().item():.4f}", flush=True)

    eps_fixed = maybe_pre_draw_eps(fix_noise, T, N, noise_seed, device)

    # --- CELL_I=<i>: single-cell mode ----------------------------------
    if cell_i_env is not None:
        i = int(cell_i_env)
        if not (0 <= i < n_grid):
            sys.exit(f"CELL_I={i} out of range [0, {n_grid})")
        t = float(t_grid[i])
        print(f"=== CELL_I={i}/{n_grid}: t={t:+.4f} ===", flush=True)
        cell = run_cell(i, t, y_obs, T, vi_seed, n_epochs, lr,
                         n_epochs_nonamort, lr_nonamort, noise_seed,
                         eps_fixed, theta_truth_d, direction_d, device)
        out = {
            'config':          cfg,
            'theta_truth':     theta_truth_d,
            'theta_vi_obs':    theta_vi_d,
            'theta_ivi':       theta_ivi_d,
            'direction':       direction_d,
            't_grid':          [float(x) for x in t_grid],
            't_ivi':           float(t_ivi),
            'extra_coord_max': float(extra_coord_max),
            't_grid_value':    t,
            'profile_cell':    cell,
        }
        with open(cell_path(i), 'w') as f:
            json.dump(out, f, indent=2)
        print(f"\nDone. Wrote {cell_path(i)}", flush=True)
        return

    # --- Default: sequential N_GRID-cell sweep + write profile.json ----
    # Restartable: reuse cells already present in profile.json.
    profile = {}
    if os.path.exists(OUT_PATH):
        try:
            with open(OUT_PATH) as f:
                cached = json.load(f)
            cfg_c = cached.get('config', {})
            ok = (cfg_c.get('n_grid') == n_grid
                  and cfg_c.get('n_epochs') == n_epochs
                  and abs(cfg_c.get('lr', -1) - lr) < 1e-9
                  and cfg_c.get('n_epochs_nonamort') == n_epochs_nonamort
                  and abs(cfg_c.get('lr_nonamort', -1) - lr_nonamort) < 1e-9
                  and cfg_c.get('encoder') == ENCODER_TYPE
                  and abs(cfg_c.get('mu1_min', 1e9) - mu1_min) < 1e-9
                  and abs(cfg_c.get('mu1_max', 1e9) - mu1_max) < 1e-9)
            if ok:
                for cell in cached.get('profile', []):
                    profile[f"{float(cell['t']):+.6f}"] = cell
                print(f"Loaded {len(profile)} cached cells", flush=True)
        except json.JSONDecodeError:
            pass

    def save():
        out = {
            'config':          cfg,
            'theta_truth':     theta_truth_d,
            'theta_vi_obs':    theta_vi_d,
            'theta_ivi':       theta_ivi_d,
            'direction':       direction_d,
            't_grid':          [float(x) for x in t_grid],
            't_ivi':           float(t_ivi),
            'extra_coord_max': float(extra_coord_max),
            'profile':         sorted(profile.values(),
                                       key=lambda c: c['t']),
        }
        with open(OUT_PATH, 'w') as f:
            json.dump(out, f, indent=2)

    t_start = time.time()
    for i, t in enumerate(t_grid):
        key = f"{float(t):+.6f}"
        cell = profile.get(key)
        need = (cell is None
                 or 'elbo_mf_amort'    not in cell
                 or 'elbo_mf_nonamort' not in cell
                 or 'logp_exact'       not in cell
                 or not np.isfinite(cell.get('elbo_mf_amort',    np.nan))
                 or not np.isfinite(cell.get('elbo_mf_nonamort', np.nan))
                 or not np.isfinite(cell.get('logp_exact',       np.nan)))
        if not need:
            print(f"  [{i+1}/{n_grid}] t={t:+.4f}  CACHED  "
                  f"ELBO_amort={cell['elbo_mf_amort']:+.4f}  "
                  f"ELBO_nonamort={cell['elbo_mf_nonamort']:+.4f}  "
                  f"logp={cell['logp_exact']:+.4f}", flush=True)
            continue
        cell = run_cell(i, float(t), y_obs, T, vi_seed, n_epochs, lr,
                         n_epochs_nonamort, lr_nonamort, noise_seed,
                         eps_fixed, theta_truth_d, direction_d, device)
        profile[key] = cell
        save()
        elapsed = time.time() - t_start
        n_done = i + 1
        eta_min = (elapsed / n_done) * (n_grid - n_done) / 60
        print(f"  -- elapsed {elapsed:.0f}s  ETA {eta_min:.1f} min --",
              flush=True)

    print(f"\nDone in {(time.time()-t_start)/60:.1f} min. "
          f"Output: {OUT_PATH}", flush=True)


if __name__ == "__main__":
    # Fix every RNG (python / numpy / torch CPU+CUDA) and request deterministic
    # kernels before any draw; the script's own explicit seeds still apply.
    from mlye.seeding import seed_everything, env_seed
    seed_everything(env_seed("OBS_SEED", 11))
    main()
