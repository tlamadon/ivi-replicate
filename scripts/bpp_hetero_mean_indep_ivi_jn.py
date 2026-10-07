r"""IVI fit (Picard + joint-normal-extra inner posterior) of the
hetero-MEAN (independent) + sinh-MA(1) model on the BPP PSID earnings
panel.

Sister script to `bpp_hetero_scale_ivi_jn.py`; model class is the
hetero-mean-with-independence variant of `bpp_hetero_mean_indep_vi_jn.py`
(see that script for the verbose model description):
  z_t   ~ poly-2 mu + poly-2 sigma (no softplus kink, law_model='poly')
  z_1   ~ SinhArcsinh (all three z1 params free)
  alpha_i ~ N(beta_a0, exp(2*log_sigma_a))          [INDEPENDENT of z_1]
  y_t   = z_t + alpha_i + sigma_eps * MA(1)-SinhArcsinh(0,1,alpha_eps,beta)
          (MASinhEmissionHetero, hetero_mode='mean';
           theta, log_beta, alpha_eps, log_sigma all free)

Inner encoder: JointNormalConfig(type='joint_normal_extra', dim=T,
  hidden_dim=64, extra_latents=1, regularize=1e-3, sd_clamp=3.0).

Estimator: Indirect VI with the binding-equation outer loop
  (estimators.md \S3):

    Phase 1: theta_VI_obs <- inner VI on y_obs (20,000 ep, lr=1e-2, CRN).
    Phase 2: for k = 1..K_outer (10 iters):
       y_sim_k  = reparam_simulate(theta_k, u_z, u_alpha, u_y)
       theta_VI_sim_k = inner VI on y_sim_k (16,000 ep, lr=1e-2, CRN).
       g_k      = theta_VI_obs - theta_VI_sim_k                 [binding form]
       theta_{k+1} = theta_k + alpha_picard * g_k               [Picard]

  Defaults follow the K=40 no-inflation production design
  (hut.bpp_hetero_scale_ivi_k40.json): METHOD=picard, alpha=0.6,
  BPP_HETERO_MEAN_NDRAWS=40 (y tiled 40x with 40 distinct CRN eps
  blocks -- a 40-draw ELBO in one batched pass), n_sim = N = 741 on
  BOTH sides of the binding equation (no bootstrap inflation).

Phase 3 (post-fit diagnostics) at converged theta_K:
  - IWAE-at-converged-theta with a fresh joint-normal-extra h=64 encoder
    re-fit on the actual y (prior + decoder frozen at theta_K). K=200.
  - BHHH SEs from per-individual FIVO scores at the same re-fit encoder.

Output: output/bundles/cells/employment-bpp-hetero-mean-indep-ivi-jn/bpp_hetero_mean_indep_ivi_jn.json
"""
import os
import sys
import time
import json

import numpy as np
import torch
import torch.nn.functional as F

torch.distributions.Distribution.set_default_validate_args(False)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from smc_truth_profile import _Bundle  # noqa: E402,F401  (kept for parity)

from mlye.models.full_model import FullModel  # noqa: E402
from mlye.utils import sinh_arcsinh_mean  # noqa: E402

from bpp_hockey_common import compute_iwae_ref_log_p, compute_bhhh_se_fivo  # noqa: E402

import psid_smc_em_fivo_sinhz1 as base  # noqa: E402

# Model factory, encoder, table assembler and PARAM_NAMES are shared
# with the VI sibling so the two entries fit the *same* model by
# construction.
from bpp_hetero_mean_indep_vi_jn import (  # noqa: E402
    make_model, build_encoder, assemble_table_hetero_mean, PARAM_NAMES,
)

# Reuse Anderson step from the simulation IVI scoping script (kept for
# the METHOD=anderson escape hatch; the default here is Picard).
from _scale_hetero_correlation_ivi import anderson_step  # noqa: E402


METHOD_TAG = 'ivi'
ENCODER_LABEL = 'joint_normal_extra'
ENCODER_CONFIG_STR = (
    "JointNormalConfig(type='joint_normal_extra', dim=T, hidden_dim=64, "
    "extra_latents=1, regularize=1e-3, sd_clamp=3.0)"
)
DEFAULT_OUT_DIR = "output/bundles/cells/employment-bpp-hetero-mean-indep-ivi-jn"
DEFAULT_OUT_JSON = os.path.join(DEFAULT_OUT_DIR,
                                "bpp_hetero_mean_indep_ivi_jn.json")
DATA_PATH = "output/data/bpp_y_matrix.npy"


# -------------------------- theta <-> vec utilities ---------------------

def extract_theta_dict(prior, decoder):
    """Read the 15-vec hetero-mean-indep parameter vector off
    prior+decoder (PARAM_NAMES order)."""
    return {
        'mu0':          float(prior.net_mu.coeffs[0].item()),
        'mu1':          float(prior.net_mu.coeffs[1].item()),
        'mu2':          float(prior.net_mu.coeffs[2].item()),
        'sigma0':       float(prior.net_sigma.coeffs[0].item()),
        'sigma1':       float(prior.net_sigma.coeffs[1].item()),
        'sigma2':       float(prior.net_sigma.coeffs[2].item()),
        'z1_log_std':   float(prior.z1_log_std.item()),
        'z1_skew':      float(prior.z1_skew.item()),
        'z1_log_tail':  float(prior.z1_log_tail.item()),
        'log_beta':     float(decoder.log_beta.item()),
        'theta':        float(decoder.theta.item()),
        'alpha_eps':    float(decoder.alpha_eps.item()),
        'log_sigma':    float(decoder.log_sigma.item()),
        'beta_a0':      float(prior.net_extra_mu.coeffs[0].item()),
        'log_sigma_a_cond': float(prior.net_extra_logsigma.coeffs[0].item()),
    }


def theta_to_vec(theta):
    return np.array([float(theta[k]) for k in PARAM_NAMES], dtype=np.float64)


def vec_to_theta(vec):
    return {k: float(v) for k, v in zip(PARAM_NAMES, vec)}


def set_prior_decoder_from_dict(prior, decoder, theta):
    """Write the 15-vec theta into prior + decoder (in-place, no grad)."""
    with torch.no_grad():
        prior.net_mu.coeffs.copy_(torch.tensor(
            [theta['mu0'], theta['mu1'], theta['mu2']],
            dtype=prior.net_mu.coeffs.dtype, device=prior.net_mu.coeffs.device))
        prior.net_sigma.coeffs.copy_(torch.tensor(
            [theta['sigma0'], theta['sigma1'], theta['sigma2']],
            dtype=prior.net_sigma.coeffs.dtype, device=prior.net_sigma.coeffs.device))
        prior.z1_log_std.fill_(theta['z1_log_std'])
        prior.z1_skew.fill_(theta['z1_skew'])
        prior.z1_log_tail.fill_(theta['z1_log_tail'])
        prior.net_extra_mu.coeffs.data[0] = theta['beta_a0']
        prior.net_extra_logsigma.coeffs.data[0] = theta['log_sigma_a_cond']
        decoder.log_beta.fill_(theta['log_beta'])
        decoder.theta.fill_(theta['theta'])
        decoder.alpha_eps.fill_(theta['alpha_eps'])
        decoder.log_sigma.fill_(theta['log_sigma'])


# --------- differentiable simulator (hetero-mean-indep + MA(1)) ----------

def reparam_simulate(theta_vec, u_z, u_alpha, u_y):
    """Differentiable simulate y (N, T) under the BPP hetero-mean-indep
    + MA(1) DGP from the 15-vec theta_vec (PARAM_NAMES order).

    Forward emission (matches MASinhEmissionHetero.log_likelihood under
    hetero_mode='mean'):
        w_t    ~ sinh-arcsinh(0, 1, alpha_eps, beta), recentred (iid in t)
        e_t    = sigma_eps * w_t                      (standardised innovation)
        r_t    = e_t + theta_ma * e_{t-1}             (MA(1) L-application)
        y_t    = z_t + alpha_i + r_t
    with alpha_i = beta_a0 + sigma_a * u_alpha, INDEPENDENT of z_1.
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
    theta_ma     = theta_vec[10]
    alpha_eps    = theta_vec[11]
    log_sigma    = theta_vec[12]
    beta_a0      = theta_vec[13]
    log_sigma_a_cond = theta_vec[14]

    z1_scale = F.softplus(z1_log_std)
    z1_tail  = torch.exp(z1_log_tail)
    sigma_a  = torch.exp(log_sigma_a_cond)
    beta     = torch.exp(log_beta)
    sigma_eps = torch.exp(log_sigma)

    N, T = u_z.shape
    z = torch.zeros(N, T, device=u_z.device, dtype=u_z.dtype)
    # z_1 ~ sinh-arcsinh with mean-preserving recentring (matches the
    # prior's loc = -mu_z1 convention in markov.py::get_eta0).
    mu_z1 = sinh_arcsinh_mean(z1_scale, z1_skew, z1_tail)
    z[:, 0] = -mu_z1 + z1_scale * torch.sinh(
        (torch.asinh(u_z[:, 0]) + z1_skew) * z1_tail)
    # z_t | z_{t-1}: poly-2 mu, softplus(poly-2) sigma — no kink.
    for t in range(1, T):
        zlag = z[:, t - 1].clone()
        mu_t = (mu0 + mu1 * zlag + mu2 * zlag ** 2).clamp(-10.0, 10.0)
        sig_raw = sigma0 + sigma1 * zlag + sigma2 * zlag ** 2
        sig_t = F.softplus(sig_raw).clamp(0.0, 10.0) + 1e-3
        z[:, t] = mu_t + sig_t * u_z[:, t]
    # alpha: INDEPENDENT of z_1 (this is the whole point of the variant).
    alpha = (beta_a0 + sigma_a * u_alpha).unsqueeze(1)  # (N, 1)
    # sinh-arcsinh iid noise with free skew alpha_eps + mean-preserving
    # recentring (matches MASinhEmissionHetero.log_likelihood's mu_W
    # subtraction; see decoders/ma.py::_residual_mean).
    mu_W = sinh_arcsinh_mean(
        torch.ones((), device=u_y.device, dtype=u_y.dtype),
        alpha_eps, beta,
    )
    eps_y = -mu_W + torch.sinh(
        (torch.asinh(u_y) + alpha_eps) * beta)  # (N, T)
    e = sigma_eps * eps_y
    # Apply MA(1) lower-triangular L: residual_t = e_t + theta * e_{t-1}
    residuals = torch.zeros_like(e)
    residuals[:, 0] = e[:, 0]
    for t in range(1, T):
        residuals[:, t] = e[:, t] + theta_ma * e[:, t - 1]
    y = z + alpha + residuals
    return y


# --------------------------- VI fit helpers -----------------------------

class NaNFailure(RuntimeError):
    pass


def _retry_seeds(base_seed, n_retries):
    return [base_seed + i for i in range(n_retries)]


def fit_vi_inner(y, n_epochs, lr, log_every=500, clip=5.0,
                 label='vi', vi_seed=11, n_retries=5, eps_fixed=None):
    """Cold-start full-joint VI fit (prior + decoder + encoder all free).
    Returns (theta_dict, history, used_seed, prior, decoder, encoder, model)."""
    last_err = None
    T = y.shape[1]
    for s in _retry_seeds(vi_seed, n_retries):
        try:
            torch.manual_seed(s); np.random.seed(s)
            prior, decoder = make_model(T, y.device, seed=s)
            encoder = build_encoder(T, y.device, seed=s)
            model = FullModel(encoder=encoder, decoder=decoder, prior=prior).to(y.device)
            params = list(model.parameters())
            opt = torch.optim.AdamW(params, lr=lr)
            history = []
            t0 = time.time()
            nan_streak = 0
            for epoch in range(n_epochs):
                opt.zero_grad()
                if eps_fixed is not None:
                    loss = -model.elbo(y, ndraws=1, eps_all=eps_fixed)
                else:
                    loss = -model.elbo(y, ndraws=1)
                if not torch.isfinite(loss):
                    nan_streak += 1
                    if epoch < 500 and nan_streak >= 50:
                        raise NaNFailure(f"NaN at epoch {epoch}")
                    continue
                nan_streak = 0
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, clip)
                opt.step()
                if epoch % log_every == 0 or epoch == n_epochs - 1:
                    history.append({'epoch': epoch, 'elbo': float(-loss.item())})
                    if epoch % (log_every * 5) == 0 or epoch == n_epochs - 1:
                        print(
                            f"    [{label}/s{s}] ep {epoch:5d}  "
                            f"ELBO={-loss.item():+.3f}  "
                            f"log_b={float(decoder.log_beta.item()):+.3f}  "
                            f"theta={float(decoder.theta.item()):+.3f}  "
                            f"sig_eps={float(torch.exp(decoder.log_sigma).item()):.3f}  "
                            f"ba0={float(prior.net_extra_mu.coeffs[0].item()):+.3f}  "
                            f"({time.time()-t0:.1f}s)", flush=True)
            theta_dict = extract_theta_dict(prior, decoder)
            return theta_dict, history, s, prior, decoder, encoder, model
        except NaNFailure as e:
            print(f"    [{label}/s{s}] NaN bail: {e}; retrying", flush=True)
            last_err = e
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    raise RuntimeError(f"inner VI failed after {n_retries} seed retries "
                       f"(last err: {last_err})")


def fit_encoder_only_at_theta(theta, y, n_epochs, lr, device, vi_seed=11,
                              n_retries=5):
    """Fresh joint-normal-extra h=64 encoder with prior+decoder frozen at
    `theta`. Used by Phase 3 to evaluate IWAE-at-theta_K and to compute
    BHHH SE at theta_K. Returns (prior, decoder, encoder, model)."""
    last_err = None
    T = y.shape[1]
    for s in _retry_seeds(vi_seed, n_retries):
        try:
            torch.manual_seed(s); np.random.seed(s)
            prior, decoder = make_model(T, device, seed=s)
            encoder = build_encoder(T, device, seed=s)
            model = FullModel(encoder=encoder, decoder=decoder, prior=prior).to(device)
            set_prior_decoder_from_dict(prior, decoder, theta)
            for p in prior.parameters():
                p.requires_grad_(False)
            for p in decoder.parameters():
                p.requires_grad_(False)
            seed_dim = encoder.get_seed_dim()
            torch.manual_seed(s + 1)
            eps_fixed_local = torch.randn(
                (1, y.shape[0], seed_dim),
                dtype=torch.float32, device=device)
            opt = torch.optim.AdamW(
                [p for p in encoder.parameters() if p.requires_grad], lr=lr)
            t0 = time.time()
            nan_streak = 0
            for epoch in range(n_epochs):
                opt.zero_grad()
                loss = -model.elbo(y, ndraws=1, eps_all=eps_fixed_local)
                if not torch.isfinite(loss):
                    nan_streak += 1
                    if epoch < 500 and nan_streak >= 50:
                        raise NaNFailure(f"NaN at epoch {epoch}")
                    continue
                nan_streak = 0
                loss.backward()
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5.0)
                opt.step()
                if (epoch + 1) % 2000 == 0 or epoch == 0:
                    print(f"  [enc@theta_K/s{s}] ep {epoch+1:5d}: "
                          f"ELBO={-loss.item():+.3f}  "
                          f"({time.time()-t0:.1f}s)", flush=True)
            # Re-enable grads on prior/decoder so BHHH-FIVO can take grads
            # of the FIVO bound wrt prior+decoder params at theta_K.
            for p in prior.parameters():
                p.requires_grad_(True)
            for p in decoder.parameters():
                p.requires_grad_(True)
            return prior, decoder, encoder, model
        except NaNFailure as e:
            print(f"  [enc@theta_K/s{s}] NaN bail: {e}; retrying", flush=True)
            last_err = e
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    raise RuntimeError(f"encoder-only fit failed after {n_retries} retries "
                       f"(last err: {last_err})")


# ------------------------------- main -----------------------------------

def main():
    n_epochs_vi_obs = int(os.environ.get('BPP_HETERO_MEAN_N_EPOCHS_VI_OBS', 20000))
    n_epochs_inner  = int(os.environ.get('BPP_HETERO_MEAN_N_EPOCHS_INNER', 16000))
    n_iters_outer   = int(os.environ.get('BPP_HETERO_MEAN_N_ITERS_OUTER', 10))
    n_epochs_encoder = int(os.environ.get('BPP_HETERO_MEAN_N_EPOCHS_ENCODER', 8000))
    lr              = float(os.environ.get('BPP_HETERO_MEAN_LR', 1e-2))
    lr_encoder      = float(os.environ.get('BPP_HETERO_MEAN_LR_ENCODER', 1e-3))
    picard_alpha    = float(os.environ.get('BPP_HETERO_MEAN_PICARD_ALPHA', 0.6))
    m_mem           = int(os.environ.get('BPP_HETERO_MEAN_M_MEM', 4))
    log_every       = int(os.environ.get('BPP_HETERO_MEAN_LOG_EVERY', 500))
    seed_base       = int(os.environ.get('BPP_HETERO_MEAN_SEED', 11))
    noise_seed      = int(os.environ.get('BPP_HETERO_MEAN_NOISE_SEED', 12345))

    # Default is Picard (per the K=40 no-inflation production design);
    # METHOD=anderson is kept as an escape hatch for sweeps.
    METHOD = os.environ.get('BPP_HETERO_MEAN_METHOD', 'picard')
    assert METHOD in ('anderson', 'picard'), f"unknown METHOD={METHOD}"
    RESTART_FACTOR = 1.5

    out_dir  = os.environ.get('BPP_HETERO_MEAN_OUT_DIR', DEFAULT_OUT_DIR)
    out_json = os.environ.get('BPP_HETERO_MEAN_OUT_JSON', DEFAULT_OUT_JSON)
    if 'BPP_HETERO_MEAN_OUT_DIR' in os.environ \
            and 'BPP_HETERO_MEAN_OUT_JSON' not in os.environ:
        out_json = os.path.join(out_dir, os.path.basename(DEFAULT_OUT_JSON))

    print(f"Loading BPP PSID earnings panel from {DATA_PATH}.", flush=True)
    y_np = np.load(DATA_PATH)
    y = torch.from_numpy(y_np).float()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    y = y.to(device)
    N, T = y.shape
    # Nonparametric-bootstrap replicate hook (for future SE replicates;
    # unset in the headline run).
    resample_seed = os.environ.get('BPP_HETERO_MEAN_RESAMPLE_SEED', '')
    if resample_seed:
        rng_rs = np.random.default_rng(int(resample_seed))
        idx_rs = torch.from_numpy(rng_rs.integers(0, N, N)).to(device)
        y = y[idx_rs]
        print(f"[bootstrap] household resample (with replacement, N={N}) "
              f"seed={resample_seed}", flush=True)
    n_sim = int(os.environ.get('BPP_HETERO_MEAN_N_SIM', N))
    # Bootstrap-inflation hook kept for parity with the hetero-scale
    # sibling, but the headline design here is K=40 multidraw with NO
    # inflation (both sides at N=741).
    n_bootstrap = int(os.environ.get('BPP_HETERO_MEAN_BOOTSTRAP_N', N))
    bootstrap_seed = int(os.environ.get('BPP_HETERO_MEAN_BOOTSTRAP_SEED', 4242))
    y_actual = y
    if n_bootstrap != N:
        g = torch.Generator(device='cpu').manual_seed(bootstrap_seed)
        idx = torch.randint(0, N, (n_bootstrap,), generator=g)
        y = y_actual[idx.to(device)]
        N_phase1 = n_bootstrap
        print(f"  bootstrap-resampled y for Phase 1: N_boot={n_bootstrap} "
              f"(seed={bootstrap_seed})", flush=True)
    else:
        N_phase1 = N
    print(f"  shape: N={N} (actual), N_phase1={N_phase1}, "
          f"T={T}, device={device}", flush=True)
    print(f"  inner-loop simulator N_sim={n_sim} "
          f"({'matches Phase 1 N' if n_sim == N_phase1 else 'MISMATCH ' + f'(x{n_sim/N_phase1:.1f})'})",
          flush=True)
    print(f"  empirical mean={y_actual.mean().item():+.4f}, std={y_actual.std().item():.4f} "
          f"(actual; Phase 1 sees bootstrap if active)",
          flush=True)
    print(f"\nIVI config: method={METHOD}  alpha={picard_alpha}  m_mem={m_mem}  "
          f"restart_factor={RESTART_FACTOR}\n"
          f"  n_epochs_vi_obs={n_epochs_vi_obs}  n_iters_outer={n_iters_outer}  "
          f"n_epochs_inner={n_epochs_inner}  lr={lr}", flush=True)

    vi_seed = seed_base * 1000 + 7  # = 11007 by default

    # Multi-draw ELBO: K CRN draws per individual, implemented by tiling
    # y K times with K distinct eps blocks (one batched pass; identical
    # mean and gradients to FullModel.elbo's ndraws loop). Default K=40
    # per the no-inflation production design.
    ndraws = int(os.environ.get('BPP_HETERO_MEAN_NDRAWS', 40))

    # --- Pre-draw CRN epsilon for Phase 1 inner VI (seed_dim = T + 1). ---
    seed_dim_phase1 = T + 1
    torch.manual_seed(noise_seed)
    eps_fixed_phase1 = torch.randn(
        (1, ndraws * N_phase1, seed_dim_phase1), dtype=torch.float32,
        device=device)
    if ndraws > 1:
        y = y.repeat(ndraws, 1)
        print(f"[multidraw] K={ndraws}: Phase-1 y tiled to "
              f"{tuple(y.shape)}", flush=True)
    print(f"[crn] eps_fixed (phase 1)  shape={tuple(eps_fixed_phase1.shape)}",
          flush=True)

    # IVI inner phase uses an independently sampled CRN eps reused across
    # outer iters (encoder is rebuilt from scratch each outer iter, so the
    # same eps gives a step-to-step common-random-number lock).
    torch.manual_seed(noise_seed)
    eps_fixed_inner = torch.randn(
        (1, ndraws * n_sim, T + 1), dtype=torch.float32, device=device)

    # --- Differentiable-simulator noise (CRN across all outer iters). ---
    torch.manual_seed(seed_base * 10000)
    u_z     = torch.randn(n_sim, T, device=device, dtype=torch.float32)
    u_alpha = torch.randn(n_sim,    device=device, dtype=torch.float32)
    u_y     = torch.randn(n_sim, T, device=device, dtype=torch.float32)

    # ================================================================
    # Phase 1: theta_VI_obs pre-fit on observed y.
    # ================================================================
    print(f"\n=== Phase 1: theta_VI_obs pre-fit "
          f"({n_epochs_vi_obs} ep, lr={lr}, CRN, K={ndraws}) ===", flush=True)
    (theta_VI_obs, vi_obs_history, used_seed_phase1,
     prior_obs, decoder_obs, encoder_obs, model_VI_obs) = fit_vi_inner(
        y, n_epochs_vi_obs, lr=lr, log_every=log_every,
        label='VI_obs', vi_seed=vi_seed, eps_fixed=eps_fixed_phase1,
    )
    theta_VI_obs_vec = theta_to_vec(theta_VI_obs)
    print(f"\ntheta_VI_obs: log_beta={theta_VI_obs['log_beta']:+.4f}  "
          f"theta={theta_VI_obs['theta']:+.4f}  "
          f"log_sigma={theta_VI_obs['log_sigma']:+.4f}  "
          f"beta_a0={theta_VI_obs['beta_a0']:+.4f}  "
          f"log_sigma_a={theta_VI_obs['log_sigma_a_cond']:+.4f}", flush=True)
    del model_VI_obs, prior_obs, decoder_obs, encoder_obs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ================================================================
    # Phase 2: outer loop (binding-equation form).
    # ================================================================
    theta_k = dict(theta_VI_obs)
    history_anderson = []
    prev_g_norm = None

    trajectory = [{
        'iter': 0,
        'iter_wall_s': None,
        'theta_k': {k: float(theta_k[k]) for k in PARAM_NAMES},
        'theta_VI': None,
        'g_norm': None,
        'restart': False,
        'gamma': None,
        'used_seed': None,
    }]

    print(f"\n=== Phase 2: {METHOD} outer loop "
          f"({n_iters_outer} iters, alpha={picard_alpha}) ===",
          flush=True)
    for k in range(1, n_iters_outer + 1):
        iter_t0 = time.time()
        print(f"\n[iter {k}/{n_iters_outer}] simulate y_sim_k + inner VI "
              f"({n_epochs_inner} ep)...", flush=True)

        theta_t = torch.tensor(
            [theta_k[name] for name in PARAM_NAMES],
            dtype=torch.float32, device=device)
        with torch.no_grad():
            y_sim_k = reparam_simulate(theta_t, u_z, u_alpha, u_y)
            if ndraws > 1:
                y_sim_k = y_sim_k.repeat(ndraws, 1)

        (theta_VI_sim_k, inner_history, used_seed,
         _p, _d, _e, _m) = fit_vi_inner(
            y_sim_k, n_epochs_inner, lr=lr, log_every=log_every,
            label=f'inner/it{k}', vi_seed=vi_seed,
            eps_fixed=eps_fixed_inner,
        )
        del _p, _d, _e, _m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        iter_wall = time.time() - iter_t0
        theta_VI_sim_vec = theta_to_vec(theta_VI_sim_k)
        x_k = theta_to_vec(theta_k)

        # Binding-equation form: g_k = theta_VI_obs - theta_VI_sim_k.
        g_k = theta_VI_obs_vec - theta_VI_sim_vec
        g_norm = float(np.linalg.norm(g_k))

        restart = False
        if METHOD == 'anderson':
            if prev_g_norm is not None and g_norm > RESTART_FACTOR * prev_g_norm:
                print(f"  [restart] ||g||={g_norm:.4f} (was {prev_g_norm:.4f})",
                      flush=True)
                history_anderson = []
                restart = True
            x_next, gamma = anderson_step(
                x_k, g_k, history_anderson, m=m_mem, beta=picard_alpha)
            history_anderson.append({'x': x_k.copy(), 'g': g_k.copy()})
            prev_g_norm = g_norm
        else:  # picard
            x_next = x_k + picard_alpha * g_k
            gamma = None
            prev_g_norm = g_norm

        theta_next = vec_to_theta(x_next)
        print(f"  inner VI(sim): log_beta={theta_VI_sim_k['log_beta']:+.4f}  "
              f"theta={theta_VI_sim_k['theta']:+.4f}  "
              f"beta_a0={theta_VI_sim_k['beta_a0']:+.4f}", flush=True)
        print(f"  theta_{k}:       log_beta={theta_next['log_beta']:+.4f}  "
              f"theta={theta_next['theta']:+.4f}  "
              f"beta_a0={theta_next['beta_a0']:+.4f}  "
              f"||g||={g_norm:.4f}  iter_wall={iter_wall:.1f}s",
              flush=True)
        g_top = sorted(zip(PARAM_NAMES, g_k), key=lambda kv: -abs(kv[1]))[:5]
        print("  objective g_k (top |components|): " + "  ".join(
            f"{name}={val:+.4f}" for name, val in g_top), flush=True)

        trajectory.append({
            'iter':         k,
            'iter_wall_s':  iter_wall,
            'theta_k':      {kk: float(theta_next[kk]) for kk in PARAM_NAMES},
            'theta_VI':     {kk: float(theta_VI_sim_k[kk]) for kk in PARAM_NAMES},
            'g':            {kk: float(v)
                             for kk, v in zip(PARAM_NAMES, g_k)},
            'g_norm':       g_norm,
            'restart':      bool(restart),
            'gamma':        (gamma.tolist()
                              if (gamma is not None and len(gamma) > 0)
                              else None),
            'used_seed':    int(used_seed) if used_seed is not None else None,
            'inner_history': inner_history,
        })
        theta_k = theta_next

    print(f"\nFinal theta_K (iter {n_iters_outer}):", flush=True)
    for name in PARAM_NAMES:
        print(f"  {name:<22} {theta_k[name]:+.4f}", flush=True)

    # ================================================================
    # Phase 3: IWAE-at-theta_K + BHHH SEs (fresh encoder at theta_K).
    # ================================================================
    skip_phase3 = os.environ.get('BPP_HETERO_MEAN_SKIP_PHASE3', '0') == '1'
    if skip_phase3:
        print("\n=== Phase 3 skipped (BPP_HETERO_MEAN_SKIP_PHASE3=1) ===",
              flush=True)
        rows_K, derived_K = [], []
        iwae_logp, iwae_meta = None, None
    else:
        print(f"\n=== Phase 3: fresh encoder@theta_K ({n_epochs_encoder} ep, "
              f"lr={lr_encoder}) + IWAE + BHHH on actual y (N={N}) ===",
              flush=True)
        prior_K, decoder_K, encoder_K, model_K = fit_encoder_only_at_theta(
            theta_k, y_actual, n_epochs_encoder, lr=lr_encoder, device=device,
            vi_seed=vi_seed,
        )

        # IWAE log p(y; theta_K) — joint MA(1) Cholesky density, K=200.
        iwae_logp, iwae_meta = compute_iwae_ref_log_p(
            prior_K, decoder_K, y_actual, encoder=encoder_K, K=200,
        )

        print(f"\n=== BHHH SEs at IVI fit "
              f"(per-individual FIVO scores, K=500, 3 reps) ===", flush=True)
        se_K, _ = compute_bhhh_se_fivo(prior_K, decoder_K, y_actual, K=500,
                                        n_reps=3, seed_base=seed_base)
        rows_K, derived_K = assemble_table_hetero_mean(prior_K, decoder_K,
                                                       se_K)
        base.print_table(f"IVI ({METHOD_TAG}) fit (hetero-mean-indep BPP) — "
                         f"theta_K", rows_K, derived_K)

    # ================================================================
    # Write JSON output.
    # ================================================================
    os.makedirs(out_dir, exist_ok=True)
    payload = {
        'data': {
            'source': 'bpp',
            'path': DATA_PATH,
            'N': int(N), 'T': int(T),
            'mean_y': float(y_actual.mean().item()),
            'std_y': float(y_actual.std().item()),
            'bootstrap_N': int(n_bootstrap),
            'bootstrap_seed': int(bootstrap_seed),
            'resample_seed': int(resample_seed) if resample_seed else None,
        },
        'model': {
            'prior': 'MarkovNormalConditionalPolyPrior(poly_degree=2, '
                      'z1_distr=sinh, law_model=poly, extra_het=True, '
                      "extra_prior_type='iid', net_extra_mu=Polynomial(0))",
            'decoder': "MASinhEmissionHetero(hetero_mode='mean', "
                       'theta free, beta free, alpha_eps free, log_sigma free)',
            'hetero': {
                'hetero_mode': 'mean',
                'extra_prior_type': 'iid',
                'extra_latents': 1,
                'alpha_independent_of_z1': True,
            },
            'param_names': PARAM_NAMES,
            'note': 'Hetero-mean with alpha_i STRUCTURALLY independent of '
                    'z_1 (net_extra_mu is degree-0). One-last-try variant '
                    'after the correlated hetero-mean line was dropped for '
                    'identification concerns.',
        },
        'training': {
            METHOD_TAG: {
                'encoder': ENCODER_LABEL,
                'encoder_config': ENCODER_CONFIG_STR,
                'hidden_dim': 64,
                'freeze_z1_skew': os.environ.get(
                    'BPP_HETERO_MEAN_FREEZE_Z1_SKEW', '0') == '1',
                'n_sim': n_sim,
                'n_epochs_vi_obs': n_epochs_vi_obs,
                'n_epochs_inner': n_epochs_inner,
                'n_iters_outer': n_iters_outer,
                'n_epochs_encoder': n_epochs_encoder,
                'lr': lr,
                'lr_encoder': lr_encoder,
                'picard_alpha': picard_alpha,
                'm_mem': m_mem,
                'restart_factor': RESTART_FACTOR,
                'ndraws': ndraws,
                'crn': True,
                'noise_seed': noise_seed,
                'seed_base': seed_base,
                'vi_seed': vi_seed,
                'method': METHOD,
                'vi_obs_history': vi_obs_history,
                'used_seed_phase1': int(used_seed_phase1)
                                      if used_seed_phase1 is not None else None,
            },
        },
        'estimates': {
            METHOD_TAG: {
                'method': METHOD,
                'picard_alpha': picard_alpha,
                'm_mem': m_mem,
                'restart_factor': RESTART_FACTOR,
                'theta_VI_obs': {k: float(theta_VI_obs[k]) for k in PARAM_NAMES},
                'theta_K': {k: float(theta_k[k]) for k in PARAM_NAMES},
                'trajectory': trajectory,
                'raw': [{'name': n, 'estimate': v, 'se': s}
                          for n, v, s in rows_K],
                'derived': [{'name': n, 'estimate': v, 'se': s}
                              for n, v, s in derived_K],
                'iwae_log_p': iwae_logp,
                'iwae_log_p_meta': iwae_meta,
            },
        },
        'se_method': {
            'method': 'BHHH (per-individual FIVO scores at theta_K)',
            'K_scores': 500, 'n_reps_scores': 3, 'seed_base': seed_base,
            'note': 'FIVO bootstrap PF marginalises alpha; hetero block '
                    'SEs are pinv-derived from a near-singular information; '
                    'decoder.theta SE is also pinv-derived (FIVO bound is '
                    'flat in theta).',
        },
    }
    with open(out_json, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {out_json}")


if __name__ == "__main__":
    # Fix every RNG (python / numpy / torch CPU+CUDA) and request deterministic
    # kernels before any draw; the script's own explicit seeds still apply.
    from mlye.seeding import seed_everything, env_seed
    seed_everything(env_seed("BPP_HETERO_MEAN_SEED", 11))
    main()
