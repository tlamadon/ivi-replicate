r"""IVI + VI-only runner on the simulation-hetero-scale DGP.

Consolidates `_scale_hetero_correlation_sweep.py` (joint-VI fit) and
`_scale_hetero_correlation_ivi.py` (Anderson outer loop) into a single
five-cell script that mirrors `_simulation_hockeystick_ivi_sweep.py`'s
METHOD x ENCODER branching, with the DGP swapped for the
scale-heterogeneity model:

  z_t   ~ poly-2 mu + softplus poly-2 sigma (NO softplus kink)
  z_1   ~ SinhArcsinh
  alpha | z_1 ~ N(beta_a0 + beta_a1 z_1, sigma_a_cond)
  y_t   = z_t + exp(alpha_i) * SinhArcsinh(0,1,0,beta) noise

12 free parameters in PARAM_NAMES order (z1_skew pinned at 0 per
specs/compute-simulation-hetero-scale.md, matching the hockey-stick
sibling):
  mu0, mu1, mu2, sigma0, sigma1, sigma2,
  z1_log_std, z1_log_tail,
  log_beta, beta_a0, beta_a1, log_sigma_a_cond

The conditional (beta_a0, beta_a1, log_sigma_a_cond) and the marginal
(mu_alpha_marg, sigma_alpha_marg, rho) are related by the
marginal-preserving reparametrization in
`_scale_hetero_correlation_sweep.calibrate_extra_hetero`.

Env vars (with defaults; spec headers map 1-to-1):
  METHOD               picard_cold | anderson | vi_only      (anderson)
  ENCODER              jn | mean_field | struct_markov | tjn (jn)
                       Required when METHOD=vi_only; ignored otherwise.
  PICARD_ALPHA         damping (also Anderson beta)          (0.6)
                       Ignored when METHOD=vi_only.
  M_MEM                Anderson memory                       (4)
  N_EPOCHS_INNER       inner-VI epoch budget (IVI cells)     (16000)
  N_ITERS_OUTER        outer iter count                      (10)
  LOG_EVERY            snapshot frequency                    (200)
  LR                   inner-VI learning rate                (1e-2)
  LR_TAG               filename tag for LR                   ('1e-2')
  FIX_NOISE            pre-draw eps once + reuse             (1)
  NOISE_SEED           eps draw seed                         (12345)
  N_EPOCHS_VI_OBS      epochs for the theta_VI_obs pre-fit   (20000)
  TJN_WARM_DECODER_FREEZE_EPOCHS                              (1000)
  N                    panel size                            (30000)
  T                    panel periods                         (6)

  Truth overrides (default = simulation-hetero-scale spec calibration):
  RHO                  correlation target (default 0.30)
  MU_ALPHA_MARG        log(0.10) ~= -2.303
  SIGMA_ALPHA_MARG     0.30

Output:
  output/bundles/cells/_simulation_hetero_scale_ivi_sweep/
    IVI:     <method>_alpha<a>[_m<m>]_ep<n>_lr<tag>[_crn]_rho<r>.json
    vi_only:    vi_only_<encoder>_h64_ep<n_vi_obs>_lr<tag>[_crn]_rho<r>.json
    mle_direct: mle_direct_no_me_ep<mle_ep>_lr<tag>_rho<r>.json

Phase 3 (post-fit diagnostics) appends a `diagnostics` block to the JSON:
  elbo_at_vi, iwae_at_vi      : at theta_VI_obs (phase-1 encoder reused)
  elbo_at_ivi, iwae_at_ivi    : at theta_K (fresh joint-normal-extra h=64
                                 encoder, frozen prior+decoder, 8000 ep,
                                 lr=1e-3, CRN). IVI methods only;
                                 null for METHOD=vi_only.
  elbo_at_truth, iwae_at_truth: at theta_truth (same encoder recipe).
  smc_log_p_at_truth          : hetero-aware bootstrap PF at K=1000 if
                                 available; else None.
                                 NOTE: the bootstrap PF in mlye/eval/smc.py
                                 currently assumes a centred residual
                                 density (no per-individual alpha latent).
                                 Until the hetero-scale variant lands,
                                 this field is None with a printed warning.
"""
import os
import sys
import time
import json
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.distributions.Distribution.set_default_validate_args(False)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mlye.models.encoders.base import JointNormalConfig, MarkovConfig
from mlye.models.encoders.normal import (
    JointNormalPosterior,
    TransformedJointNormalPosterior,
)
from mlye.models.encoders.markov import ConditionalNormalMarkovPosterior
from mlye.models.encoders.base import Encoder
from mlye.models.priors.markov import MarkovNormalConditionalPolyPrior
from mlye.models.decoders.ma import MASinhEmissionHetero
from mlye.models.full_model import FullModel

# Reuse DGP helpers + extract_params + calibration from the scoping script.
from _scale_hetero_correlation_sweep import (
    TRUTH_BASE,
    MU_ALPHA_MARG as MU_ALPHA_MARG_DEFAULT,
    SIGMA_ALPHA_MARG as SIGMA_ALPHA_MARG_DEFAULT,
    calibrate_extra_hetero,
    set_truth,
    simulate_panel,
    extract_params,
)
# Reuse differentiable simulator + Anderson update from the IVI scoping script.
from _scale_hetero_correlation_ivi import (
    PARAM_NAMES as _PARAM_NAMES_UPSTREAM,  # 13-vec (with z1_skew at idx 7)
    reparam_simulate as _reparam_simulate_upstream,
    anderson_step,
)

# Local PARAM_NAMES omits z1_skew (pinned at 0 in the estimator per
# specs/compute-simulation-hetero-scale.md). The upstream simulator still
# expects the 13-vec with z1_skew at index 7, so we wrap it by padding a zero.
PARAM_NAMES = [
    'mu0', 'mu1', 'mu2',
    'sigma0', 'sigma1', 'sigma2',
    'z1_log_std', 'z1_log_tail',
    'log_beta',
    'beta_a0', 'beta_a1', 'log_sigma_a_cond',
]


def reparam_simulate(theta_vec, u_z, u_alpha, u_y):
    z1_skew = torch.zeros(
        1, dtype=theta_vec.dtype, device=theta_vec.device)
    theta_13 = torch.cat([theta_vec[:7], z1_skew, theta_vec[7:]])
    return _reparam_simulate_upstream(theta_13, u_z, u_alpha, u_y)


# ----------------------------------------------------------------- env / labels

# Map spec's ENCODER label to a build directive. For IVI cells the encoder is
# always 'jn' (joint-normal-extra h=64); for vi_only cells the label drives
# the encoder family.
ENCODER_LABEL_TO_FILENAME_TAG = {
    'jn':            'jn',
    'mean_field':    'meanfield',
    'struct_markov': 'struct_markov',
    'tjn':           'tjn',
}


# ---------------------------- encoder builders (cell-aware, extra_latents=1)

def _build_jn_extra(T, hidden_dim=64, regularize=1e-3, sd_clamp=3.0):
    """Joint-normal h=64 with one extra latent for alpha (dim T+1)."""
    return JointNormalConfig(
        type='joint_normal_extra', dim=T,
        hidden_dim=hidden_dim, regularize=regularize,
        sd_clamp=sd_clamp, extra_latents=1,
    ).build()


def _build_meanfield_extra(T, hidden_dim=64, regularize=1e-3, sd_clamp=3.0):
    """Mean-field (diagonal) normal posterior with one extra latent.

    JointNormalConfig with type='normal_diagonal' forces diagonal=True
    internally; pairing with extra_latents=1 yields a diagonal (T+1)-vec
    posterior.
    """
    return JointNormalConfig(
        type='joint_normal_extra', dim=T,
        hidden_dim=hidden_dim, regularize=regularize,
        sd_clamp=sd_clamp, extra_latents=1, diagonal=True,
    ).build()


class TJNExtraPosterior(Encoder):
    r"""Transformed-joint-normal posterior over the joint (z_{1:T}, alpha)
    of dim T+1. Wraps `TransformedJointNormalPosterior(dim=T+1)` and
    projects y (which has shape (B, T)) to (B, T+1) by appending a
    fixed pseudo-observation channel (the empirical mean of y per
    individual) so the TJN's input-dim contract is satisfied.

    The skip connection `mu = y_proj + ...` then injects the y-mean as
    the alpha-channel anchor; the learned `self.mu(h)` head adjusts
    away from this anchor.
    """

    def __init__(self, T, hidden=64, regularize=1e-3):
        super().__init__()
        self.T = T
        self.tjn = TransformedJointNormalPosterior(
            dim=T + 1, hidden=hidden, regularize=regularize,
        )
        self.regularize = regularize

    def get_seed_dim(self) -> int:
        return self.T + 1

    def _project_y(self, y):
        # Append the per-individual mean as a pseudo-observation for the
        # alpha latent.
        y_mean = y.mean(dim=1, keepdim=True)
        return torch.cat([y, y_mean], dim=1)

    def draw_and_logprob(self, y, u, logpr_draw=False):
        y_proj = self._project_y(y)
        return self.tjn.draw_and_logprob(y_proj, u, logpr_draw=logpr_draw)

    def stats(self) -> dict:
        return self.tjn.stats()


def _build_tjn_extra(T, hidden=64, regularize=1e-3):
    """Transformed-joint-normal over (z_{1:T}, alpha) of dim T+1."""
    return TJNExtraPosterior(T=T, hidden=hidden, regularize=regularize)


class StructuredMarkovExtraPosterior(Encoder):
    r"""Conditional-Markov posterior on z_{1:T} plus an independent
    Gaussian head for the alpha latent.

    Composition pattern: q(z_{1:T}, alpha | y) = q_markov(z_{1:T} | y) *
    q_alpha(alpha | y). The alpha head is a small MLP on y that emits
    (mu_alpha, log_sigma_alpha); the (T+1)-vec latent matches the
    convention in `mlye/models/encoders/base.split_latent`.

    Mirrors the spec's `struct_markov` cell at h=64. The alpha head is a
    single hidden layer at the same width.
    """

    def __init__(self, T, hidden_dim=64, regularize=1e-3):
        super().__init__()
        self.dim = T  # nt for the time-indexed block (matches split_latent)
        self.regularize = regularize
        self.markov = ConditionalNormalMarkovPosterior(
            dim=T, hidden_dim=hidden_dim, regularize=regularize,
        )
        # alpha head: y -> (mu, log_sigma)
        self.alpha_fc = nn.Linear(T, hidden_dim)
        self.alpha_mu = nn.Linear(hidden_dim, 1)
        self.alpha_log_sigma = nn.Linear(hidden_dim, 1)
        # Tiny init so first epoch is well-behaved.
        with torch.no_grad():
            self.alpha_log_sigma.weight.mul_(0.01)
            self.alpha_log_sigma.bias.fill_(-1.5)
        self.mean_var = torch.tensor(0.0)

    def get_seed_dim(self) -> int:
        # T noise dims for the markov chain + 1 for alpha
        return self.dim + 1

    def draw_and_logprob(self, y, u, logpr_draw=False):
        # Markov over z_{1:T} uses the first T noise dims.
        u_z = u[:, :self.dim]
        u_a = u[:, self.dim:]
        z_markov, log_qz = self.markov.draw_and_logprob(y, u_z, logpr_draw=logpr_draw)
        # Alpha head conditioned on y.
        h = F.softplus(self.alpha_fc(y))
        mu_a = self.alpha_mu(h)
        sigma_a = F.softplus(self.alpha_log_sigma(h)) + self.regularize
        alpha = mu_a + sigma_a * u_a
        from torch.distributions import Normal
        log_qa = Normal(mu_a, sigma_a).log_prob(alpha).squeeze(1)
        z_full = torch.cat([z_markov, alpha], dim=1)
        return z_full, log_qz + log_qa

    def stats(self) -> dict:
        return {
            'encoder_type': 'StructuredMarkovExtraPosterior',
            'mean_var': float(self.markov.mean_var.detach().mean().item())
                if hasattr(self.markov.mean_var, 'detach') else 0.0,
        }


def build_encoder_for_cell(encoder_label, T, device, hidden_dim=64,
                            regularize=1e-3, sd_clamp=3.0):
    if encoder_label == 'jn':
        enc = _build_jn_extra(T, hidden_dim=hidden_dim,
                                regularize=regularize, sd_clamp=sd_clamp)
    elif encoder_label == 'mean_field':
        enc = _build_meanfield_extra(T, hidden_dim=hidden_dim,
                                        regularize=regularize, sd_clamp=sd_clamp)
    elif encoder_label == 'struct_markov':
        enc = StructuredMarkovExtraPosterior(
            T=T, hidden_dim=hidden_dim, regularize=regularize)
    elif encoder_label == 'tjn':
        enc = _build_tjn_extra(T, hidden=hidden_dim, regularize=regularize)
    else:
        raise ValueError(f"unknown encoder_label: {encoder_label}")
    return enc.to(device)


def build_full_model(T, device, encoder_label='jn', hidden_dim=64,
                       regularize=1e-3, sd_clamp=3.0):
    prior = MarkovNormalConditionalPolyPrior(
        nt=T, poly_degree=2, law_model='poly',
        z1_distr='sinh', extra_heterogeneity=True,
        extra_prior_type='iid').to(device)
    with torch.no_grad():
        prior.z1_skew.zero_()
    prior.z1_skew.requires_grad_(False)
    decoder = MASinhEmissionHetero(theta=0.0, fix_theta=True,
                                     sigma_eps=0.1,  # unused under hetero_mode='scale'
                                     hetero_mode='scale').to(device)
    encoder = build_encoder_for_cell(
        encoder_label, T, device,
        hidden_dim=hidden_dim, regularize=regularize, sd_clamp=sd_clamp)
    return FullModel(encoder=encoder, decoder=decoder, prior=prior).to(device)


# ---------------------------------------- truth + theta utilities

def _build_truth_dict(rho, mu_alpha_marg, sigma_alpha_marg):
    beta_a0, beta_a1, log_sigma_a_cond = calibrate_extra_hetero(
        rho, mu_alpha_marg, sigma_alpha_marg, TRUTH_BASE['z1_log_std'])
    truth = dict(TRUTH_BASE)
    truth['beta_a0'] = beta_a0
    truth['beta_a1'] = beta_a1
    truth['log_sigma_a_cond'] = log_sigma_a_cond
    return truth


def theta_to_vec(theta):
    return np.array([float(theta[k]) for k in PARAM_NAMES], dtype=np.float64)


def vec_to_theta(vec):
    return {k: float(v) for k, v in zip(PARAM_NAMES, vec)}


def L2_dist(a, b):
    return float(np.linalg.norm(a - b))


def _set_prior_decoder_from_dict(model, theta):
    """Write the 12-vec theta into prior + decoder. Encoder untouched.
    z1_skew is pinned at 0 (frozen in build_full_model) and absent from theta."""
    with torch.no_grad():
        p = model.prior
        p.net_mu.coeffs.copy_(torch.tensor(
            [theta['mu0'], theta['mu1'], theta['mu2']],
            dtype=p.net_mu.coeffs.dtype, device=p.net_mu.coeffs.device))
        p.net_sigma.coeffs.copy_(torch.tensor(
            [theta['sigma0'], theta['sigma1'], theta['sigma2']],
            dtype=p.net_sigma.coeffs.dtype, device=p.net_sigma.coeffs.device))
        p.z1_log_std.fill_(theta['z1_log_std'])
        p.z1_log_tail.fill_(theta['z1_log_tail'])
        # Extra-hetero (alpha | z_1) coefficients.
        p.net_extra_mu.coeffs.data[0] = theta['beta_a0']
        p.net_extra_mu.coeffs.data[1] = theta['beta_a1']
        p.net_extra_logsigma.coeffs.data[0] = theta['log_sigma_a_cond']
        model.decoder.log_beta.fill_(theta['log_beta'])


# ---------------------------------------- VI fit helpers (with retries)

class NaNFailure(RuntimeError):
    pass


def _retry_seeds(base_seed, n_retries):
    return [base_seed + i for i in range(n_retries)]


def fit_vi_inner(y, T, n_epochs, lr, log_every,
                  label='vi', vi_seed=None, n_retries=5,
                  eps_fixed=None, encoder_label='jn',
                  hidden_dim=64):
    """Cold-start a VI fit with the chosen encoder family. Returns
    (final_theta, trace, used_seed, model)."""
    last_err = None
    for s in _retry_seeds(vi_seed, n_retries):
        try:
            torch.manual_seed(s); np.random.seed(s)
            model = build_full_model(
                T, y.device, encoder_label=encoder_label, hidden_dim=hidden_dim)
            opt = torch.optim.AdamW(model.parameters(), lr=lr)
            trace = []
            t0 = time.time()
            nan_streak = 0
            prev_theta_vec = None
            last_grad_norm = float('nan')
            for epoch in range(n_epochs):
                opt.zero_grad()
                loss = -model.elbo(y, ndraws=1, eps_all=eps_fixed)
                if not torch.isfinite(loss):
                    nan_streak += 1
                    if epoch < 500 and nan_streak >= 50:
                        raise NaNFailure(f"NaN at epoch {epoch}")
                    continue
                nan_streak = 0
                loss.backward()
                grad_sq = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        grad_sq += float(p.grad.detach().pow(2).sum().item())
                last_grad_norm = float(np.sqrt(grad_sq))
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                if (epoch == 0 or (epoch + 1) % log_every == 0
                        or epoch == n_epochs - 1):
                    theta = extract_params(model)
                    theta_canon = {k: theta[k] for k in PARAM_NAMES}
                    theta_vec = theta_to_vec(theta_canon)
                    delta = (float(np.linalg.norm(theta_vec - prev_theta_vec))
                              if prev_theta_vec is not None else None)
                    prev_theta_vec = theta_vec
                    trace.append({
                        'epoch':            epoch + 1,
                        'elbo':             float(-loss.item()),
                        'theta':            theta_canon,
                        'grad_norm':        last_grad_norm,
                        'param_delta_norm': delta,
                    })
                    if ((epoch + 1) % max(1, log_every * 10) == 0
                            or epoch == 0 or epoch == n_epochs - 1):
                        print(f"    [{label}/s{s}] ep {epoch+1:5d}  "
                              f"ELBO={trace[-1]['elbo']:+.3f}  "
                              f"log_b={theta['log_beta']:+.3f}  "
                              f"mu1={theta['mu1']:+.3f}  "
                              f"mu_a={theta.get('_mu_alpha_marg_implied',0):+.3f}  "
                              f"sd_a={theta.get('_sigma_alpha_marg_implied',0):.3f}  "
                              f"({time.time()-t0:.1f}s)", flush=True)
            final = extract_params(model)
            theta_canon = {k: final[k] for k in PARAM_NAMES}
            return theta_canon, trace, s, model
        except NaNFailure as e:
            print(f"    [{label}/s{s}] NaN bail: {e}; retrying", flush=True)
            last_err = e
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    raise RuntimeError(f"inner VI failed after {n_retries} seed retries "
                        f"(last err: {last_err})")


def fit_tjn_warm(y, T, n_epochs, freeze_epochs, lr, log_every, vi_seed,
                   eps_fixed, jn_theta, hidden_dim=64, n_retries=5):
    """TJN two-phase warm-start fit on the hetero-scale model.

    Builds a TJN h=64 FullModel over dim T+1, copies prior+decoder from
    `jn_theta`, trains for `n_epochs` total. The first `freeze_epochs`
    epochs pin `decoder.log_beta`; afterwards both unfreeze.
    """
    last_err = None
    for s in _retry_seeds(vi_seed, n_retries):
        try:
            torch.manual_seed(s); np.random.seed(s)
            model = build_full_model(
                T, y.device, encoder_label='tjn', hidden_dim=hidden_dim)
            _set_prior_decoder_from_dict(model, jn_theta)
            for p in model.decoder.parameters():
                p.requires_grad_(False)
            opt = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad], lr=lr)
            trace = []
            t0 = time.time()
            nan_streak = 0
            prev_theta_vec = None
            last_grad_norm = float('nan')
            thawed = False
            for epoch in range(n_epochs):
                if not thawed and epoch >= freeze_epochs:
                    for p in model.decoder.parameters():
                        p.requires_grad_(True)
                    opt = torch.optim.AdamW(
                        [p for p in model.parameters() if p.requires_grad], lr=lr)
                    thawed = True
                    print(f"    [tjn/s{s}] ep {epoch+1}: decoder thawed",
                          flush=True)
                opt.zero_grad()
                loss = -model.elbo(y, ndraws=1, eps_all=eps_fixed)
                if not torch.isfinite(loss):
                    nan_streak += 1
                    if epoch < 500 and nan_streak >= 50:
                        raise NaNFailure(f"NaN at epoch {epoch}")
                    continue
                nan_streak = 0
                loss.backward()
                grad_sq = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        grad_sq += float(p.grad.detach().pow(2).sum().item())
                last_grad_norm = float(np.sqrt(grad_sq))
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                if (epoch == 0 or (epoch + 1) % log_every == 0
                        or epoch == n_epochs - 1):
                    theta = extract_params(model)
                    theta_canon = {k: theta[k] for k in PARAM_NAMES}
                    theta_vec = theta_to_vec(theta_canon)
                    delta = (float(np.linalg.norm(theta_vec - prev_theta_vec))
                              if prev_theta_vec is not None else None)
                    prev_theta_vec = theta_vec
                    trace.append({
                        'epoch':            epoch + 1,
                        'elbo':             float(-loss.item()),
                        'theta':            theta_canon,
                        'grad_norm':        last_grad_norm,
                        'param_delta_norm': delta,
                        'decoder_frozen':   not thawed,
                    })
                    if ((epoch + 1) % max(1, log_every * 10) == 0
                            or epoch == 0 or epoch == n_epochs - 1):
                        print(f"    [tjn/s{s}] ep {epoch+1:5d}  "
                              f"ELBO={trace[-1]['elbo']:+.3f}  "
                              f"log_b={theta['log_beta']:+.3f}  "
                              f"frozen={not thawed}  "
                              f"({time.time()-t0:.1f}s)", flush=True)
            final = extract_params(model)
            theta_canon = {k: final[k] for k in PARAM_NAMES}
            return theta_canon, trace, s, model
        except NaNFailure as e:
            print(f"    [tjn/s{s}] NaN bail: {e}; retrying", flush=True)
            last_err = e
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    raise RuntimeError(f"TJN warm-start failed after {n_retries} seed retries "
                        f"(last err: {last_err})")


def fit_encoder_only_at_theta(theta, y, T, vi_seed, n_epochs, lr, device,
                                  n_retries=5):
    """Fresh joint-normal-extra h=64 encoder with prior+decoder frozen
    at `theta`. Used by Phase 3 to evaluate ELBO/IWAE at theta_K or
    theta_truth."""
    last_err = None
    for s in _retry_seeds(vi_seed, n_retries):
        try:
            torch.manual_seed(s); np.random.seed(s)
            model = build_full_model(T, device, encoder_label='jn')
            _set_prior_decoder_from_dict(model, theta)
            for p in model.prior.parameters():
                p.requires_grad_(False)
            for p in model.decoder.parameters():
                p.requires_grad_(False)
            seed_dim = model.encoder.get_seed_dim()
            torch.manual_seed(s + 1)
            eps_fixed_local = torch.randn(
                (1, y.shape[0], seed_dim),
                dtype=torch.float32, device=device)
            opt = torch.optim.AdamW(
                [p for p in model.encoder.parameters() if p.requires_grad], lr=lr)
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
                torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), 5.0)
                opt.step()
                if (epoch + 1) % 2000 == 0 or epoch == 0:
                    print(f"  [enc@theta/s{s}] ep {epoch+1:5d}: "
                          f"ELBO={-loss.item():+.3f}  "
                          f"({time.time()-t0:.1f}s)", flush=True)
            return model
        except NaNFailure as e:
            print(f"  [enc@theta/s{s}] NaN bail: {e}; retrying", flush=True)
            last_err = e
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    raise RuntimeError(f"encoder-only fit failed after {n_retries} retries "
                        f"(last err: {last_err})")


# ----------------------------------------------------------------- main

def main():
    method            = os.environ.get('METHOD', 'anderson')
    encoder_label     = os.environ.get('ENCODER', 'jn')
    emission          = os.environ.get('EMISSION', 'sinh')
    alpha             = float(os.environ.get('PICARD_ALPHA', 0.6))
    n_epochs          = int(os.environ.get('N_EPOCHS_INNER', 16000))
    n_iters           = int(os.environ.get('N_ITERS_OUTER', 10))
    m_mem             = int(os.environ.get('M_MEM', 4))
    log_every         = int(os.environ.get('LOG_EVERY', 200))
    lr                = float(os.environ.get('LR', 1e-2))
    fix_noise         = int(os.environ.get('FIX_NOISE', 1)) != 0
    noise_seed        = (int(os.environ.get('NOISE_SEED', 12345))
                          if fix_noise else None)
    n_epochs_vi_obs   = int(os.environ.get('N_EPOCHS_VI_OBS', 20000))
    n_epochs_encoder  = int(os.environ.get('N_EPOCHS_ENCODER', 8000))
    n_epochs_mle      = int(os.environ.get('N_EPOCHS_MLE', 20000))
    lr_encoder        = float(os.environ.get('LR_ENCODER', 1e-3))
    L_iwae            = int(os.environ.get('L_IWAE', 100))
    K_smc             = int(os.environ.get('K_SMC', 1000))
    N_override        = int(os.environ.get('N', 30000))
    T_override        = int(os.environ.get('T', 6))
    tjn_freeze_epochs = int(os.environ.get(
        'TJN_WARM_DECODER_FREEZE_EPOCHS', 1000))

    rho               = float(os.environ.get('RHO', 0.30))
    mu_alpha_marg     = float(os.environ.get('MU_ALPHA_MARG',
                                              MU_ALPHA_MARG_DEFAULT))
    sigma_alpha_marg  = float(os.environ.get('SIGMA_ALPHA_MARG',
                                              SIGMA_ALPHA_MARG_DEFAULT))

    assert method in ('picard_cold', 'anderson', 'vi_only', 'mle_direct'), \
        f"unknown METHOD: {method}"
    assert emission in ('sinh', 'degenerate'), \
        f"unknown EMISSION: {emission}"
    if method == 'mle_direct':
        assert emission == 'degenerate', \
            "METHOD=mle_direct requires EMISSION=degenerate"
    else:
        emission = 'sinh'
    is_vi_only = method == 'vi_only'
    is_mle_direct = method == 'mle_direct'
    if not is_mle_direct:
        assert encoder_label in ENCODER_LABEL_TO_FILENAME_TAG, \
            f"unknown ENCODER: {encoder_label}"
    if method in ('picard_cold', 'anderson'):
        encoder_label = 'jn'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device={device}", flush=True)
    print(f"DGP: simulation-hetero-scale (poly-2 mu/sigma + sinh z1 + "
          f"sinh-arcsinh emission, with alpha-in-scale heterogeneity).",
          flush=True)
    print(f"  rho={rho:+.4f}  mu_a^marg={mu_alpha_marg:+.4f}  "
          f"sigma_a^marg={sigma_alpha_marg:.4f}", flush=True)
    print(f"method={method} encoder={encoder_label} alpha={alpha} "
          f"n_iters={n_iters} n_epochs_inner={n_epochs} m_mem={m_mem} "
          f"log_every={log_every} lr={lr} fix_noise={fix_noise} "
          f"noise_seed={noise_seed}", flush=True)
    print(f"n_epochs_vi_obs={n_epochs_vi_obs}", flush=True)
    if encoder_label == 'tjn':
        print(f"tjn_warm_decoder_freeze_epochs={tjn_freeze_epochs}", flush=True)

    N, T = N_override, T_override
    OBS_SEED = 11
    vi_seed = OBS_SEED * 1000 + 7  # = 11007

    # --- Build truth dict + simulate y_obs ---
    truth = _build_truth_dict(rho, mu_alpha_marg, sigma_alpha_marg)
    truth_vec = theta_to_vec(truth)

    print(f"\n=== Phase 0: simulate y_obs (N={N}, T={T}) ===", flush=True)
    torch.manual_seed(OBS_SEED); np.random.seed(OBS_SEED)
    truth_model = build_full_model(T, device, encoder_label='jn')
    set_truth(truth_model, TRUTH_BASE,
                truth['beta_a0'], truth['beta_a1'],
                truth['log_sigma_a_cond'])
    y_obs, _z_true, _alpha_true = simulate_panel(
        truth_model, N, T, OBS_SEED, device)
    print(f"  y_obs: std={y_obs.std().item():+.4f}  "
          f"alpha emp mean={_alpha_true.mean().item():+.4f}  "
          f"sd={_alpha_true.std().item():.4f}", flush=True)
    del truth_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if is_mle_direct:
        # METHOD=mle_direct + EMISSION=degenerate (per spec 2026-06-22).
        # No encoder; the inference model is the homogeneous prior alone
        # (no extra-hetero branch, since α enters only through the emission).
        # Fit by direct MLE on the closed-form prior log-likelihood at
        # z = y_obs. The 8 free params are mu0/1/2, sigma0/1/2, z1_log_std,
        # z1_log_tail; z1_skew is pinned at 0 (matching the IVI/VI cells
        # and the hockey-stick sibling); log_beta and the three
        # extra-hetero params do not exist in this model class.
        print(f"\n=== Phase 0b: MLE direct (no measurement error) ===",
              flush=True)
        torch.manual_seed(vi_seed); np.random.seed(vi_seed)
        prior = MarkovNormalConditionalPolyPrior(
            nt=T, poly_degree=2, law_model='poly',
            z1_distr='sinh', extra_heterogeneity=False).to(device)
        with torch.no_grad():
            prior.z1_skew.zero_()
        prior.z1_skew.requires_grad_(False)
        opt = torch.optim.AdamW(
            [p for p in prior.parameters() if p.requires_grad], lr=lr)
        mle_trace = []
        t0 = time.time()
        prev_theta_vec = None
        nan_streak = 0
        free_keys = ['mu0', 'mu1', 'mu2', 'sigma0', 'sigma1', 'sigma2',
                      'z1_log_std', 'z1_log_tail']
        for epoch in range(n_epochs_mle):
            opt.zero_grad()
            loss = -prior.log_prob(y_obs).mean()
            if not torch.isfinite(loss):
                nan_streak += 1
                if epoch < 200 and nan_streak >= 10:
                    raise RuntimeError(
                        f"mle_direct NaN streak at epoch {epoch}")
                continue
            nan_streak = 0
            loss.backward()
            grad_sq = 0.0
            for p in prior.parameters():
                if p.grad is not None:
                    grad_sq += float(p.grad.detach().pow(2).sum().item())
            last_grad_norm = float(np.sqrt(grad_sq))
            torch.nn.utils.clip_grad_norm_(prior.parameters(), 5.0)
            opt.step()
            if epoch == 0 or (epoch + 1) % log_every == 0 \
                    or epoch == n_epochs_mle - 1:
                t = {
                    'mu0':              float(prior.net_mu.coeffs[0].item()),
                    'mu1':              float(prior.net_mu.coeffs[1].item()),
                    'mu2':              float(prior.net_mu.coeffs[2].item()),
                    'sigma0':           float(prior.net_sigma.coeffs[0].item()),
                    'sigma1':           float(prior.net_sigma.coeffs[1].item()),
                    'sigma2':           float(prior.net_sigma.coeffs[2].item()),
                    'z1_log_std':       float(prior.z1_log_std.item()),
                    'z1_log_tail':      float(prior.z1_log_tail.item()),
                    'log_beta':         float('nan'),
                    'beta_a0':          float('nan'),
                    'beta_a1':          float('nan'),
                    'log_sigma_a_cond': float('nan'),
                }
                t_vec_free = np.array(
                    [t[k] for k in free_keys], dtype=np.float64)
                delta = (float(np.linalg.norm(t_vec_free - prev_theta_vec))
                          if prev_theta_vec is not None else None)
                prev_theta_vec = t_vec_free
                mle_trace.append({
                    'epoch':            epoch + 1,
                    'log_p':            float(-loss.item()),
                    'theta':            t,
                    'grad_norm':        last_grad_norm,
                    'param_delta_norm': delta,
                })
                if ((epoch + 1) % (log_every * 10) == 0
                        or epoch == 0 or epoch == n_epochs_mle - 1):
                    print(f"  [mle/{epoch+1:5d}]  log_p={-loss.item():+.3f}  "
                          f"mu1={t['mu1']:+.3f}  sigma0={t['sigma0']:+.3f}  "
                          f"sigma2={t['sigma2']:+.3f}  "
                          f"||g||={last_grad_norm:.3f}  "
                          f"({time.time()-t0:.1f}s)", flush=True)
        theta_MLE = mle_trace[-1]['theta']
        truth = _build_truth_dict(rho, mu_alpha_marg, sigma_alpha_marg)
        L2_MLE = float(np.linalg.norm(
            np.array([theta_MLE[k] for k in free_keys], dtype=np.float64) -
            np.array([truth[k]     for k in free_keys], dtype=np.float64)))
        print(f"\ntheta_MLE: mu1={theta_MLE['mu1']:+.4f}  "
              f"sigma0={theta_MLE['sigma0']:+.4f}  "
              f"sigma2={theta_MLE['sigma2']:+.4f}  "
              f"L2(8 free prior+z1)={L2_MLE:.4f}", flush=True)
        prior_log_p_at_mle = float(-loss.item())

        # Closed-form prior log-likelihood at the homogeneous projection of
        # truth on y_obs (degenerate-emission reference; not directly
        # comparable to the IVI cells' smc_log_p_at_truth).
        torch.manual_seed(vi_seed + 1)
        prior_truth = MarkovNormalConditionalPolyPrior(
            nt=T, poly_degree=2, law_model='poly',
            z1_distr='sinh', extra_heterogeneity=False).to(device)
        with torch.no_grad():
            prior_truth.z1_skew.zero_()
            prior_truth.net_mu.coeffs.copy_(torch.tensor(
                [truth['mu0'], truth['mu1'], truth['mu2']],
                dtype=prior_truth.net_mu.coeffs.dtype,
                device=prior_truth.net_mu.coeffs.device))
            prior_truth.net_sigma.coeffs.copy_(torch.tensor(
                [truth['sigma0'], truth['sigma1'], truth['sigma2']],
                dtype=prior_truth.net_sigma.coeffs.dtype,
                device=prior_truth.net_sigma.coeffs.device))
            prior_truth.z1_log_std.fill_(truth['z1_log_std'])
            prior_truth.z1_log_tail.fill_(truth['z1_log_tail'])
        prior_truth.z1_skew.requires_grad_(False)
        with torch.no_grad():
            prior_log_p_at_truth = float(
                prior_truth.log_prob(y_obs).mean().item())
        print(f"  prior_log_p_at_mle  = {prior_log_p_at_mle:+.4f}", flush=True)
        print(f"  prior_log_p_at_truth= {prior_log_p_at_truth:+.4f}",
              flush=True)
        del prior, prior_truth
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Output filename: mle_direct_no_me_ep<N>_lr<tag>_rho<r>.json
        lr_tag = os.environ.get('LR_TAG', f'{lr:.0e}')
        fname = f"mle_direct_no_me_ep{n_epochs_mle}_lr{lr_tag}_rho{rho:.2f}.json"
        out_dir = "output/bundles/cells/_simulation_hetero_scale_ivi_sweep"
        os.makedirs(out_dir, exist_ok=True)
        fname = os.environ.get('OUT_NAME', '') or fname
        out_path = os.path.join(out_dir, fname)

        config = {
            'dgp':              'simulation-hetero-scale',
            'method':           method,
            'encoder':          None,
            'emission':         emission,
            'alpha':            None,
            'm_mem':            None,
            'n_iters_outer':    None,
            'n_epochs_inner':   None,
            'n_epochs_vi_obs':  None,
            'n_epochs_encoder': None,
            'n_epochs_mle':     n_epochs_mle,
            'tjn_warm_decoder_freeze_epochs': None,
            'log_every':        log_every,
            'lr':               lr,
            'lr_encoder':       None,
            'fix_noise':        None,
            'noise_seed':       None,
            'L_iwae':           L_iwae,
            'K_smc':            K_smc,
            'N':                N, 'T': T,
            'obs_seed':         OBS_SEED,
            'vi_seed':          vi_seed,
            'restart_factor':   None,
            'rho':              rho,
            'mu_alpha_marg':    mu_alpha_marg,
            'sigma_alpha_marg': sigma_alpha_marg,
            'extra_latents':    None,
        }
        out = {
            'config':                 config,
            'truth':                  {k: float(v) for k, v in truth.items()},
            'theta_VI_obs':           theta_MLE,
            'theta_VI_obs_trace':     None,
            'theta_VI_obs_jn_trace':  None,
            'mle_trace':              mle_trace,
            'L2_VI':                  L2_MLE,
            'trajectory':             [],
            'final_theta':            theta_MLE,
            'final_L2':               L2_MLE,
            'diagnostics': {
                'elbo_at_vi':          None,
                'iwae_at_vi':          None,
                'elbo_at_ivi':         None,
                'iwae_at_ivi':         None,
                'elbo_at_truth':       None,
                'iwae_at_truth':       None,
                'smc_log_p_at_truth':  None,
                'prior_log_p_at_mle':   prior_log_p_at_mle,
                'prior_log_p_at_truth': prior_log_p_at_truth,
            },
        }
        with open(out_path, 'w') as f:
            json.dump(out, f, indent=2)
        print(f"\nDone. L2_MLE={L2_MLE:.4f}", flush=True)
        print(f"Wrote {out_path}", flush=True)
        return

    # --- Pre-draw the CRN epsilon (shape depends on encoder) ---
    # The phase-1 encoder is the cell's chosen encoder family; for IVI
    # phases (2 onward), the encoder is always JN, so seed_dim = T + 1.
    # We pre-draw at the cell's encoder seed_dim.
    if encoder_label == 'tjn':
        # For TJN cell, Phase 1 has two stages: an internal JN h=64 warm
        # fit (seed_dim T+1) then the TJN fit (seed_dim T+1). Same shape.
        seed_dim_phase1 = T + 1
    elif encoder_label == 'struct_markov':
        seed_dim_phase1 = T + 1  # StructuredMarkovExtraPosterior
    else:
        seed_dim_phase1 = T + 1  # joint_normal_extra, normal_diagonal w/ extra
    if fix_noise:
        torch.manual_seed(noise_seed)
        eps_fixed_phase1 = torch.randn(
            (1, N, seed_dim_phase1), dtype=torch.float32, device=device)
        print(f"\n[crn] eps_fixed (phase 1) pre-drawn at noise_seed={noise_seed}  "
              f"shape={tuple(eps_fixed_phase1.shape)}", flush=True)
    else:
        eps_fixed_phase1 = None

    # IVI phase 2: encoder always JN (seed_dim T+1).
    if fix_noise:
        torch.manual_seed(noise_seed)
        eps_fixed_inner = torch.randn(
            (1, N, T + 1), dtype=torch.float32, device=device)
    else:
        eps_fixed_inner = None

    # === Phase 1: theta_VI_obs ===
    print(f"\n=== Phase 1: fit theta_VI_obs ({n_epochs_vi_obs} ep, "
          f"encoder={encoder_label}) ===", flush=True)
    vi_obs_jn_trace = None
    if encoder_label == 'tjn':
        # Internal JN warm-start fit first.
        print(f"\n[tjn] internal JN warm-start ({n_epochs_vi_obs} ep) ...",
              flush=True)
        theta_VI_jn, jn_trace, _, _model_jn = fit_vi_inner(
            y_obs, T, n_epochs_vi_obs, lr=lr, log_every=log_every,
            label='VI_obs_jn', vi_seed=vi_seed,
            eps_fixed=eps_fixed_phase1, encoder_label='jn')
        vi_obs_jn_trace = jn_trace
        del _model_jn
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"\n[tjn] TJN fit ({n_epochs_vi_obs} ep, "
              f"freeze_decoder_first={tjn_freeze_epochs}) ...", flush=True)
        theta_VI_obs, vi_obs_trace, _, model_VI_obs = fit_tjn_warm(
            y_obs, T, n_epochs=n_epochs_vi_obs,
            freeze_epochs=tjn_freeze_epochs, lr=lr,
            log_every=log_every, vi_seed=vi_seed,
            eps_fixed=eps_fixed_phase1, jn_theta=theta_VI_jn)
    else:
        theta_VI_obs, vi_obs_trace, _, model_VI_obs = fit_vi_inner(
            y_obs, T, n_epochs_vi_obs, lr=lr, log_every=log_every,
            label='VI_obs', vi_seed=vi_seed,
            eps_fixed=eps_fixed_phase1, encoder_label=encoder_label)
    theta_VI_obs_vec = theta_to_vec(theta_VI_obs)
    L2_VI = L2_dist(theta_VI_obs_vec, truth_vec)
    print(f"\ntheta_VI_obs:  log_beta={theta_VI_obs['log_beta']:+.4f}  "
          f"beta_a0={theta_VI_obs['beta_a0']:+.4f}  "
          f"sigma0={theta_VI_obs['sigma0']:+.4f}  "
          f"L2={L2_VI:.4f}", flush=True)
    print(f"truth:         log_beta={truth['log_beta']:+.4f}  "
          f"beta_a0={truth['beta_a0']:+.4f}  "
          f"sigma0={truth['sigma0']:+.4f}", flush=True)

    # --- Output path / cell filename ---
    lr_tag = os.environ.get('LR_TAG', f'{lr:.0e}')
    suffixes = [f'lr{lr_tag}']
    if fix_noise:
        suffixes.append('crn')
    suffixes.append(f"rho{rho:.2f}")
    suffix = '_'.join(suffixes)
    out_dir = "output/bundles/cells/_simulation_hetero_scale_ivi_sweep"
    os.makedirs(out_dir, exist_ok=True)
    if is_vi_only:
        enc_tag = ENCODER_LABEL_TO_FILENAME_TAG[encoder_label]
        fname = f"vi_only_{enc_tag}_h64_ep{n_epochs_vi_obs}_{suffix}.json"
    elif method == 'anderson':
        fname = f"anderson_alpha{alpha:.2f}_m{m_mem}_ep{n_epochs}_{suffix}.json"
    else:
        fname = f"{method}_alpha{alpha:.2f}_ep{n_epochs}_{suffix}.json"
    # OUT_NAME overrides the filename; replicate.py --smoke uses it to keep the
    # canonical cell names while shrinking the epoch counts encoded in them.
    fname = os.environ.get('OUT_NAME', '') or fname
    out_path = os.path.join(out_dir, fname)

    RESTART_FACTOR = 1.5

    config = {
        'dgp':              'simulation-hetero-scale',
        'method':           method,
        'encoder':          encoder_label,
        'alpha':            None if is_vi_only else alpha,
        'm_mem':            m_mem if method == 'anderson' else None,
        'n_iters_outer':    None if is_vi_only else n_iters,
        'n_epochs_inner':   None if is_vi_only else n_epochs,
        'n_epochs_vi_obs':  n_epochs_vi_obs,
        'n_epochs_encoder': n_epochs_encoder,
        'tjn_warm_decoder_freeze_epochs': (
            tjn_freeze_epochs if encoder_label == 'tjn' else None),
        'log_every':        log_every,
        'lr':               lr,
        'lr_encoder':       lr_encoder,
        'fix_noise':        fix_noise,
        'noise_seed':       noise_seed,
        'L_iwae':           L_iwae,
        'K_smc':            K_smc,
        'N':                N, 'T': T,
        'obs_seed':         OBS_SEED,
        'vi_seed':          vi_seed,
        'restart_factor':   None if is_vi_only else RESTART_FACTOR,
        'rho':              rho,
        'mu_alpha_marg':    mu_alpha_marg,
        'sigma_alpha_marg': sigma_alpha_marg,
        'extra_latents':    1,
        'param_names':      PARAM_NAMES,
    }
    base_out = {
        'config':                 config,
        'truth':                  {k: float(v) for k, v in truth.items()},
        'theta_VI_obs':           theta_VI_obs,
        'theta_VI_obs_trace':     vi_obs_trace,
        'theta_VI_obs_jn_trace':  vi_obs_jn_trace,
        'L2_VI':                  L2_VI,
    }

    # === Phase 2: IVI outer loop (skipped for vi_only) ===
    if is_vi_only:
        print(f"\n=== Phase 2 SKIPPED (METHOD=vi_only) ===", flush=True)
        out = {
            **base_out,
            'trajectory':  None,
            'final_theta': None,
            'final_L2':    None,
        }
        theta_k_for_phase3 = None
    else:
        # Differentiable simulator noise (CRN across all outer iters).
        torch.manual_seed(OBS_SEED * 1000)
        u_z     = torch.randn(N, T, device=device, dtype=torch.float32)
        u_alpha = torch.randn(N,    device=device, dtype=torch.float32)
        u_y     = torch.randn(N, T, device=device, dtype=torch.float32)

        theta_k = dict(theta_VI_obs)
        history = []
        prev_g_norm = None

        trajectory = [{'iter': 0,
                        'theta_k':  {k: float(theta_k[k]) for k in PARAM_NAMES},
                        'L2_truth': L2_VI}]

        print(f"\n=== Phase 2: IVI outer loop ({n_iters} iters) ===",
              flush=True)
        L2_to_truth = L2_VI
        for k in range(1, n_iters + 1):
            iter_t0 = time.time()
            print(f"\n[iter {k}] simulate y_sim + inner VI "
                  f"({n_epochs} ep, method={method})...", flush=True)

            theta_t = torch.tensor([theta_k[name] for name in PARAM_NAMES],
                                    dtype=torch.float32, device=device)
            with torch.no_grad():
                y_sim = reparam_simulate(theta_t, u_z, u_alpha, u_y)

            theta_VI, trace, used_seed, _ = fit_vi_inner(
                y_sim, T, n_epochs, lr=lr, log_every=log_every,
                label=f"{method[:6]}/it{k}",
                vi_seed=vi_seed, eps_fixed=eps_fixed_inner,
                encoder_label='jn')

            iter_wall = time.time() - iter_t0
            theta_VI_vec = theta_to_vec(theta_VI)
            x_k = theta_to_vec(theta_k)

            restart = False
            if method == 'anderson':
                g_k = theta_VI_obs_vec - theta_VI_vec
                g_norm = float(np.linalg.norm(g_k))
                if prev_g_norm is not None and g_norm > RESTART_FACTOR * prev_g_norm:
                    print(f"    [restart] ||g||={g_norm:.4f} "
                          f"(was {prev_g_norm:.4f})", flush=True)
                    history = []
                    restart = True
                x_next, gamma = anderson_step(
                    x_k, g_k, history, m=m_mem, beta=alpha)
                history.append({'x': x_k.copy(), 'g': g_k.copy()})
                prev_g_norm = g_norm
            else:
                x_next = x_k + alpha * (theta_VI_obs_vec - theta_VI_vec)
                g_k = theta_VI_obs_vec - theta_VI_vec
                g_norm = float(np.linalg.norm(g_k))
                gamma = None

            theta_next = vec_to_theta(x_next)
            L2_to_truth = L2_dist(x_next, truth_vec)

            print(f"  inner VI final: log_beta={theta_VI['log_beta']:+.4f}  "
                  f"beta_a0={theta_VI['beta_a0']:+.4f}  "
                  f"sigma0={theta_VI['sigma0']:+.4f}", flush=True)
            print(f"  theta_{k}: log_beta={theta_next['log_beta']:+.4f}  "
                  f"beta_a0={theta_next['beta_a0']:+.4f}  "
                  f"L2={L2_to_truth:.4f}  ||g||={g_norm:.4f}  "
                  f"iter_wall={iter_wall:.1f}s", flush=True)

            trajectory.append({
                'iter':            k,
                'iter_wall_s':     iter_wall,
                'theta_k':         {kk: float(theta_next[kk]) for kk in PARAM_NAMES},
                'theta_VI':        {kk: float(theta_VI[kk]) for kk in PARAM_NAMES},
                'L2_truth':        L2_to_truth,
                'g_norm':          g_norm,
                'restart':         bool(restart),
                'gamma':           (gamma.tolist()
                                    if (gamma is not None and len(gamma) > 0)
                                    else None),
                'used_seed':       int(used_seed) if used_seed is not None else None,
                'inner_trace':     trace,
            })
            theta_k = theta_next

            out = {
                **base_out,
                'trajectory':  trajectory,
                'final_theta': {kk: float(theta_k[kk]) for kk in PARAM_NAMES},
                'final_L2':    L2_to_truth,
            }
            with open(out_path, 'w') as f:
                json.dump(out, f, indent=2)
            print(f"  [cache write: {out_path}]", flush=True)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        theta_k_for_phase3 = theta_k

    # === Phase 3: post-fit diagnostics ===
    print(f"\n=== Phase 3: post-fit diagnostics ===", flush=True)

    # 3a. ELBO + IWAE at theta_VI_obs (phase-1 encoder reused).
    print(f"\n[3a] ELBO/IWAE at theta_VI_obs (phase-1 encoder, L={L_iwae}) ...",
          flush=True)
    diag_vi = model_VI_obs.elbo_diagnostics(y_obs, ndraws=L_iwae)
    elbo_at_vi = float(diag_vi['elbo'])
    iwae_at_vi = float(diag_vi['iwae'])
    print(f"  elbo_at_vi={elbo_at_vi:+.4f}  iwae_at_vi={iwae_at_vi:+.4f}  "
          f"gap={iwae_at_vi - elbo_at_vi:+.4f}", flush=True)
    del model_VI_obs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 3b. ELBO + IWAE at theta_K (IVI cells only).
    if is_vi_only:
        elbo_at_ivi = None
        iwae_at_ivi = None
    else:
        print(f"\n[3b] fit encoder@theta_K + ELBO/IWAE "
              f"({n_epochs_encoder} ep, lr={lr_encoder}) ...", flush=True)
        model_ivi = fit_encoder_only_at_theta(
            theta_k_for_phase3, y_obs, T, vi_seed,
            n_epochs=n_epochs_encoder, lr=lr_encoder, device=device)
        diag_ivi = model_ivi.elbo_diagnostics(y_obs, ndraws=L_iwae)
        elbo_at_ivi = float(diag_ivi['elbo'])
        iwae_at_ivi = float(diag_ivi['iwae'])
        print(f"  elbo_at_ivi={elbo_at_ivi:+.4f}  "
              f"iwae_at_ivi={iwae_at_ivi:+.4f}  "
              f"gap={iwae_at_ivi - elbo_at_ivi:+.4f}", flush=True)
        del model_ivi
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 3c. ELBO + IWAE at theta_truth.
    print(f"\n[3c] fit encoder@truth + ELBO/IWAE "
          f"({n_epochs_encoder} ep, lr={lr_encoder}) ...", flush=True)
    model_truth = fit_encoder_only_at_theta(
        truth, y_obs, T, vi_seed, n_epochs=n_epochs_encoder,
        lr=lr_encoder, device=device)
    diag_truth = model_truth.elbo_diagnostics(y_obs, ndraws=L_iwae)
    elbo_at_truth = float(diag_truth['elbo'])
    iwae_at_truth = float(diag_truth['iwae'])
    print(f"  elbo_at_truth={elbo_at_truth:+.4f}  "
          f"iwae_at_truth={iwae_at_truth:+.4f}  "
          f"gap={iwae_at_truth - elbo_at_truth:+.4f}", flush=True)

    # 3d. SMC log p_hat at truth: requires an alpha-aware bootstrap PF that
    # samples alpha from its conditional prior given z_1 for each particle.
    # The current `mlye/eval/smc.py` BootstrapParticleFilter assumes a
    # centred residual density (no per-individual scale latent), so it
    # cannot be used as-is. Until a hetero-aware variant lands we emit
    # None and print a warning. See spec Layer 1 Phase 3.5 note.
    smc_log_p_at_truth = None
    print(f"\n[3d] SMC log p_hat at truth: SKIPPED (hetero-aware bootstrap "
          f"PF not yet implemented; see follow-up).", flush=True)
    del model_truth
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    out['diagnostics'] = {
        'elbo_at_vi':         elbo_at_vi,
        'iwae_at_vi':         iwae_at_vi,
        'elbo_at_ivi':        elbo_at_ivi,
        'iwae_at_ivi':        iwae_at_ivi,
        'elbo_at_truth':      elbo_at_truth,
        'iwae_at_truth':      iwae_at_truth,
        'smc_log_p_at_truth': smc_log_p_at_truth,
    }
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)

    if is_vi_only:
        print(f"\nDone. L2_VI={L2_VI:.4f}", flush=True)
    else:
        print(f"\nDone. Final L2={trajectory[-1]['L2_truth']:.4f}", flush=True)
    print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    # Fix every RNG (python / numpy / torch CPU+CUDA) and request deterministic
    # kernels before any draw; the script's own explicit seeds still apply.
    from mlye.seeding import seed_everything, env_seed
    seed_everything(env_seed("OBS_SEED", 11))
    main()
