r"""AR(1) + Normal helper module --- the well-specified Gaussian baseline.

DGP (canonical AR(1) preset, per simulation-ar1n.md):
  - mu(z)    = mu_1 z                       (mu_0 = mu_2 = 0 pinned)
  - sigma(z) = softplus(sigma_0)            (sigma_1 = sigma_2 = 0 pinned)
  - z_1      ~ N(0, softplus(z1_log_std))   (normal initial state)
  - y_t      = z_t + e_t,  e_t ~ N(0, sigma_eps)  (Gaussian emission)
  - No MA, no extra heterogeneity.

Headline calibration:
  mu_1 = 0.9, sigma_0 = -1.48, z1_log_std = -0.733 (so sigma_z1 = 0.40),
  log_sigma_eps = -1.470 (so sigma_eps = 0.23). N = 30000, T = 6.

This file is a fresh sibling of `softplus_sinh_posterior_comparison.py`
that bakes in the AR(1)+Normal DGP. It exposes the same module-level
surface (TRUTH, PARAM_NAMES, build_prior_decoder, build_full_model,
build_encoder, set_prior_decoder_to_truth, extract_params,
simulate_from_truth, reparam_simulate, NaNFailure, _retry_seeds) so
that the Layer 1a sweep script can import it as ``M`` and reuse the
hockey-stick precedent's plumbing verbatim.

Naming convention: PARAM_NAMES is the same 13-element list as the
hockey-stick sibling so the JSON schema is trivially compatible. The
five "pinned" entries (alpha0, log_alpha1, z1_skew, z1_log_tail,
log_beta) are constants of the inference model class and are emitted
into the JSON as recorded constants. The eight free entries are
mu0 (idx 2), mu1 (idx 3), mu2 (idx 4), sigma0 (idx 5), sigma1 (idx 6),
sigma2 (idx 7), z1_log_std (idx 8), log_sigma_eps (idx 11). Note: the
four "extra" poly coords mu0, mu2, sigma1, sigma2 equal 0 at the
canonical AR(1) truth but are FREE during estimation (initialised at
0 and let drift) per spec `specs/compute-simulation-ar1n.md` Model A.
"""
import time

import numpy as np
import torch
import torch.nn.functional as F

from mlye.models.encoders.base import (
    JointNormalConfig, MarkovConfig,
)
from mlye.models.encoders.normal import TransformedJointNormalPosterior
from mlye.models.priors.markov import MarkovNormalConditionalPolyPrior
from mlye.models.decoders.ma import MASinhEmission
from mlye.models.full_model import FullModel


# ----------------------------------------------------------------------
# Truth (AR(1) + Normal canonical calibration)
# ----------------------------------------------------------------------
# mu_1 = 0.9, sigma_0 = -1.48 (so sigma(z) = softplus(-1.48) ~ 0.205),
# sigma_z1 = 0.40  =>  z1_log_std = log(exp(0.40) - 1) ~ -0.733,
# sigma_eps = 0.23 =>  log_sigma_eps = log(0.23) ~ -1.470.

MU1_DEFAULT            =  0.9000
SIGMA0_DEFAULT         = -1.4800
SIGMA_Z1_DEFAULT       = -0.7330      # softplus(.) = 0.4002
LOG_SIGMA_EPS_DEFAULT  = -1.4700      # exp(.) = 0.2299

TRUTH = {
    'alpha0':         0.0,            # pinned (not used: law_model='poly')
    'log_alpha1':     0.0,            # pinned (not used)
    'mu0':            0.0,            # pinned
    'mu1':            MU1_DEFAULT,    # free
    'mu2':            0.0,            # pinned
    'sigma0':         SIGMA0_DEFAULT, # free
    'sigma1':         0.0,            # pinned
    'sigma2':         0.0,            # pinned
    'z1_log_std':     SIGMA_Z1_DEFAULT,  # free
    'z1_skew':        0.0,            # pinned (normal z_1)
    'z1_log_tail':    0.0,            # pinned (normal z_1, log(1)=0)
    'log_sigma_eps':  LOG_SIGMA_EPS_DEFAULT,  # free
    'log_beta':       0.0,            # pinned (beta = 1, Gaussian emission)
}
PARAM_NAMES = list(TRUTH.keys())

# Free-parameter indices in PARAM_NAMES order (Model A: eight free params).
# All six poly-2 coefficients are free; only the four shape parameters
# (z1_skew, z1_log_tail, log_beta) plus the unused alpha0 / log_alpha1
# remain pinned. See specs/compute-simulation-ar1n.md "Model A".
FREE_INDICES = [2, 3, 4, 5, 6, 7, 8, 11]
# mu0, mu1, mu2, sigma0, sigma1, sigma2, z1_log_std, log_sigma_eps
FREE_NAMES   = [PARAM_NAMES[i] for i in FREE_INDICES]


# ----------------------------------------------------------------------
# Encoder + model builders
# ----------------------------------------------------------------------
ENCODER_TYPES = [
    'normal_diagonal',           # mean-field
    'joint_normal',              # joint normal
    'tridiag_joint_normal',      # tridiagonal-precision joint normal
    'conditional_markov',        # structured Markov
    'transformed_joint_normal',  # TJN
]


def build_encoder(encoder_type, T, regularize=1e-3, hidden_dim=32):
    """Build an encoder of the requested family.

    Matches `softplus_sinh_posterior_comparison.build_encoder` modulo
    the TJN branch (instantiates the class directly because
    `FlowConfig` does not currently expose `hidden`).
    """
    if encoder_type in ('joint_normal', 'normal_diagonal',
                          'tridiag_joint_normal'):
        return JointNormalConfig(dim=T, type=encoder_type,
                                  regularize=regularize,
                                  hidden_dim=hidden_dim).build()
    if encoder_type == 'conditional_markov':
        return MarkovConfig(dim=T, type=encoder_type,
                              regularize=regularize,
                              hidden_dim=hidden_dim).build()
    if encoder_type == 'transformed_joint_normal':
        return TransformedJointNormalPosterior(
            dim=T, hidden=hidden_dim, regularize=regularize)
    raise ValueError(f"unknown encoder_type: {encoder_type}")


def build_prior_decoder(T, device):
    """Build the Model-A inference prior + Gaussian decoder.

    Per spec `specs/compute-simulation-ar1n.md` Model A: the full
    poly-2 prior is the inference class. All six polynomial
    coefficients (mu_0, mu_1, mu_2, sigma_0, sigma_1, sigma_2) are
    FREE during estimation. The four "extra" coords
    (mu_0, mu_2, sigma_1, sigma_2) happen to equal 0 at the AR(1)
    truth, but they drift freely from a zero init.

    `beta_init=1.0, fix_beta=True` pins the emission's tailweight at 1
    (Gaussian emission). `theta=0.0, fix_theta=True` drops MA(1).
    `log_sigma_eps` is the only trainable decoder param.
    `z1_distr='normal'` removes z1_skew / z1_log_tail entirely (no
    tensors to pin on the prior side).
    """
    prior = MarkovNormalConditionalPolyPrior(
        nt=T, poly_degree=2, law_model='poly',
        z1_distr='normal', extra_heterogeneity=False).to(device)
    decoder = MASinhEmission(
        sigma_eps=float(np.exp(LOG_SIGMA_EPS_DEFAULT)),
        theta=0.0, fix_theta=True,
        sigma_floor=0.0, beta_init=1.0, fix_beta=True).to(device)
    return prior, decoder


def _apply_pinning_masks(model, device):
    """Pin the four shape parameters that stay at truth for Model A:
    log_beta = 0, theta_MA = 0, z1_skew = 0, z1_log_tail = 0.

    Per spec `specs/compute-simulation-ar1n.md` Model A, all six
    polynomial coefficients (mu_0, mu_1, mu_2, sigma_0, sigma_1,
    sigma_2) are FREE. The four "extra" coords
    (mu_0, mu_2, sigma_1, sigma_2) are initialised to truth (= 0) but
    NOT frozen --- they receive un-masked gradients and may drift.

    Most of the truth-side pinning is wired in at construction time:
      - `MASinhEmission(theta=0, fix_theta=True, beta_init=1.0,
         fix_beta=True)` -> decoder.log_beta, decoder.theta both have
         `requires_grad=False` and value 0.
      - `MarkovNormalConditionalPolyPrior(z1_distr='normal')` does
         not register `z1_skew` / `z1_log_tail` tensors at all.

    This function is therefore (a) defensive: it asserts that the
    four shape parameters above are at their truth values and not
    trainable, and (b) the public hook the callers reach for after
    `model.to(device)`. Idempotent. Argument `device` is kept for API
    stability (no longer needed since no mask tensors are allocated).
    """
    del device  # no longer needed; kept for API stability

    # Decoder shape pins (must be zero and non-trainable).
    if hasattr(model.decoder, 'log_beta'):
        assert not model.decoder.log_beta.requires_grad, \
            "Model A requires fix_beta=True (decoder.log_beta frozen)"
        assert float(model.decoder.log_beta.item()) == 0.0, \
            "Model A requires log_beta = 0 (Gaussian emission)"
    if hasattr(model.decoder, 'theta'):
        assert not model.decoder.theta.requires_grad, \
            "Model A requires fix_theta=True (decoder.theta frozen)"
        assert float(model.decoder.theta.item()) == 0.0, \
            "Model A requires theta_MA = 0 (no MA(1))"

    # Prior z_1 shape pins: under z1_distr='normal' these tensors
    # don't exist; under 'sinh' they do and must be 0 + frozen.
    if hasattr(model.prior, 'z1_skew'):
        with torch.no_grad():
            model.prior.z1_skew.fill_(0.0)
        model.prior.z1_skew.requires_grad_(False)
    if hasattr(model.prior, 'z1_log_tail'):
        with torch.no_grad():
            model.prior.z1_log_tail.fill_(0.0)
        model.prior.z1_log_tail.requires_grad_(False)

    # NOTE: the six poly coefficients (mu_0..mu_2, sigma_0..sigma_2)
    # are intentionally left ALONE here -- they are all free under
    # Model A. Earlier versions of this helper pinned mu_0, mu_2,
    # sigma_1, sigma_2 via gradient hooks; that pinning has been
    # removed to match the spec.


def build_full_model(encoder_type, T, device):
    encoder = build_encoder(encoder_type, T).to(device)
    prior, decoder = build_prior_decoder(T, device)
    model = FullModel(encoder=encoder, decoder=decoder, prior=prior).to(device)
    _apply_pinning_masks(model, device)
    return model


# ----------------------------------------------------------------------
# Parameter <-> model dict
# ----------------------------------------------------------------------
# Helpers that are robust to (a) `law_model='poly'` (no `b`, `log_gamma`)
# and (b) `z1_distr='normal'` (no `z1_skew`, `z1_log_tail`; the scale
# parameter is named `log_std` rather than `z1_log_std`).

def set_prior_decoder_to_truth(model, truth):
    with torch.no_grad():
        # mu / sigma poly coeffs (pinned entries written too; the mask
        # holds them at zero on the grad side).
        model.prior.net_mu.coeffs.copy_(torch.tensor(
            [truth['mu0'], truth['mu1'], truth['mu2']],
            dtype=model.prior.net_mu.coeffs.dtype,
            device=model.prior.net_mu.coeffs.device))
        model.prior.net_sigma.coeffs.copy_(torch.tensor(
            [truth['sigma0'], truth['sigma1'], truth['sigma2']],
            dtype=model.prior.net_sigma.coeffs.dtype,
            device=model.prior.net_sigma.coeffs.device))
        # z_1: 'normal' family uses `log_std`.
        if hasattr(model.prior, 'log_std'):
            model.prior.log_std.fill_(truth['z1_log_std'])
        elif hasattr(model.prior, 'z1_log_std'):
            # 'sinh' family; not used by AR1N but kept for safety.
            model.prior.z1_log_std.fill_(truth['z1_log_std'])
            if hasattr(model.prior, 'z1_skew'):
                model.prior.z1_skew.fill_(truth.get('z1_skew', 0.0))
            if hasattr(model.prior, 'z1_log_tail'):
                model.prior.z1_log_tail.fill_(truth.get('z1_log_tail', 0.0))
        # Decoder.
        model.decoder.log_sigma.fill_(truth['log_sigma_eps'])
        # `log_beta` is non-trainable (fix_beta=True). copy_ requires a
        # tensor; this is a no-op once `log_beta == 0.0`.
        if abs(float(truth.get('log_beta', 0.0))) > 1e-9:
            # If the caller asks for a nonzero log_beta on a Gaussian
            # decoder, write it anyway (it's a leaf tensor); the
            # gradient is masked by `requires_grad=False`.
            model.decoder.log_beta.data.fill_(truth['log_beta'])


def extract_params(model) -> dict:
    p = dict(TRUTH)   # start from pinned defaults so all 13 keys exist
    p['mu0'] = float(model.prior.net_mu.coeffs[0].item())
    p['mu1'] = float(model.prior.net_mu.coeffs[1].item())
    p['mu2'] = float(model.prior.net_mu.coeffs[2].item())
    p['sigma0'] = float(model.prior.net_sigma.coeffs[0].item())
    p['sigma1'] = float(model.prior.net_sigma.coeffs[1].item())
    p['sigma2'] = float(model.prior.net_sigma.coeffs[2].item())
    if hasattr(model.prior, 'log_std'):
        p['z1_log_std'] = float(model.prior.log_std.item())
    elif hasattr(model.prior, 'z1_log_std'):
        p['z1_log_std'] = float(model.prior.z1_log_std.item())
    if hasattr(model.prior, 'z1_skew'):
        p['z1_skew'] = float(model.prior.z1_skew.item())
    if hasattr(model.prior, 'z1_log_tail'):
        p['z1_log_tail'] = float(model.prior.z1_log_tail.item())
    p['log_sigma_eps'] = float(model.decoder.log_sigma.item())
    p['log_beta']      = float(model.decoder.log_beta.item())
    # alpha0 / log_alpha1 are not present on `law_model='poly'`; keep
    # the TRUTH defaults (zeros).
    return p


# ----------------------------------------------------------------------
# Simulation (differentiable, CRN-friendly)
# ----------------------------------------------------------------------
def reparam_simulate(theta, z_noise, y_noise):
    r"""Differentiable simulate y(theta, z_noise, y_noise) for the
    AR(1)+Normal class.

    theta: (13,) tensor in PARAM_NAMES order. Only the 4 free entries
      mu1 (idx 3), sigma0 (idx 5), z1_log_std (idx 8), log_sigma_eps
      (idx 11) and the pinned mu0/mu2/sigma1/sigma2/z1_skew/z1_log_tail/log_beta
      are read; the alpha0 / log_alpha1 entries are ignored (the
      law-of-motion has no softplus head here).

    z_noise, y_noise: (N, T) iid N(0, 1).

    Mirrors `MarkovNormalConditionalPolyPrior(law_model='poly',
    z1_distr='normal')` + `MASinhEmission(theta=0, fix_beta=True)`.
    """
    mu0           = theta[2]
    mu1           = theta[3]
    mu2           = theta[4]
    sigma0        = theta[5]
    sigma1        = theta[6]
    sigma2        = theta[7]
    z1_log_std    = theta[8]
    log_sigma_eps = theta[11]

    z1_scale  = F.softplus(z1_log_std)
    sigma_eps = torch.exp(log_sigma_eps)

    N, T = z_noise.shape
    z = torch.zeros(N, T, device=z_noise.device, dtype=z_noise.dtype)
    # z_1 ~ N(0, z1_scale)
    z[:, 0] = z1_scale * z_noise[:, 0]
    for t in range(1, T):
        zlag = z[:, t-1].clone()
        mu_t = mu0 + mu1 * zlag + mu2 * zlag ** 2
        mu_t = mu_t.clamp(-10.0, 10.0)
        poly_sig = sigma0 + sigma1 * zlag + sigma2 * zlag ** 2
        sig_t = F.softplus(poly_sig).clamp(0.0, 10.0) + 1e-3
        z[:, t] = mu_t + sig_t * z_noise[:, t]
    # Gaussian emission: y = z + sigma_eps * y_noise. Equivalent to
    # MASinhEmission(beta=1, theta=0): sinh(asinh(y_noise) * 1) = y_noise.
    eps = sigma_eps * y_noise
    return z + eps


def simulate_from_truth(N, T, truth, seed, device):
    torch.manual_seed(seed)
    z_noise = torch.randn(N, T, device=device)
    y_noise = torch.randn(N, T, device=device)
    theta = torch.tensor([truth[k] for k in PARAM_NAMES], device=device,
                          dtype=torch.float32)
    with torch.no_grad():
        y = reparam_simulate(theta, z_noise, y_noise)
    return y


# ----------------------------------------------------------------------
# Closed-form marginal log-likelihood
# ----------------------------------------------------------------------
def _as_tensor(v, device, dtype=torch.float64):
    """Coerce a Python float or torch scalar tensor to a 1-element tensor
    on `device` with `dtype`."""
    if isinstance(v, torch.Tensor):
        return v.detach().to(device=device, dtype=dtype)
    return torch.tensor(float(v), device=device, dtype=dtype)


def ar1n_marginal_logp(y, mu1, sigma0, z1_log_std, log_sigma_eps):
    r"""Exact marginal log-likelihood per individual for the
    AR(1) + normal-z_1 + Gaussian-emission DGP, averaged over N rows.

    The DGP is linear-Gaussian:
      z_1 ~ N(0, sigma_z1^2)      with  sigma_z1  = softplus(z1_log_std)
      z_t = mu_1 z_{t-1} + sigma_eta eta_t,
                                    sigma_eta = softplus(sigma_0),
                                    eta_t ~ N(0, 1)
      y_t = z_t + sigma_eps eps_t,  sigma_eps = exp(log_sigma_eps),
                                    eps_t ~ N(0, 1)
    so y_{1:T} ~ N(0, Sigma(theta)) with
      V_1 = sigma_z1^2,  V_t = mu_1^2 V_{t-1} + sigma_eta^2,
      Sigma_{ts} = mu_1^{|t - s|} V_{min(t, s)} + sigma_eps^2 1{t == s}.

    Args:
      y:               (N, T) torch tensor of observations.
      mu1:             scalar (torch tensor or Python float).
      sigma0:          scalar (torch tensor or Python float).
      z1_log_std:      scalar (torch tensor or Python float).
      log_sigma_eps:   scalar (torch tensor or Python float).

    Returns:
      Python float: mean over the N rows of log p(y_i | theta), where
      p is the exact marginal density on R^T. Computation is done in
      float64 (well-conditioned at T = 6) and the .item() is returned
      on CPU. Stateless and noise-free (zero variance across calls).
    """
    device = y.device
    mu1_t   = _as_tensor(mu1,           device)
    sig0_t  = _as_tensor(sigma0,        device)
    z1lst_t = _as_tensor(z1_log_std,    device)
    lse_t   = _as_tensor(log_sigma_eps, device)
    y64 = y.to(dtype=torch.float64)

    N, T = y64.shape
    sigma_z1  = F.softplus(z1lst_t)
    sigma_eta = F.softplus(sig0_t)
    sigma_eps = torch.exp(lse_t)
    V1   = sigma_z1 ** 2
    V_eta = sigma_eta ** 2
    V_eps = sigma_eps ** 2

    # State variances V_t.
    Vs = [V1]
    for _t in range(1, T):
        Vs.append(mu1_t ** 2 * Vs[-1] + V_eta)
    Vs_t = torch.stack(Vs)               # (T,)

    # Sigma_{ts} = mu_1^{|t - s|} * V_{min(t, s)} + V_eps * 1{t == s}
    idx = torch.arange(T, device=device, dtype=torch.float64)
    diff = (idx.view(T, 1) - idx.view(1, T)).abs()       # (T, T)
    min_idx = torch.minimum(idx.view(T, 1), idx.view(1, T)).long()
    Sigma = (mu1_t ** diff) * Vs_t[min_idx]
    eye = torch.eye(T, device=device, dtype=torch.float64)
    Sigma = Sigma + V_eps * eye
    # Small jitter for safety (well below the leading eigenvalues at T=6).
    Sigma = Sigma + 1e-12 * eye

    L = torch.linalg.cholesky(Sigma)
    log_det = 2.0 * torch.log(torch.diagonal(L)).sum()
    # Solve L z = y^T, with y^T shape (T, N).
    z = torch.linalg.solve_triangular(L, y64.transpose(0, 1), upper=False)
    sq_per_i = (z ** 2).sum(dim=0)        # (N,)

    log_2pi = float(np.log(2.0 * np.pi))
    logp_per_i = -0.5 * T * log_2pi - 0.5 * log_det - 0.5 * sq_per_i
    return float(logp_per_i.mean().item())


def poly2_chain_logp(y, mu0, mu1, mu2, sigma0, sigma1, sigma2, z1_log_std):
    r"""Per-individual log-likelihood under Model B (no measurement error).

    Spec: `specs/compute-simulation-ar1n.md` "Model B" --- the
    degenerate emission y_t = z_t, sigma_eps = 0. The poly-2 prior is
    arbitrary (NOT restricted to AR(1)); each conditional density is
    closed-form Gaussian, so

      log p(y_{1:T}; theta)
        = log N(y_1; 0, sigma_{z_1}^2)
          + sum_{t=2}^{T} log N(y_t; mu(y_{t-1}), sigma(y_{t-1})^2),

    where
      mu(z)    = mu0 + mu1 * z + mu2 * z**2
      sigma(z) = softplus(sigma0 + sigma1 * z + sigma2 * z**2)
      sigma_{z_1} = softplus(z1_log_std).

    Args:
      y:           (N, T) tensor of observations, treated as latent
                   states under Model B.
      mu0..mu2:    poly-2 mean coefficients (scalars; tensor or float).
      sigma0..sigma2: poly-2 raw scale coefficients (pre-softplus;
                   tensor or float).
      z1_log_std:  raw initial-state scale (pre-softplus; tensor or
                   float).

    Returns:
      (N,) tensor of per-individual log-densities. Autograd-compatible
      end-to-end if the scalar arguments are leaf tensors with
      `requires_grad=True`. Caller averages over N.
    """
    device = y.device
    dtype  = y.dtype

    def _scalar(v):
        if isinstance(v, torch.Tensor):
            return v.to(device=device, dtype=dtype)
        return torch.tensor(float(v), device=device, dtype=dtype)

    mu0_t  = _scalar(mu0)
    mu1_t  = _scalar(mu1)
    mu2_t  = _scalar(mu2)
    sig0_t = _scalar(sigma0)
    sig1_t = _scalar(sigma1)
    sig2_t = _scalar(sigma2)
    z1lst_t = _scalar(z1_log_std)

    log_2pi = float(np.log(2.0 * np.pi))

    # Term 1: log N(y_1; 0, sigma_z1^2).
    sigma_z1 = F.softplus(z1lst_t)
    y1 = y[:, 0]
    log_sigma_z1 = torch.log(sigma_z1)
    term1 = -0.5 * log_2pi - log_sigma_z1 - 0.5 * (y1 / sigma_z1) ** 2

    # Chain terms: log N(y_t; mu(y_{t-1}), sigma(y_{t-1})^2) for t=2..T.
    if y.shape[1] > 1:
        ylag  = y[:, :-1]                  # (N, T-1)
        ynext = y[:, 1:]                   # (N, T-1)
        mu_t = mu0_t + mu1_t * ylag + mu2_t * ylag ** 2
        sig_t = F.softplus(sig0_t + sig1_t * ylag + sig2_t * ylag ** 2)
        log_sig_t = torch.log(sig_t)
        chain = -0.5 * log_2pi - log_sig_t \
                - 0.5 * ((ynext - mu_t) / sig_t) ** 2
        chain_sum = chain.sum(dim=1)        # (N,)
    else:
        chain_sum = torch.zeros_like(term1)

    return term1 + chain_sum


# ----------------------------------------------------------------------
# NaN-retry helpers (lifted verbatim from softplus_sinh_posterior_comparison)
# ----------------------------------------------------------------------
class NaNFailure(RuntimeError):
    pass


def _retry_seeds(base_seed, n_retries):
    """Stable offsets 0, 1, 2, ..., n_retries-1 (deterministic)."""
    return [base_seed + i for i in range(n_retries)]
