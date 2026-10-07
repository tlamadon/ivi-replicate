"""Bootstrap particle filter for log-marginal-likelihood evaluation.

For our state-space model
    z_t | z_{t-1} ~ p_prior(z_t | z_{t-1})        (Markov; given by prior.get_mu_sigma)
    y_t | z_t     ~ p_decoder(y_t | z_t)          (per-period; given by decoder)
the marginal log-likelihood is

    log p(y_{1:T}) = log ∫ p(z_1) Π_t p(z_t | z_{t-1}) p(y_t | z_t) dz_{1:T}

A bootstrap particle filter (SIR with prior as proposal) gives a Monte-Carlo
estimate that is unbiased in the *marginal* p(y) (and slightly biased downward
in the *log* of it), with bias going to zero as K → ∞.

The advantage over IWAE here: SMC builds a local proposal at each timestep
(propagating particles through the prior, then reweighting by the local
likelihood), so heavy-tailed observations don't require a global Q to track
the whole posterior. Q-quality concerns disappear.

Currently assumes:
- Prior exposes .get_eta0() (z_1 distribution) and .get_mu_sigma(z) for transitions.
- Decoder exposes .get_distribution() returning a torch.distributions object that
  evaluates p(y_t - z_t) (i.e., a centred residual density). True for sinh_ma,
  Normal, etc.; not currently true for ma_normal with theta!=0 (cross-period
  coupling — would require a different formulation).
"""
from __future__ import annotations

import math
import torch
from torch.distributions import Normal


class BootstrapParticleFilter:
    def __init__(self, model, K: int = 1000, resample_every_step: bool = True,
                 resample_method: str = "systematic"):
        """
        Args:
            model: object with .prior and .decoder (or a FullModel).
            K: number of particles per individual.
            resample_every_step: run the resample step after every filter
                update (the standard SIR schedule). Almost always True; the
                False branch is retained for diff studies against the
                pre-2026-06-30 SIS behaviour (see specs/estimators.md §1.1
                and journal/2026-06-30-smc-ffbs-resampling-fix.md).
            resample_method: "systematic" (default; low-variance, one uniform
                draw per individual, deterministic stratified inverse-CDF)
                or "multinomial" (per-torch.multinomial; higher variance,
                retained for backward-compat and audit reproduction of older
                results such as those under results/legacy/).
        """
        self.model = model
        self.K = K
        self.resample_every_step = resample_every_step
        if resample_method not in ("systematic", "multinomial"):
            raise ValueError(
                f"resample_method must be 'systematic' or 'multinomial'; "
                f"got {resample_method!r}")
        self.resample_method = resample_method

    @torch.no_grad()
    def log_marginal(self, y: torch.Tensor) -> torch.Tensor:
        """Return per-individual log p(y_{1:T}) estimate. y: (N, T) → (N,)."""
        N, T = y.shape
        K = self.K
        device = y.device
        log_K = math.log(K)

        # --- t = 1 ---
        z = self._draw_z1(N, K, device)                      # (N, K)
        log_w = self._log_lik_yt_given_zt(y[:, 0:1], z)      # (N, K)
        log_marg = torch.logsumexp(log_w, dim=1) - log_K     # (N,)
        if self.resample_every_step:
            z = self._resample(log_w, z)

        # --- t = 2..T ---
        for t in range(1, T):
            z = self._propagate(z)                            # (N, K)
            log_w = self._log_lik_yt_given_zt(y[:, t:t+1], z) # (N, K)
            log_marg = log_marg + torch.logsumexp(log_w, dim=1) - log_K
            # Skip resampling at the very last step — never used downstream.
            if self.resample_every_step and t < T - 1:
                z = self._resample(log_w, z)

        return log_marg

    # ----- helpers --------------------------------------------------------

    def _draw_z1(self, N: int, K: int, device) -> torch.Tensor:
        """Draw (N, K) particles from the marginal p(z_1)."""
        z1_dist = self.model.prior.get_eta0()
        samples = z1_dist.sample((N * K,))      # (N*K, ...)
        if samples.dim() == 2:
            samples = samples.squeeze(-1)
        return samples.view(N, K).to(device)

    def _propagate(self, z_prev: torch.Tensor) -> torch.Tensor:
        """Sample z_t ~ p(z_t | z_{t-1}) for each particle. (N,K) → (N,K).
        Uses prior.transition_sample so non-Gaussian transitions
        (e.g. sinh-arcsinh) work out of the box."""
        N, K = z_prev.shape
        z_flat = z_prev.reshape(-1, 1)
        z_new = self.model.prior.transition_sample(z_flat)
        return z_new.view(N, K)

    def _log_lik_yt_given_zt(self, y_t: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        """log p(y_t | z_t) per (i, k). y_t: (N,1) or (N,), z_t: (N,K)."""
        if y_t.dim() == 1:
            y_t = y_t.unsqueeze(-1)
        residuals = y_t - z_t                                 # (N, K)
        dist = self.model.decoder.get_distribution()
        log_p = dist.log_prob(residuals)
        # SinhArcsinh and similar may return shape (N, K, 1); squeeze.
        if log_p.dim() == 3:
            log_p = log_p.squeeze(-1)
        return log_p

    def _resample(self, log_w: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Per-individual resampling. Method selected by
        `self.resample_method`; see the class docstring."""
        # Normalise weights in log space, then exponentiate.
        log_w = log_w - torch.logsumexp(log_w, dim=1, keepdim=True)
        w = torch.exp(log_w)
        # Guard against numerical degeneracy: any-row-all-zero → uniform.
        bad = (w.sum(dim=1) <= 0)
        if bad.any():
            w = w.clone()
            w[bad] = 1.0 / w.shape[1]
        if self.resample_method == "systematic":
            idx = _systematic_resample_per_row(w)
        else:
            idx = torch.multinomial(w, num_samples=self.K, replacement=True)  # (N, K)
        return torch.gather(z, dim=1, index=idx)


@torch.no_grad()
def _systematic_resample_per_row(w: torch.Tensor) -> torch.Tensor:
    """Per-row systematic resampling of a (N, K) normalized weight tensor.

    Returns (N, K) ancestor indices. Draws one uniform u ~ U[0, 1/K) per
    individual, then walks the CDF at strides 1/K to pick ancestors —
    the standard stratified inverse-CDF sampling. This is bitwise
    equivalent to the resampling scheme used by
    `experiments/smc_smoothers.smc_ffbs_sample_resampled`, so the two SMC
    kernels in the project (this class for likelihood evaluation and the
    smoother for the SMC-EM E-step) share one resampling routine.
    """
    N, K = w.shape
    device = w.device
    u01 = torch.rand(N, 1, device=device)
    grid = torch.arange(K, device=device, dtype=w.dtype).unsqueeze(0)
    u = (u01 + grid) / K                                              # (N, K)
    cdf = torch.cumsum(w, dim=1)
    cdf = cdf / cdf[:, -1:].clamp_min(1e-30)                          # guard round-off on last col
    return torch.searchsorted(cdf, u, right=False).clamp(max=K - 1)
