"""SMC forward-backward smoothers shared across the project's fit scripts.

This module is the canonical home for the FFBS posterior sampler used by
SMC-EM E-steps. It exists because `scripts/climb_per_step.smc_ffbs_sample`
is, historically, a sequential-importance sampler that does *not* resample
between forward propagation steps — at long T (e.g. T=40 in the longt /
ar1n-smc-em-K-sweep specs) the IS weights collapse exponentially and the
backward FFBS draw is dominated by a handful of effective trajectories. The
result is a smoothed-score bias that does **not** decay with K (empirically
∝ K^(-0.2) instead of the predicted K^(-1)).

The function below adds a per-step systematic resample inside the forward
loop. It is otherwise drop-in compatible with `climb_per_step.smc_ffbs_sample`:
same signature `(prior, decoder, y, K) -> (N, T) tensor` and same backward
FFBS sweep. With resampling enabled the bias decays as K^(-1) until it hits
the M-step stochastic-gradient floor (see specs/compute-simulation-ar1n_smc_em_K_sweep.md
"Wall-time estimate" + the journal entry for the AR(1)n K-sweep).

Two entry points:
- `smc_ffbs_sample_resampled(prior, decoder, y, K)`: systematic resample at
  every forward step.
- `select_smc_smoother(resample: str)`: returns the right sampler given a
  string config — `"systematic"` (default) or `"none"` (the legacy SIS path,
  kept for backward compat and the published K-sweep baseline).
"""
from __future__ import annotations

import numpy as np
import torch

# Re-export the legacy SIS smoother so callers can switch via
# `select_smc_smoother()` without two import lines.
from climb_per_step import smc_ffbs_sample as smc_ffbs_sample_no_resample


__all__ = [
    "smc_ffbs_sample_resampled",
    "smc_ffbs_sample_no_resample",
    "smc_ffbs_sample_genealogy",
    "select_smc_smoother",
]


@torch.no_grad()
def _systematic_resample_per_row(w: torch.Tensor) -> torch.Tensor:
    """Per-row systematic resampling of a (N, K) weight tensor.

    Returns (N, K) ancestor indices. One u01 draw per individual.
    """
    N, K = w.shape
    device = w.device
    u01 = torch.rand(N, 1, device=device)
    grid = torch.arange(K, device=device, dtype=w.dtype).unsqueeze(0)
    u = (u01 + grid) / K
    cdf = torch.cumsum(w, dim=1)
    cdf = cdf / cdf[:, -1:].clamp_min(1e-30)
    idx = torch.searchsorted(cdf, u, right=False).clamp(max=K - 1)
    return idx


@torch.no_grad()
def smc_ffbs_sample_resampled(prior, decoder, y, K):
    """SMC FFBS smoother with systematic resampling at every forward step.

    Forward pass:
      - t = 0: draw K particles per individual from p(z_1), compute local
        likelihood w_0 = p(y_1 | z_1), store both.
      - For t = 1..T-1: systematic-resample using the previous step's
        local likelihood (so post-resample particles are uniformly weighted),
        propagate via the prior transition, then weight by local likelihood
        and store both.

    Backward pass: standard FFBS — sample the terminal particle with prob
    ∝ w_{T-1}^k, then walk backward picking z_t^j with prob ∝ w_t^j ·
    p(z_{t+1} | z_t^j). Because the forward pass resampled before each
    propagation, `log_w_hist[t]` stores the **local** per-step weight (not
    cumulative). That is the correct filter weight for the backward kernel.

    Args:
        prior, decoder: model components (no encoder needed). The prior
            must expose `get_eta0()` and `get_mu_sigma(z)`. The decoder
            must expose `get_distribution()`.
        y: (N, T) observations.
        K: number of particles per individual.

    Returns: (N, T) smoothed sample.
    """
    N, T = y.shape
    device = y.device

    z1_dist = prior.get_eta0()
    z = z1_dist.sample((N * K,)).view(N, K).to(device)
    dist_e = decoder.get_distribution()

    log_lik = dist_e.log_prob((y[:, 0:1] - z).unsqueeze(-1)).squeeze(-1)
    if log_lik.dim() == 3:
        log_lik = log_lik.squeeze(-1)

    z_hist = [z.clone()]
    log_w_hist = [log_lik.clone()]                            # local likelihood at t=0

    for t in range(1, T):
        # Resample using the previous step's local likelihood.
        prev_logw = log_w_hist[-1]
        prev_logw_norm = prev_logw - torch.logsumexp(prev_logw, dim=1, keepdim=True)
        w_prev = torch.exp(prev_logw_norm).clamp_min(1e-30)
        idx = _systematic_resample_per_row(w_prev)
        z = z.gather(1, idx)

        # Propagate.
        mu, sigma = prior.get_mu_sigma(z.reshape(-1, 1))
        z = (mu + sigma * torch.randn_like(mu)).view(N, K)

        # Local weight for time t.
        log_lik = dist_e.log_prob((y[:, t:t+1] - z).unsqueeze(-1)).squeeze(-1)
        if log_lik.dim() == 3:
            log_lik = log_lik.squeeze(-1)

        z_hist.append(z.clone())
        log_w_hist.append(log_lik.clone())

    # Backward FFBS sweep — same formula as the legacy smoother, but with
    # log_w_hist storing per-step (not cumulative) likelihoods because of
    # the forward resampling.
    final_logw = log_w_hist[-1] - torch.logsumexp(log_w_hist[-1], dim=1, keepdim=True)
    final_w = torch.exp(final_logw).clamp_min(1e-30)
    idx_T = torch.multinomial(final_w, num_samples=1).squeeze(-1)
    smoothed = [None] * T
    smoothed[T - 1] = z_hist[T - 1].gather(1, idx_T.unsqueeze(1)).squeeze(1)
    for t in range(T - 2, -1, -1):
        z_t = z_hist[t]
        log_w_filt = log_w_hist[t] - torch.logsumexp(log_w_hist[t], dim=1, keepdim=True)
        z_next = smoothed[t + 1].unsqueeze(1)
        mu, sigma = prior.get_mu_sigma(z_t.reshape(-1, 1))
        mu = mu.view(N, K); sigma = sigma.view(N, K)
        log_trans = (-0.5 * ((z_next - mu) / sigma) ** 2
                     - torch.log(sigma) - 0.5 * float(np.log(2 * np.pi)))
        log_back = log_w_filt + log_trans
        log_back = log_back - torch.logsumexp(log_back, dim=1, keepdim=True)
        probs = torch.exp(log_back).clamp_min(1e-30)
        idx = torch.multinomial(probs, num_samples=1).squeeze(-1)
        smoothed[t] = z_t.gather(1, idx.unsqueeze(1)).squeeze(1)
    return torch.stack(smoothed, dim=1)


@torch.no_grad()
def smc_ffbs_sample_genealogy(prior, decoder, y, K):
    """SMC smoother with systematic-resampled forward + **ancestry-tracking**
    backward pass (i.e. the "genealogy" convention of specs/estimators.md
    §1.2).

    Forward pass is identical to `smc_ffbs_sample_resampled`. The
    difference is entirely in the backward pass:

      - `smc_ffbs_sample_resampled` (proper FFBS) draws z_t^(s)
        multinomially from all K candidates with weights
        ∝ w_t^k * p(z_{t+1}^(s) | z_t^k).
      - This function walks the resampling ancestry: at each backward
        step, z_t^(s) is fixed to `z_hist[t][ancestry[t+1][k_current]]`,
        i.e. the direct ancestor of the tracked terminal particle.
        No transition-density re-weighting, no candidate mixing.

    Both convention share the same forward pass and the same terminal-
    particle draw. So this is a controlled A/B against
    `smc_ffbs_sample_resampled` on the smoother backward-choice axis
    alone — matching the FFBS-vs-genealogy scaling contrast that
    `scripts/smc_smoothing_scaling.py` measured on the LGSSM ridge
    (slope 1 for FFBS, slope 2 for genealogy on the smoothed-additive-
    functional variance).

    Returns: (N, T) smoothed sample.
    """
    N, T = y.shape
    device = y.device

    z1_dist = prior.get_eta0()
    z = z1_dist.sample((N * K,)).view(N, K).to(device)
    dist_e = decoder.get_distribution()

    log_lik = dist_e.log_prob((y[:, 0:1] - z).unsqueeze(-1)).squeeze(-1)
    if log_lik.dim() == 3:
        log_lik = log_lik.squeeze(-1)

    z_hist = [z.clone()]
    log_w_hist = [log_lik.clone()]
    # ancestry[t] : (N, K) tensor of indices into z_hist[t-1]. Defined for
    # t = 1..T-1 (there's no resample before t=0). ancestry[t][:, k] gives
    # the parent index at time t-1 that produced particle k at time t.
    ancestry = [None]                                     # placeholder for t=0

    for t in range(1, T):
        # Systematic resample using the previous step's local likelihood.
        prev_logw = log_w_hist[-1]
        prev_logw_norm = prev_logw - torch.logsumexp(prev_logw, dim=1, keepdim=True)
        w_prev = torch.exp(prev_logw_norm).clamp_min(1e-30)
        idx = _systematic_resample_per_row(w_prev)          # (N, K)
        ancestry.append(idx.clone())                        # save the ancestry link
        z = z.gather(1, idx)

        # Propagate.
        mu, sigma = prior.get_mu_sigma(z.reshape(-1, 1))
        z = (mu + sigma * torch.randn_like(mu)).view(N, K)

        # Local weight for time t.
        log_lik = dist_e.log_prob((y[:, t:t+1] - z).unsqueeze(-1)).squeeze(-1)
        if log_lik.dim() == 3:
            log_lik = log_lik.squeeze(-1)

        z_hist.append(z.clone())
        log_w_hist.append(log_lik.clone())

    # Backward: walk the resampling ancestry back from the terminal particle.
    final_logw = log_w_hist[-1] - torch.logsumexp(log_w_hist[-1], dim=1, keepdim=True)
    final_w = torch.exp(final_logw).clamp_min(1e-30)
    idx_T = torch.multinomial(final_w, num_samples=1).squeeze(-1)  # (N,)

    smoothed = [None] * T
    smoothed[T - 1] = z_hist[T - 1].gather(1, idx_T.unsqueeze(1)).squeeze(1)
    k_cur = idx_T
    for t in range(T - 2, -1, -1):
        # Ancestor at time t of the particle currently tracked at time t+1.
        k_cur = ancestry[t + 1].gather(1, k_cur.unsqueeze(1)).squeeze(1)
        smoothed[t] = z_hist[t].gather(1, k_cur.unsqueeze(1)).squeeze(1)
    return torch.stack(smoothed, dim=1)


def select_smc_smoother(resample: str):
    """Pick a smoother by config string.

    `"systematic"` -> `smc_ffbs_sample_resampled` (proper SMC + FFBS
                       backward — the contract default).
    `"genealogy"`  -> `smc_ffbs_sample_genealogy` (proper SMC forward,
                       ancestry-tracking backward — the T²/K variance
                       comparator; see specs/estimators.md §1.2).
    `"none"`       -> `smc_ffbs_sample_no_resample` (the legacy SIS path
                       imported from `climb_per_step`).
    """
    if resample == "systematic":
        return smc_ffbs_sample_resampled
    if resample == "genealogy":
        return smc_ffbs_sample_genealogy
    if resample == "none":
        return smc_ffbs_sample_no_resample
    raise ValueError(
        f"unknown RESAMPLE={resample!r}; expected one of "
        f"'systematic' | 'genealogy' | 'none'"
    )
