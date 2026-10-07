"""Filtering Variational Objective (FIVO) — differentiable bootstrap particle
filter for use as a training objective.

Background
----------
For a state-space model $z_t \mid z_{t-1} \sim p(z_t \mid z_{t-1})$ and
$y_t \mid z_t \sim p(y_t \mid z_t)$, the IWAE bound with K iid trajectories
suffers from compounding weight variance: at large T even K=100 leaves a
non-trivial gap to $\log p(y)$ (we observe 0.18 nats at T=40 in the joint-VI
baseline). FIVO (Maddison, Lawson, Tucker, Mnih 2017; Naesseth, Linderman,
Ranganath, Blei 2018) replaces iid trajectories with K particles run through
a particle filter: at each step, particles are propagated, reweighted by the
local likelihood, and resampled. Resampling re-concentrates particles on
configurations consistent with $y_{1:t}$, so the per-step weight variance
stays bounded as T grows. The objective

    $\log \hat p_{\mathrm{SMC}}(y) = \sum_{t=1}^{T} \log \tfrac{1}{K}
    \sum_{k=1}^{K} w_t^k$

is a tighter lower bound on $\log p(y)$ than IWAE_K for any K, and converges
to $\log p(y)$ as $K \to \infty$.

Bootstrap proposal
------------------
This implementation uses the prior as the proposal:
$q(z_t \mid z_{t-1}) = p(z_t \mid z_{t-1})$. The transition density and the
proposal density cancel in the importance weights, so the per-step weight
reduces to $w_t^k = p(y_t \mid z_t^k)$. **No encoder is needed.** The
trade-off vs an auxiliary-encoder FIVO is that the proposal is data-blind,
so for very informative observations a sequential encoder would do better.

Differentiability through resampling
------------------------------------
Multinomial resampling is non-differentiable. We use the standard
biased-gradient trick (Maddison et al., 2017): stop_grad on the sampled
indices, but allow autograd through `torch.gather`, so gradients flow back
to the surviving particles. The gradient of the resampling step itself is
ignored (= zero). The bias is generally small in practice for SSMs and is
the price of a clean, low-variance gradient.
"""
from __future__ import annotations

import math
import torch


class FilteringVariationalObjective:
    """Bootstrap FIVO. `model` must expose `.prior` (with `get_eta0()` and
    `get_mu_sigma(z)`) and `.decoder` (with `get_distribution()`)."""

    def __init__(self, model, K: int = 100):
        self.model = model
        self.K = K

    def fivo_bound(self, y: torch.Tensor) -> torch.Tensor:
        r"""Per-individual FIVO bound. y: (N, T) → (N,) tensor of
        $\log\hat p_{\mathrm{SMC}}(y_i)$ estimates with full autograd."""
        N, T = y.shape
        K = self.K
        device = y.device
        log_K = math.log(K)

        # ---- t = 1 -----------------------------------------------------
        z = self._draw_z1(N, K, device)                       # (N, K)
        log_w = self._log_lik_yt(y[:, 0:1], z)                # (N, K)
        log_p = torch.logsumexp(log_w, dim=1) - log_K         # (N,)

        # ---- t = 2..T --------------------------------------------------
        for t in range(1, T):
            z = self._resample(log_w, z)                      # (N, K), uniform weights now
            z = self._propagate(z)                            # (N, K)
            log_w = self._log_lik_yt(y[:, t:t+1], z)          # (N, K)
            log_p = log_p + (torch.logsumexp(log_w, dim=1) - log_K)

        return log_p

    # ----- helpers --------------------------------------------------------

    def _draw_z1(self, N: int, K: int, device) -> torch.Tensor:
        r"""Reparameterised draw from $p(z_1)$, shape (N, K)."""
        z1_dist = self.model.prior.get_eta0()
        samples = z1_dist.rsample((N * K,))
        if samples.dim() == 2:
            samples = samples.squeeze(-1)
        return samples.view(N, K).to(device)

    def _propagate(self, z_prev: torch.Tensor) -> torch.Tensor:
        r"""$z_t \sim p(z_t \mid z_{t-1})$, reparameterised. (N, K) → (N, K).
        Uses prior.transition_sample so non-Gaussian transitions (e.g.
        sinh-arcsinh) work out of the box."""
        N, K = z_prev.shape
        z_flat = z_prev.reshape(-1, 1)
        z_new = self.model.prior.transition_sample(z_flat)
        return z_new.view(N, K)

    def _log_lik_yt(self, y_t: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        r"""$\log p(y_t \mid z_t)$ per (i, k)."""
        if y_t.dim() == 1:
            y_t = y_t.unsqueeze(-1)
        residuals = y_t - z_t
        dist = self.model.decoder.get_distribution()
        log_p = dist.log_prob(residuals)
        if log_p.dim() == 3:
            log_p = log_p.squeeze(-1)
        return log_p

    def _resample(self, log_w: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Biased-gradient multinomial resampling: stop_grad on indices,
        autograd flows through `gather` to the surviving particles."""
        # Indices are sampled under no_grad — they are categorical draws and
        # carry no path-derivative anyway.
        with torch.no_grad():
            log_w_norm = log_w - torch.logsumexp(log_w, dim=1, keepdim=True)
            w = torch.exp(log_w_norm)
            bad = (w.sum(dim=1) <= 0)
            if bad.any():
                w = w.clone()
                w[bad] = 1.0 / w.shape[1]
            idx = torch.multinomial(w, num_samples=self.K, replacement=True)
        # gather IS differentiable in `z`.
        return torch.gather(z, dim=1, index=idx)
