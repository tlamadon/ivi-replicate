"""Laplace approximation encoder (non-amortised).

For each individual i, find the posterior mode

    ẑ_i^* = argmax_z [ log p(z) + log p(y_i | z) ]

via batched Newton's method, then approximate Q(z|y_i) ≈ N(ẑ_i^*, H_i^{-1})
where H_i = -∇²_z [log p(z) + log p(y_i|z)] |_{ẑ_i^*}.

Compared to the Kalman smoother encoder, the Hessian uses the *actual*
decoder log-likelihood (e.g., sinh-arcsinh for tails) rather than a
Gaussian-equivalent linearisation. For tail observations the heavy-tailed
log-density has lower curvature, so Σ = H^{-1} is wider, sampled residuals
y - z are larger, and the decoder gets gradient signal to learn β > 1.

Implementation:
- Per-individual mode cache as a buffer; refined via 1-3 Newton iterations
  (warm-started from cache) at each forward call.
- Hessians computed by T=dim_latent backward passes (cheap at T~6).
- Levenberg-Marquardt regularisation (small λI added to -H) ensures the
  step is well-defined even when -H is not strictly PSD.
- Mode and Σ are detached from θ's autograd graph: the M-step gradient
  on θ flows only through log p(z) and log p(y|z) at fixed z samples.
  This is EM proper (E-step takes θ as fixed).
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

from .base import Encoder


class LaplaceEncoder(Encoder):
    def __init__(
        self,
        dim: int,
        num_newton_steps: int = 3,
        lm_lambda: float = 1e-3,
        cache_modes: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.dim_latent = dim
        self.num_newton_steps = num_newton_steps
        self.lm_lambda = lm_lambda
        self.cache_modes = cache_modes

        # Refs to prior + decoder (set by FullModel via set_components()).
        # Wrap in lists so nn.Module doesn't try to register them as submodules.
        self._prior_ref = []
        self._decoder_ref = []

        # Mode cache. Initialised lazily on the first forward call.
        # Stored as a plain attribute (not a buffer) since size may not be
        # known at construction time.
        self._mode_cache = None  # (N, T)
        self.mean_var = torch.tensor(0.0)

    def set_components(self, prior, decoder):
        self._prior_ref = [prior]
        self._decoder_ref = [decoder]

    @property
    def prior(self):
        if not self._prior_ref:
            raise RuntimeError("LaplaceEncoder needs set_components(prior, decoder) before use")
        return self._prior_ref[0]

    @property
    def decoder(self):
        if not self._decoder_ref:
            raise RuntimeError("LaplaceEncoder needs set_components(prior, decoder) before use")
        return self._decoder_ref[0]

    def get_seed_dim(self) -> int:
        return self.dim_latent

    # ----- log-joint, gradient, Hessian -----------------------------------

    def _log_joint(self, y, z):
        """Per-sample log p(z) + log p(y|z). Returns (N,)."""
        log_pz = self.prior.log_prob(z)
        log_py_z = self.decoder.log_likelihood(y, z)
        return log_pz + log_py_z

    def _grad_only(self, y, z):
        """Compute g = ∇_z log_joint, batched per i."""
        z_req = z.detach().requires_grad_(True)
        with torch.enable_grad():
            log_joint = self._log_joint(y, z_req)
            g = torch.autograd.grad(log_joint.sum(), z_req,
                                    create_graph=False)[0]
        return g.detach()

    def _grad_and_hessian(self, y, z):
        """Compute g = ∇_z log_joint and H = ∇²_z log_joint, batched per i."""
        T = z.shape[1]
        z_req = z.detach().requires_grad_(True)
        with torch.enable_grad():
            log_joint = self._log_joint(y, z_req)
            g = torch.autograd.grad(log_joint.sum(), z_req,
                                    create_graph=True)[0]
            H_cols = []
            for j in range(T):
                H_j = torch.autograd.grad(
                    g[:, j].sum(), z_req,
                    retain_graph=(j < T - 1),
                )[0]
                H_cols.append(H_j)
            H = torch.stack(H_cols, dim=2)
        return g.detach(), H.detach()

    def _log_joint_no_grad(self, y, z):
        with torch.no_grad():
            return self._log_joint(y, z)

    @staticmethod
    def _clip_eig(M, eig_min: float = 1e-3, eig_max: float = 1e4):
        """Per-batch eigenvalue clipping, returning a well-conditioned PSD M.

        Symmetrises, eigendecomposes, clamps eigenvalues to [eig_min, eig_max],
        and reassembles. eig_min ensures PSD; eig_max prevents pathological
        Hessians from heavy-tail regions where curvature can explode (sinh-
        arcsinh has very tiny σ_eff in the centre or huge near tails). For
        T~6 the eigendecomposition is essentially free.
        """
        M = 0.5 * (M + M.transpose(-1, -2))
        # Pre-condition slightly so eigh doesn't fail on borderline matrices.
        T = M.shape[-1]
        I = torch.eye(T, device=M.device, dtype=M.dtype)
        try:
            eigvals, eigvecs = torch.linalg.eigh(M + 1e-6 * I)
        except torch._C._LinAlgError:
            # Last-ditch: heavily regularised eigh.
            eigvals, eigvecs = torch.linalg.eigh(M + 1.0 * I)
        eigvals = eigvals.clamp(min=eig_min, max=eig_max)
        M_clipped = eigvecs @ torch.diag_embed(eigvals) @ eigvecs.transpose(-1, -2)
        M_clipped = 0.5 * (M_clipped + M_clipped.transpose(-1, -2))
        return M_clipped, eigvals, eigvecs

    @staticmethod
    def _solve_with_eig(eigvals, eigvecs, g):
        """Solve M δ = g where M = V diag(λ) V^T."""
        Vt_g = (eigvecs.transpose(-1, -2) @ g.unsqueeze(-1)).squeeze(-1)
        delta = (eigvecs @ (Vt_g / eigvals).unsqueeze(-1)).squeeze(-1)
        return delta

    @staticmethod
    def _sigma_chol_with_eig(eigvals, eigvecs):
        """Σ = V diag(1/λ) V^T, then Cholesky."""
        T = eigvals.shape[-1]
        Sigma = eigvecs @ torch.diag_embed(1.0 / eigvals) @ eigvecs.transpose(-1, -2)
        Sigma = 0.5 * (Sigma + Sigma.transpose(-1, -2))
        I = torch.eye(T, device=Sigma.device, dtype=Sigma.dtype)
        # Tiny positive jitter for the Cholesky.
        L = torch.linalg.cholesky(Sigma + 1e-8 * I)
        return Sigma, L

    def _refine_mode(self, y, z_init):
        """Refine z toward the mode of log_joint via gradient ascent with
        per-batch backtracking line search.

        We avoid Newton steps in the inner loop: the Hessian can be very
        ill-conditioned far from the mode (heavy-tailed regions, edge cases),
        and Newton there is more trouble than it's worth. Instead we take
        normalised gradient steps with a step size that's halved for any
        batch element where log_joint failed to improve.

        We use the Hessian only at the *final* point to define Q's covariance,
        where a single eigenvalue clip suffices.
        """
        z = z_init
        # Initial step size per batch element. Tuned so a step of size step *
        # gradient direction is reasonable for typical residuals (~σ_eps).
        step = 0.1 * torch.ones(y.shape[0], device=y.device, dtype=y.dtype)

        for _ in range(self.num_newton_steps):
            g = self._grad_only(y, z)
            # Normalised gradient direction so step size has consistent units.
            g_norm = g.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            direction = g / g_norm
            # Compute log_joint at current z and a candidate z + step * direction.
            ll_cur = self._log_joint_no_grad(y, z)
            for _bls in range(5):
                z_new = z + step.unsqueeze(-1) * direction
                ll_new = self._log_joint_no_grad(y, z_new)
                improved = ll_new > ll_cur - 1e-6
                if improved.all():
                    z = torch.where(improved.unsqueeze(-1), z_new, z)
                    break
                # Halve step where it failed; accept where it succeeded.
                z = torch.where(improved.unsqueeze(-1), z_new, z)
                step = torch.where(improved, step, step * 0.5)
                ll_cur = torch.where(improved, ll_new, ll_cur)
            else:
                # All 5 backtracks failed for some elements; just leave them.
                pass
        return z.detach()

    def _ensure_mode_cache(self, y):
        N, T = y.shape
        need_init = (
            not self.cache_modes
            or self._mode_cache is None
            or self._mode_cache.shape != (N, T)
            or self._mode_cache.device != y.device
        )
        if need_init:
            # Cold init: ẑ_0 = y. Reasonable when σ_ε is small relative to
            # the residual; Newton refines from here.
            self._mode_cache = y.detach().clone()

    # ----- Encoder interface ----------------------------------------------

    def forward(self, y):
        T = y.shape[1]
        device = y.device
        dtype = y.dtype
        I = torch.eye(T, device=device, dtype=dtype)

        # E-step: refine the mode. Cache is updated.
        self._ensure_mode_cache(y)
        z_star = self._refine_mode(y, self._mode_cache)
        if self.cache_modes:
            self._mode_cache = z_star.detach()

        # Hessian at refined mode for the Σ that defines Q.
        _, H = self._grad_and_hessian(y, z_star)
        # Clip Hessian eigenvalues from both sides for numerical safety.
        # eig_min ensures PSD; eig_max prevents pathological scales from
        # producing an essentially point-mass Q.
        _, eigvals, eigvecs = self._clip_eig(-H + self.lm_lambda * I,
                                             eig_min=self.lm_lambda,
                                             eig_max=1.0 / (1e-4))  # σ_min ~ 0.01
        Sigma_post, L = self._sigma_chol_with_eig(eigvals, eigvecs)
        Sigma_post = Sigma_post.detach()
        L = L.detach()

        self.mean_var = Sigma_post.diagonal(dim1=-2, dim2=-1).mean().detach()

        # Both mu and L are detached from θ-graph, so M-step gradient on θ
        # flows only via log p(z) and log p(y|z) at fixed z samples.
        return z_star.detach(), L.detach()

    def draw_and_logprob(self, y, u, logpr_draw=False):
        mu, L = self.forward(y)
        z = mu + (L @ u.unsqueeze(-1)).squeeze(-1)
        q = MultivariateNormal(mu, scale_tril=L)
        if logpr_draw:
            log_qz = q.log_prob(z)
        else:
            log_qz = -q.entropy()
        return z, log_qz

    def stats(self) -> dict:
        return {"mean_var": float(self.mean_var.item())}
