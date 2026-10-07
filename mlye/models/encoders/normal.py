import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, MultivariateNormal
from torch.distributions import constraints
import torch.profiler

from ...utils import evalSinhArcsinhNormal
from .base import Encoder, EncoderConfig

class JointNormalPosterior(Encoder):
    def __init__(
            self, 
            dim:int, 
            dim_latent:int | None = None,
            regularize:float=1e-3, 
            hidden_dim:int=32, 
            diagonal:bool=False, 
            sd_clamp:float=3.0):
        super().__init__()
        self.dim = dim

        # Note: diagonal=True is handled by normal_diagonal encoder type

        # unless sepecified, we assume it is the same ( input = output )
        if dim_latent is None:
            self.dim_latent = dim
        else:
            self.dim_latent = dim_latent

        self.fc = nn.Linear(dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, self.dim_latent)
        self.diagonal = diagonal
        self.regularize = regularize
        self.sd_clamp = sd_clamp

        self.diag_head = nn.Linear(hidden_dim, self.dim_latent)
        if not diagonal:
            self.off_head = nn.Linear(hidden_dim, self.dim_latent*(self.dim_latent-1)//2)

        with torch.no_grad():
            if not diagonal:
                self.off_head.weight.mul_(0.01)
                if self.off_head.bias is not None:
                    self.off_head.bias.mul_(0.01)

        self.mean_var = torch.tensor(0.0)  # for stats

    def get_seed_dim(self) -> int:
        return self.dim_latent

    def forward(self, y):
        h  = F.relu(self.fc(y))
        mu = self.mu(h)

        # we center the first latent that represent z aroudn y
        d_y = y.size(-1)
        mu[..., :d_y] += y

        diag_raw = self.diag_head(h)
        L_diag = F.softplus(diag_raw) + self.regularize  # >0
        if not self.diagonal:
            off_raw = self.off_head(h)
            idx_i, idx_j = torch.tril_indices(self.dim_latent, self.dim_latent, offset=-1)
            L = torch.zeros(y.size(0), self.dim_latent, self.dim_latent, device=y.device)
            L[:, idx_i, idx_j] = off_raw.clamp(-5.0, 5.0)
            L[:, range(self.dim_latent), range(self.dim_latent)] = L_diag.clamp(max=self.sd_clamp)
        else:
            L = torch.diag_embed(L_diag.clamp(max=self.sd_clamp))
        return mu, L

    def draw_and_logprob(self, y, u, logpr_draw=False):
        mu, L = self.forward(y)
        z = mu + torch.matmul(L, u.unsqueeze(-1)).squeeze(-1)  # eps ~ N(0,I)
        q = MultivariateNormal(mu, scale_tril=L)

        if logpr_draw:
            log_qz = q.log_prob(z)
        else:
            # Better: use closed-form KL for ELBO and skip log_qz entirely.
            log_qz = - q.entropy() #log_prob(z)

        diag_Sigma = torch.sum(L**2, dim=-1)
        self.mean_var = diag_Sigma.mean()

        return z, log_qz

    def stats(self) ->  dict:
        return {
            'mean_var': self.mean_var.item()
        }

    @classmethod
    def from_config(cls, config: EncoderConfig) -> 'JointNormalPosterior':
        """Create instance from configuration using type-based dispatch."""
        from .base import JointNormalConfig
        
        # Type-based dispatch
        if isinstance(config, JointNormalConfig):
            # Specialized config - use all fields directly
            kwargs = {
                'dim': config.dim,
                'regularize': config.regularize,
                'hidden_dim': config.hidden_dim,
                'sd_clamp': config.sd_clamp,
                'diagonal': config.diagonal if config.type != 'normal_diagonal' else True
            }
            
            if config.type == 'joint_normal_extra':
                if config.extra_latents <= 0:
                    raise ValueError("extra_latents must be positive for joint_normal_extra encoder")
                kwargs['dim_latent'] = config.dim + config.extra_latents
        else:
            # Base config - use fallbacks
            kwargs = {
                'dim': config.dim,
                'regularize': config.regularize,
                'hidden_dim': 32,  # Default
                'sd_clamp': 3.0,   # Default
                'diagonal': True if config.type == 'normal_diagonal' else False
            }
            
            if config.type == 'joint_normal_extra':
                # Default to 1 extra latent for base config
                kwargs['dim_latent'] = config.dim + 1
            
        return cls(**kwargs)

class TridiagJointNormalPosterior(Encoder):
    """Joint normal posterior whose precision matrix is tridiagonal.

    q(z_{1:T}) = N(mu, Lambda^{-1}) where Lambda is symmetric tridiagonal
    and positive-definite **by construction**.

    Parameterisation: the precision matrix is built from its bidiagonal
    Cholesky factor L (lower-triangular, only diagonal + sub-diagonal
    non-zero):
        Lambda = L @ L.T
    with L_{t,t}  = softplus(raw_diag) + regularize  (strictly positive)
         L_{t,t-1} = raw_sub                          (free sign).
    Lambda is then symmetric tridiagonal and PD unconditionally — exactly
    the HMM-shaped posterior conditional-independence structure, with no
    soft-constraint or safety-margin hack.

    The encoder head emits 3T - 1 outputs per individual:
        T   mean entries
        T   raw Cholesky-factor diagonal entries (-> softplus + jitter)
        T-1 raw Cholesky-factor sub-diagonal entries (free sign)
    """

    def __init__(
        self,
        dim: int,
        regularize: float = 1e-3,
        hidden_dim: int = 32,
    ):
        super().__init__()
        self.dim = dim
        self.dim_latent = dim
        self.regularize = regularize

        self.fc = nn.Linear(dim, hidden_dim)
        # 3T - 1 outputs: T mean + T raw L_diag + (T-1) raw L_subdiag
        self.head = nn.Linear(hidden_dim, 3 * dim - 1)

        with torch.no_grad():
            # Shrink head weights so the precision starts near identity-ish
            # and the sub-diagonal of L starts near zero.
            self.head.weight.mul_(0.01)
            if self.head.bias is not None:
                self.head.bias.mul_(0.01)

        self.mean_var = torch.tensor(0.0)

    def get_seed_dim(self) -> int:
        return self.dim_latent

    def forward(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (mu, L_diag, L_subdiag) per individual.

        mu:        (B, T)
        L_diag:    (B, T)   strictly positive (Cholesky-factor diagonal)
        L_subdiag: (B, T-1) free sign (Cholesky-factor sub-diagonal)
        """
        h = F.relu(self.fc(y))
        out = self.head(h)

        T = self.dim
        mu = out[..., :T].clone()
        raw_diag = out[..., T:2 * T]
        raw_sub = out[..., 2 * T:3 * T - 1]

        d_y = y.size(-1)
        mu[..., :d_y] = mu[..., :d_y] + y

        L_diag = F.softplus(raw_diag) + self.regularize
        L_subdiag = raw_sub

        return mu, L_diag, L_subdiag

    def _build_L(self, L_diag: torch.Tensor, L_subdiag: torch.Tensor) -> torch.Tensor:
        """Construct the dense (B, T, T) lower-triangular bidiagonal L."""
        B = L_diag.size(0)
        T = self.dim
        L = torch.zeros(B, T, T, device=L_diag.device, dtype=L_diag.dtype)
        idx = torch.arange(T, device=L_diag.device)
        L[:, idx, idx] = L_diag
        if T > 1:
            idx_sub = torch.arange(1, T, device=L_diag.device)
            L[:, idx_sub, idx_sub - 1] = L_subdiag
        return L

    def draw_and_logprob(self, y: torch.Tensor, u: torch.Tensor, logpr_draw: bool = False):
        mu, L_diag, L_subdiag = self.forward(y)
        L = self._build_L(L_diag, L_subdiag)

        # Precision = L L^T (tridiagonal, PD unconditionally)
        Lambda = L @ L.transpose(-1, -2)
        # Symmetrise for numerical safety in MultivariateNormal
        Lambda = 0.5 * (Lambda + Lambda.transpose(-1, -2))

        # Reparameterised sample: z = mu + L^{-T} u
        # Cov(z) = L^{-T} L^{-1} = (L L^T)^{-1} = Lambda^{-1}, as required.
        # Solve L^T x = u (upper-triangular system) -> x = L^{-T} u.
        z = mu + torch.linalg.solve_triangular(
            L.transpose(-1, -2), u.unsqueeze(-1), upper=True
        ).squeeze(-1)

        q = MultivariateNormal(mu, precision_matrix=Lambda)
        if logpr_draw:
            log_qz = q.log_prob(z)
        else:
            log_qz = -q.entropy()

        # Stat: mean marginal variance = mean of diag(Sigma) where Sigma = Lambda^{-1}.
        # diag(Sigma) = diag(L^{-T} L^{-1}) = column-wise sum of squares of L^{-1}.
        I = torch.eye(self.dim, device=y.device, dtype=L.dtype).expand_as(L)
        Linv = torch.linalg.solve_triangular(L, I, upper=False)
        sigma_diag = (Linv ** 2).sum(dim=-2)
        self.mean_var = sigma_diag.mean()

        return z, log_qz

    def stats(self) -> dict:
        return {'mean_var': float(self.mean_var.item())}

    @classmethod
    def from_config(cls, config: EncoderConfig) -> 'TridiagJointNormalPosterior':
        from .base import JointNormalConfig

        if isinstance(config, JointNormalConfig):
            return cls(
                dim=config.dim,
                regularize=config.regularize,
                hidden_dim=config.hidden_dim,
            )
        return cls(
            dim=config.dim,
            regularize=config.regularize,
            hidden_dim=32,
        )


class JointNormalPosteriorRestricted(Encoder):
    def __init__(self, dim:int, sigma_eps:float, regularize=1e-3, fix_sigma=False, hidden_dim=32):
        super().__init__()
        self.dim = dim
        self.fc = nn.Linear(dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, dim) # dim means
        self.regularize = regularize

        if fix_sigma:
            self.log_std = np.log(sigma_eps) * torch.ones(dim)
        else:
            self.log_std = nn.Parameter( np.log(sigma_eps) * torch.ones(dim), requires_grad=True)

    def forward(self, y):
        h = F.relu(self.fc(y))
        mu = self.mu(h)
        # return mu, torch.eye(self.dim) * torch.exp(self.log_std)
        return mu, torch.diag(torch.exp(self.log_std))

    def draw_and_logprob(self, y, u, logpr_draw=False):

        mu, L = self.forward(y)
        z = mu + torch.matmul(L, u.unsqueeze(-1)).squeeze(-1)
        q = MultivariateNormal(mu, scale_tril=L)
        log_qz = q.log_prob(z)

        if logpr_draw:
            return z, log_qz
        else:
            return z, -q.entropy()

        return z, log_qz

    def get_seed_dim(self) -> int:
        return self.dim

    def stats(self):
        return {
            'mean_var': torch.exp(2 * self.log_std).mean().item()
        }

    @classmethod  
    def from_config(cls, config: EncoderConfig) -> 'JointNormalPosteriorRestricted':
        """Create instance from configuration using type-based dispatch."""
        from .base import RestrictedNormalConfig
        
        if isinstance(config, RestrictedNormalConfig):
            # Specialized config
            return cls(
                dim=config.dim,
                sigma_eps=config.sigma_eps,
                regularize=config.regularize,
                fix_sigma=config.fix_sigma,
                hidden_dim=config.hidden_dim
            )
        else:
            # Base config with defaults
            return cls(
                dim=config.dim,
                sigma_eps=1.0,  # Default
                regularize=config.regularize,
                fix_sigma=False,  # Default
                hidden_dim=32  # Default
            )

class JointNormalTribandPrecisionPosterior(Encoder):
    def __init__(
        self,
        dim: int,
        regularize: float = 1e-3,
        diagonal: bool = False,
        hidden_dim: int = 32,
        eps: float = 1e-6,   # jitter for strictly positive diag
    ):
        super().__init__()
        self.dim = dim
        self.regularize = regularize
        self.diagonal = diagonal
        self.eps = eps

        self.fc = nn.Linear(dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, dim)                 # mean
        self.l_triband = nn.Linear(hidden_dim, dim + (dim-1))  # symmetric tri-band precision factor params

        with torch.no_grad():
            self.l_triband.weight *= 0.01

        # for stats() robustness
        self.register_buffer("mean_var", torch.tensor(0.0))

    def get_seed_dim(self) -> int:
        return self.dim

    def _build_precision_chol(self, l_raw: torch.Tensor, B: int, device) -> torch.Tensor:
        """
        Build lower-triangular L (tri-band) such that precision J = L^T L.
        l_raw: (B, dim + (dim-1)) with first dim entries for diagonal, next (dim-1) for subdiagonal.
        Returns L of shape (B, dim, dim).
        """
        N = self.dim
        L = torch.zeros(B, N, N, device=device)

        # diagonal (strictly positive)
        ii = 0
        for i in range(N):
            L[:, i, i] = F.softplus(l_raw[:, ii]) + self.eps
            ii += 1

        # sub-diagonal (optionally zero if diagonal=True)
        if not self.diagonal:
            for i in range(1, N):
                L[:, i, i-1] = l_raw[:, ii]
                ii += 1

        # optional extra stabilization on the diagonal
        if self.regularize > 0:
            for i in range(N):
                L[:, i, i] = L[:, i, i] + self.regularize

        return L

    def forward(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            mu: (B, dim)
            scale_tril: (B, dim, dim)  # Cholesky of covariance (Sigma) = inv(precision)
        """
        with torch.profiler.record_function("POSTERIOR FORWARD"):
            h = F.relu(self.fc(y))
            mu: torch.Tensor = y + self.mu(h)

            # precision-cholesky parameters
            l_raw = self.l_triband(h)
            B = y.size(0)
            device = y.device

            # L such that precision J = L^T L
            L_prec = self._build_precision_chol(l_raw, B, device)

            # scale_tril of covariance is inv(L_prec) via triangular solve: L_prec * X = I -> X = L_prec^{-1}
            I = torch.eye(self.dim, device=device).expand(B, self.dim, self.dim)
            scale_tril = torch.linalg.solve_triangular(
                L_prec, I, upper=False, left=True
            )

            # stats
            self.mean_var = torch.diagonal(scale_tril, dim1=-2, dim2=-1).pow(2).mean()

            return mu, scale_tril

    def draw_and_logprob(self, y: torch.Tensor, u: torch.Tensor, logpr_draw=False):
        """
        u: standard normal noise with shape (B, dim)
        """
        with torch.profiler.record_function("POSTERIOR LOGPROB"):
            mu, scale_tril = self.forward(y)

            # reparameterize: z = mu + scale_tril @ u
            z = mu + torch.matmul(scale_tril, u.unsqueeze(-1)).squeeze(-1)

            q = MultivariateNormal(mu, scale_tril=scale_tril)
            log_qz = q.log_prob(z)

        return z, log_qz

    def stats(self) -> dict:
        return {"mean_var": float(self.mean_var.item())}

    @classmethod
    def from_config(cls, config: EncoderConfig) -> 'JointNormalTribandPrecisionPosterior':
        """Create instance from configuration using isinstance dispatch."""
        from .base import TribandConfig
        
        if isinstance(config, TribandConfig):
            # Specialized config - use all fields directly
            return cls(
                dim=config.dim,
                regularize=config.regularize,
                diagonal=config.diagonal
            )
        else:
            # Base config - use defaults
            return cls(
                dim=config.dim,
                regularize=config.regularize,
                diagonal=False  # Default
            )


# class TransformedJointNormalPosterior(Encoder):
#     """
#     Amortized encoder that parameterizes a multivariate normal posterior with a learnable
#     lower-triangular scale (Cholesky factor) and applies a per-dimension sinh–arcsinh
#     transformation to capture skewness and tail-heaviness.

#     This module maps an input y ∈ R^dim to:
#     - mu: mean of the base multivariate normal
#     - L: lower-triangular scale_tril with strictly positive diagonal
#     - skew: per-dimension skew parameter of the sinh–arcsinh transform
#     - tail: per-dimension positive tail parameter of the sinh–arcsinh transform

#     The latent sample is produced via reparameterization and a marginal flow:
#     1) z1 = mu + L u, where u ~ N(0, I)
#     2) z2, log_flow = evalSinhArcsinhNormal(mu, stddev, skew, tail, z0) with
#         z0 = (z1 - mu) / stddev and stddev = sqrt(diag(L L^T))

#     The method draw_and_logprob returns (z2, log_qz), where log_qz is computed as the
#     expected base log-density term of the Gaussian (via -entropy) plus the sum of the
#     log-Jacobian contributions from the sinh–arcsinh transformation.

#     Args:
#          dim (int): Dimensionality of y and the latent z.
#          regularize (float, optional): Positive jitter added to the softplus-transformed
#               diagonal of L to ensure numerical stability. Default: 1e-3.
#          diagonal (bool, optional): If True, intend to restrict covariance to diagonal.
#               Note: currently not used in forward; kept for interface compatibility.

#     Attributes:
#          dim (int): Problem dimension.
#          fc (nn.Linear): Feature extractor mapping R^dim → R^32 with ReLU.
#          mu (nn.Linear): Produces a residual shift added to y for the mean.
#          l_tri (nn.Linear): Produces entries used to fill the lower-triangular scale L.
#          net_skew (nn.Linear): Produces per-dimension skew parameters.
#          net_log_tail (nn.Linear): Produces log tail parameters (exp ensures positivity).
#          regularize (float): Diagonal jitter.
#          diagonal (bool): Flag reserved for diagonal-covariance variants.

#     Forward inputs/outputs:
#          forward(y):
#               Args:
#                     y (Tensor): Shape (B, dim).
#               Returns:
#                     mu (Tensor): Shape (B, dim).
#                     L (Tensor): Shape (B, dim, dim), lower-triangular with positive diagonal.
#                     skew (Tensor): Shape (B, dim).
#                     tail (Tensor): Shape (B, dim), strictly positive.

#     Sampling and log-density:
#          draw_and_logprob(y, u):
#               Args:
#                     y (Tensor): Shape (B, dim).
#                     u (Tensor): Base noise ~ N(0, I), shape (B, dim).
#               Returns:
#                     z (Tensor): Transformed latent sample, shape (B, dim).
#                     log_qz (Tensor): Per-sample log-density under q(z | y) up to the
#                          flow-corrected term, shape (B,).

#     Statistics:
#          stats():
#               Returns:
#                     dict with:
#                          - mean_var (float): Mean of the diagonal variances (L_ii^2).
#                          - mean_alpha (float): Mean skew parameter across batch and dims.

#     Notes:
#     - The diagonal of L is enforced positive via softplus plus `regularize`.
#     - Tail parameters are kept positive by exponentiating `net_log_tail` outputs.
#     - Requires an implementation of `evalSinhArcsinhNormal` returning (z, log_flow)
#       with log_flow per-dimension contributions.
#     - Constraint checks are performed for debugging; violations are printed to stdout.
#     """
#     def __init__(
#             self, 
#             dim:int, 
#             regularize:float=1e-3,             
#         ):
#         super().__init__()
#         self.dim = dim
#         self.fc = nn.Linear(dim, 32)
#         self.mu = nn.Linear(32, dim) # dim means
#         self.l_tri = nn.Linear(32, dim*(dim-1))  # dim*(dim-1) lower triangle
#         self.regularize = regularize

#         # transform marginals through sinh-arcsinh transformation
#         self.net_skew     = nn.Linear(32, dim) # transform parameter
#         self.net_log_tail = nn.Linear(32, dim) # transform parameter

#         with torch.no_grad():
#             self.l_tri.weight *= 0.01  # shrink weights
#             if self.l_tri.bias is not None:
#                 self.l_tri.bias.data *= 0.01

#             self.net_skew.weight *= 0.01  # shrink weights
#             if self.net_skew.bias is not None:
#                 self.net_skew.bias.data *= 0.01 

#             self.net_log_tail.weight *= 0.01  # shrink weights
#             if self.net_log_tail.bias is not None:
#                 self.net_log_tail.bias.data *= 0.01

#     def forward(self, y:torch.Tensor) ->  tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
#         with torch.profiler.record_function("POSTERIOR FORWARD"):
#             h = F.relu( self.fc(y) )
#             mu:torch.Tensor = y + self.mu(h)
#             skew = self.net_skew(h).clamp(-0.5, 0.5)  # clamp skewness to [-3, 3] for stability
#             tail = torch.exp(self.net_log_tail(h)).clamp(0.3, 1.7) # clamp tail to (0.1, 1.9) for stability

#             l_raw = self.l_tri(h)
#             L = torch.zeros(y.size(0), self.dim, self.dim, device=y.device)

#             ii = 0
#             for i in range(self.dim):
#                 for j in range(i, self.dim):                    
#                     if (i==j): # ON diagonal
#                         L[:, i, i] = F.softplus(l_raw[:, ii]).clamp(0, 3)  + self.regularize # we add a little to the diagonal to avoid entropy diverging
#                     else: # OFF diagonal
#                         L[:, j, i] = l_raw[:, ii].clamp(-3, 3)
#                     ii += 1

#             constraint = constraints.independent(constraints.real, 1)
#             # Boolean mask of invalid entries
#             mask = ~constraint.check(L)
#             if mask.any():
#                 print("Constraint violated at indices:", mask.nonzero())
#                 print("Invalid values:", L[mask])

#         return mu, L, skew, tail

#     def draw_and_logprob(self, y, u, logpr_draw=False):
#         with torch.profiler.record_function("POSTERIOR LOGPROB"):

#             mu, L, skew, tail = self.forward(y)

#             # force to normal
#             # skew = torch.zeros_like(skew)
#             # tail = torch.ones_like(tail)  # we do not use skewness and tail in this implementation

#             Sigma_diag = torch.sum(L**2, dim=1)  # diagonal of Sigma
#             stddev = torch.sqrt(Sigma_diag)

#             # print("shapes:", L.shape, stddev.shape)
#             Lcor = L / stddev.unsqueeze(-1)  # normalize L by stddev

#             # z1 = mu + torch.matmul(Lcor, u.unsqueeze(-1)).squeeze(-1)
#             z0 = torch.matmul(Lcor, u.unsqueeze(-1)).squeeze(-1)  # z0 = Lcor * u
#             q = MultivariateNormal(torch.zeros_like(mu), scale_tril=Lcor)
#             log_qz = q.log_prob(z0)
#             #log_qz = - q.entropy()

#             # applying sinh-arcsinh transformation to draws
#             z2, log_flow = evalSinhArcsinhNormal(mu, stddev, skew, tail, z0, include_base=False)

#             # print("z2 shape:", log_qz.shape, "log_flow shape:", log_flow.shape)
#             log_qz += torch.sum( log_flow, dim=1) 

#             # storing mean variance of posterior for stats
#             self.mean_var = torch.diagonal(L, dim1=-2, dim2=-1).pow(2).mean()
#             self.mean_alpha = skew.mean()

#         return z2, log_qz

#     def stats(self) ->  dict:
#         return {
#             'mean_var': self.mean_var.item(),
#             'mean_alpha': self.mean_alpha.item(),
#         }
    

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

# Assumed to exist in your codebase:
# def evalSinhArcsinhNormal(mu, stddev, skew, tail, z0, include_base: bool): ...
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

# Assumed to exist in your codebase:
# def evalSinhArcsinhNormal(mu, stddev, skew, tail, z0, include_base: bool):
#   -> returns (z, log_flow_per_dim), where log_flow is typically log|dz0/dz| per-dim

class TransformedJointNormalPosterior(Encoder):
    """
    Multivariate Gaussian encoder with learnable lower-triangular scale (Cholesky)
    and per-dimension sinh–arcsinh transform. The lower-triangle is parameterized
    with two separate heads:
      - l_diag:      diagonal entries (positivity via softplus + jitter)
      - l_offdiag:   strictly lower-tri entries (clipped)

    Uses torch.tril_indices (row-major) for vectorized scatter.
    """

    def __init__(
        self,
        dim: int,
        regularize: float = 1e-3,
        hidden: int = 32,
        offdiag_clip: float = 3.0,
        skew_clip: float = 0.5,
        tail_min: float = 0.3,
        tail_max: float = 1.7,
    ):
        super().__init__()
        self.dim = dim
        self.regularize = regularize
        self.offdiag_clip = offdiag_clip
        self.skew_clip = skew_clip
        self.tail_min = tail_min
        self.tail_max = tail_max

        # Shared features
        self.fc = nn.Linear(dim, hidden)

        # Heads
        self.mu = nn.Linear(hidden, dim)
        self.l_diag = nn.Linear(hidden, dim)                     # diagonal (D)
        self.l_offdiag = nn.Linear(hidden, dim * (dim - 1) // 2) # strictly lower (D*(D-1)/2)
        self.net_skew = nn.Linear(hidden, dim)
        self.net_log_tail = nn.Linear(hidden, dim)

        # Running stats
        self.mean_var = torch.tensor(0.0)
        self.mean_alpha = torch.tensor(0.0)

        # Small init for stability
        with torch.no_grad():
            for lin in (self.l_diag, self.l_offdiag, self.net_skew, self.net_log_tail):
                lin.weight.mul_(0.01)
                if lin.bias is not None:
                    lin.bias.mul_(0.01)

        # ---- Precompute row-major lower-tri indices ----
        # Diagonal indices (i, i)
        diag = torch.arange(dim, dtype=torch.long)
        # Strictly lower triangle (row > col), row-major order
        lt = torch.tril_indices(row=dim, col=dim, offset=-1)
        self.register_buffer("_diag_idx", diag, persistent=False)
        self.register_buffer("_lt_row", lt[0], persistent=False)
        self.register_buffer("_lt_col", lt[1], persistent=False)

    def get_seed_dim(self) -> int:
        return self.dim

    @torch.no_grad()
    def stats(self) -> dict:
        return {"mean_var": float(self.mean_var), "mean_alpha": float(self.mean_alpha)}

    # ---- Vectorized Cholesky builder using separate diag/offdiag heads ----
    def _build_L(self, h: torch.Tensor) -> torch.Tensor:
        """
        h: (B, H)
        returns L: (B, D, D) lower-tri with positive diagonal
        """
        B, D = h.shape[0], self.dim
        device, dtype = h.device, h.dtype

        # Diagonal (positivity via softplus + jitter); no upper clamp by default
        raw_d = self.l_diag(h)                                   # (B, D)
        diag = F.softplus(raw_d) + self.regularize               # (B, D)

        # Off-diagonals (strictly lower), optional clip
        raw_o = self.l_offdiag(h)                                # (B, D*(D-1)/2)
        if self.offdiag_clip is not None:
            raw_o = raw_o.clamp(-self.offdiag_clip, self.offdiag_clip)

        # Scatter into L
        L = torch.zeros(B, D, D, device=device, dtype=dtype)
        L[:, self._diag_idx, self._diag_idx] = diag
        L[:, self._lt_row, self._lt_col] = raw_o
        return L

    def forward(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            mu:   (B, D)
            L:    (B, D, D) lower-tri with positive diag
            skew: (B, D)
            tail: (B, D) positive
        """
        with torch.profiler.record_function("POSTERIOR FORWARD"):
            h = F.relu(self.fc(y))
            mu = y + self.mu(h)

            skew = self.net_skew(h).clamp(-self.skew_clip, self.skew_clip)
            tail = torch.exp(self.net_log_tail(h)).clamp(self.tail_min, self.tail_max)

            L = self._build_L(h)

        return mu, L, skew, tail

    def draw_and_logprob(self, y: torch.Tensor, u: torch.Tensor, logpr_draw: bool = False):
        """
        Args:
            y: (B, D)
            u: (B, D) ~ N(0, I)
        Returns:
            z:      (B, D) transformed sample
            log_qz: (B,)   log-density under q(z|y)
        """
        with torch.profiler.record_function("POSTERIOR LOGPROB"):
            mu, L, skew, tail = self.forward(y)                  # L: (B, D, D)

            # diag(Σ) for Σ = L L^T  (sum over columns)
            Sigma_diag = (L ** 2).sum(dim=2)                     # (B, D)
            stddev = Sigma_diag.clamp_min(1e-12).sqrt()          # (B, D)

            # Normalize rows to unit marginal stds: diag(Lcor Lcor^T) = 1
            Lcor = L / stddev.unsqueeze(-1)                      # (B, D, D)

            # Base Gaussian at correlation scale
            z0 = torch.matmul(Lcor, u.unsqueeze(-1)).squeeze(-1) # (B, D)
            q0 = MultivariateNormal(loc=torch.zeros_like(mu), scale_tril=Lcor, validate_args=False)
            log_qz = q0.log_prob(z0)                             # (B,)

            # Apply per-dim sinh–arcsinh transform
            z, log_flow = evalSinhArcsinhNormal(mu, stddev, skew, tail, z0, include_base=False)
            log_qz = log_qz + log_flow.sum(dim=1)

            # Stats
            self.mean_var = torch.diagonal(L, dim1=-2, dim2=-1).pow(2).mean()
            self.mean_alpha = skew.mean()

        return z, log_qz

    @classmethod
    def from_config(cls, config: EncoderConfig) -> 'TransformedJointNormalPosterior':
        """Create instance from configuration using isinstance dispatch."""
        from .base import FlowConfig
        
        if isinstance(config, FlowConfig):
            # Specialized config - use all fields directly
            return cls(
                dim=config.dim,
                regularize=config.regularize
            )
        else:
            # Base config - use defaults
            return cls(
                dim=config.dim,
                regularize=config.regularize
            )
