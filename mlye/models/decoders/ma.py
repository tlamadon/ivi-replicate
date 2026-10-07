import numpy as np
import torch
import torch.nn as nn
from torch.distributions import StudentT
from scipy.stats import skew, kurtosis

from mlye.models.encoders.base import split_latent
from ...utils import SinhArcsinh
from .base import Decoder


class MASinhEmission(Decoder):
    def __init__(self, sigma_eps=0.1, theta = 0.0, fix_sigma=False, fix_theta=False,
                 sigma_floor: float = 0.0, beta_init: float = 1.0, fix_beta: bool = False,
                 alpha_eps: float = 0.0, fix_skew: bool = True):
        super().__init__()

        self.log_sigma = nn.Parameter(np.log(sigma_eps) * torch.ones(1), requires_grad=not fix_sigma)
        # beta_init=1.0 → Gaussian; beta_init>1 → heavier tails. fix_beta=True
        # freezes β at the init value (used for β-profile experiments).
        self.log_beta = nn.Parameter(np.log(beta_init) * torch.ones(1), requires_grad=not fix_beta)
        self.theta = nn.Parameter(theta * torch.ones(1), requires_grad=not fix_theta)
        # Residual skewness of the sinh-arcsinh emission. Default 0 + frozen
        # preserves the previous behaviour (centred symmetric residual).
        # When fix_skew=False, log_likelihood recentres so the conditional
        # mean E[y_t | z_t] = z_t stays preserved at any alpha_eps.
        self.alpha_eps = nn.Parameter(alpha_eps * torch.ones(1),
                                       requires_grad=not fix_skew)
        # sigma_floor: enforce scale >= sigma_floor via softplus around the floor.
        # Prevents the decoder from collapsing to σ→0 when the encoder drives
        # residuals to zero (the failure mode observed under KL annealing).
        self.sigma_floor = float(sigma_floor)

    def _effective_scale(self) -> torch.Tensor:
        # Additive floor reparameterisation: σ_eff = floor + exp(log_sigma).
        # log_sigma → -∞ gives σ_eff → floor; log_sigma moderate gives
        # σ_eff ≈ exp(log_sigma) when exp(log_sigma) >> floor. Smooth, full
        # gradient, and effective σ is bounded below by floor by construction.
        sigma = torch.exp(self.log_sigma)
        if self.sigma_floor > 0.0:
            sigma = self.sigma_floor + sigma
        return sigma

    def get_distribution(self) -> torch.distributions.Distribution:
        """Return the SinhArcsinhNormal distribution with current parameters.

        ``skew=self.alpha_eps`` is 0 by default (frozen); when free, the
        log-likelihood path subtracts the closed-form mean so the
        conditional mean E[y_t | z_t] = z_t stays at z_t.
        """
        # SinhArcsinhNormal expects loc, scale, skewness, and tailweight
        return SinhArcsinh(
            loc = torch.zeros(1, device=self.log_sigma.device),
            scale = self._effective_scale(),
            skew = self.alpha_eps,
            tailweight = torch.exp(self.log_beta)
        )

    def _residual_mean(self) -> torch.Tensor:
        """E[W] for W ~ SinhArcsinh(0, sigma_eff, alpha_eps, exp(log_beta)).

        Uses the Jones & Pewsey (2009) decomposition
            E[scale * sinh(delta * asinh(Z) + delta * eps)]
                = scale * sinh(delta * eps) * P_delta,
        with P_delta = E[cosh(delta * asinh(Z))] for Z ~ N(0, 1). When
        alpha_eps = 0, sinh(0) = 0 makes this exactly 0 — so the
        ``fix_skew=True`` (default) path is bit-identical to the old
        no-recentring behaviour.
        """
        tailweight = torch.exp(self.log_beta)
        scale = self._effective_scale()
        device = self.log_sigma.device
        dtype = self.log_sigma.dtype
        n = 64
        base = torch.distributions.Normal(
            torch.zeros((), dtype=dtype, device=device),
            torch.ones((), dtype=dtype, device=device),
        )
        u = torch.linspace(1.0 / (2 * n), 1.0 - 1.0 / (2 * n), n,
                           device=device, dtype=dtype)
        z = base.icdf(u)
        p_delta = torch.cosh(tailweight * torch.asinh(z)).mean(0)
        return scale * torch.sinh(self.alpha_eps * tailweight) * p_delta

    def log_likelihood(self, y, z):
        """Compute joint log-likelihood using MA(1) Cholesky decomposition

        For MA(1): Sigma = sigma^2 * L @ L^T where L has 1's on diagonal, theta on lower off-diagonal
        L^{-1} has 1's on diagonal, -theta on lower off-diagonal
        Apply L^{-1} to residuals: (L^{-1} @ residuals^T)^T
        Log determinant: log|Sigma| = log|sigma^2 L L^T| = 2T * log(sigma) + log|L|^2
        For MA(1) L matrix, log|L| = 0 (determinant is 1)

        """
        batch_size, T = y.shape
        device = y.device

        # get underlying distribution
        dist = self.get_distribution()

        # Compute residuals
        residuals = y - z  # (batch_size, T)

        # Apply L^{-1} transformation without in-place operations
        transformed_residuals = []
        transformed_residuals.append(residuals[:, 0])

        for t in range(1, T):
            transformed_t = residuals[:, t] - self.theta * transformed_residuals[t-1]
            transformed_residuals.append(transformed_t)

        # Stack to create tensor of shape (batch_size, T)
        transformed_residuals = torch.stack(transformed_residuals, dim=1)

        # Mean-preserving recentring: under the model y_t = z_t + eps_t with
        # eps_t a centred sinh-arcsinh, the standardised residual W = eps_t + mu_W
        # is distributed as ``dist``. mu_W = 0 exactly when alpha_eps = 0, so the
        # fix_skew=True default is unchanged.
        mu_W = self._residual_mean()
        # Compute log probability of marginal density for each transformed residual
        log_prob_per_element = dist.log_prob(transformed_residuals + mu_W)

        # Sum over time dimension to get total log-likelihood for each batch element
        log_likelihood = torch.sum(log_prob_per_element, dim=1)

        return log_likelihood

    def draw(self, z):
        dist = self.get_distribution()
        return dist.sample(z.shape)

    def get_eps_std(self):
        return 0.0

    def stats(self):
        """Return a dictionary of statistics"""

        eps = self.draw(torch.zeros(50000, 1, device=self.log_sigma.device)).detach().cpu().numpy()
        return {
            'mean': np.mean(eps).item(),
            'std': np.std(eps).item(),
            'skewness': skew(eps).item(),
            'kurtosis': kurtosis(eps).item(),
            'theta': self.theta.item(),
            'distribution': 'ma_sinh_arcsinh_normal'
        }


class MANormalEmission(Decoder):
    def __init__(self, sigma_eps=0.1, theta=0.5, fix_sigma=False, fix_theta=False,
                 distribution='normal', df=5.0, fix_df=False):
        super().__init__()

        # Register all scalar state as nn.Parameter (with requires_grad
        # following the fix_* flag). nn.Parameter is moved by ``.to(device)``,
        # while a plain attribute tensor is not -- so a plain-tensor branch
        # for the fix_* case silently leaves the value on CPU after a GPU
        # move, blowing up later inside ``StudentT.log_prob``. Matches the
        # pattern already used for ``self.theta`` below.
        self.log_sigma_eps = nn.Parameter(np.log(sigma_eps) * torch.ones(1),
                                           requires_grad=not fix_sigma)
        self.theta = nn.Parameter(0.1 * theta * torch.ones(1),
                                   requires_grad=not fix_theta)

        self.distribution = distribution

        if distribution == 'student_t':
            self.log_df = nn.Parameter(np.log(df) * torch.ones(1),
                                        requires_grad=not fix_df)




    def _log_marginal_density(self, x):
        """Compute log probability of marginal density for transformed residuals

        Supports normal and Student-t distributions.
        For Student-t: allows for heavier tails and kurtosis through df parameter.

        Args:
            x: transformed residuals of shape (batch_size, T)

        Returns:
            log probabilities of shape (batch_size, T)
        """
        if self.distribution == 'normal':
            # log N(x; 0, sigma^2) = -0.5 * (x^2/sigma^2 + log(2*pi*sigma^2))
            return -0.5 * ((x ** 2) * torch.exp(-2 * self.log_sigma_eps) +
                          2 * self.log_sigma_eps +
                          np.log(2 * np.pi))

        elif self.distribution == 'student_t':
            df = torch.exp(self.log_df)
            scale = torch.exp(self.log_sigma_eps)
            student_t = StudentT(df=df, loc=torch.zeros_like(df), scale=scale)
            return student_t.log_prob(x)

        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")

    def log_likelihood(self, y, z):
        """Compute joint log-likelihood using MA(1) Cholesky decomposition

        For MA(1): Sigma = sigma^2 * L @ L^T where L has 1's on diagonal, theta on lower off-diagonal
        L^{-1} has 1's on diagonal, -theta on lower off-diagonal
        Apply L^{-1} to residuals: (L^{-1} @ residuals^T)^T
        Log determinant: log|Sigma| = log|sigma^2 L L^T| = 2T * log(sigma) + log|L|^2
        For MA(1) L matrix, log|L| = 0 (determinant is 1)

        """
        batch_size, T = y.shape
        device = y.device

        # Compute residuals
        residuals = y - z  # (batch_size, T)

        # Apply L^{-1} transformation without in-place operations
        transformed_residuals = []
        transformed_residuals.append(residuals[:, 0])

        for t in range(1, T):
            transformed_t = residuals[:, t] - self.theta * transformed_residuals[t-1]
            transformed_residuals.append(transformed_t)

        # Stack to create tensor of shape (batch_size, T)
        transformed_residuals = torch.stack(transformed_residuals, dim=1)

        # Compute log probability of marginal density for each transformed residual
        log_prob_per_element = self._log_marginal_density(transformed_residuals)

        # Sum over time dimension to get total log-likelihood for each batch element
        log_likelihood = torch.sum(log_prob_per_element, dim=1)

        return log_likelihood

    def draw(self, z):
        """Draw from joint MA(1) distribution using closed-form Cholesky"""
        batch_size, T = z.shape
        device = z.device

        if self.distribution == 'normal':
            # Sample white noise from normal distribution
            white_noise = torch.randn(batch_size, T, device=device)

        elif self.distribution == 'student_t':
            # Sample white noise from Student-t distribution
            df = torch.exp(self.log_df).item()  # Convert to scalar
            student_t = StudentT(df=df, loc=0.0, scale=1.0)
            white_noise = student_t.sample((batch_size, T)).to(device)

        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")

        # Apply L transformation: L @ white_noise^T where L has 1's on diagonal, theta on lower off-diagonal
        ma_noise = torch.zeros_like(white_noise)
        ma_noise[:, 0] = white_noise[:, 0]
        for t in range(1, T):
            ma_noise[:, t] = white_noise[:, t] + self.theta * ma_noise[:, t-1]

        # Scale by sigma
        ma_errors = torch.exp(self.log_sigma_eps) * ma_noise

        return z + ma_errors

    def get_eps_std(self):
        """Return marginal standard deviation"""
        if self.distribution == 'normal':
            # For normal: sigma * sqrt(1 + theta^2)
            return torch.exp(self.log_sigma_eps) * torch.sqrt(1 + self.theta ** 2)
        elif self.distribution == 'student_t':
            # For Student-t: scale * sqrt(df/(df-2)) * sqrt(1 + theta^2) for df > 2
            df = torch.exp(self.log_df)
            scale = torch.exp(self.log_sigma_eps)
            # Student-t variance is scale^2 * df/(df-2) for df > 2
            var_multiplier = torch.where(df > 2, df / (df - 2), torch.ones_like(df))
            return scale * torch.sqrt(var_multiplier) * torch.sqrt(1 + self.theta ** 2)
        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")

    def moments(self):
        """Return the first two moments of the distribution"""
        if self.distribution == 'normal':
            return {
                'mean': 0.0,
                'variance': (torch.exp(2 * self.log_sigma_eps) * (1 + self.theta ** 2)).item(),
                'skewness': 0.0,
                'kurtosis': 3.0,  # Normal distribution has kurtosis of
            }
        elif self.distribution == 'student_t':
            df = torch.exp(self.log_df)
            scale = torch.exp(self.log_sigma_eps)
            variance = scale ** 2 * df / (df - 2) * (1 + self.theta ** 2) if df > 2 else torch.Tensor(float('inf'))
            return {
                'mean': 0.0,
                'variance': variance.item(),
                'skewness': 0.0,
                'kurtosis': 3.0 + 6.0 / (df.item() - 4) if df.item() > 4 else float('inf'),  # Adjusted kurtosis for Student-t
            }
        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")

    def get_theta(self):
        return self.theta

    def get_df(self):
        """Return degrees of freedom parameter for Student-t distribution"""
        if hasattr(self, 'log_df'):
            return torch.exp(self.log_df)
        else:
            raise AttributeError("degrees of freedom parameter not available for normal distribution")

    def get_distribution(self):
        """Return the distribution type"""
        return self.distribution

    def stats(self):
        """Return a dictionary of statistics"""
        moments = self.moments()
        stats = {
            'distribution': self.distribution,
            'std': moments['variance'] ** 0.5,
            'skewness': 0.0,
            'kurtosis': moments['kurtosis'],  # Normal distribution has kurtosis of
            'theta': self.theta.item(),
        }

        return stats


class MAStudentEmission(MANormalEmission):
    """Student-$t$ MA(1) emission (specs/models.md §3.2).

    Thin specialisation of :class:`MANormalEmission` with the distribution
    hard-coded to ``'student_t'`` and ``theta`` defaulted to $0$ pinned --
    the canonical configuration used by the PSID Student-$t$ entry.
    """

    def __init__(self, sigma_eps=0.1, theta=0.0, fix_sigma=False, fix_theta=True,
                 df=5.0, fix_df=False):
        super().__init__(
            sigma_eps=sigma_eps, theta=theta,
            fix_sigma=fix_sigma, fix_theta=fix_theta,
            distribution='student_t', df=df, fix_df=fix_df,
        )

    def get_distribution(self) -> torch.distributions.Distribution:
        # Override the parent (which returns the string ``'student_t'``):
        # SMC-FFBS calls ``dist.log_prob(y - z)``, so it needs an actual
        # zero-mean residual distribution with ``sigma_eps`` baked into the
        # scale (matches ``MASinhEmission.get_distribution`` convention).
        device = self.log_sigma_eps.device
        return torch.distributions.StudentT(
            df=torch.exp(self.log_df),
            loc=torch.zeros(1, device=device),
            scale=torch.exp(self.log_sigma_eps),
        )


class MASinhEmissionHetero(Decoder):
    """Heteroscedastic sinh-arcsinh MA(1) decoder.

    The extra latent z_extra (per individual) enters the error either as:
      - hetero_mode="scale": log_sigma_i = z_extra, i.e. error sd is exp(z_extra).
        Likelihood uses residual / exp(z_extra) and a -log_sigma Jacobian per period.
      - hetero_mode="mean":  z_extra is a permanent additive shift on the error,
        with a single learnable log_sigma for the error sd.
        Likelihood uses (residual - z_extra) / exp(log_sigma) and -log_sigma per period.
    """

    def __init__(self, theta=0.0, fix_theta=False, sigma_eps=0.1, fix_sigma=False,
                 hetero_mode="scale", alpha_eps=0.0, fix_skew=True):
        super().__init__()

        if hetero_mode not in ("scale", "mean"):
            raise ValueError(f"hetero_mode must be 'scale' or 'mean', got {hetero_mode!r}")
        self.hetero_mode = hetero_mode

        self.log_beta = nn.Parameter(np.log(1.0) * torch.ones(1), requires_grad=True)
        self.theta = nn.Parameter(theta * torch.ones(1), requires_grad=not fix_theta)
        # Residual skewness of the sinh-arcsinh emission. Default 0 + frozen
        # preserves the previous behaviour (centred symmetric residual).
        # When fix_skew=False, log_likelihood recentres so the conditional
        # mean E[y_t | z_t, alpha_i] = z_t stays preserved at any alpha_eps.
        self.alpha_eps = nn.Parameter(alpha_eps * torch.ones(1),
                                       requires_grad=not fix_skew)

        # Scale parameter: only used when heterogeneity enters the mean (otherwise
        # z_extra plays the role of log_sigma directly, and the base scale is 1).
        if hetero_mode == "mean":
            self.log_sigma = nn.Parameter(np.log(sigma_eps) * torch.ones(1),
                                          requires_grad=not fix_sigma)

        self.stats_scale_var = 0.0
        self.stats_values = {'var': 0.0}

        self.hetero_net = False
        if self.hetero_net:
            self.net_log_sigma = nn.Sequential(nn.Linear(2, 1))

    def get_distribution(self) -> torch.distributions.Distribution:
        """Return the SinhArcsinh base distribution (loc=0, scale=1).

        ``skew=self.alpha_eps`` is 0 by default (frozen); when free, the
        log-likelihood path subtracts the closed-form mean so the
        conditional mean E[y_t | z_t, alpha_i] = z_t stays at z_t.
        """
        return SinhArcsinh(
            loc = torch.zeros(1, device=self.log_beta.device),
            scale = torch.ones(1, device=self.log_beta.device),
            skew = self.alpha_eps,
            tailweight = torch.exp(self.log_beta)
        )

    def _residual_mean(self) -> torch.Tensor:
        """E[W] for W ~ SinhArcsinh(0, 1, alpha_eps, exp(log_beta)).

        Uses the Jones & Pewsey (2009) decomposition
            E[sinh(delta * asinh(Z) + delta * eps)] = sinh(delta * eps) * P_delta,
        where P_delta = E[cosh(delta * asinh(Z))] for Z ~ N(0, 1). When
        alpha_eps = 0, sinh(0) = 0 makes this exactly 0 — so the
        ``fix_skew=True`` (default) path is bit-identical to the old
        no-recentring behaviour. P_delta is computed via 64-node quasi-
        Monte Carlo on symmetric normal quantiles (differentiable in
        log_beta).
        """
        tailweight = torch.exp(self.log_beta)
        device = self.log_beta.device
        dtype = self.log_beta.dtype
        n = 64
        base = torch.distributions.Normal(
            torch.zeros((), dtype=dtype, device=device),
            torch.ones((), dtype=dtype, device=device),
        )
        # Symmetric mid-quantile sample => P_delta is exact-to-FP-precision.
        u = torch.linspace(1.0 / (2 * n), 1.0 - 1.0 / (2 * n), n,
                           device=device, dtype=dtype)
        z = base.icdf(u)
        p_delta = torch.cosh(tailweight * torch.asinh(z)).mean(0)
        return torch.sinh(self.alpha_eps * tailweight) * p_delta

    def log_likelihood(self, y, z):
        """Compute joint log-likelihood using MA(1) Cholesky decomposition."""
        batch_size, T = y.shape

        # Split z into the time-indexed latent and the individual-level extra latent.
        z_latent, z_extra = split_latent(z, y.shape[1])

        if self.hetero_net:
            z1 = z_latent[:, 0].unsqueeze(1)
            eff = self.net_log_sigma(torch.cat((z1, z_extra), dim=1))
        else:
            eff = z_extra

        # Residuals depend on the mode: in "mean" mode, z_extra shifts the residual.
        if self.hetero_mode == "mean":
            residuals = y - z_latent - eff
            # Scalar log_sigma applies uniformly across batch and time; broadcast later.
            log_sigmas = self.log_sigma.expand_as(eff)
        else:
            residuals = y - z_latent
            log_sigmas = eff

        # Diagnostics: interpretation differs by mode (scale of errors vs. permanent shift).
        self.stats_scale_var = torch.mean(torch.exp(2 * log_sigmas.detach())).item()
        self.stats_values['var'] = torch.exp(2 * log_sigmas).mean().item()
        self.stats_values['scale_var'] = eff.var().item()
        self.stats_values['scale_mean'] = eff.mean().item()
        self.stats_values['scale_cov'] = ((z_latent[:, 0].unsqueeze(1) - z_latent[:, 0].unsqueeze(1).mean()) * (y - y.mean())).mean().item()

        dist = self.get_distribution()

        # Apply L^{-1} to residuals (MA(1) inverse): transformed_t = residuals_t - theta * transformed_{t-1}
        transformed_residuals = [residuals[:, 0]]
        for t in range(1, T):
            transformed_t = residuals[:, t] - self.theta * transformed_residuals[t-1]
            transformed_residuals.append(transformed_t)
        transformed_residuals = torch.stack(transformed_residuals, dim=1)

        # Mean-preserving recentring: under the model y_t = z_t + exp(alpha_i) * eps_t
        # with eps_t ~ centred sinh-arcsinh, the standardised residual W = eps_t + mu_W
        # is distributed as ``dist``. Adding mu_W here shifts the argument of log_prob
        # so the residual we model has mean 0 at every alpha_eps. mu_W = 0 exactly
        # when alpha_eps = 0, so the fix_skew=True default is unchanged.
        mu_W = self._residual_mean()
        log_prob_per_element = dist.log_prob(
            transformed_residuals / torch.exp(log_sigmas) + mu_W
        ) - log_sigmas
        log_likelihood = torch.sum(log_prob_per_element, dim=1)

        return log_likelihood

    def draw(self, z):
        dist = self.get_distribution()
        return dist.sample(z.shape)

    def get_eps_std(self):
        return 0.0

    def stats(self):
        """Return a dictionary of statistics"""

        eps = self.draw(torch.zeros(50000, 1, device=self.log_beta.device)).detach().cpu().numpy()

        out = {
            'mean': np.mean(eps).item(),
            'std': np.std(eps).item() * self.stats_scale_var**0.5,
            'skewness': skew(eps).item(),
            'kurtosis': kurtosis(eps).item(),
            'theta': self.theta.item(),
            'distribution': 'ma_sinh_arcsinh_normal',
            'hetero_mode': self.hetero_mode,
            'scale_var': self.stats_values['scale_var'],
            'scale_cov': self.stats_values['scale_cov'],
            'scale_mean': self.stats_values['scale_mean'],
        }
        if self.hetero_mode == "mean":
            out['log_sigma'] = self.log_sigma.item()
        return out