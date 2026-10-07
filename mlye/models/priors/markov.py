from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import torch.profiler

from mlye.models.encoders.base import split_latent

from .base import Prior
from .utils import Polynomial, skewnorm_moments_torch
from ...utils import SinhArcsinh, sinh_arcsinh_mean
import structlog

# ----------------------------------------
# Flexible Neural Prior:  p(z3|z2) * p(z2|z1) * p(z1)
# ----------------------------------------
class MarkovNormalConditionalPolyPrior(Prior):
    r"""Markov prior over a length-T latent path with optional extra-latent.

    Naming caveat (read me!):
        Several parameters are named ``*_log_std`` / ``log_sigma_*`` for
        historical reasons, but they DO NOT enter via ``exp(.)``.  The
        prior applies a *softplus* (``self.std_transform``) so that the
        actual standard deviation/scale fed to the downstream
        distribution is ``softplus(param)``, not ``exp(param)``.

        Concretely:
          * ``self.z1_log_std`` -> scale used by SinhArcsinh / Normal
            for z_1 is ``softplus(self.z1_log_std)``.
          * ``net_sigma`` polynomial output -> ``sigma(z) =
            softplus(net_sigma(z))``  (transition std).

        So to pin ``z_1`` to a SinhArcsinh with scale 0.34 (as in the
        canonical simulator preset, ``EtaDensityConfig("symetric")``),
        set ``self.z1_log_std`` to ``log(exp(0.34) - 1) ~= -0.904``,
        NOT to ``log(0.34) ~= -1.079``.

        Likewise, ``self.z1_log_tail`` does pass through ``exp(.)``,
        so ``self.z1_log_tail = log(tail)``.

        The misleading names are kept for backwards compatibility with
        cached state dicts and prior reports.
    """

    def __init__(self,
                 nt=3,
                 hidden_dim=32,
                 poly_degree=0,
                 z1_distr:Literal['normal', 'sinh'] = 'normal',
                 law_model:Literal['mlp', 'poly','softplus', 'ar1'] = 'poly',
                 extra_heterogeneity:bool = False,
                 extra_prior_type:Literal['iid', 'ar1_t'] = 'iid',
                 sigma_clamp: float = 10.0,
                 mu_clamp: float = 10.0,
                 skewed=False):

        super().__init__()
        self.nt = nt
        self.skewed = skewed
        self.max_skew = 1.5  # maximum skewness for the skewed normal distribution
        self.regularize = 1e-3  # regularization term for the prior
        self.z1_distr = z1_distr
        self.law_model = law_model
        self.sigma_clamp = sigma_clamp
        self.mu_clamp = mu_clamp
        self.extra_heterogeneity = extra_heterogeneity
        self.extra_prior_type = extra_prior_type

        self.logger = structlog.get_logger("MarkovNormalConditionalPolyPrior")

        # Parametrizing Law of Motion
        # --------------------------------
        # NOTE: std_transform is softplus (NOT exp).  Parameters named
        # *_log_std / log_sigma_* are pre-softplus; the actual std is
        # softplus(param).  See class docstring for inversion formulas.
        self.std_transform = F.softplus
        self.mu_transform  = lambda t: t  # identity by default; overridden for 'softplus' law

        if self.skewed:
            self.net_omega = Polynomial(2, init_scale=0.05)  # omega for skewed normal

        # we model the extra heterogeneity as a Normal conditional on N(mu(z1), sigma(z1))
        if self.extra_heterogeneity:
            if self.extra_prior_type == 'iid':
                # Single per-individual extra latent, regressed on z1.
                self.net_extra_mu       = Polynomial(1)
                self.net_extra_logsigma = Polynomial(0)
            elif self.extra_prior_type == 'ar1_t':
                # T per-period extra latents (interpreted as log-volatility),
                # AR(1) prior:  v_1 ~ N(0, exp(extra_v0_log_std))
                #               v_t = rho_v * v_{t-1} + N(0, exp(extra_v_log_std))
                # rho_v parameterised through tanh to keep |rho_v| < 1.
                self.extra_v_logit_rho = nn.Parameter(torch.zeros(1), requires_grad=True)
                self.extra_v_log_std   = nn.Parameter(torch.full((1,), -1.0), requires_grad=True)
                self.extra_v0_log_std  = nn.Parameter(torch.full((1,), -1.0), requires_grad=True)
            else:
                raise ValueError(f"Unsupported extra_prior_type: {self.extra_prior_type}")

        if law_model == 'mlp':
            self.net_mu = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1)  # mu
            )
            self.net_sigma = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1)  # log_sigma
            )
            with torch.no_grad():
                # Initialize weights and biases small
                for m in self.net_sigma:
                    if isinstance(m, nn.Linear):
                        nn.init.normal_(m.weight, mean=0.0, std=1e-3)  # very small weights
                        nn.init.constant_(m.bias, 0.0)                  # zero bias

        elif law_model == 'poly':
            print(f"Using polynomial prior with degree {poly_degree}")
            self.net_mu = Polynomial(poly_degree)
            self.net_sigma = Polynomial(poly_degree)

        elif law_model == 'softplus':
            print(f"Using softplus prior")
            self.net_mu = Polynomial(poly_degree)
            self.net_sigma = Polynomial(poly_degree)
            self.b = nn.Parameter(torch.zeros(1), requires_grad=True)  # bias for the softplus
            self.log_gamma = nn.Parameter(torch.zeros(1), requires_grad=True)  # scale for the softplus
            self.mu_transform = lambda t: self.b + F.softplus( (t - self.b) * torch.exp( -self.log_gamma ) ) * torch.exp(self.log_gamma)  # softplus transformation

        elif law_model == 'ar1':
            print(f"Using AR(1) prior")
            self.net_mu = Polynomial(1)
            self.net_sigma = Polynomial(2)

        else:
            raise ValueError(f"Unsupported law of motion: {law_model}")

        # Parametrizing Initial Distribution
        # ----------------------------------

        if z1_distr == 'sinh':
            # NOTE: pre-softplus; SinhArcsinh scale = softplus(z1_log_std).
            # To pin scale = s set z1_log_std = log(exp(s) - 1).
            self.z1_log_std = nn.Parameter(torch.zeros(1), requires_grad=True)
            self.z1_skew    = nn.Parameter(torch.zeros(1), requires_grad=True)
            # log_tail uses exp(.): SinhArcsinh tailweight = exp(z1_log_tail).
            self.z1_log_tail= nn.Parameter(torch.zeros(1), requires_grad=True)

        elif z1_distr == 'normal':
            # NOTE: pre-softplus; Normal std = softplus(log_std).
            # To pin std = s set log_std = log(exp(s) - 1).
            self.log_std = nn.Parameter(torch.zeros(1), requires_grad=True)

        else:
            raise ValueError(f"Unsupported z1_distr: {z1_distr}")

    def epoch_logging(self):
        self.logger.info("Prior parameter values ")

    def get_eta0(self) -> torch.distributions.Distribution:
        if self.z1_distr == 'sinh':
            # Mean-preserving recentring: under SinhArcsinh(loc, scale,
            # skew, tail), E[z_1] = loc + scale * sinh(skew * tail) * P_tail
            # (Jones & Pewsey 2009; see mlye.utils.sinh_arcsinh_mean).
            # Setting loc = -mu_z1 pins E[z_1] = 0 at every (skew, tail),
            # parallel to the decoder's residual recentring. When
            # z1_skew = 0, sinh(0) = 0 makes mu_z1 = 0 to FP precision,
            # so existing fits that pin z1_skew = 0 are bit-identical.
            scale = self.std_transform(self.z1_log_std)
            tail = torch.exp(self.z1_log_tail)
            mu_z1 = sinh_arcsinh_mean(scale, self.z1_skew, tail)
            return SinhArcsinh(
                -mu_z1,
                scale,
                self.z1_skew,
                tail,
            )
        elif self.z1_distr == 'normal':
            return Normal(torch.zeros(1, device=self.log_std.device), self.std_transform( self.log_std ))
        else:
            raise ValueError(f"Unsupported z1_distr: {self.z1_distr}")

    def log_prob_eta0(self, z):
        """
        Compute the log probability of the initial latent variable z0 under the prior distribution.
        z: tensor of shape (batch_size, 1)
        """
        eta0 = self.get_eta0()
        return eta0.log_prob(z[:, 0:1]).squeeze(1)

    def draw_eta0(self, n:int) -> torch.Tensor:
        """
        Draw samples from the prior distribution.
        eps: tensor of shape (batch_size, 1) for z1
        """
        eta0 = self.get_eta0()
        return eta0.sample((n,)).unsqueeze(1)

    """
        returns mu and sigma
        mu: mean of the distribution eta|eta_1
        sigma: standard deviation of the distribution eta|eta_1
    """
    def get_mu_sigma(self, z:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.mu_transform(self.net_mu(z)).clamp(-self.mu_clamp,self.mu_clamp), self.std_transform(self.net_sigma(z)).clamp(0,self.mu_clamp) + self.regularize

    def get_mu_sigma_detached(self, z):
        mu,sigma = self.get_mu_sigma(z)
        return mu.detach().cpu(), sigma.detach().cpu()

    def get_moments(self, z) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, sigma = self.get_mu_sigma(z)
        skewness = torch.Tensor(0)
        kurtosis = torch.Tensor(0)

        if self.skewed:
            omega = self.net_omega(z)
            mu, sigma, skewness, kurtosis = skewnorm_moments_torch(mu, sigma, omega)
            return mu, sigma, skewness, kurtosis
        else:
            return mu, sigma, skewness, kurtosis

    def log_prob(self, z):
        with torch.profiler.record_function("MARKOV logpr"):
            logp_total = self.log_prob_eta0(z)

            # extra heterogeneity term
            if self.extra_heterogeneity:
                zz, ze = split_latent(z, self.nt)
                if self.extra_prior_type == 'iid':
                    z1 = zz[:, 0:1]
                    extra_mu    = self.net_extra_mu(z1)
                    extra_sigma = torch.exp( self.net_extra_logsigma(z1) ) + self.regularize
                    logp_extra  = Normal(extra_mu, extra_sigma).log_prob( ze ).squeeze(1)
                    logp_total += logp_extra
                elif self.extra_prior_type == 'ar1_t':
                    # ze: (batch, T) — interpret as v_{1:T} log-volatility sequence.
                    rho_v = torch.tanh(self.extra_v_logit_rho)
                    sigma_v0 = torch.exp(self.extra_v0_log_std) + self.regularize
                    sigma_v  = torch.exp(self.extra_v_log_std)  + self.regularize
                    # v_1 ~ N(0, sigma_v0)
                    logp_v = Normal(torch.zeros_like(ze[:, 0]), sigma_v0).log_prob(ze[:, 0])
                    # v_t | v_{t-1} ~ N(rho_v * v_{t-1}, sigma_v)  for t=2..T
                    if ze.shape[1] > 1:
                        means = rho_v * ze[:, :-1]
                        logp_v_rest = Normal(means, sigma_v).log_prob(ze[:, 1:])
                        logp_v = logp_v + logp_v_rest.sum(dim=1)
                    logp_total += logp_v

                # this should be simple heterogeneity
                # logp_total += Normal(0.0, 1.0).log_prob(ze)  # Added log probability for standard normal

            for t in range(1,self.nt):
                # p(z2 | z1)
                z_cur = z[:, t:(t+1)]
                z_lag = z[:, (t-1):t]
                mu, sigma = self.get_mu_sigma(z_lag)
                mu   = torch.nan_to_num(mu,   nan=0.0, posinf=1e6, neginf=-1e6)
                logp2 = Normal(mu, sigma).log_prob(z_cur).squeeze(1)

                # we can add a skewed distribution here, using the skewed normal
                if self.skewed:
                    omega = self.net_omega(z_lag)
                    logp2 += torch.special.log_ndtr( omega * (z_cur - mu)/sigma ).squeeze(1)  # skewed log probability

                logp_total += logp2

        return logp_total
