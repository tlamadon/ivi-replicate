import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import torch.profiler

from mlye.models.encoders.base import split_latent
from mlye.models.priors.utils import Polynomial

from .base import Prior


# ----------------------------------------
class AR1Prior(Prior):
    def __init__(self, nt=3, extra_heterogeneity: bool = False):
        super().__init__()
        self.nt = nt
        self.extra_heterogeneity = extra_heterogeneity
        self.regularize = 1e-3  # regularization term for the prior

        # z1 density (normal)
        self.mean    = nn.Parameter(torch.zeros(1), requires_grad=True)  # mean for numerical stability
        self.log_std = nn.Parameter(torch.zeros(1), requires_grad=True)  # log for numerical stability

        # zt | z_{t-1} density (normal)
        self.mu       = nn.Parameter(torch.zeros(1), requires_grad=True)  # mean for numerical stability
        self.rho       = nn.Parameter(torch.zeros(1), requires_grad=True)  # mean for numerical stability
        self.log_std_u = nn.Parameter(torch.zeros(1), requires_grad=True)  # log for numerical stability

        # we model the extra heterogeneity as a Normal conditional on N(mu(z1), sigma(z1))
        if self.extra_heterogeneity:
            self.net_extra_mu       = Polynomial(1)
            self.net_extra_logsigma = Polynomial(0)


    def get_eta0(self):
        return Normal(self.mean, F.softplus( self.log_std ))

    def draw_eta0(self, n:int) -> torch.Tensor:
        """
        Draw samples from the prior distribution.
        eps: tensor of shape (batch_size, 1) for z1
        """
        return Normal(self.mean, F.softplus( self.log_std )).sample((n,)).unsqueeze(1)

    """
        returns mu and sigma
        mu: mean of the distribution eta|eta_1
        sigma: standard deviation of the distribution eta|eta_1
    """
    def get_mu_sigma(self, z):
        return self.mu + self.rho * z, torch.exp(self.log_std_u) * torch.ones_like(z)

    def get_mu_sigma_detached(self, z):
        return (self.mu + self.rho * z).detach().cpu(), (torch.exp(self.log_std_u)* torch.ones_like(z)).detach().cpu()

    def log_prob(self, z):
        with torch.profiler.record_function("MARKOV logpr"):
            logp_total = Normal(self.mean, F.softplus( self.log_std )).log_prob(z[:, 0:1]).squeeze(1)

            # extra heterogeneity term
            if self.extra_heterogeneity:
                zz, ze = split_latent(z, self.nt)
                z1 = zz[:, 0:1]

                extra_mu    = self.net_extra_mu(z1)
                extra_sigma = torch.exp( self.net_extra_logsigma(z1) ) + self.regularize
                logp_extra  = Normal(extra_mu, extra_sigma).log_prob( ze ).squeeze(-1)
                logp_total += logp_extra
                
                # logp_total += Normal(0.0, 1.0).log_prob( ze.squeeze(-1) )  # Added log probability for standard normal

            for t in range(1,self.nt):
                # p(z2 | z1)
                z_cur = z[:, t:(t+1)]
                z_lag = z[:, (t-1):t]
                mu = self.mu + z_lag * self.rho
                sigma = torch.exp(self.log_std_u)
                logp2 = Normal(mu, sigma).log_prob(z_cur).squeeze(1)
                logp_total += logp2

        return logp_total


    def log_prob_eta0(self, z):
        """
        Compute the log probability of the initial latent variable z0 under the prior distribution.
        z: tensor of shape (batch_size, 1)
        """
        eta0 = self.get_eta0()
        return eta0.log_prob(z[:, 0:1]).squeeze(1)
