import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from ...utils import evalSinhArcsinhNormal
from .base import Encoder, EncoderConfig
from .layers import MaskedLinear


class ConditionalNormalMarkovPosterior(Encoder):
    def __init__(self, dim:int, hidden_dim:int = 32, regularize=1e-3):
        super().__init__()

        self.dim = dim
        self.hidden_dim = hidden_dim
        self.regularize = regularize

        # mean and variance for the first z_1 - 1D input y_1 - instead we use all y
        self.fc0 = nn.Linear(dim, self.hidden_dim)
        self.mu0 = nn.Linear(self.hidden_dim, 1)        # dim means
        self.log_sigma0 = nn.Linear(self.hidden_dim, 1) # dim means

        # mean and variance for later z (t >=2 ), 2D inputs z_{t-1}, y_t
        self.fc1 = MaskedLinear(1 + dim, self.hidden_dim)
        # self.fc1 = nn.Linear(1 + dim + 1, self.hidden_dim)
        self.mu1        = nn.Linear(self.hidden_dim, dim - 1) # dim means
        self.log_sigma1 = nn.Linear(self.hidden_dim, dim - 1) # dim means

        # allowing for an MA term
        # self.ma_coef = nn.Parameter(torch.zeros(1)) 
        self.mean_var = torch.zeros(1)  # Placeholder for mean variance of posterior

    def get_seed_dim(self) -> int:
        """Return the dimension of the seed noise input required."""
        return self.dim

    def create_mask(self, y, t, pad:int = 0):

        T = y.size(1)  # number of time steps

        # we create mask that only cinludes the future
        # Create a mask of shape (T,) where True if t >= p
        col_mask = torch.arange(y.size(1), device=y.device) >= t  # shape (T,)

        # we add one column for z_{t-1} and pads
        # concat it with a vector of True values of shape (T,)
        col_mask = torch.cat([col_mask.bool(), torch.ones(1 + pad, device=y.device, dtype=torch.bool)])

        # Expand to (N, T) by repeating for each row
        # mask = col_mask.unsqueeze(0).expand(y.size(0), T + 1)

        return col_mask        

    # we return the transformed draws as well its log-probability
    def draw_and_logprob(self, y, u, logpr_draw=False):

        z         = torch.zeros(y.size(0), self.dim, device=y.device)
        log_prob  = torch.zeros(y.size(0), device=y.device)

        h0        = F.softplus(self.fc0(y))
        mu0       = self.mu0(h0)
        sigma0    = F.softplus(self.log_sigma0(h0)) #+ self.regularize

        z[:,0:1]  = mu0 + sigma0 * u[:,0:1]
        log_prob  = log_prob + Normal(mu0, sigma0).log_prob(z[:,0:1]).squeeze()
        #log_prob  = log_prob - Normal(mu0, sigma0).entropy().squeeze()
        y_mean = torch.mean(y, dim=1, keepdim=True)
        # resid = y[:,0:1] - z[:,0:1]

        for t in range(1, self.dim):

            z_t_lag = z[:, (t-1):t]
            mask = self.create_mask(y, t)
            inputs = torch.cat( [y, z_t_lag], dim=1)
            
            # inputs    = torch.cat((z_t_lag, y[:, t:t+1], y_mean), dim=1)
            # inputs    = torch.cat((z_t_lag, y, resid), dim=1)

            tmp = self.fc1(inputs, mask)
            # tmp = self.fc1(inputs)
            # h1        = F.relu(tmp)
            h1        = F.softplus(tmp)
            mu        = self.mu1(h1)[:,(t-1):t]
            log_sigma = self.log_sigma1(h1)[:,(t-1):t]
            sigma     = F.softplus(log_sigma) + self.regularize
            z[:, t:t+1] = mu + sigma * u[:, t:t+1]            
            log_prob  = log_prob + Normal(mu, sigma).log_prob(z[:, t:t+1]).squeeze()
            # log_prob  = log_prob - torch.log(sigma)
            # log_prob  = log_prob - Normal(mu, sigma).entropy().squeeze()
            # resid = y[:,t:t+1] - z[:,t:t+1] - self.ma_coef * resid

            # saving posterior variance for stats
            self.mean_var = sigma.pow(2).mean()

        return z, log_prob

    def stats(self) -> dict:
        return {
            'encoder_type': 'ConditionalNormalMarkovPosterior',
            'dim': self.dim,
            'regularize': self.regularize,
            'mean_var': self.mean_var.item()
        }

    @classmethod
    def from_config(cls, config: EncoderConfig) -> 'ConditionalNormalMarkovPosterior':
        """Create instance from configuration."""
        return cls(
            dim=config.dim,
            hidden_dim=getattr(config, 'hidden_dim', 32),
            regularize=getattr(config, 'regularize', 1e-3)
        )


class ConditionalNormalMarkovMAPosterior(ConditionalNormalMarkovPosterior):
    def __init__(self, dim:int, hidden_dim:int = 32, regularize=1e-3):
        super().__init__(dim=dim, hidden_dim=hidden_dim, regularize=regularize)

        # allowing for an MA term
        self.fc1 = MaskedLinear(2 + dim, self.hidden_dim) 
        self.ma_coef = nn.Parameter(torch.zeros(1)) 


    # we return the transformed draws as well its log-probability
    def draw_and_logprob(self, y, u, logpr_draw=False):

        z         = torch.zeros(y.size(0), self.dim, device=y.device)
        log_prob  = torch.zeros(y.size(0), device=y.device)

        h0        = F.relu(self.fc0(y))
        mu0       = self.mu0(h0)
        sigma0    = F.softplus(self.log_sigma0(h0)) #+ self.regularize

        z[:,0:1]  = mu0 + sigma0 * u[:,0:1]
        log_prob  = log_prob + Normal(mu0, sigma0).log_prob(z[:,0:1]).squeeze()
        #log_prob  = log_prob - Normal(mu0, sigma0).entropy().squeeze()
        resid = y[:,0:1] - z[:,0:1]

        for t in range(1, self.dim):

            z_t_lag = z[:, (t-1):t]
            mask = self.create_mask(y, t, pad=1)
            inputs = torch.cat( [y, z_t_lag, resid], dim=1)
            
            tmp = self.fc1(inputs, mask)
            h1        = F.relu(tmp)
            mu        = self.mu1(h1)[:,(t-1):t]
            log_sigma = self.log_sigma1(h1)[:,(t-1):t]
            sigma     = F.softplus(log_sigma) #+ self.regularize
            z[:, t:t+1] = mu + sigma * u[:, t:t+1]            
            log_prob  = log_prob + Normal(mu, sigma).log_prob(z[:, t:t+1]).squeeze()
            resid = y[:,t:t+1] - z[:,t:t+1] - self.ma_coef * resid

            # saving posterior variance for stats
            self.mean_var = sigma.pow(2).mean()

        return z, log_prob

    def stats(self) -> dict:
        return {
            'encoder_type': 'ConditionalNormalMarkovMAPosterior',
            'dim': self.dim,
            'regularize': self.regularize,
            'mean_var': self.mean_var.item()
        }

    @classmethod
    def from_config(cls, config: EncoderConfig) -> 'ConditionalNormalMarkovMAPosterior':
        """Create instance from configuration."""
        return cls(
            dim=config.dim,
            hidden_dim=getattr(config, 'hidden_dim', 32),
            regularize=getattr(config, 'regularize', 1e-3)
        )


class ConditionalSinhMarkovPosterior(Encoder):
    def __init__(self, dim:int, hidden_dim:int = 32, regularize=1e-3):
        super().__init__()

        self.dim = dim
        self.hidden_dim = hidden_dim
        self.regularize = regularize

        # mean and variance for the first z_1 - 1D input y_1 - instead we use all y
        self.fc0 = nn.Linear(dim, self.hidden_dim)

        # mean and variance for later z (t >=2 ), 2D inputs z_{t-1}, y_t
        self.fc1 = MaskedLinear(1 + dim, self.hidden_dim)

        # all parameters
        self.mu1 = nn.Linear(self.hidden_dim, dim) # dim means
        self.log_sigma1 = nn.Linear(self.hidden_dim, dim) # dim means

        # skewness and kurtosis using sinh-arcsinh transformation
        self.alpha1 = nn.Linear(self.hidden_dim, dim)  # skew
        self.beta1 = nn.Linear(self.hidden_dim, dim)   # kurtosis

        # allowing for an MA term
        # self.ma_coef = nn.Parameter(torch.zeros(1)) 

        self.mean_var = torch.zeros(1)  # Placeholder for mean variance of posterior

        with torch.no_grad():
            self.alpha1.weight *= 0.01  # shrink weights
            if self.alpha1.bias is not None:
                self.alpha1.bias.data *= 0.01
            self.beta1.weight *= 0.01  # shrink weights
            if self.beta1.bias is not None:
                self.beta1.bias.data *= 0.01                

    def get_seed_dim(self) -> int:
        """Return the dimension of the seed noise input required."""
        return self.dim

    def create_mask(self, y, t):

        T = y.size(1)  # number of time steps

        # Create a mask of shape (T,) where True if t >= p
        col_mask = torch.arange(y.size(1), device=y.device) >= t  # shape (T,)

        # concat it with a vector of True values of shape (T,)
        col_mask = torch.cat([col_mask.bool(), torch.ones(1, device=y.device, dtype=torch.bool)])

        # Expand to (N, T) by repeating for each row
        # mask = col_mask.unsqueeze(0).expand(y.size(0), T + 1)

        return col_mask        

    # we return the transformed draws as well its log-probability
    def draw_and_logprob(self, y, u, logpr_draw=False):

        z         = torch.zeros(y.size(0), self.dim, device=y.device)
        log_prob  = torch.zeros(y.size(0), device=y.device)

        h0        = F.relu(self.fc0(y))
        mu0       = self.mu1(h0)[:,0:1]
        sigma0    = F.softplus(self.log_sigma1(h0))[:,0:1] #+ self.regularize
        alpha0    = self.alpha1(h0)[:,0:1]
        beta0     = torch.exp(self.beta1(h0))[:,0:1]

        # we draw the first z_0
        z0, log_prob = evalSinhArcsinhNormal(mu0, sigma0, alpha0, beta0, u[:,0:1])
        z[:,0:1]  = z0
        log_prob = log_prob.squeeze()

        # we draw all the following z_t
        for t in range(1, self.dim):
            
            z_t_lag = z[:, (t-1):t]
            mask = self.create_mask(y, t)
            inputs = torch.cat( [y, z_t_lag], dim=1)
            
            tmp = self.fc1(inputs, mask)
            h1        = F.relu(tmp)
            mu        = self.mu1(h1)[:,t:t+1]
            log_sigma = self.log_sigma1(h1)[:,t:t+1]
            sigma     = F.softplus(log_sigma) + self.regularize
            alpha     = self.alpha1(h1)[:,t:t+1] 
            beta      = torch.exp(self.beta1(h1)[:,t:t+1]) + self.regularize

            z[:, t:t+1], log_prob2 = evalSinhArcsinhNormal(mu, sigma, alpha, beta, u[:, t:t+1])
            log_prob  = log_prob + log_prob2.squeeze()

            # saving posterior variance for stats
            self.mean_var = sigma.pow(2).mean()

        return z, log_prob

    def stats(self) -> dict:
        return {
            'encoder_type': 'ConditionalSinhMarkovPosterior',
            'dim': self.dim,
            'regularize': self.regularize,
            'mean_var': self.mean_var.item()
        }

    @classmethod
    def from_config(cls, config: EncoderConfig) -> 'ConditionalSinhMarkovPosterior':
        """Create instance from configuration."""
        return cls(
            dim=config.dim,
            hidden_dim=getattr(config, 'hidden_dim', 32),
            regularize=getattr(config, 'regularize', 1e-3)
        )