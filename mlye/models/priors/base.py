from abc import abstractmethod
from typing import Literal
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Distribution, Normal
from scipy.stats import skew

from ...utils import kurtosis


class Prior(nn.Module):
    nt: int

    def draw_eta0(self, n: int) -> torch.Tensor:
        """
        Draw samples from the prior distribution.
        eps: tensor of shape (batch_size, 1) for z1
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def log_prob(self, z):
        """
        Compute the log probability of the latent variable z under the prior distribution.
        z: tensor of shape (batch_size, latent_dim)
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    @abstractmethod
    def get_eta0(self) -> Distribution:
        NotImplementedError("This method should be implemented by subclasses.")
        pass

    @abstractmethod
    def log_prob_eta0(self, z) -> torch.Tensor:
        """
        Compute the log probability of the initial latent variable z0 under the prior distribution.
        z: tensor of shape (batch_size, 1)
        """
        NotImplementedError("This method should be implemented by subclasses.")
        return z

    def parameters_as_dict(self):
        """
        Return a dictionary of parameters of the prior.
        This can include mean, standard deviation, skewness, kurtosis, etc.
        """
        return self.state_dict()

    def gradient_norm(self):
        """
        Compute the gradient norm of the prior parameters.
        Returns a scalar tensor representing the gradient norm.
        """
        total_norm = 0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)  # L2 norm
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        return total_norm

    def draw_from_all_z(self,n):
        all_eta = []
        eta = self.draw_eta0(100_000)
        all_eta.append(eta.detach().numpy().flatten())

        for t in range(1, self.nt):
            mu, sigma = self.get_mu_sigma(eta)
            eta = mu + sigma * torch.randn_like(mu)
            all_eta.append(eta.detach().numpy().flatten())

        all_eta = np.concat(all_eta)
        return all_eta

    @abstractmethod
    def get_mu_sigma_detached(self, z) -> tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def get_mu_sigma(self, z) -> tuple[torch.Tensor, torch.Tensor]:
        pass

    # Default Gaussian transitions; non-Gaussian priors override.
    def transition_sample(self, z_prev: torch.Tensor) -> torch.Tensor:
        mu, sigma = self.get_mu_sigma(z_prev)
        return mu + sigma * torch.randn_like(mu)

    def transition_log_prob(self, z_cur: torch.Tensor,
                              z_prev: torch.Tensor) -> torch.Tensor:
        mu, sigma = self.get_mu_sigma(z_prev)
        return Normal(mu, sigma).log_prob(z_cur)

    def stats(self):
        """
        Return a dictionary of statistics for the prior.
        This can include mean, standard deviation, skewness, kurtosis, etc.
        """
        z1 = self.draw_eta0(50000).detach().cpu().numpy()
        return {
            'z1_mean': np.mean(z1).item(),
            'z1_std': np.std(z1).item(),
            'z1_skewness': skew(z1).item(),
            'z1_kurtosis': kurtosis(z1).item()
        }
