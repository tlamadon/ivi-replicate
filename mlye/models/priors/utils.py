import torch
import torch.nn as nn


class Polynomial(nn.Module):
    def __init__(self, degree, init_scale=0.1):
        super().__init__()
        # Learnable coefficients a_0, a_1, ..., a_degree
        self.coeffs = nn.Parameter(init_scale * torch.randn(degree + 1), requires_grad=True)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        result = torch.zeros_like(x)
        for i, coeff in enumerate(self.coeffs):
            result += coeff * x**i
        return result


def skewnorm_moments_torch(xi, omega, alpha):
    """
    Compute elementwise mean, std, skewness, and excess kurtosis of skew-normal distribution in PyTorch.

    Parameters:
    - xi: location (tensor)
    - omega: scale (tensor, must be > 0)
    - alpha: shape (tensor)

    Returns:
    - mean, std, skewness, kurtosis (tensors)
    """
    delta = alpha / torch.sqrt(1 + alpha**2)
    sqrt2_over_pi = torch.sqrt(torch.tensor(2.0 / torch.pi, dtype=xi.dtype, device=xi.device))

    # Mean
    mean = xi + omega * delta * sqrt2_over_pi

    # Variance and std
    var = omega**2 * (1 - (2 * delta**2) / torch.pi)
    std = torch.sqrt(var)

    # Skewness
    skewness = ((4 - torch.pi) / 2) * ((delta * sqrt2_over_pi) ** 3) / (1 - (2 * delta**2) / torch.pi)**(3/2)

    # Excess kurtosis
    kurtosis = 2 * (torch.pi - 3) * (delta * sqrt2_over_pi)**4 / (1 - (2 * delta**2) / torch.pi)**2

    return mean, std, skewness, kurtosis
