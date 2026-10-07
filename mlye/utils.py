import numpy as np
import torch
import torch.distributions.constraints
import torch.nn as nn
from scipy import stats

import torch.nn.functional as F
from torch.distributions import Distribution
from torch.distributions.utils import broadcast_all
from math import gamma, lgamma, pi

def printGradNorm(module: nn.Module, str: str):
    total_norm = 0
    for p in module.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)  # L2 norm
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    #print(f"{str}: {total_norm:.6f}")
    return(total_norm)


def skew_normal_skewness(alpha):
    delta = alpha / np.sqrt(1 + alpha**2)
    sqrt_2_over_pi = np.sqrt(2 / np.pi)
    numerator = (4 - np.pi) / 2 * (delta * sqrt_2_over_pi)
    denominator = (1 - 2 * delta**2 / np.pi) ** (3/2)
    return numerator / denominator


def assert_no_nan_in_parameters(model, model_name: str = "model"):
    for name, param in model.named_parameters():
        if param.requires_grad:
            if torch.isnan(param).any():
                raise ValueError(f"NaN detected in parameter: {name}")
            if torch.isinf(param).any():
                raise ValueError(f"Inf detected in parameter: {name}")


def assert_no_nan_in_gradients(model, model_name: str = "model"):
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                raise ValueError(f"NaN detected in gradient of: {name}")
            if torch.isinf(param.grad).any():
                raise ValueError(f"Inf detected in gradient of: {name}")

def kurtosis(x):
    return stats.kurtosis(x, bias=False, fisher=False)

def skewnorm_moments(xi, omega, alpha):
    """
    Compute the mean, std deviation, skewness, and kurtosis of a skew-normal distribution.

    Parameters:
    - xi: location parameter
    - omega: scale parameter (must be > 0)
    - alpha: shape (skewness) parameter

    Returns:
    - mean, std_dev, skewness, excess_kurtosis
    """
    delta = alpha / np.sqrt(1 + alpha**2)
    sqrt2_over_pi = np.sqrt(2 / np.pi)

    # Mean
    mean = xi + omega * delta * sqrt2_over_pi

    # Variance and standard deviation
    var = omega**2 * (1 - (2 * delta**2) / np.pi)
    std_dev = np.sqrt(var)

    # Skewness
    skewness = ((4 - np.pi) / 2) * ((delta * sqrt2_over_pi) ** 3) / (1 - (2 * delta**2) / np.pi)**(3/2)

    # Excess Kurtosis
    kurtosis = 2 * (np.pi - 3) * (delta * sqrt2_over_pi)**4 / (1 - (2 * delta**2) / np.pi)**2

    return mean, std_dev, skewness, kurtosis


class GeneralizedNormal(Distribution):
    @property
    def support(self):
        return torch.distributions.constraints.real
    has_rsample = True  # allows reparameterization
    
    def __init__(self, loc, scale, beta, validate_args=None):
        self.loc, self.scale, self.beta = broadcast_all(loc, scale, beta)
        super().__init__(self.loc.size(), validate_args=validate_args)

    def sample(self, sample_shape=torch.Size()):
        u = torch.randn(sample_shape + self.loc.shape, device=self.loc.device)
        return self.loc + self.scale * torch.sign(u) * torch.abs(u)**(1/self.beta)

    def rsample(self, sample_shape=torch.Size()):
        # approximate reparameterization
        return self.sample(sample_shape)

    def log_prob(self, value):
        z = (value - self.loc) / self.scale
        norm_const = self.beta / (2 * self.scale * torch.exp(torch.lgamma(1 / self.beta)))
        return torch.log(norm_const) - torch.abs(z) ** self.beta

    def entropy(self):
        return (1 / self.beta) - torch.log(self.beta / (2 * self.scale)) + torch.lgamma(1 / self.beta)


import torch
from torch.distributions import Normal, TransformedDistribution, constraints
from torch.distributions.transforms import Transform
import torch
from torch.distributions import (
    Normal,
    TransformedDistribution,
    Transform,
    constraints,
    AffineTransform
)
import math

# class SinhArcsinhTransform(Transform):
#     """
#     y = sinh((asinh(x) + epsilon) / delta)
#     inverse: x = sinh(delta * asinh(y) - epsilon)
#     """
#     domain = constraints.real
#     codomain = constraints.real
#     bijective = True
#     sign = +1

#     def __init__(self, 
#                  epsilon:float|torch.Tensor=0.0, 
#                  delta:float|torch.Tensor=1.0, 
#                  cache_size=1):
#         super().__init__(cache_size=cache_size)
#         self.epsilon = epsilon
#         self.delta = delta

#     # def __eq__(self, other):
#     #     return isinstance(other, SinhArcsinhTransform) and \
#     #            self.epsilon == other.epsilon and \
#     #            self.delta == other.delta

#     def _call(self, x):
#         return torch.sinh((torch.asinh(x) + self.epsilon) / self.delta)

#     def _inverse(self, y):
#         return torch.sinh(self.delta * torch.asinh(y) - self.epsilon)

#     def log_abs_det_jacobian(self, x, y):
#         # ∂y/∂x = cosh((asinh(x)+ε)/δ) / (δ * sqrt(1 + x^2))
#         z = torch.asinh(x)
#         inner = (z + self.epsilon) / self.delta
#         return torch.log(torch.cosh(inner)) - torch.log(self.delta) - 0.5 * torch.log1p(x**2)


# def SinhArcsinhNormal(loc, scale, skewness, tailweight):
#     base = Normal(loc, scale)
#     transform = SinhArcsinhTransform(skewness, tailweight)
#     return TransformedDistribution(base, [transform])

# class SinhArcsinhNormal(TransformedDistribution):
#     def __init__(self, 
#                  loc:float|torch.Tensor=0.0, 
#                  scale:float|torch.Tensor=1.0, 
#                  epsilon:float|torch.Tensor=0.0, 
#                  delta:float|torch.Tensor=1.0, 
#                  validate_args=None):
#         device = getattr(scale, "device", torch.device("cpu"))
#         base_dist = Normal(torch.tensor(0.0, device=device), torch.tensor(1.0, device=device))
#         transforms = [
#             SinhArcsinhTransform(epsilon=epsilon, delta=delta),
#             AffineTransform(loc, scale)
#         ]
#         super().__init__(base_dist, transforms, validate_args=validate_args)
#         self.loc = loc
#         self.scale = scale
#         self.epsilon = epsilon
#         self.delta = delta
    
#     def rvs(self, num_samples: int):
#         return self.rsample(torch.Size([num_samples])).detach().cpu().numpy().astype(np.float64)

def evalSinhArcsinhNormal(loc, scale, skewness, tailweight, u, include_base=True):
    """
    Applies the rule forward and computes the log probability of the input z
    
    Parameters:
    - loc: location parameter
    - scale: scale parameter
    - skewness: skewness parameter (epsilon)
    - tailweight: tail weight parameter (delta)
    - num_samples: number of samples to generate for evaluation
    
    Returns:
    - samples: generated samples from the distribution
    - log_prob contribution
    """
    # Convert parameters to tensors to ensure proper broadcasting and operations
    loc = torch.as_tensor(loc, dtype=u.dtype, device=u.device)
    scale = torch.as_tensor(scale, dtype=u.dtype, device=u.device)
    skewness = torch.as_tensor(skewness, dtype=u.dtype, device=u.device)
    tailweight = torch.as_tensor(tailweight, dtype=u.dtype, device=u.device)

    # Apply the transformation
    z2 = loc + scale * torch.sinh( tailweight * (torch.arcsinh( u ) + skewness ))

    # Compute the log probability contribution
    # ∂z/∂u = scale * cosh(tailweight * (asinh(u) + skewness)) * tailweight / sqrt(1 + u^2)
    log_qz = -torch.log(scale) \
             - torch.log(torch.cosh( tailweight * (torch.arcsinh(u) + skewness))) \
             - torch.log(tailweight) + 0.5 * torch.log(1 + u**2) 

    # add constribution of the base distribution
    if include_base:
        log_qz += Normal(0, 1).log_prob(u)    

    return(z2, log_qz)

def trim_outliers(data, factor=3):
    """
    Trims the values that are more than `factor` times the standard deviation away from the mean.
    
    Args:
        data (numpy.ndarray): Input data array.
        factor (float): The number of standard deviations to use as a threshold. Defaults to 3.
        
    Returns:
        numpy.ndarray: The data with outliers removed.
    """
    # Calculate mean and standard deviation
    mean = np.mean(data)
    std_dev = np.std(data)
    
    # Calculate z-scores and filter out values that are beyond the threshold
    z_scores = (data - mean) / std_dev
    trimmed_data = data[np.abs(z_scores) <= factor]
    
    return trimmed_data


def trim_outliers_tensor(data, factor=3):
    """
    Trims the values that are more than `factor` times the standard deviation away from the mean in a PyTorch tensor.
    
    Args:
        data (torch.Tensor): Input tensor.
        factor (float): The number of standard deviations to use as a threshold. Defaults to 3.
        
    Returns:
        torch.Tensor: The tensor with outliers removed.
    """
    # Calculate mean and standard deviation
    mean = torch.mean(data)
    std_dev = torch.std(data)
    
    # Calculate z-scores
    z_scores = (data - mean) / std_dev
    
    # Trim the outliers
    trimmed_data = data[torch.abs(z_scores) <= factor]
    
    return trimmed_data


import torch
from torch import Tensor
from torch.distributions import Distribution, constraints
from torch.distributions.utils import broadcast_all
from torch.distributions.normal import Normal
from collections import namedtuple

def _log_cosh(x: Tensor) -> Tensor:
    # Stable log(cosh(x)) = |x| + log1p(exp(-2|x|)) - log(2)
    ax = x.abs()
    return ax + torch.log1p(torch.exp(-2 * ax)) - torch.log(torch.tensor(2.0, dtype=x.dtype, device=x.device))


def sinh_arcsinh_mean(scale: Tensor, skew: Tensor, tailweight: Tensor,
                       n_quad: int = 64) -> Tensor:
    """E[Y] for Y ~ SinhArcsinh(loc=0, scale, skew, tailweight).

    Jones & Pewsey (2009) decomposition:
        E[scale * sinh(τ * asinh(Z) + τ * α)]
            = scale * sinh(τ * α) * P_τ,
    with P_τ = E[cosh(τ * asinh(Z))] for Z ~ N(0, 1). When skew = 0,
    sinh(0) = 0 makes this exactly 0 — so passing skew=0 returns 0 to
    FP precision and is a bit-identical no-op for centred cases.

    Implementation: n_quad-node symmetric mid-quantile QMC for P_τ,
    fully differentiable in (scale, skew, tailweight).

    Same recipe as `mlye/models/decoders/ma.py::_residual_mean` so the
    z1-prior and emission-decoder recentrings stay numerically aligned.
    """
    device = tailweight.device
    dtype = tailweight.dtype
    base = Normal(
        torch.zeros((), dtype=dtype, device=device),
        torch.ones((), dtype=dtype, device=device),
    )
    u = torch.linspace(1.0 / (2 * n_quad), 1.0 - 1.0 / (2 * n_quad),
                        n_quad, device=device, dtype=dtype)
    z = base.icdf(u)
    p_tau = torch.cosh(tailweight * torch.asinh(z)).mean(0)
    return scale * torch.sinh(skew * tailweight) * p_tau


class SinhArcsinh(Distribution):
    """
    y = loc + scale * sinh((asinh(z) + skew) * tailweight),  z ~ N(0,1)
    """
    @property
    def arg_constraints(self):
        return {
            "loc": constraints.real,
            "scale": constraints.positive,
            "skew": constraints.real,
            "tailweight": constraints.positive,
        }
    @property
    def support(self):
        return constraints.real
    has_rsample = True

    def __init__(self, loc: Tensor = torch.zeros(1), scale: Tensor = torch.ones(1), skew: Tensor = torch.zeros(1), tailweight: Tensor = torch.ones(1), validate_args=None):
        loc, scale, skew, tailweight = broadcast_all(loc, scale, skew, tailweight)
        self.loc, self.scale, self.skew, self.tailweight = loc, scale, skew, tailweight
        batch_shape = torch.broadcast_shapes(loc.shape, scale.shape, skew.shape, tailweight.shape)
        super().__init__(batch_shape=batch_shape, event_shape=torch.Size(), validate_args=validate_args)
        self._base = Normal(torch.zeros((), dtype=loc.dtype, device=loc.device),
                            torch.ones((), dtype=loc.dtype, device=loc.device))

    # -------- helpers --------
    def _forward(self, z: Tensor) -> Tensor:
        return self.loc + self.scale * torch.sinh((torch.asinh(z) + self.skew) * self.tailweight)

    def _inverse(self, y: Tensor) -> Tensor:
        w = (y - self.loc) / self.scale
        return torch.sinh(torch.asinh(w) / self.tailweight - self.skew)

    # -------- core API --------
    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        z = self._base.rsample(shape)  # reparameterized standard normal
        return self._forward(z)

    def sample(self, sample_shape=torch.Size()):
        with torch.no_grad():
            return self.rsample(sample_shape)

    def rvs(self, num_samples: int):
        return self.rsample(torch.Size([num_samples])).squeeze().detach().cpu().numpy().astype(np.float64)

    def log_prob(self, value: Tensor) -> Tensor:
        if self._validate_args:
            self._validate_sample(value)
        z = self._inverse(value)
        log_base = self._base.log_prob(z)

        # log|dz/dy| = -[log σ + log τ + log cosh(h)] + 0.5*log(1+z^2),
        # with h = (asinh(z) + α)*τ
        h = (torch.asinh(z) + self.skew) * self.tailweight
        log_abs_det = -torch.log(self.scale) - torch.log(self.tailweight) - _log_cosh(h) \
                      + 0.5 * torch.log1p(z.pow(2))
        return log_base + log_abs_det

    # -------- extras (handy and cheap) --------
    def cdf(self, value: Tensor) -> Tensor:
        z = self._inverse(value)
        return self._base.cdf(z)

    def icdf(self, value: Tensor) -> Tensor:
        z = self._base.icdf(value)
        return self._forward(z)

    @property
    def mean(self):
        # No simple closed form in general.
        # Provide a differentiable Monte Carlo estimate for convenience.
        # (Keep it cheap; increase n for accuracy.)
        n = 64
        z = self._base.icdf(torch.linspace(1/(2*n), 1-1/(2*n), n, device=self.loc.device, dtype=self.loc.dtype))
        y = self._forward(z).mean(0)  # averages over quadrature-like points
        return y.expand(self.batch_shape)

    @property
    def variance(self):
        n = 64
        z = self._base.icdf(torch.linspace(1/(2*n), 1-1/(2*n), n, device=self.loc.device, dtype=self.loc.dtype))
        y = self._forward(z)
        m = y.mean(0)
        return (y.pow(2).mean(0) - m.pow(2)).clamp_min(0).expand(self.batch_shape)

    def entropy(self):
        # H(Y) = H(Z) + E[log|dy/dz|] with Z~N(0,1)
        # where log|dy/dz| = log σ + log τ + log cosh(h) - 0.5*log(1+z^2)
        n = 256
        z = self._base.icdf(torch.linspace(1/(2*n), 1-1/(2*n), n, device=self.loc.device, dtype=self.loc.dtype))
        h = (torch.asinh(z) + self.skew) * self.tailweight
        log_dydz = torch.log(self.scale) + torch.log(self.tailweight) + _log_cosh(h) - 0.5*torch.log1p(z.pow(2))
        return self._base.entropy() + log_dydz.mean(0)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(SinhArcsinh, _instance)
        loc = self.loc.expand(batch_shape)
        scale = self.scale.expand(batch_shape)
        skew = self.skew.expand(batch_shape)
        tailweight = self.tailweight.expand(batch_shape)
        SinhArcsinh.__init__(new, loc, scale, skew, tailweight, validate_args=False)
        new._validate_args = self._validate_args
        return new


def state_dict_to_json_dict(state_dict):
    """
    Convert a PyTorch state_dict (or similar mapping) into a JSON-serializable dict.
    Preserves dtype, device, and shape, plus data as a nested list.
    """
    json_dict = {}
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            value = np.array(v.cpu().tolist()).flatten().tolist()
            print(value)
            # If the value is a list of length one, replace with the unique value
            if isinstance(value, list) and len(value) == 1:
                json_dict[k] = value[0]
            elif isinstance(value, list) and len(value) <= 6:
                json_dict[k] = value
            else:
                json_dict[k] = 'skipped (too long)'
        else:
            # In case of non-tensor entries (rare, but possible)
            json_dict[k] = v
    return json_dict


def find_region_for_level(Z, X, Y, min_level_threshold=0.5):
    max_level = np.max(Z)
    # Find the region where the probability is above the minimum level threshold
    min_level_threshold = max_level * 0.5
    mask = Z >= min_level_threshold

    # Get the indices where the mask is True
    y_indices, x_indices = np.where(mask)

    if len(x_indices) > 0 and len(y_indices) > 0:
        # Calculate the bounds based on the actual grid coordinates
        x_min, x_max = X[y_indices, x_indices].min(), X[y_indices, x_indices].max()
        y_min, y_max = Y[y_indices, x_indices].min(), Y[y_indices, x_indices].max()
        
        # Add some padding
        x_padding = (x_max - x_min) * 0.1
        y_padding = (y_max - y_min) * 0.1
        
        x_min -= x_padding
        x_max += x_padding
        y_min -= y_padding
        y_max += y_padding
    else:
        # Fallback to original limits if no region found
        x_min, x_max = X.min(), X.max()
        y_min, y_max = Y.min(), Y.max()

    RegionBounds = namedtuple("RegionBounds", ["x_min", "x_max", "y_min", "y_max"])
    return RegionBounds(x_min, x_max, y_min, y_max)



