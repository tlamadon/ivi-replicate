"""Base classes and configurations for encoders with config.build() pattern."""

from typing import Literal, Union, TYPE_CHECKING
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from pydantic import BaseModel

if TYPE_CHECKING:
    from . import Encoder


class Encoder(ABC, nn.Module):
    """Abstract base class for all encoders."""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_seed_dim(self) -> int:
        """Return the dimension of the seed noise input required (usually T, but can be something else)."""
        pass

    @abstractmethod
    def draw_and_logprob(self, y: torch.Tensor, u: torch.Tensor, logpr_draw: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw samples and compute log probabilities."""
        pass

    @abstractmethod
    def stats(self) -> dict:
        """Return statistics for monitoring."""
        pass


PosteriorType = Literal[
    'joint_normal',
    'joint_normal_restricted',
    'conditional_markov',
    'conditional_sinh_markov',
    'conditional_markov_ma',
    'triband',
    'transformed_joint_normal',
    'normal_diagonal',
    'joint_normal_extra',
    'tridiag_joint_normal',
    'laplace',
]


# Base configuration class
class EncoderConfig(BaseModel):
    """Base configuration for all encoders."""
    type: PosteriorType = 'joint_normal'
    dim: int = 6  # number of latent variables
    regularize: float = 1e-3

    def __str__(self):
        return f"{self.__class__.__name__}(type={self.type}, dim={self.dim}, regularize={self.regularize})"

    def build(self) -> 'Encoder':
        """Build encoder from configuration. Override in subclasses or dispatch based on type."""
        raise NotImplementedError(f"Config type {self.type} does not have a build() method")


# Specialized configuration classes with build() methods
class JointNormalConfig(EncoderConfig):
    """Configuration for joint normal encoders."""
    type: Literal['joint_normal', 'normal_diagonal', 'joint_normal_extra', 'tridiag_joint_normal'] = 'joint_normal'
    diagonal: bool = False
    hidden_dim: int = 32
    sd_clamp: float = 3.0
    extra_latents: int = 0  # For 'joint_normal_extra' encoder

    def build(self) -> 'Encoder':
        """Build JointNormalPosterior from this configuration."""
        from .normal import JointNormalPosterior, TridiagJointNormalPosterior

        if self.type == 'tridiag_joint_normal':
            return TridiagJointNormalPosterior(
                dim=self.dim,
                hidden_dim=self.hidden_dim,
                regularize=self.regularize,
            )

        kwargs = {
            'dim': self.dim,
            'regularize': self.regularize,
            'hidden_dim': self.hidden_dim,
            'sd_clamp': self.sd_clamp,
            'diagonal': self.diagonal if self.type != 'normal_diagonal' else True
        }

        if self.type == 'joint_normal_extra':
            if self.extra_latents <= 0:
                raise ValueError("extra_latents must be positive for joint_normal_extra encoder")
            kwargs['dim_latent'] = self.dim + self.extra_latents

        return JointNormalPosterior(**kwargs)


class RestrictedNormalConfig(EncoderConfig):
    """Configuration for restricted normal encoders."""
    type: Literal['joint_normal_restricted'] = 'joint_normal_restricted'
    sigma_eps: float = 1.0
    fix_sigma: bool = False
    hidden_dim: int = 32

    def build(self) -> 'Encoder':
        """Build JointNormalPosteriorRestricted from this configuration."""
        from .normal import JointNormalPosteriorRestricted

        return JointNormalPosteriorRestricted(
            dim=self.dim,
            sigma_eps=self.sigma_eps,
            regularize=self.regularize,
            fix_sigma=self.fix_sigma,
            hidden_dim=self.hidden_dim
        )


class MarkovConfig(EncoderConfig):
    """Configuration for Markov-based encoders."""
    type: Literal['conditional_markov', 'conditional_markov_ma', 'conditional_sinh_markov'] = 'conditional_markov'
    hidden_dim: int = 32

    def build(self) -> 'Encoder':
        """Build appropriate Markov encoder from this configuration."""
        from .markov import (
            ConditionalNormalMarkovPosterior,
            ConditionalNormalMarkovMAPosterior,
            ConditionalSinhMarkovPosterior
        )

        if self.type == 'conditional_markov':
            return ConditionalNormalMarkovPosterior(
                dim=self.dim,
                regularize=self.regularize,
                hidden_dim=self.hidden_dim
            )
        elif self.type == 'conditional_markov_ma':
            return ConditionalNormalMarkovMAPosterior(
                dim=self.dim,
                regularize=self.regularize,
                hidden_dim=self.hidden_dim
            )
        elif self.type == 'conditional_sinh_markov':
            return ConditionalSinhMarkovPosterior(
                dim=self.dim,
                regularize=self.regularize,
                hidden_dim=self.hidden_dim
            )
        else:
            raise ValueError(f"Unknown Markov encoder type: {self.type}")


class FlowConfig(EncoderConfig):
    """Configuration for the transformed (sinh-arcsinh) joint-normal encoder."""
    type: Literal['transformed_joint_normal'] = 'transformed_joint_normal'

    def build(self) -> 'Encoder':
        """Build TransformedJointNormalPosterior from this configuration."""
        from .normal import TransformedJointNormalPosterior

        return TransformedJointNormalPosterior(
            dim=self.dim,
            regularize=self.regularize
        )

class LaplaceConfig(EncoderConfig):
    """Configuration for the Laplace approximation encoder.

    No learnable encoder parameters: Q is the Laplace approximation around
    the per-individual posterior mode, with curvature derived from the
    actual model log-density (prior + decoder) at the current θ.
    """
    type: Literal['laplace'] = 'laplace'
    num_newton_steps: int = 3
    lm_lambda: float = 1e-3
    cache_modes: bool = True

    def build(self) -> 'Encoder':
        from .laplace import LaplaceEncoder
        return LaplaceEncoder(
            dim=self.dim,
            num_newton_steps=self.num_newton_steps,
            lm_lambda=self.lm_lambda,
            cache_modes=self.cache_modes,
        )


class TribandConfig(EncoderConfig):
    """Configuration for triband precision encoders."""
    type: Literal['triband'] = 'triband'
    diagonal: bool = False

    def build(self) -> 'Encoder':
        """Build JointNormalTribandPrecisionPosterior from this configuration."""
        from .normal import JointNormalTribandPrecisionPosterior

        return JointNormalTribandPrecisionPosterior(
            dim=self.dim,
            regularize=self.regularize,
            diagonal=self.diagonal
        )


# Union type for all possible configs
AnyEncoderConfig = Union[
    JointNormalConfig,
    RestrictedNormalConfig,
    MarkovConfig,
    FlowConfig,
    TribandConfig,
    LaplaceConfig,
    EncoderConfig  # Fallback to base
]

import torch

def split_latent(z: torch.Tensor, nt: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Split latent variable tensor z into latent part and extra part.

    Works for both 2D (batch, dim) and 3D (batch, T, dim) tensors.

    Args:
        z: Tensor of shape (batch, dim_latent + extra_latents)
           or (batch, T, dim_latent + extra_latents)
        nt: Dimension of the main latent variable

    Returns:
        A tuple (latent_part, extra_part)
    """
    if z.ndim == 2:
        latent_part = z[:, :nt]
        extra_part = z[:, nt:]
    elif z.ndim == 3:
        latent_part = z[:, :, :nt]
        extra_part = z[:, :, nt:]
    else:
        raise ValueError(f"Expected 2D or 3D tensor, got {z.ndim}D")

    return latent_part, extra_part


def build_encoder(config: AnyEncoderConfig) -> 'Encoder':
    """
    Build encoder from configuration using config.build() method.

    This is a convenience function that delegates to the config's build() method.
    """
    return config.build()


def create_encoder_config(
    posterior_type: PosteriorType,
    dim: int = 6,
    regularize: float = 1e-3,
    **kwargs
) -> AnyEncoderConfig:
    """
    Factory function to create the correct encoder config class for a given posterior type.

    This is necessary because Pydantic Union types are class-based, not field-based.
    Simply changing `config.encoder.type` doesn't change the config object's CLASS,
    which means calling `build()` would invoke the wrong class's method.

    Args:
        posterior_type: The type of posterior encoder to create
        dim: Number of latent variables (default: 6)
        regularize: Regularization strength (default: 1e-3)
        **kwargs: Additional type-specific parameters (e.g., hidden_dim, diagonal, etc.)

    Returns:
        An encoder config object of the appropriate class

    Example:
        >>> # Create a joint normal encoder config
        >>> config = create_encoder_config('joint_normal', dim=6, regularize=1e-3)
        >>> encoder = config.build()  # Creates JointNormalPosterior

        >>> # Create a Markov encoder config
        >>> config = create_encoder_config('conditional_markov', dim=6, hidden_dim=64)
        >>> encoder = config.build()  # Creates ConditionalNormalMarkovPosterior

    Raises:
        ValueError: If posterior_type is not recognized
    """
    # Extract common parameters
    base_params = {'type': posterior_type, 'dim': dim, 'regularize': regularize}

    # Map posterior types to their config classes
    if posterior_type in ['joint_normal', 'normal_diagonal', 'joint_normal_extra', 'tridiag_joint_normal']:
        return JointNormalConfig(**{**base_params, **kwargs})

    elif posterior_type == 'joint_normal_restricted':
        return RestrictedNormalConfig(**{**base_params, **kwargs})

    elif posterior_type in ['conditional_markov', 'conditional_markov_ma', 'conditional_sinh_markov']:
        return MarkovConfig(**{**base_params, **kwargs})

    elif posterior_type == 'transformed_joint_normal':
        return FlowConfig(**{**base_params, **kwargs})

    elif posterior_type == 'triband':
        return TribandConfig(**{**base_params, **kwargs})

    elif posterior_type == 'laplace':
        return LaplaceConfig(**{**base_params, **kwargs})

    else:
        # List valid types for helpful error message
        valid_types = [
            'joint_normal', 'normal_diagonal', 'joint_normal_extra',
            'tridiag_joint_normal',
            'joint_normal_restricted', 'conditional_markov', 'conditional_markov_ma',
            'conditional_sinh_markov', 'transformed_joint_normal',
            'triband', 'laplace'
        ]
        raise ValueError(
            f"Unknown posterior type: '{posterior_type}'. "
            f"Valid types are: {', '.join(valid_types)}"
        )
