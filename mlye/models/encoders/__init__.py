"""Encoder package for approximate posterior distributions q(z|y)."""

# Base classes and configurations
from .base import (
    Encoder,
    EncoderConfig,
    JointNormalConfig,
    RestrictedNormalConfig,
    MarkovConfig,
    FlowConfig,
    TribandConfig,
    AnyEncoderConfig,
    PosteriorType,
    create_encoder_config,
)

# Neural network layers
from .layers import MaskedLinear

# Normal-based encoders
from .normal import (
    JointNormalPosterior,
    JointNormalPosteriorRestricted,
    JointNormalTribandPrecisionPosterior,
    TransformedJointNormalPosterior,
    TridiagJointNormalPosterior,
)

# Conditional Markov encoders
from .markov import (
    ConditionalNormalMarkovPosterior,
    ConditionalNormalMarkovMAPosterior,
    ConditionalSinhMarkovPosterior
)

__all__ = [
    # Base classes
    'Encoder',
    'EncoderConfig',
    'JointNormalConfig',
    'RestrictedNormalConfig',
    'MarkovConfig',
    'FlowConfig',
    'TribandConfig',
    'AnyEncoderConfig',
    'PosteriorType',
    'MaskedLinear',
    'create_encoder_config',

    # Normal-based encoders
    'JointNormalPosterior',
    'JointNormalPosteriorRestricted',
    'JointNormalTribandPrecisionPosterior',
    'TransformedJointNormalPosterior',
    'TridiagJointNormalPosterior',

    # Conditional Markov encoders
    'ConditionalNormalMarkovPosterior',
    'ConditionalNormalMarkovMAPosterior',
    'ConditionalSinhMarkovPosterior',
]
