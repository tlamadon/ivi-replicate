"""
Models package - Core estimator components.

- encoders: Approximate posterior q(z|y)
- decoders: Likelihood p(y|z)
- priors: Prior distributions p(z)
- full_model: FullModel composition (encoder + decoder + prior)
"""

from .full_model import FullModel
from . import encoders
from . import decoders
from . import priors

__all__ = [
    'FullModel',
    'encoders',
    'decoders',
    'priors',
]
