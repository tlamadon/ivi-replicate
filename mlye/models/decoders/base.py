from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Literal

class Decoder(ABC, nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def log_likelihood(self, y:torch.Tensor, z:torch.Tensor) -> torch.Tensor:
        """Draw samples and compute log probabilities."""
        pass

    @abstractmethod
    def draw(self, z: torch.Tensor) -> torch.Tensor:
        """Generate samples from the distribution given latent variable z."""
        pass

    @abstractmethod
    def get_distribution(self) -> torch.distributions.Distribution:
        """Return the underlying distribution object."""
        pass

    @abstractmethod
    def stats(self) -> dict:
        """Return statistics for monitoring."""
        pass

    def recenter(self):
        """Optional: Recenter the distribution if applicable (e.g., for flow-based decoders)."""
        pass
