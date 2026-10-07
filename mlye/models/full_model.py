"""
Full variational inference model combining encoder, decoder, and prior.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import warnings
from typing import Optional

from .encoders import Encoder
from .decoders import Decoder
from .priors import Prior

import structlog
logger = structlog.get_logger(__name__)


class FullModel(nn.Module):
    """
    Complete variational inference model.

    Combines three components:
    - encoder: q(z|y) - approximate posterior
    - decoder: p(y|z) - likelihood/emission
    - prior: p(z) - prior distribution

    Provides ELBO computation and various diagnostic methods.
    """

    def __init__(
            self,
            encoder: Encoder,
            decoder: Decoder,
            prior: Prior,
            beta: float = 1.0):
        super().__init__()

        self.encoder = encoder
        self.decoder: Decoder = decoder
        self.prior: Prior = prior
        self.ESS = torch.tensor(0.0)
        self.beta = beta

        # Some encoders (e.g. KalmanSmootherEncoder) need refs to the prior
        # and decoder to derive Q structurally from the current model. Wire
        # them up after construction so the encoder doesn't have to know
        # about prior/decoder construction order.
        if hasattr(encoder, "set_components"):
            encoder.set_components(prior=prior, decoder=decoder)

    def load_all(self, state_dict):
        """Load state dict for all three components."""
        self.encoder.load_state_dict(state_dict['model_encoder'])
        self.decoder.load_state_dict(state_dict['model_decoder'])
        self.prior.load_state_dict(state_dict['model_prior'])

    def draw_encoder_seed(self, y: torch.Tensor, eps_all, ndraws: int = 1) -> torch.Tensor:
        """Draw or validate random seeds for encoder."""
        if eps_all is None:
            eps_all = torch.randn((ndraws, y.shape[0], self.encoder.get_seed_dim())).to(torch.float32).to(y.device)
        else:
            # Check that the size matches demanded size
            if eps_all.shape[1] != y.shape[0]:
                raise ValueError(f"Size mismatch: eps_all has shape {eps_all.shape[1:]}, but y has shape {y.shape[0]}")
            if eps_all.shape[2] != self.encoder.get_seed_dim():
                raise ValueError(f"Size mismatch: eps_all has shape {eps_all.shape[1:]}, but encoder seed dim is {self.encoder.get_seed_dim()}")
        return eps_all

    def elbo(self,
             y: torch.Tensor,
             ndraws: int = 1,
             eps_all: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute Evidence Lower Bound (ELBO).

        Args:
            y: Observed data [batch, dim]
            ndraws: Number of Monte Carlo samples
            eps_all: Optional pre-drawn random seeds [ndraws, batch, seed_dim]

        Returns:
            ELBO estimate (scalar)
        """
        eps_all = self.draw_encoder_seed(y, eps_all, ndraws)

        total = torch.zeros(1, device=y.device)
        ndraws = eps_all.shape[0]
        ESS = torch.tensor(0.0, device=y.device)

        for d in range(int(ndraws)):
            # Reshape eps to match the batch size of y
            eps = eps_all[d, :, :].reshape(y.shape[0], -1)

            # get latent draws and probabilities
            z, log_qz = self.encoder.draw_and_logprob(y, eps)

            # compute prior and likelihood
            log_pz = self.prior.log_prob(z)
            log_py_given_z = self.decoder.log_likelihood(y, z)

            # Accumulate the ELBO for this draw
            total += (log_py_given_z + self.beta * (log_pz - log_qz)).mean()

            # Compute the importance weights
            log_w = log_py_given_z + log_pz - log_qz       # [K] log importance weights
            w = torch.exp(log_w - torch.max(log_w))        # numerically stable
            ESS += (w.sum())**2 / (w**2).sum()

        self.ESS = ESS / (y.size(0) * ndraws)  # normalize ESS by batch size and number of draws
        return total / ndraws

    def elbo_diagnostics(self,
                        y: torch.Tensor,
                        ndraws: int = 10):
        """
        Combined diagnostic method that computes:
        1. ELBO decomposition (reconstruction, prior, entropy terms)
        2. IWAE bound and gap (diagnostic for variational quality)
        3. Effective Sample Size (ESS)

        Returns:
            dict with keys:
                - recon: E_q[log p(y|z)] - reconstruction term
                - prior: E_q[log p(z)] - prior term
                - entropy: E_q[log q(z|y)] - entropy term
                - elbo: standard ELBO estimate
                - mi: mutual information (elbo - recon)
                - iwae: IWAE bound (tighter than ELBO)
                - gap: IWAE - ELBO (smaller is better)
                - ess: Effective Sample Size (normalized)
        """
        K = int(ndraws)

        with torch.no_grad():
            # Collect log-probabilities for all draws
            recon_terms = []   # log p(y|z)
            prior_terms = []   # log p(z)
            entropy_terms = [] # log q(z|y)
            log_ws = []        # log weights for IWAE

            for _ in range(K):
                eps = self.draw_encoder_seed(y, None, 1)[0]
                z, log_qz = self.encoder.draw_and_logprob(y, eps, logpr_draw=True)
                log_pz = self.prior.log_prob(z)
                log_py_given_z = self.decoder.log_likelihood(y, z)

                # For ELBO decomposition
                recon_terms.append(log_py_given_z.unsqueeze(0))   # [1, batch]
                prior_terms.append(log_pz.unsqueeze(0))           # [1, batch]
                entropy_terms.append(log_qz.unsqueeze(0))         # [1, batch]

                # For IWAE computation
                log_w = log_py_given_z + log_pz - log_qz
                log_ws.append(log_w.unsqueeze(0))                 # [1, batch]

            # Stack over draws -> [K, batch]
            recon_stack = torch.cat(recon_terms, dim=0)
            prior_stack = torch.cat(prior_terms, dim=0)
            entropy_stack = torch.cat(entropy_terms, dim=0)
            log_ws = torch.cat(log_ws, dim=0)

            # ELBO decomposition: Monte Carlo expectations under q(z|y)
            recon_mean = recon_stack.mean(dim=0)     # [batch]
            prior_mean = prior_stack.mean(dim=0)     # [batch]
            entropy_mean = entropy_stack.mean(dim=0) # [batch]
            elbo_per_example = recon_mean + prior_mean - entropy_mean  # [batch]

            # IWAE bound: log(1/K sum_k exp(log_w_k))
            iwae_per_example = torch.logsumexp(log_ws, dim=0) - np.log(K)  # [batch]

            # Compute ESS (Effective Sample Size)
            ws = torch.exp(log_ws - torch.max(log_ws, dim=0, keepdim=True)[0])
            ess_numer = (ws.sum(dim=0))**2
            ess_denom = (ws**2).sum(dim=0)
            ess = ess_numer / ess_denom
            ess = ess[torch.isfinite(ess)]
            ESS = ess.mean() / K if ess.numel() > 0 else torch.tensor(0.0, device=y.device)

            # Batch aggregates (scalars) for logging
            recon_batch = recon_mean.mean().item()
            prior_batch = prior_mean.mean().item()
            entropy_batch = entropy_mean.mean().item()
            elbo_batch = elbo_per_example.mean().item()
            iwae_batch = iwae_per_example.mean().item()
            gap = iwae_batch - elbo_batch

            return {
                "recon": recon_batch,
                "prior": prior_batch,
                "entropy": entropy_batch,
                "elbo": elbo_batch,
                "mi": elbo_batch - recon_batch,
                "iwae": iwae_batch,
                "gap": gap,
                "ess": ESS.item() if isinstance(ESS, torch.Tensor) else ESS
            }

    def elbo_iwae(self,
             y: torch.Tensor,
             ndraws: int = 1,
             eps_all: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute Importance Weighted Autoencoder (IWAE) bound.

        Provides a tighter bound than standard ELBO.
        """
        eps_all = self.draw_encoder_seed(y, eps_all, ndraws)
        total = torch.zeros(1, device=y.device)
        ndraws = eps_all.shape[0]

        # Importance Weighted Autoencoder (IWAE) objective
        log_ws = []
        for d in range(int(ndraws)):
            eps = eps_all[d, :, :].reshape(y.shape[0], -1)
            z, log_qz = self.encoder.draw_and_logprob(y, eps, logpr_draw=True)
            log_pz = self.prior.log_prob(z)
            log_py_given_z = self.decoder.log_likelihood(y, z)
            log_w = log_py_given_z + log_pz - log_qz  # [batch]
            log_ws.append(log_w.unsqueeze(0))  # shape: [1, batch]
        log_ws = torch.cat(log_ws, dim=0)  # shape: [ndraws, batch]
        # IWAE bound: log(1/K sum_k exp(log_w_k))
        iwae = torch.logsumexp(log_ws, dim=0) - np.log(ndraws)
        total = iwae.mean()  # already the per-individual IWAE; do NOT divide by ndraws again.
        # Effective Sample Size (ESS), normalised to [1/ndraws, 1].
        ws = torch.exp(log_ws - torch.max(log_ws, dim=1, keepdim=True)[0])
        self.ESS = ((ws.sum(dim=0))**2 / (ws**2).sum(dim=0)).mean() / ndraws

        return total

    def log_joint(self, y: torch.Tensor, z: torch.Tensor):
        """Compute the log joint probability p(y, z) = p(y|z) * p(z)"""
        log_py_given_z = self.decoder.log_likelihood(y, z)
        log_pz = self.prior.log_prob(z)
        return log_py_given_z + log_pz

    def log_joint_wo_noise(self, y: torch.Tensor):
        """Compute the log joint probability p(y, z) = p(y|z) * p(z)"""
        log_pz = self.prior.log_prob(y)
        return log_pz

    def log_pr(self, y):
        """Compute log probability by marginalizing over decoder noise."""
        # draw some errors
        z_sim = self.decoder.draw(torch.zeros_like(y))

        # evaluate the log_likeliood
        li = self.prior.log_prob(y - z_sim)

        # return the mean
        return (torch.logsumexp(li, 0) - np.log(len(li)))

    def draw_latent(self, y: torch.Tensor, ndraws: int = 1, save: bool = True) -> torch.Tensor:
        """Draw latent base for later ELBO computation."""
        eps = torch.randn((ndraws, y.shape[0], y.shape[1])).to(torch.float32).to(y.device)

        # save internally
        if save:
            self.eps = eps

        return eps

    def train_on_truth(self,
                       y: torch.Tensor,
                       z_true: torch.Tensor,
                       lr: float = 1e-3,
                       epochs=5000) -> float:
        """Train the model using the true latent states (for benchmarking)."""

        # Initialize the model, optimizer, and other training components
        num_epochs = epochs

        optimizer = torch.optim.AdamW(
            list(self.decoder.parameters()) + list(self.prior.parameters()),
            lr=lr)

        loss = - self.log_joint(y, z_true).mean()

        # Training loop
        for epoch in range(num_epochs):

            optimizer.zero_grad()

            # Compute loss (negative ELBO)
            loss = - self.log_joint(y, z_true).mean()

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Print epoch loss every 50 epochs
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")

        return loss.item()

    def train_posterior_to_prior(self,
                       y: torch.Tensor,
                       lr: float = 1e-3,
                       epochs=5000) -> None:
        """Train encoder to match prior (for debugging/benchmarking)."""

        optimizer = torch.optim.Adam(
            self.encoder.parameters(),
            lr=lr)

        U = torch.normal(0, 1, size=(1, y.size(0), y.size(1)))
        print(f"U shape: {U.shape}")
        print(f"ytensor shape: {y.shape}")
        num_epochs = 2000

        # Training loop
        for epoch in range(num_epochs):

            optimizer.zero_grad()

            # Compute loss (negative ELBO)
            U = torch.normal(0, 1, size=(1, y.size(0), y.size(1)))
            loss = - self.elbo(y, eps_all=U)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Print epoch loss every 50 epochs
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}, post var: {self.encoder.stats()['mean_var']:.4f}")
