"""SMC β-profile with all OTHER parameters held at the DGP truth.

DGP (no softplus anywhere):
    z_1 ~ N(0, sigma_z1)               sigma_z1 = sigma_z / sqrt(1-rho^2)
    z_t = rho * z_{t-1} + sigma_z * u_t,  u_t ~ N(0,1)
    y_t = z_t + eps_t,                 eps_t ~ SinhArcsinh(0, scale_eps, beta_truth)

Truth model used by SMC:
    Same prior (rho, sigma_z, sigma_z1 fixed at truth).
    Decoder: SinhArcsinh(0, scale_eps fixed at truth, beta = grid value).

For each beta in the grid we evaluate log p(y) by bootstrap PF with K=5000.
If the model class is correctly identified the curve should peak at beta_truth.
"""
import os
import time
import json

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from mlye.utils import SinhArcsinh
from mlye.models.priors.base import Prior
from mlye.eval import BootstrapParticleFilter


# ----- truth-model components ----------------------------------------------

class TruthLinearAR1Prior(Prior):
    """Exact linear-Gaussian AR(1) prior with stationary z_1 — no parameters
    learned, just exposes the SMC interface (`get_eta0`, `get_mu_sigma`)."""

    def __init__(self, rho: float, sigma_z: float, sigma_z1: float, nt: int):
        super().__init__()
        self.nt = nt
        self.rho = float(rho)
        self.sigma_z = float(sigma_z)
        self.sigma_z1 = float(sigma_z1)
        self._device = torch.device('cpu')

    def to(self, device):
        self._device = torch.device(device)
        return super().to(device)

    def get_eta0(self):
        loc = torch.zeros(1, device=self._device)
        scl = torch.full((1,), self.sigma_z1, device=self._device)
        return Normal(loc, scl)

    def get_mu_sigma(self, z):
        mu = self.rho * z
        sigma = torch.full_like(z, self.sigma_z)
        return mu, sigma

    def get_mu_sigma_detached(self, z):
        mu, sigma = self.get_mu_sigma(z)
        return mu.detach().cpu(), sigma.detach().cpu()

    def log_prob_eta0(self, z):
        return self.get_eta0().log_prob(z[:, 0:1]).squeeze(1)

    def log_prob(self, z):
        # z: (batch, T). Sums log p(z_1) + sum_t log N(z_t | rho z_{t-1}, sigma_z).
        log_p = self.log_prob_eta0(z)
        for t in range(1, self.nt):
            mu = self.rho * z[:, t-1:t]
            sigma = torch.full_like(mu, self.sigma_z)
            log_p = log_p + Normal(mu, sigma).log_prob(z[:, t:t+1]).squeeze(1)
        return log_p


class TruthSinhDecoder(nn.Module):
    """Sinh-arcsinh emission with scale fixed at truth, β configurable."""

    def __init__(self, scale_eps: float, beta: float):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([scale_eps]), requires_grad=False)
        self.log_beta = nn.Parameter(torch.tensor([np.log(beta)]), requires_grad=False)

    def get_distribution(self):
        return SinhArcsinh(
            loc=torch.zeros(1, device=self.scale.device),
            scale=self.scale,
            skew=torch.zeros(1, device=self.scale.device),
            tailweight=torch.exp(self.log_beta),
        )

    def log_likelihood(self, y, z):
        # MA(0): residuals = y - z; sum over T of marginal log_prob.
        residuals = y - z
        log_p = self.get_distribution().log_prob(residuals)
        if log_p.dim() == 3:
            log_p = log_p.squeeze(-1)
        return log_p.sum(dim=1)

    def draw(self, z):
        return self.get_distribution().sample(z.shape)

    def recenter(self):
        # No-op (no learnable scalars to recenter).
        pass


class _Bundle:
    def __init__(self, prior, decoder):
        self.prior = prior
        self.decoder = decoder


# ----- DGP -----------------------------------------------------------------

def simulate_truth(N, T, rho, sigma_z, scale_eps, beta_truth, seed):
    rng = np.random.default_rng(seed)
    sigma_z1 = sigma_z / np.sqrt(1 - rho * rho)
    # Persistent component
    z = np.zeros((N, T))
    z[:, 0] = sigma_z1 * rng.standard_normal(N)
    for t in range(1, T):
        z[:, t] = rho * z[:, t-1] + sigma_z * rng.standard_normal(N)
    # Transitory: SinhArcsinh; sample on torch using a deterministic generator
    g = torch.Generator().manual_seed(seed + 1)
    u = torch.randn((N, T), generator=g)  # standard normal
    eps = scale_eps * torch.sinh(torch.asinh(u) * beta_truth)  # location 0, skew 0
    y = torch.from_numpy(z).float() + eps.float()
    return y, sigma_z1


# ----- main ----------------------------------------------------------------

def main():
    # Truth parameters
    rho = 0.95
    sigma_z = 0.20
    scale_eps = 0.10
    beta_truth = 2.13
    N, T = 10_000, 10
    seed = 11

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"DGP (linear AR(1), no softplus): N={N}, T={T}, seed={seed}")
    print(f"  rho={rho}, sigma_z={sigma_z}, scale_eps={scale_eps}, beta_truth={beta_truth}")

    y, sigma_z1 = simulate_truth(N, T, rho, sigma_z, scale_eps, beta_truth, seed)
    y = y.to(device)
    print(f"  sigma_z1 (stationary)={sigma_z1:.4f}")
    print(f"  empirical std(y)={y.std().item():.4f}  std(Δy)={torch.diff(y, dim=1).std().item():.4f}")

    beta_grid = [1.0, 1.2, 1.5, 1.8, 2.0, 2.13, 2.3, 2.6, 3.0]
    K = 5000

    print(f"\n=== SMC β-profile at truth (K={K}, all other params at truth) ===")
    profile = []
    prior = TruthLinearAR1Prior(rho=rho, sigma_z=sigma_z, sigma_z1=sigma_z1, nt=T).to(device)
    for beta in beta_grid:
        decoder = TruthSinhDecoder(scale_eps=scale_eps, beta=beta).to(device)
        model = _Bundle(prior=prior, decoder=decoder)
        pf = BootstrapParticleFilter(model, K=K)
        # Repeat a few times to assess MC noise.
        runs = []
        for rep in range(3):
            torch.manual_seed(1000 * rep + 7)
            t0 = time.time()
            log_p = pf.log_marginal(y).mean().item()
            runs.append(log_p)
        mean = float(np.mean(runs))
        sd = float(np.std(runs))
        elapsed = time.time() - t0
        marker = " <-- truth" if abs(beta - beta_truth) < 1e-9 else ""
        print(f"  β={beta:.2f}  log p(y)={mean:.4f} ± {sd:.4f}  ({elapsed:.1f}s/rep){marker}")
        profile.append({'beta': beta, 'log_p_mean': mean, 'log_p_sd': sd, 'reps': runs})

    # Best β
    best = max(profile, key=lambda r: r['log_p_mean'])
    print(f"\nargmax  β={best['beta']:.2f}  log p(y)={best['log_p_mean']:.4f}")
    print(f"truth   β={beta_truth:.2f}  log p(y)={[r for r in profile if r['beta']==beta_truth][0]['log_p_mean']:.4f}")

    out_dir = "output/bundles/cells/smc_id-smc_truth_profile"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "smc_truth_profile_linear_ar1.json")
    with open(out_path, "w") as f:
        json.dump({
            'dgp': {'rho': rho, 'sigma_z': sigma_z, 'sigma_z1': sigma_z1,
                    'scale_eps': scale_eps, 'beta_truth': beta_truth, 'N': N, 'T': T, 'seed': seed},
            'K': K, 'profile': profile,
        }, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
