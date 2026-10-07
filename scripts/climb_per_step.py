"""How fast does each inference method climb the likelihood surface?

For each method, train from a common init for 500 epochs. At checkpoints
{0, 50, 100, 250, 500}, evaluate SMC log p̂(y; θ) (K=5000) at the *current*
parameters (no encoder / no inference-side noise), giving the real
log-likelihood per individual.

Reports:
  - log p(y) trajectory per method
  - climb rate (Δ log p / epoch) on the segment [50, 500]
  - per-epoch wall-clock for context (re-bench inside this script)

Same DGP as §1: ρ=0.95, σ_z=0.20, σ_ε=0.10, β=2.13. N=10000, T=10.
On GPU."""
import os, sys, time, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from smc_truth_profile import (TruthLinearAR1Prior, TruthSinhDecoder, _Bundle,
                                simulate_truth)

from mlye.models.encoders import JointNormalConfig, ConditionalSinhMarkovPosterior
from mlye.models.encoders.base import LaplaceConfig
from mlye.models.decoders.ma import MASinhEmission
from mlye.models.priors.ar import AR1Prior
from mlye.models.full_model import FullModel
from mlye.eval import BootstrapParticleFilter, FilteringVariationalObjective

@torch.no_grad()
def smc_ffbs_sample(prior, decoder, y, K):
    """Generic FFBS via prior.get_mu_sigma — works for any AR-style prior."""
    N, T = y.shape
    device = y.device
    z1_dist = prior.get_eta0()
    z = z1_dist.sample((N * K,)).view(N, K).to(device)
    log_w_t = decoder.get_distribution().log_prob(
        (y[:, 0:1] - z).unsqueeze(-1)).squeeze(-1)
    if log_w_t.dim() == 3:
        log_w_t = log_w_t.squeeze(-1)
    z_hist = [z.clone()]
    log_w_hist = [log_w_t.clone()]
    for t in range(1, T):
        mu, sigma = prior.get_mu_sigma(z.reshape(-1, 1))
        z = (mu + sigma * torch.randn_like(mu)).view(N, K)
        log_lik = decoder.get_distribution().log_prob(
            (y[:, t:t+1] - z).unsqueeze(-1)).squeeze(-1)
        if log_lik.dim() == 3:
            log_lik = log_lik.squeeze(-1)
        log_w_t = log_w_t + log_lik
        z_hist.append(z.clone())
        log_w_hist.append(log_w_t.clone())
    final_logw = log_w_hist[-1] - torch.logsumexp(log_w_hist[-1], dim=1, keepdim=True)
    final_w = torch.exp(final_logw).clamp_min(1e-30)
    idx_T = torch.multinomial(final_w, num_samples=1).squeeze(-1)
    smoothed = [None] * T
    smoothed[T - 1] = z_hist[T - 1].gather(1, idx_T.unsqueeze(1)).squeeze(1)
    for t in range(T - 2, -1, -1):
        z_t = z_hist[t]
        log_w_filt = log_w_hist[t] - torch.logsumexp(log_w_hist[t], dim=1, keepdim=True)
        z_next = smoothed[t + 1].unsqueeze(1)
        mu, sigma = prior.get_mu_sigma(z_t.reshape(-1, 1))
        mu = mu.view(N, K); sigma = sigma.view(N, K)
        log_trans = (-0.5 * ((z_next - mu) / sigma) ** 2
                      - torch.log(sigma) - 0.5 * np.log(2 * np.pi))
        log_back = log_w_filt + log_trans
        log_back = log_back - torch.logsumexp(log_back, dim=1, keepdim=True)
        probs = torch.exp(log_back).clamp_min(1e-30)
        idx = torch.multinomial(probs, num_samples=1).squeeze(-1)
        smoothed[t] = z_t.gather(1, idx.unsqueeze(1)).squeeze(1)
    return torch.stack(smoothed, dim=1)


def init_prior_at_data(y, prior):
    with torch.no_grad():
        prior.mean.fill_(float(y.mean()))
        prior.log_std.fill_(np.log(max(float(y.std()), 1e-3)))
        prior.mu.fill_(0.0)
        prior.rho.fill_(0.5)
        prior.log_std_u.fill_(np.log(max(float(y.std()) * 0.3, 1e-3)))
    return prior


def smc_logp(prior, decoder, y, K=5000):
    pf = BootstrapParticleFilter(_Bundle(prior=prior, decoder=decoder), K=K)
    return float(pf.log_marginal(y).mean().item())


def make_model(method, T, device, K_iwae=None):
    """Returns (model_or_bundle, optimizer, step_fn).

    model has .prior and .decoder. step_fn(y) does one SGD step and returns
    the loss value. For FIVO we use a _Bundle (no encoder)."""
    decoder = MASinhEmission(sigma_eps=0.20, theta=0.0, fix_theta=True,
                              sigma_floor=0.0, beta_init=1.0,
                              fix_beta=False).to(device)
    prior = AR1Prior(nt=T).to(device)

    if method == 'ELBO_K1':
        encoder = JointNormalConfig(dim=T, type='joint_normal',
                                      regularize=1e-3).build()
        model = FullModel(encoder=encoder, decoder=decoder, prior=prior).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        def step(y):
            opt.zero_grad()
            loss = -model.elbo(y, ndraws=1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            return loss.item()
        return model, opt, step

    if method == 'IWAE_K10':
        encoder = JointNormalConfig(dim=T, type='joint_normal',
                                      regularize=1e-3).build()
        model = FullModel(encoder=encoder, decoder=decoder, prior=prior).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        def step(y):
            opt.zero_grad()
            loss = -model.elbo_iwae(y, ndraws=10)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            return loss.item()
        return model, opt, step

    if method == 'ELBO_markov_sinh':
        encoder = ConditionalSinhMarkovPosterior(dim=T, hidden_dim=32,
                                                  regularize=1e-3)
        model = FullModel(encoder=encoder, decoder=decoder, prior=prior).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        def step(y):
            opt.zero_grad()
            loss = -model.elbo(y, ndraws=1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            return loss.item()
        return model, opt, step

    if method == 'FIVO_K10':
        bundle = _Bundle(prior=prior, decoder=decoder)
        opt = torch.optim.AdamW(list(prior.parameters()) +
                                  list(decoder.parameters()), lr=1e-3)
        fivo = FilteringVariationalObjective(bundle, K=10)
        def step(y):
            opt.zero_grad()
            loss = -fivo.fivo_bound(y).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(prior.parameters()) +
                                              list(decoder.parameters()), 5.0)
            opt.step()
            return loss.item()
        return bundle, opt, step

    if method == 'FIVO_K100':
        bundle = _Bundle(prior=prior, decoder=decoder)
        opt = torch.optim.AdamW(list(prior.parameters()) +
                                  list(decoder.parameters()), lr=1e-3)
        fivo = FilteringVariationalObjective(bundle, K=100)
        def step(y):
            opt.zero_grad()
            loss = -fivo.fivo_bound(y).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(prior.parameters()) +
                                              list(decoder.parameters()), 5.0)
            opt.step()
            return loss.item()
        return bundle, opt, step

    if method == 'Laplace_EM':
        encoder = LaplaceConfig(dim=T, num_newton_steps=3).build()
        model = FullModel(encoder=encoder, decoder=decoder, prior=prior).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        def step(y):
            opt.zero_grad()
            loss = -model.elbo(y, ndraws=1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            return loss.item()
        return model, opt, step

    if method == 'SMC_EM_K10' or method == 'SMC_EM_K100':
        K = 10 if method.endswith('K10') else 100
        bundle = _Bundle(prior=prior, decoder=decoder)
        opt = torch.optim.AdamW(list(prior.parameters()) +
                                  list(decoder.parameters()), lr=1e-3)
        def step(y):
            opt.zero_grad()
            with torch.no_grad():
                z_sample = smc_ffbs_sample(prior, decoder, y, K)
            log_pz = prior.log_prob(z_sample)
            log_py_z = decoder.log_likelihood(y, z_sample)
            loss = -(log_py_z + log_pz).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(prior.parameters()) +
                                              list(decoder.parameters()), 5.0)
            opt.step()
            return loss.item()
        return bundle, opt, step

    raise ValueError(method)


def run_method(method, y, n_epochs, checkpoints, T, device, log_p_truth):
    print(f"  --- {method} ---", flush=True)
    np.random.seed(11); torch.manual_seed(11)
    model, opt, step = make_model(method, T, device)
    # ELBO methods have FullModel; FIVO has _Bundle. Both have prior & decoder.
    prior = model.prior
    decoder = model.decoder
    init_prior_at_data(y, prior)
    log_p_track = []
    epoch_track = []
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t_start = time.time()
    last_check = 0

    for epoch in range(n_epochs + 1):
        if epoch in checkpoints:
            with torch.no_grad():
                lp = smc_logp(prior, decoder, y, K=5000)
            wall = time.time() - t_start
            beta_h = float(torch.exp(decoder.log_beta).item())
            sigma_h = float(torch.exp(decoder.log_sigma).item())
            print(f"    epoch {epoch:4d}  log p(y)={lp:.4f}  "
                  f"(gap to truth {log_p_truth - lp:+.4f})  "
                  f"β={beta_h:.3f}  σ_ε={sigma_h:.4f}  ({wall:.1f}s)",
                  flush=True)
            log_p_track.append(lp)
            epoch_track.append(epoch)
        if epoch < n_epochs:
            step(y)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    total_time = time.time() - t_start
    return {
        'epochs': epoch_track, 'log_p': log_p_track,
        'beta_final': float(torch.exp(decoder.log_beta).item()),
        'sigma_eps_final': float(torch.exp(decoder.log_sigma).item()),
        'rho_final': float(prior.rho.item()),
        'sigma_z_final': float(torch.exp(prior.log_std_u).item()),
        'total_time_s': total_time,
    }


def main():
    rho_t, sigma_z_t, scale_eps_t, beta_truth = 0.95, 0.20, 0.10, 2.13
    N, T, seed = 10_000, 10, 11
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    np.random.seed(seed); torch.manual_seed(seed)
    y, sigma_z1_t = simulate_truth(N, T, rho_t, sigma_z_t, scale_eps_t,
                                     beta_truth, seed)
    y = y.to(device)

    truth_prior = TruthLinearAR1Prior(rho=rho_t, sigma_z=sigma_z_t,
                                        sigma_z1=sigma_z1_t, nt=T).to(device)
    truth_decoder = TruthSinhDecoder(scale_eps=scale_eps_t, beta=beta_truth).to(device)
    log_p_truth = smc_logp(truth_prior, truth_decoder, y, K=5000)
    print(f"SMC log p̂(y) @ truth = {log_p_truth:.4f}\n")

    methods = ['ELBO_K1', 'IWAE_K10', 'ELBO_markov_sinh',
                'Laplace_EM', 'SMC_EM_K10', 'SMC_EM_K100',
                'FIVO_K10', 'FIVO_K100']
    checkpoints = {0, 50, 100, 250, 500}
    n_epochs = 500

    out_dir = "output/bundles/cells/speed-climb_per_step"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "climb_per_step.json")
    base = {'truth': {'rho': rho_t, 'sigma_z': sigma_z_t, 'sigma_eps': scale_eps_t,
                      'beta': beta_truth, 'N': N, 'T': T, 'seed': seed,
                      'log_p_truth': log_p_truth},
            'results': {}}
    results = base['results']
    for m in methods:
        results[m] = run_method(m, y, n_epochs=n_epochs, checkpoints=checkpoints,
                                  T=T, device=device, log_p_truth=log_p_truth)
        with open(out_path, 'w') as f:
            json.dump(base, f, indent=2)
        print()
    print(f"Wrote {out_path}\n")

    # Summary
    print(f"{'method':<22s} | " + " | ".join(f"ep={e:>4d}".rjust(10) for e in [0, 50, 100, 250, 500])
           + " | " + f"{'β̂':>5s} {'σ̂_ε':>5s} | {'time(s)':>8s}")
    print(f"{'truth':<22s} | " + " | ".join(f"{log_p_truth:>10.3f}" for _ in [0,50,100,250,500])
           + f" | {beta_truth:>5.2f} {scale_eps_t:>5.3f} | --")
    for m in methods:
        r = results[m]
        cells = [f"{lp:>10.3f}" for lp in r['log_p']]
        print(f"{m:<22s} | " + " | ".join(cells) +
               f" | {r['beta_final']:>5.2f} {r['sigma_eps_final']:>5.3f} | {r['total_time_s']:>8.1f}")

    print("\nClimb rate (nats/epoch on segment [50, 500]):")
    for m in methods:
        r = results[m]
        i50 = r['epochs'].index(50)
        i500 = r['epochs'].index(500)
        rate = (r['log_p'][i500] - r['log_p'][i50]) / (500 - 50)
        print(f"  {m:<22s}  {rate:+.5f} nats/epoch")


if __name__ == "__main__":
    main()
