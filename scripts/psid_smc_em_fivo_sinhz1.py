"""Same as psid_smc_em_fivo.py but with **sinh-arcsinh z_1**.

Model:
  prior: MarkovNormalConditionalPolyPrior(poly_degree=2, law_model='poly',
                                          z1_distr='sinh',
                                          extra_heterogeneity=False)
    -> mu(z)    = a0 + a1 z + a2 z^2          (mu_3 effectively set to 0)
    -> sigma(z) = softplus(b0 + b1 z + b2 z^2) (sigma_3 effectively set to 0)
    -> z_1     ~ SinhArcsinh(loc=0, scale=softplus(z1_log_std),
                              skew=z1_skew, tailweight=exp(z1_log_tail))
  decoder: MASinhEmission(theta=0 fixed, beta free, sigma_eps free).

Fit by SMC EM and bootstrap FIVO; report BHHH SEs (FIVO scores). MA(1)
disabled (theta=0) for the same reasons as the parent script.

Writes JSON to output/bundles/cells/employment-psid_smc_em_fivo_sinhz1/.
"""
import os
import sys
import time
import json

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from smc_truth_profile import _Bundle
from climb_per_step import smc_ffbs_sample

from mlye.models.priors.markov import MarkovNormalConditionalPolyPrior
from mlye.models.decoders.ma import MASinhEmission
from mlye.models.encoders import JointNormalConfig
from mlye.models.full_model import FullModel
from mlye.eval import FilteringVariationalObjective, BootstrapParticleFilter


# ----- model construction --------------------------------------------------

def make_model(T: int, device, init_sigma: float = 0.10,
                init_beta: float = 1.5, seed: int = 11):
    """Poly(degree=2) prior, **sinh** z_1, sinh-arcsinh decoder (theta=0)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    prior = MarkovNormalConditionalPolyPrior(
        nt=T, hidden_dim=32, poly_degree=2,
        z1_distr='sinh', law_model='poly',
        extra_heterogeneity=False,
        sigma_clamp=10.0, mu_clamp=10.0, skewed=False,
    ).to(device)
    # Init AR(0.9), modest spread, near-Gaussian z_1.
    with torch.no_grad():
        prior.net_mu.coeffs.copy_(torch.tensor([0.0, 0.9, 0.0]))
        prior.net_sigma.coeffs.copy_(torch.tensor([-1.0, 0.0, 0.0]))
        # softplus(0)~0.69 for the z_1 scale; neutral skew; tail=1 (Gaussian).
        prior.z1_log_std.fill_(0.0)
        prior.z1_skew.fill_(0.0)
        prior.z1_log_tail.fill_(0.0)
    decoder = MASinhEmission(
        sigma_eps=init_sigma, theta=0.0, fix_theta=True,
        sigma_floor=0.0, beta_init=init_beta, fix_beta=False,
    ).to(device)
    return prior, decoder


# ----- training loops ------------------------------------------------------

def fit_fivo(y: torch.Tensor, K: int = 200, n_epochs: int = 8000,
              lr: float = 1e-2, log_every: int = 500,
              clip: float = 5.0, seed: int = 11):
    T = y.shape[1]
    prior, decoder = make_model(T, y.device, seed=seed)
    bundle = _Bundle(prior=prior, decoder=decoder)
    params = list(prior.parameters()) + list(decoder.parameters())
    opt = torch.optim.AdamW(params, lr=lr)
    fivo = FilteringVariationalObjective(bundle, K=K)

    history = []
    t0 = time.time()
    for epoch in range(n_epochs):
        opt.zero_grad()
        loss = -fivo.fivo_bound(y).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, clip)
        opt.step()
        if epoch % log_every == 0 or epoch == n_epochs - 1:
            history.append({'epoch': epoch, 'fivo_bound': float(-loss.item())})
            print(f"  [FIVO  ] epoch {epoch:5d}  bound={-loss.item():.4f}  "
                  f"sigma_eps={float(torch.exp(decoder.log_sigma).item()):.4f}  "
                  f"beta={float(torch.exp(decoder.log_beta).item()):.3f}  "
                  f"z1_skew={float(prior.z1_skew.item()):+.3f}  "
                  f"z1_tail={float(torch.exp(prior.z1_log_tail).item()):.3f}",
                  flush=True)
    print(f"  FIVO trained in {time.time()-t0:.1f}s", flush=True)
    return prior, decoder, history


def fit_smc_em(y: torch.Tensor, K: int = 200, n_epochs: int = 4000,
                lr: float = 1e-2, log_every: int = 250,
                clip: float = 5.0, seed: int = 11):
    T = y.shape[1]
    prior, decoder = make_model(T, y.device, seed=seed)
    params = list(prior.parameters()) + list(decoder.parameters())
    opt = torch.optim.AdamW(params, lr=lr)

    history = []
    t0 = time.time()
    for epoch in range(n_epochs):
        opt.zero_grad()
        with torch.no_grad():
            z_sample = smc_ffbs_sample(prior, decoder, y, K)
        log_pz = prior.log_prob(z_sample)
        log_py_z = decoder.log_likelihood(y, z_sample)
        loss = -(log_py_z + log_pz).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, clip)
        opt.step()
        if epoch % log_every == 0 or epoch == n_epochs - 1:
            history.append({'epoch': epoch, 'm_step_obj': float(-loss.item())})
            print(f"  [SMC EM] epoch {epoch:5d}  M-step obj={-loss.item():.4f}  "
                  f"sigma_eps={float(torch.exp(decoder.log_sigma).item()):.4f}  "
                  f"beta={float(torch.exp(decoder.log_beta).item()):.3f}  "
                  f"z1_skew={float(prior.z1_skew.item()):+.3f}  "
                  f"z1_tail={float(torch.exp(prior.z1_log_tail).item()):.3f}",
                  flush=True)
    print(f"  SMC EM trained in {time.time()-t0:.1f}s", flush=True)
    return prior, decoder, history


def fit_vi_iwae(y: torch.Tensor, hidden_dim: int = 32,
                 n_epochs: int = 8000, lr: float = 1e-3,
                 ndraws_iwae: int = 100, log_every: int = 200,
                 clip: float = 5.0, seed: int = 11):
    """Joint VI fit (encoder + prior + decoder, ELBO objective), evaluated
    by the IWAE bound at L=ndraws_iwae at convergence. Per estimators.md
    §2.1 (joint-normal H=32) and §2.2 (IWAE L=100)."""
    T = y.shape[1]
    prior, decoder = make_model(T, y.device, seed=seed)
    torch.manual_seed(seed)
    encoder = JointNormalConfig(
        dim=T, type='joint_normal', regularize=1e-3, hidden_dim=hidden_dim,
    ).build().to(y.device)
    model = FullModel(encoder=encoder, decoder=decoder, prior=prior).to(y.device)
    params = list(model.parameters())
    opt = torch.optim.AdamW(params, lr=lr)

    history = []
    t0 = time.time()
    for epoch in range(n_epochs):
        opt.zero_grad()
        loss = -model.elbo(y, ndraws=1)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, clip)
        opt.step()
        if epoch % log_every == 0 or epoch == n_epochs - 1:
            history.append({'epoch': epoch, 'elbo': float(-loss.item())})
            print(f"  [IWAE  ] epoch {epoch:5d}  ELBO={-loss.item():.4f}  "
                  f"sigma_eps={float(torch.exp(decoder.log_sigma).item()):.4f}  "
                  f"beta={float(torch.exp(decoder.log_beta).item()):.3f}  "
                  f"z1_skew={float(prior.z1_skew.item()):+.3f}  "
                  f"z1_tail={float(torch.exp(prior.z1_log_tail).item()):.3f}",
                  flush=True)
    print(f"  VI trained in {time.time()-t0:.1f}s", flush=True)

    with torch.no_grad():
        iwae_bound = float(model.elbo_iwae(y, ndraws=ndraws_iwae).item())
    print(f"  IWAE bound (L={ndraws_iwae}) at convergence: {iwae_bound:.4f}",
          flush=True)
    return prior, decoder, encoder, history, iwae_bound


# ----- parameter listing / transforms --------------------------------------

def named_param_list(prior, decoder):
    out = []
    for n, p in prior.named_parameters():
        if p.requires_grad:
            out.append((f"prior.{n}", p))
    for n, p in decoder.named_parameters():
        if p.requires_grad:
            out.append((f"decoder.{n}", p))
    return out


def flatten_raw_params(prior, decoder):
    return np.concatenate([
        p.detach().double().cpu().numpy().flatten()
        for _, p in named_param_list(prior, decoder)
    ])


def flat_index_map(prior, decoder):
    mapping = {}
    offset = 0
    for n, p in named_param_list(prior, decoder):
        k = p.numel()
        mapping[n] = (offset, offset + k)
        offset += k
    return mapping


def compute_bhhh_se(prior, decoder, y, K: int = 500, n_reps: int = 3,
                      seed_base: int = 11):
    bundle = _Bundle(prior=prior, decoder=decoder)
    fivo = FilteringVariationalObjective(bundle, K=K)
    params = [p for _, p in named_param_list(prior, decoder)]
    N = y.shape[0]
    P = sum(p.numel() for p in params)
    device = y.device

    scores_avg = torch.zeros(N, P, device=device, dtype=torch.float64)
    t0 = time.time()
    for rep in range(n_reps):
        torch.manual_seed(seed_base * 1000 + rep)
        for i in range(N):
            y_i = y[i:i + 1]
            log_p_i = fivo.fivo_bound(y_i).mean()
            grads = torch.autograd.grad(log_p_i, params, retain_graph=False)
            s_i = torch.cat([g.detach().flatten() for g in grads]).double()
            scores_avg[i] += s_i / n_reps
        print(f"  BHHH rep {rep+1}/{n_reps} done ({time.time()-t0:.1f}s)", flush=True)

    I_hat = scores_avg.T @ scores_avg
    I_hat_np = I_hat.cpu().numpy()
    cov = np.linalg.pinv(I_hat_np)
    se = np.sqrt(np.diag(cov))
    return se, I_hat_np


def assemble_table(prior, decoder, se):
    raw_vals = flatten_raw_params(prior, decoder)
    idx = flat_index_map(prior, decoder)
    rows = []
    for name in idx:
        s, e = idx[name]
        for k in range(s, e):
            sub = ""
            if e - s > 1:
                sub = f"[{k - s}]"
            rows.append((f"{name}{sub}", float(raw_vals[k]), float(se[k])))

    derived = []
    log_sigma_idx = idx['decoder.log_sigma'][0]
    log_beta_idx  = idx['decoder.log_beta'][0]
    sigma_eps = float(np.exp(raw_vals[log_sigma_idx]))
    beta      = float(np.exp(raw_vals[log_beta_idx]))
    se_sigma_eps = sigma_eps * float(se[log_sigma_idx])
    se_beta      = beta      * float(se[log_beta_idx])
    derived.append(("sigma_eps (=exp(log_sigma))", sigma_eps, se_sigma_eps))
    derived.append(("beta (=exp(log_beta))",        beta,      se_beta))

    mu_s, _ = idx['prior.net_mu.coeffs']
    sg_s, _ = idx['prior.net_sigma.coeffs']
    z1_ls = idx['prior.z1_log_std'][0]
    z1_lt = idx['prior.z1_log_tail'][0]
    z1_sk = idx['prior.z1_skew'][0]
    mu1   = float(raw_vals[mu_s + 1])
    se_mu1 = float(se[mu_s + 1])
    sg0 = float(raw_vals[sg_s])
    sigma_at_zero = float(np.log1p(np.exp(sg0))) + 1e-3 if sg0 < 30 else sg0 + 1e-3
    deriv_at_zero = float(1.0 / (1.0 + np.exp(-sg0)))
    se_sigma_at_zero = deriv_at_zero * float(se[sg_s])
    raw_ls = float(raw_vals[z1_ls])
    sigma_z1 = float(np.log1p(np.exp(raw_ls))) if raw_ls < 30 else raw_ls
    deriv_ls = float(1.0 / (1.0 + np.exp(-raw_ls)))
    se_sigma_z1 = deriv_ls * float(se[z1_ls])
    raw_lt = float(raw_vals[z1_lt])
    tail_z1 = float(np.exp(raw_lt))
    se_tail_z1 = tail_z1 * float(se[z1_lt])
    derived.append(("mu_1 (AR slope at z=0)",                mu1,           se_mu1))
    derived.append(("sigma(0) (=softplus(sigma_0)+1e-3)",    sigma_at_zero, se_sigma_at_zero))
    derived.append(("sigma_{z_1} (=softplus(z1_log_std))",   sigma_z1,      se_sigma_z1))
    derived.append(("z1_skew",                                float(raw_vals[z1_sk]),
                                                              float(se[z1_sk])))
    derived.append(("z1_tail (=exp(z1_log_tail))",            tail_z1,      se_tail_z1))

    return rows, derived


def print_table(label, rows, derived):
    print(f"\n=== {label} ===")
    print(f"{'parameter':<36}  {'estimate':>10}  {'SE':>10}")
    print("-" * 64)
    for name, v, s in rows:
        print(f"{name:<36}  {v:>10.4f}  {s:>10.4f}")
    print("\n  ---- derived ----")
    for name, v, s in derived:
        print(f"{name:<36}  {v:>10.4f}  {s:>10.4f}")


def main():
    print("Loading PSID earnings panel.", flush=True)
    y_np = np.load("data/abb_y_matrix.npy")
    y = torch.from_numpy(y_np).float()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    y = y.to(device)
    N, T = y.shape
    print(f"  shape: N={N}, T={T}, device={device}", flush=True)
    print(f"  empirical mean={y.mean().item():+.4f}, std={y.std().item():.4f}, "
          f"std(diff y)={torch.diff(y, dim=1).std().item():.4f}", flush=True)

    print("\n=== Fitting FIVO (poly2 + sinh-z1, K=200) ===")
    prior_fivo, decoder_fivo, hist_fivo = fit_fivo(
        y, K=200, n_epochs=8000, lr=1e-2)

    bundle_fivo = _Bundle(prior=prior_fivo, decoder=decoder_fivo)
    pf = BootstrapParticleFilter(bundle_fivo, K=5000)
    smc_logp_fivo = float(pf.log_marginal(y).mean().item())
    print(f"  SMC log p(y) at FIVO fit: {smc_logp_fivo:.4f}")

    print("\n=== BHHH SEs at the FIVO fit (FIVO scores, K=500, 3 reps) ===")
    se_fivo, info_fivo = compute_bhhh_se(prior_fivo, decoder_fivo, y,
                                           K=500, n_reps=3, seed_base=11)
    rows_fivo, derived_fivo = assemble_table(prior_fivo, decoder_fivo, se_fivo)
    print_table("FIVO fit (sinh-z1)", rows_fivo, derived_fivo)

    print("\n=== Fitting SMC EM (poly2 + sinh-z1, K=200) ===")
    prior_smc, decoder_smc, hist_smc = fit_smc_em(
        y, K=200, n_epochs=4000, lr=1e-2)

    bundle_smc = _Bundle(prior=prior_smc, decoder=decoder_smc)
    pf = BootstrapParticleFilter(bundle_smc, K=5000)
    smc_logp_smc = float(pf.log_marginal(y).mean().item())
    print(f"  SMC log p(y) at SMC EM fit: {smc_logp_smc:.4f}")

    print("\n=== BHHH SEs at the SMC EM fit (FIVO scores, K=500, 3 reps) ===")
    se_smc, info_smc = compute_bhhh_se(prior_smc, decoder_smc, y,
                                         K=500, n_reps=3, seed_base=11)
    rows_smc, derived_smc = assemble_table(prior_smc, decoder_smc, se_smc)
    print_table("SMC EM fit (sinh-z1)", rows_smc, derived_smc)

    print("\n=== Side-by-side (estimate ± SE) ===")
    print(f"{'parameter':<36}  {'SMC EM':>20}  {'FIVO':>20}")
    print("-" * 84)
    for (n1, v1, s1), (n2, v2, s2) in zip(rows_smc, rows_fivo):
        assert n1 == n2
        print(f"{n1:<36}  {v1:>+9.4f} ± {s1:<7.4f}  {v2:>+9.4f} ± {s2:<7.4f}")
    print("\n  ---- derived ----")
    for (n1, v1, s1), (n2, v2, s2) in zip(derived_smc, derived_fivo):
        assert n1 == n2
        print(f"{n1:<36}  {v1:>+9.4f} ± {s1:<7.4f}  {v2:>+9.4f} ± {s2:<7.4f}")

    print("\n=== Fitting joint VI (poly2 + sinh-z1, H=32) for IWAE ===")
    prior_vi, decoder_vi, encoder_vi, hist_vi, iwae_bound = fit_vi_iwae(
        y, hidden_dim=32, n_epochs=8000, lr=1e-3, ndraws_iwae=100)

    bundle_vi = _Bundle(prior=prior_vi, decoder=decoder_vi)
    pf = BootstrapParticleFilter(bundle_vi, K=5000)
    smc_logp_vi = float(pf.log_marginal(y).mean().item())
    print(f"  SMC log p(y) at VI fit: {smc_logp_vi:.4f}")
    print(f"  IWAE bound (L=100) at VI fit: {iwae_bound:.4f}")

    print("\n=== BHHH SEs at the VI fit (FIVO scores, K=500, 3 reps) ===")
    se_vi, info_vi = compute_bhhh_se(prior_vi, decoder_vi, y,
                                       K=500, n_reps=3, seed_base=11)
    rows_vi, derived_vi = assemble_table(prior_vi, decoder_vi, se_vi)
    print_table("VI fit (sinh-z1) -- IWAE", rows_vi, derived_vi)

    print(f"\nSMC log p(y) at SMC EM fit: {smc_logp_smc:.4f}")
    print(f"SMC log p(y) at FIVO  fit: {smc_logp_fivo:.4f}")
    print(f"SMC log p(y) at VI   fit: {smc_logp_vi:.4f}")
    print(f"IWAE bound (L=100)        : {iwae_bound:.4f}")

    out_dir = "output/bundles/cells/employment-psid_smc_em_fivo_sinhz1"
    os.makedirs(out_dir, exist_ok=True)
    payload = {
        'data': {'source': 'data/abb_y_matrix.npy', 'N': N, 'T': T,
                  'mean_y': float(y.mean().item()),
                  'std_y': float(y.std().item())},
        'model': {'prior': 'MarkovNormalConditionalPolyPrior(law_model=poly, '
                            'poly_degree=2, z1_distr=sinh, '
                            'extra_heterogeneity=False)',
                   'decoder': 'MASinhEmission(beta free, theta=0 fixed, '
                              'sigma_eps free, sigma_floor=0)',
                   'note': 'mu_3 and sigma_3 implicit (set to 0 by poly_degree=2)'},
        'training': {'fivo': {'K': 200, 'n_epochs': 8000, 'lr': 1e-2,
                                'history': hist_fivo},
                      'smc_em': {'K': 200, 'n_epochs': 4000, 'lr': 1e-2,
                                  'history': hist_smc},
                      'iwae': {'encoder': 'joint_normal',
                                'hidden_dim': 32,
                                'n_epochs': 8000, 'lr': 1e-3,
                                'ndraws_iwae': 100,
                                'history': hist_vi}},
        'estimates': {
            'fivo': {
                'raw': [{'name': n, 'estimate': v, 'se': s}
                        for n, v, s in rows_fivo],
                'derived': [{'name': n, 'estimate': v, 'se': s}
                            for n, v, s in derived_fivo],
                'smc_log_p': smc_logp_fivo,
            },
            'smc_em': {
                'raw': [{'name': n, 'estimate': v, 'se': s}
                        for n, v, s in rows_smc],
                'derived': [{'name': n, 'estimate': v, 'se': s}
                            for n, v, s in derived_smc],
                'smc_log_p': smc_logp_smc,
            },
            'iwae': {
                'raw': [{'name': n, 'estimate': v, 'se': s}
                        for n, v, s in rows_vi],
                'derived': [{'name': n, 'estimate': v, 'se': s}
                            for n, v, s in derived_vi],
                'smc_log_p': smc_logp_vi,
                'iwae_bound': iwae_bound,
            },
        },
        'se_method': {'method': 'BHHH (per-individual FIVO scores)',
                       'K_scores': 500, 'n_reps_scores': 3, 'seed_base': 11},
    }
    out_path = os.path.join(out_dir, "psid_smc_em_fivo_sinhz1.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
