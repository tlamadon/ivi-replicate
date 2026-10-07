r"""Posterior comparison on the softplus-poly + sinh-z1 + sinh-emission DGP.

DGP (from the simulation-parameters table):
  - mu(z_{t-1})    = -0.25 + 0.1 log[1 + exp(10 (0.9 z + 0.25))]
                     = alpha_0 + alpha_1 log[1 + exp((1/alpha_1)
                                          (mu_0 + mu_1 z + mu_2 z^2 - alpha_0))]
                     with alpha_0 = -0.25, alpha_1 = 0.1,
                     (mu_0, mu_1, mu_2) = (0, 0.9, 0).
  - sigma(z_{t-1}) = softplus(-1.55 + 0.35 z^2)
                     with (sigma_0, sigma_1, sigma_2) = (-1.55, 0, 0.35).
  - z_1 ~ sinh-arcsinh with cdf Phi(sinh(0.89 asinh(z/0.34))),
            i.e. SinhArcsinh(loc=0, scale=0.34, skew=0, tailweight=1/0.89).
  - emission y = z + e, e ~ sinh-arcsinh
            SinhArcsinh(loc=0, scale=0.033, skew=0, tailweight=1/0.47).
  - No MA, no per-individual heterogeneity.
  - N = 30,000, T = 10.

Posteriors:
  - mean-field            (normal_diagonal)
  - joint-normal          (joint_normal)
  - markov                (conditional_markov)
  - joint-normal + sinh   (transformed_joint_normal)
  - markov + sinh         (conditional_sinh_markov)

For each posterior:
  - Full VI from random init on y_obs -> theta_VI, gap@VI, ELBO@VI.
  - Encoder-only fit at the truth (prior+decoder frozen) -> gap@truth.
  - For joint-normal additionally: 5 Picard IVI sweeps with strong CRN
    (fixed sim_noise + fixed VI init seed across the outer loop).

Output: output/bundles/cells/softplus-sinh-posterior-comparison/{encoder}.json + summary.json
"""
import os
import sys
import time
import json

import numpy as np
import torch
import torch.nn.functional as F

torch.distributions.Distribution.set_default_validate_args(False)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mlye.models.encoders.base import (
    JointNormalConfig, MarkovConfig, FlowConfig,
)
from mlye.models.priors.markov import MarkovNormalConditionalPolyPrior
from mlye.models.decoders.ma import MASinhEmission
from mlye.models.full_model import FullModel


# ----------------------------------------------------------------------
# Parameter table -> code conventions
# ----------------------------------------------------------------------
# psi_1 = 0.033, psi_2 = 0.47  for e  (SinhArcsinh scale = psi_1,
#                                       tailweight = 1/psi_2)
# gamma_1 = 0.34, gamma_2 = 0.89 for z_1.
#
# decoder.log_sigma = log(psi_1)
# decoder.log_beta  = log(1/psi_2) = -log(psi_2)
# z1 scale enters via softplus(z1_log_std) so
#   z1_log_std = log(exp(gamma_1) - 1)
# z1_log_tail = -log(gamma_2)
#
# softplus law of motion uses (b, log_gamma) for (alpha_0, log alpha_1)
# in the prior.

PSI_1   = 0.033
PSI_2   = 0.47
GAMMA_1 = 0.34
GAMMA_2 = 0.89

TRUTH = {
    'alpha0':        -0.25,
    'log_alpha1':    float(np.log(0.10)),
    'mu0':            0.0,
    'mu1':            0.9,
    'mu2':            0.0,
    'sigma0':        -1.55,
    'sigma1':         0.0,
    'sigma2':         0.35,
    # softplus(z1_log_std) = gamma_1  (scale of the sinh-arcsinh z1)
    'z1_log_std':     float(np.log(np.exp(GAMMA_1) - 1.0)),
    'z1_skew':        0.0,
    'z1_log_tail':    float(-np.log(GAMMA_2)),       # tailweight = 1/gamma_2
    'log_sigma_eps':  float(np.log(PSI_1)),
    'log_beta':       float(-np.log(PSI_2)),         # tailweight = 1/psi_2
}
PARAM_NAMES = list(TRUTH.keys())


# ----------------------------------------------------------------------
# Encoder / model builders
# ----------------------------------------------------------------------
ENCODER_TYPES = [
    'normal_diagonal',           # mean-field
    'joint_normal',              # joint normal
    'conditional_markov',        # markov
    'transformed_joint_normal',  # joint normal with sinh marginals
    'conditional_sinh_markov',   # markov with sinh marginals
]
HUMAN_LABEL = {
    'normal_diagonal':          'mean-field',
    'joint_normal':             'joint normal',
    'conditional_markov':       'Markov',
    'transformed_joint_normal': 'joint normal + sinh',
    'conditional_sinh_markov':  'Markov + sinh',
}
# Sinh-marginal encoders need lower lr to avoid blowups at random init.
ENCODER_LR = {
    'normal_diagonal':          1e-3,
    'joint_normal':             1e-3,
    'conditional_markov':       1e-3,
    'transformed_joint_normal': 2e-4,
    'conditional_sinh_markov':  2e-4,
}


def build_encoder(encoder_type, T, regularize=1e-3, hidden_dim=32):
    if encoder_type in ('joint_normal', 'normal_diagonal'):
        return JointNormalConfig(dim=T, type=encoder_type,
                                  regularize=regularize,
                                  hidden_dim=hidden_dim).build()
    if encoder_type in ('conditional_markov', 'conditional_sinh_markov'):
        return MarkovConfig(dim=T, type=encoder_type,
                              regularize=regularize,
                              hidden_dim=hidden_dim).build()
    if encoder_type == 'transformed_joint_normal':
        return FlowConfig(dim=T, type='transformed_joint_normal',
                            regularize=regularize).build()
    raise ValueError(f"unknown encoder_type: {encoder_type}")


def build_prior_decoder(T, device):
    prior = MarkovNormalConditionalPolyPrior(
        nt=T, poly_degree=2, law_model='softplus',
        z1_distr='sinh', extra_heterogeneity=False).to(device)
    decoder = MASinhEmission(sigma_eps=PSI_1, theta=0.0, fix_theta=True,
                              sigma_floor=0.0, beta_init=1.0,
                              fix_beta=False).to(device)
    return prior, decoder


def build_full_model(encoder_type, T, device):
    encoder = build_encoder(encoder_type, T).to(device)
    prior, decoder = build_prior_decoder(T, device)
    return FullModel(encoder=encoder, decoder=decoder, prior=prior).to(device)


def set_prior_decoder_to_truth(model, truth):
    with torch.no_grad():
        model.prior.b.fill_(truth['alpha0'])
        model.prior.log_gamma.fill_(truth['log_alpha1'])
        model.prior.net_mu.coeffs.copy_(torch.tensor(
            [truth['mu0'], truth['mu1'], truth['mu2']]))
        model.prior.net_sigma.coeffs.copy_(torch.tensor(
            [truth['sigma0'], truth['sigma1'], truth['sigma2']]))
        model.prior.z1_log_std.fill_(truth['z1_log_std'])
        model.prior.z1_skew.fill_(truth['z1_skew'])
        model.prior.z1_log_tail.fill_(truth['z1_log_tail'])
        model.decoder.log_sigma.fill_(truth['log_sigma_eps'])
        model.decoder.log_beta.fill_(truth['log_beta'])


def extract_params(model) -> dict:
    p = {}
    p['alpha0']       = float(model.prior.b.item())
    p['log_alpha1']   = float(model.prior.log_gamma.item())
    p['mu0']          = float(model.prior.net_mu.coeffs[0].item())
    p['mu1']          = float(model.prior.net_mu.coeffs[1].item())
    p['mu2']          = float(model.prior.net_mu.coeffs[2].item())
    p['sigma0']       = float(model.prior.net_sigma.coeffs[0].item())
    p['sigma1']       = float(model.prior.net_sigma.coeffs[1].item())
    p['sigma2']       = float(model.prior.net_sigma.coeffs[2].item())
    p['z1_log_std']   = float(model.prior.z1_log_std.item())
    p['z1_skew']      = float(model.prior.z1_skew.item())
    p['z1_log_tail']  = float(model.prior.z1_log_tail.item())
    p['log_sigma_eps']= float(model.decoder.log_sigma.item())
    p['log_beta']     = float(model.decoder.log_beta.item())
    return p


# ----------------------------------------------------------------------
# Simulation (CRN-friendly, reparam'd from a single noise tensor)
# ----------------------------------------------------------------------
def reparam_simulate(theta, z_noise, y_noise):
    r"""Differentiable simulate y(theta, z_noise, y_noise).

    theta: (len(PARAM_NAMES),) tensor in PARAM_NAMES order.
    z_noise, y_noise: (N, T) iid N(0, 1).

    Mirrors MarkovNormalConditionalPolyPrior(law_model='softplus',
    z1_distr='sinh') + MASinhEmission(theta=0).
    """
    alpha0       = theta[0]
    log_alpha1   = theta[1]
    mu0          = theta[2]
    mu1          = theta[3]
    mu2          = theta[4]
    sigma0       = theta[5]
    sigma1       = theta[6]
    sigma2       = theta[7]
    z1_log_std   = theta[8]
    z1_skew      = theta[9]
    z1_log_tail  = theta[10]
    log_sigma_eps= theta[11]
    log_beta     = theta[12]

    alpha1   = torch.exp(log_alpha1)
    z1_scale = F.softplus(z1_log_std)
    z1_tail  = torch.exp(z1_log_tail)
    sigma_eps= torch.exp(log_sigma_eps)
    beta     = torch.exp(log_beta)

    N, T = z_noise.shape
    z = torch.zeros(N, T, device=z_noise.device, dtype=z_noise.dtype)
    # z_1: sinh-arcsinh of N(0,1)
    z[:, 0] = z1_scale * torch.sinh((torch.asinh(z_noise[:, 0]) + z1_skew) * z1_tail)
    for t in range(1, T):
        zlag = z[:, t-1].clone()
        poly_mu = mu0 + mu1 * zlag + mu2 * zlag**2
        mu_t = (alpha0 + alpha1 * F.softplus((poly_mu - alpha0) / alpha1)).clamp(-10.0, 10.0)
        poly_sig = sigma0 + sigma1 * zlag + sigma2 * zlag**2
        sig_t = F.softplus(poly_sig).clamp(0.0, 10.0) + 1e-3
        z[:, t] = mu_t + sig_t * z_noise[:, t]
    eps = sigma_eps * torch.sinh(torch.asinh(y_noise) * beta)
    return z + eps


def simulate_from_truth(N, T, truth, seed, device):
    torch.manual_seed(seed)
    z_noise = torch.randn(N, T, device=device)
    y_noise = torch.randn(N, T, device=device)
    theta = torch.tensor([truth[k] for k in PARAM_NAMES], device=device,
                          dtype=torch.float32)
    with torch.no_grad():
        y = reparam_simulate(theta, z_noise, y_noise)
    return y


# ----------------------------------------------------------------------
# Fitting helpers
# ----------------------------------------------------------------------
class NaNFailure(RuntimeError):
    pass


def fit_vi(model, y, n_epochs, lr, log_every=500, label='vi',
            nan_bail_consec=10, nan_bail_window=200):
    """Fit VI. Bails with NaNFailure if NaN persists for >= nan_bail_consec
    consecutive epochs within the first nan_bail_window epochs (signals a
    bad random init that won't recover)."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    t0 = time.time()
    nan_streak = 0
    for epoch in range(n_epochs):
        opt.zero_grad()
        loss = -model.elbo(y, ndraws=1)
        if not torch.isfinite(loss):
            nan_streak += 1
            if epoch < nan_bail_window and nan_streak >= nan_bail_consec:
                raise NaNFailure(f"NaN streak {nan_streak} at epoch {epoch}")
            continue
        nan_streak = 0
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if (epoch + 1) % log_every == 0 or epoch == 0:
            elbo = -loss.item()
            p = extract_params(model)
            print(f"  [{label}] epoch {epoch+1:4d}: ELBO={elbo:+.3f}  "
                  f"mu1={p['mu1']:+.3f}  sigma2={p['sigma2']:+.3f}  "
                  f"log_b={p['log_beta']:+.3f}  log_se={p['log_sigma_eps']:+.3f}  "
                  f"({time.time()-t0:.1f}s)", flush=True)
    return model


def _retry_seeds(base_seed, n_retries):
    """Stable offsets 0, 1, 2, ..., n_retries-1 (deterministic)."""
    return [base_seed + i for i in range(n_retries)]


def fit_vi_from_init(encoder_type, y, T, vi_seed, n_epochs, lr, device,
                      init_truth=False, init_theta=None, n_retries=10):
    last_err = None
    for s in _retry_seeds(vi_seed, n_retries):
        try:
            torch.manual_seed(s); np.random.seed(s)
            model = build_full_model(encoder_type, T, device)
            if init_truth:
                set_prior_decoder_to_truth(model, TRUTH)
            elif init_theta is not None:
                with torch.no_grad():
                    for k, v in init_theta.items():
                        pass
            fit_vi(model, y, n_epochs=n_epochs, lr=lr,
                    label=f'VI/{encoder_type[:6]}/s{s}')
            return model, extract_params(model)
        except NaNFailure as e:
            print(f"  [VI/{encoder_type[:6]}/s{s}] NaN bail: {e}; retrying",
                  flush=True)
            last_err = e
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    raise RuntimeError(f"VI fit failed after {n_retries} seed retries "
                        f"(last err: {last_err})")


def fit_encoder_only_at_truth(encoder_type, y, T, vi_seed, n_epochs, lr,
                                  device, n_retries=10):
    last_err = None
    for s in _retry_seeds(vi_seed, n_retries):
        try:
            torch.manual_seed(s); np.random.seed(s)
            model = build_full_model(encoder_type, T, device)
            set_prior_decoder_to_truth(model, TRUTH)
            for p in model.prior.parameters():
                p.requires_grad_(False)
            for p in model.decoder.parameters():
                p.requires_grad_(False)
            opt = torch.optim.AdamW(
                [p for p in model.encoder.parameters() if p.requires_grad], lr=lr)
            t0 = time.time()
            nan_streak = 0
            for epoch in range(n_epochs):
                opt.zero_grad()
                loss = -model.elbo(y, ndraws=1)
                if not torch.isfinite(loss):
                    nan_streak += 1
                    if epoch < 200 and nan_streak >= 10:
                        raise NaNFailure(f"NaN at epoch {epoch}")
                    continue
                nan_streak = 0
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), 5.0)
                opt.step()
                if (epoch + 1) % 1000 == 0 or epoch == 0:
                    print(f"  [enc@truth/{encoder_type[:6]}/s{s}] "
                          f"epoch {epoch+1:4d}: ELBO={-loss.item():+.3f}  "
                          f"({time.time()-t0:.1f}s)", flush=True)
            return model
        except NaNFailure as e:
            print(f"  [enc@truth/{encoder_type[:6]}/s{s}] NaN bail: {e}; "
                  f"retrying", flush=True)
            last_err = e
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    raise RuntimeError(f"enc@truth fit failed after {n_retries} seed retries "
                        f"(last err: {last_err})")


def gap_iwae_minus_elbo(model, y, ndraws=100):
    return model.elbo_diagnostics(y, ndraws=ndraws)


# ----------------------------------------------------------------------
# Picard IVI (full VI at each step, fixed-point update, strong CRN)
# ----------------------------------------------------------------------
def picard_iter(encoder_type, theta_0_vec, theta_VI_obs_vec, N, T,
                  sim_seed, vi_seed, n_epochs, lr, device):
    """One Picard step:
      y_sim    = simulate(theta_0; fixed noise)
      theta_VI = full VI fit on y_sim (fixed init seed)
      F_x      = theta_VI_obs - (theta_VI - theta_0)
    """
    theta_0_t = torch.tensor(theta_0_vec, dtype=torch.float32, device=device)
    torch.manual_seed(sim_seed)
    z_noise = torch.randn(N, T, device=device)
    y_noise = torch.randn(N, T, device=device)
    with torch.no_grad():
        y_sim = reparam_simulate(theta_0_t, z_noise, y_noise)
    _, theta_VI = fit_vi_from_init(encoder_type, y_sim, T, vi_seed,
                                      n_epochs, lr, device)
    theta_VI_vec = np.array([theta_VI[k] for k in PARAM_NAMES],
                              dtype=np.float64)
    F_x = theta_VI_obs_vec - (theta_VI_vec - theta_0_vec)
    return F_x, theta_VI_vec


# ----------------------------------------------------------------------
# Per-encoder pipeline
# ----------------------------------------------------------------------
def run_one_encoder(encoder_type, y_obs, truth_vec, N, T, obs_seed,
                       n_epochs, lr, K_iter, device, do_picard):
    print(f"\n{'='*80}")
    print(f"== Encoder: {encoder_type}  ({HUMAN_LABEL[encoder_type]}) ==")
    print(f"{'='*80}", flush=True)
    # Use a slower lr for sinh-marginal encoders (init can blow up).
    lr_enc = ENCODER_LR.get(encoder_type, lr)
    if lr_enc != lr:
        print(f"  (using encoder-specific lr={lr_enc})", flush=True)
    record = {'encoder_type': encoder_type,
              'human_label':  HUMAN_LABEL[encoder_type],
              'lr': lr_enc}

    # ---- [A] Full VI from random init on y_obs ----
    print(f"\n[A] Full VI from random init on y_obs ...", flush=True)
    t0 = time.time()
    model_obs, theta_VI = fit_vi_from_init(encoder_type, y_obs, T, obs_seed,
                                              n_epochs, lr_enc, device)
    t_full = time.time() - t0
    theta_VI_vec = np.array([theta_VI[k] for k in PARAM_NAMES],
                              dtype=np.float64)
    L2_VI = float(np.linalg.norm(theta_VI_vec - truth_vec))
    diag_VI = gap_iwae_minus_elbo(model_obs, y_obs, ndraws=100)
    print(f"  full VI: L2={L2_VI:.4f}  ELBO={diag_VI['elbo']:+.3f}  "
          f"IWAE={diag_VI['iwae']:+.3f}  gap@VI={diag_VI['gap']:.4f}  "
          f"({t_full:.1f}s)", flush=True)
    record['theta_VI']      = {k: float(theta_VI[k]) for k in PARAM_NAMES}
    record['L2_VI']         = L2_VI
    record['elbo_VI']       = float(diag_VI['elbo'])
    record['iwae_VI']       = float(diag_VI['iwae'])
    record['gap_VI']        = float(diag_VI['gap'])
    record['wall_full_vi_s']= t_full

    del model_obs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- [B] Encoder-only fit at truth (prior+decoder frozen) ----
    print(f"\n[B] Encoder-only fit at truth (prior+decoder frozen) ...",
          flush=True)
    t0 = time.time()
    model_truth = fit_encoder_only_at_truth(
        encoder_type, y_obs, T, obs_seed + 1, n_epochs, lr_enc, device)
    t_enc = time.time() - t0
    diag_truth = gap_iwae_minus_elbo(model_truth, y_obs, ndraws=100)
    print(f"  encoder@truth: ELBO={diag_truth['elbo']:+.3f}  "
          f"IWAE={diag_truth['iwae']:+.3f}  "
          f"gap@truth={diag_truth['gap']:.4f}  ({t_enc:.1f}s)", flush=True)
    record['elbo_truth']    = float(diag_truth['elbo'])
    record['iwae_truth']    = float(diag_truth['iwae'])
    record['gap_truth']     = float(diag_truth['gap'])
    record['wall_enc_s']    = t_enc

    del model_truth
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- [C] Picard IVI sweeps (joint-normal only) ----
    if do_picard:
        print(f"\n[C] Picard IVI ({K_iter} sweeps, strong CRN) ...",
              flush=True)
        sim_seed = obs_seed * 1000          # CRN: fixed across outer loop
        vi_seed  = obs_seed * 1000 + 7      # CRN: fixed across outer loop
        x = theta_VI_vec.copy()
        traj = [{'iter': 0,
                  'theta_0': x.tolist(),
                  'L2_truth': L2_VI}]
        t0 = time.time()
        for k in range(1, K_iter + 1):
            F_x, theta_VI_x = picard_iter(encoder_type, x, theta_VI_vec, N, T,
                                                sim_seed, vi_seed,
                                                n_epochs, lr_enc, device)
            x = F_x
            L2 = float(np.linalg.norm(x - truth_vec))
            wall = time.time() - t0
            print(f"  Picard iter {k}/{K_iter}: L2={L2:.4f}  "
                  f"|theta_VI(x)-theta_VI(obs)|="
                  f"{float(np.linalg.norm(theta_VI_x - theta_VI_vec)):.4f}  "
                  f"({wall:.1f}s)", flush=True)
            traj.append({'iter': k,
                          'theta_0': x.tolist(),
                          'L2_truth': L2,
                          'wall_s': wall,
                          'theta_VI_x': theta_VI_x.tolist()})
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        theta_IVI_vec = x.copy()
        L2_IVI = float(np.linalg.norm(theta_IVI_vec - truth_vec))
        record['theta_IVI']         = {k: float(theta_IVI_vec[i])
                                          for i, k in enumerate(PARAM_NAMES)}
        record['L2_IVI']            = L2_IVI
        record['picard_trajectory'] = traj
        record['wall_picard_s']     = traj[-1]['wall_s']
        print(f"  Picard final: L2={L2_IVI:.4f}  "
              f"(improvement {L2_VI - L2_IVI:+.4f})", flush=True)
    return record


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)
    N, T, obs_seed = 30_000, 10, 11
    n_epochs, lr = 3000, 1e-3
    K_iter = 5
    truth = TRUTH
    truth_vec = np.array([truth[k] for k in PARAM_NAMES], dtype=np.float64)

    print(f"\nTRUTH (softplus law of motion + sinh z1 + sinh emission, "
          f"no MA, no extra hetero):")
    for k in PARAM_NAMES:
        print(f"  {k:<16s} = {truth[k]:+.5f}")
    print()

    out_dir = "output/bundles/cells/softplus-sinh-posterior-comparison"
    os.makedirs(out_dir, exist_ok=True)

    # ---- simulate y_obs ----
    print(f"=== Simulating y_obs (N={N}, T={T}, seed={obs_seed}) ===",
          flush=True)
    y_obs = simulate_from_truth(N, T, truth, obs_seed, device)
    print(f"  y_obs: mean={y_obs.mean().item():+.4f}  "
          f"std={y_obs.std().item():.4f}  "
          f"q05={y_obs.flatten().quantile(0.05).item():+.4f}  "
          f"q95={y_obs.flatten().quantile(0.95).item():+.4f}",
          flush=True)

    # ---- run each encoder ----
    summary = {
        'truth': truth,
        'param_names': PARAM_NAMES,
        'config': {'N': N, 'T': T, 'obs_seed': obs_seed,
                    'n_epochs': n_epochs, 'lr': lr,
                    'K_iter': K_iter,
                    'psi_1': PSI_1, 'psi_2': PSI_2,
                    'gamma_1': GAMMA_1, 'gamma_2': GAMMA_2},
        'encoders': {},
    }

    for encoder_type in ENCODER_TYPES:
        out_path = os.path.join(out_dir, f"{encoder_type}.json")
        if os.path.exists(out_path):
            try:
                rec = json.load(open(out_path))
                print(f"\nSkipping {encoder_type} (cached at {out_path})",
                      flush=True)
                summary['encoders'][encoder_type] = rec
                continue
            except json.JSONDecodeError:
                pass

        do_picard = (encoder_type == 'joint_normal')
        try:
            rec = run_one_encoder(encoder_type, y_obs, truth_vec,
                                       N, T, obs_seed, n_epochs, lr,
                                       K_iter, device, do_picard)
        except Exception as e:
            print(f"\n!! Encoder {encoder_type} failed: {e}", flush=True)
            import traceback
            traceback.print_exc()
            rec = {'encoder_type': encoder_type, 'error': str(e)}

        with open(out_path, 'w') as f:
            json.dump(rec, f, indent=2)
        print(f"  wrote {out_path}", flush=True)
        summary['encoders'][encoder_type] = rec

    # ---- summary table ----
    print(f"\n{'='*110}")
    print(f"== Summary: softplus + sinh DGP, N={N}, T={T} ==")
    print(f"{'='*110}")
    print(f"  {'encoder':<26s} {'ELBO@VI':>10s} {'IWAE@VI':>10s} "
          f"{'gap@VI':>8s} {'ELBO@truth':>11s} {'IWAE@truth':>11s} "
          f"{'gap@truth':>10s} {'L2_VI':>8s} {'L2_IVI':>8s}")
    print('-' * 110)
    for et in ENCODER_TYPES:
        rec = summary['encoders'].get(et, {})
        if 'error' in rec:
            print(f"  {HUMAN_LABEL[et]:<26s} FAILED: {rec['error']}")
            continue
        l2_ivi = rec.get('L2_IVI', float('nan'))
        print(f"  {HUMAN_LABEL[et]:<26s} {rec['elbo_VI']:>10.3f} "
              f"{rec['iwae_VI']:>10.3f} {rec['gap_VI']:>8.4f} "
              f"{rec['elbo_truth']:>11.3f} {rec['iwae_truth']:>11.3f} "
              f"{rec['gap_truth']:>10.4f} {rec['L2_VI']:>8.4f} "
              f"{l2_ivi:>8.4f}")
    print('=' * 110)

    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
