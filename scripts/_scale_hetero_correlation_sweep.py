r"""Joint-normal VI on a scale-heterogeneity DGP, sweeping the correlation
between alpha (log-residual-scale) and z_1.

DGP (drops the hockey-stick softplus kink; keeps everything else of the
simulation-hockeystick truth):
  - prior:  MarkovNormalConditionalPolyPrior(law_model='poly', poly_deg=2,
            z1_distr='sinh', extra_heterogeneity=True, extra_prior_type='iid')
  - decoder: MASinhEmissionHetero(hetero_mode='scale', theta=0, fix_theta=True)

The "extra" latent alpha_i plays the role of log(sigma_i): y_t = z_t +
exp(alpha_i) * sinh-arcsinh(0,1,0,beta) noise. We parametrize alpha so
the **marginal of alpha is held fixed** as we sweep the conditional
correlation rho = corr(alpha, z_1):

  alpha | z_1 ~ N(beta_a0 + beta_a1 * z_1, sigma_a_cond)
  with
    beta_a0     = mu_alpha_marg               (E[z_1]=0 by sinh-arcsinh sym)
    beta_a1     = rho * sigma_alpha_marg / sigma_{z_1}
    sigma_a_cond = sigma_alpha_marg * sqrt(1 - rho^2)
    sigma_{z_1} = softplus(z1_log_std)

So sweeping rho in [0, 1) keeps marginal alpha ~ N(mu_alpha_marg,
sigma_alpha_marg^2) constant; only the slope through z_1 and the
conditional spread change. This isolates the **identification of the
correlation channel** from any change in the marginal of the residual
scale.

For each rho in {0.0, 0.5, 0.9}:
  1. set truth, simulate (N, T) data;
  2. fit joint-normal h=64 encoder (with 1 extra latent for alpha) +
     prior + decoder via joint VI (Phase-1 hyperparameters: 20000 epochs,
     lr=1e-2, step-to-step CRN);
  3. report ELBO, recovered (mu_alpha, sigma_alpha_marg, rho), and L2
     against the true 16-vec.

Output: output/bundles/cells/_scale_hetero_correlation_sweep/sweep.json
"""
import os
import sys
import time
import json

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlye.models.priors.markov import MarkovNormalConditionalPolyPrior
from mlye.models.decoders.ma import MASinhEmissionHetero
from mlye.models.encoders.base import JointNormalConfig
from mlye.models.full_model import FullModel


# ---- Truth calibration (drop the hockey-stick softplus, keep the rest) ---

TRUTH_BASE = {
    # poly-2 mu (a0 + a1 z + a2 z^2) — no kink
    'mu0':          0.0,
    'mu1':          0.9,
    'mu2':          0.0,
    # poly-2 sigma  (softplus(c0 + c1 z + c2 z^2))
    'sigma0':       -1.8,
    'sigma1':       0.0,
    'sigma2':       0.35,
    # z_1 sinh-arcsinh
    'z1_log_std':   -0.904,
    'z1_skew':      0.0,
    'z1_log_tail':  0.117,
    # MASinhEmissionHetero (hetero_mode='scale') has no log_sigma_eps;
    # the role is played by mu_alpha_marg below.
    'log_beta':     0.755,
}
# Extra-hetero marginal calibration: alpha ~ N(mu_alpha_marg, sigma_alpha_marg^2).
MU_ALPHA_MARG    = float(np.log(0.10))      # marginal mean of log-sigma_i ~= -2.30
SIGMA_ALPHA_MARG = 0.30                       # marginal sd of log-sigma_i


def calibrate_extra_hetero(rho, mu_alpha_marg, sigma_alpha_marg, z1_log_std):
    """Return (beta_a0, beta_a1, log_sigma_a_cond) for alpha | z_1 ~
    N(beta_a0 + beta_a1 z_1, exp(log_sigma_a_cond)) such that the marginal
    of alpha is N(mu_alpha_marg, sigma_alpha_marg^2) and
    corr(alpha, z_1) = rho. Assumes E[z_1] = 0 under sinh-arcsinh
    with skew=0."""
    sigma_z1 = float(F.softplus(torch.tensor(z1_log_std)))
    beta_a0  = mu_alpha_marg
    beta_a1  = rho * sigma_alpha_marg / sigma_z1
    sigma_cond = sigma_alpha_marg * float(np.sqrt(max(1.0 - rho * rho, 1e-8)))
    return beta_a0, beta_a1, float(np.log(sigma_cond))


# ---- Model build + set-truth + simulate ---------------------------------

def build_model(T, device, hidden_dim=64):
    prior = MarkovNormalConditionalPolyPrior(
        nt=T, poly_degree=2, law_model='poly',
        z1_distr='sinh', extra_heterogeneity=True,
        extra_prior_type='iid').to(device)
    decoder = MASinhEmissionHetero(theta=0.0, fix_theta=True,
                                     sigma_eps=0.1,  # unused under 'scale' mode
                                     hetero_mode='scale').to(device)
    encoder = JointNormalConfig(
        type='joint_normal_extra', dim=T,
        hidden_dim=hidden_dim, extra_latents=1).build().to(device)
    model = FullModel(encoder=encoder, decoder=decoder, prior=prior).to(device)
    return model


@torch.no_grad()
def set_truth(model, truth_base, beta_a0, beta_a1, log_sigma_a_cond):
    p = model.prior
    p.net_mu.coeffs.copy_(torch.tensor(
        [truth_base['mu0'], truth_base['mu1'], truth_base['mu2']]))
    p.net_sigma.coeffs.copy_(torch.tensor(
        [truth_base['sigma0'], truth_base['sigma1'], truth_base['sigma2']]))
    p.z1_log_std.fill_(truth_base['z1_log_std'])
    p.z1_skew.fill_(truth_base['z1_skew'])
    p.z1_log_tail.fill_(truth_base['z1_log_tail'])
    # Extra-hetero coefficients
    p.net_extra_mu.coeffs.data[0] = beta_a0
    p.net_extra_mu.coeffs.data[1] = beta_a1
    p.net_extra_logsigma.coeffs.data[0] = log_sigma_a_cond
    # Decoder (only log_beta — under hetero_mode='scale' there is no log_sigma)
    model.decoder.log_beta.fill_(truth_base['log_beta'])


@torch.no_grad()
def simulate_panel(model, N, T, seed, device):
    """Simulate (y, z, alpha) from the prior/decoder under their current
    parameter values."""
    g = torch.Generator(device=device).manual_seed(seed)
    prior = model.prior
    # z_1 ~ sinh-arcsinh
    z1_scale = F.softplus(prior.z1_log_std)
    z1_skew  = prior.z1_skew
    z1_tail  = torch.exp(prior.z1_log_tail)
    u1 = torch.randn(N, generator=g, device=device)
    z1 = (z1_scale * torch.sinh(
        (torch.asinh(u1) + z1_skew) * z1_tail)).squeeze()
    # z_t | z_{t-1} ~ N(mu(z_lag), softplus(sigma_poly(z_lag)))
    z_list = [z1]
    z_cur = z1.unsqueeze(1)
    for t in range(1, T):
        mu, sigma = prior.get_mu_sigma(z_cur)
        eps = torch.randn(mu.shape, generator=g, device=device)
        z_next = mu + sigma * eps
        z_list.append(z_next.squeeze(1))
        z_cur = z_next
    z = torch.stack(z_list, dim=1)
    # alpha | z_1
    z1_col = z[:, 0:1]
    a_mu = prior.net_extra_mu(z1_col).squeeze(1)
    a_sigma = torch.exp(prior.net_extra_logsigma(z1_col)).squeeze(1) \
              + prior.regularize
    alpha = a_mu + a_sigma * torch.randn(a_mu.shape, generator=g,
                                            device=device)
    # y = z + exp(alpha) * sinh-arcsinh(0, 1, 0, beta) noise
    beta = torch.exp(model.decoder.log_beta)
    v = torch.randn(z.shape, generator=g, device=device)
    eps_y = torch.sinh(torch.asinh(v) * beta)  # sigma=1 sinh-arcsinh noise
    sigma_i = torch.exp(alpha)
    y = z + sigma_i.unsqueeze(1) * eps_y
    return y, z, alpha


# ---- Recovered-truth dict (read after fit) ------------------------------

def extract_params(model):
    p = model.prior
    out = {
        'mu0': float(p.net_mu.coeffs[0].item()),
        'mu1': float(p.net_mu.coeffs[1].item()),
        'mu2': float(p.net_mu.coeffs[2].item()),
        'sigma0': float(p.net_sigma.coeffs[0].item()),
        'sigma1': float(p.net_sigma.coeffs[1].item()),
        'sigma2': float(p.net_sigma.coeffs[2].item()),
        'z1_log_std':  float(p.z1_log_std.item()),
        'z1_skew':     float(p.z1_skew.item()),
        'z1_log_tail': float(p.z1_log_tail.item()),
        'beta_a0':     float(p.net_extra_mu.coeffs[0].item()),
        'beta_a1':     float(p.net_extra_mu.coeffs[1].item()),
        'log_sigma_a_cond': float(p.net_extra_logsigma.coeffs[0].item()),
        'log_beta':    float(model.decoder.log_beta.item()),
    }
    # Derived marginal/correlation diagnostics
    sigma_z1 = float(F.softplus(torch.tensor(out['z1_log_std'])))
    cond = float(np.exp(out['log_sigma_a_cond']))
    var_alpha = (out['beta_a1'] ** 2) * (sigma_z1 ** 2) + cond ** 2
    sigma_alpha = float(np.sqrt(var_alpha))
    rho_recovered = (out['beta_a1'] * sigma_z1 / sigma_alpha
                       if sigma_alpha > 0 else 0.0)
    out['_sigma_z1_implied'] = sigma_z1
    out['_sigma_alpha_marg_implied'] = sigma_alpha
    out['_rho_implied'] = rho_recovered
    out['_mu_alpha_marg_implied'] = out['beta_a0']  # since E[z_1]=0
    return out


# ---- Fit one rho cell ---------------------------------------------------

def fit_one_cell(rho, N, T, n_epochs, lr, hidden_dim, vi_seed, sim_seed,
                  fix_noise, noise_seed, device, dtype=torch.float32):
    print(f"\n========== rho = {rho:.2f} ==========", flush=True)
    # --- truth at this rho ---
    beta_a0, beta_a1, log_sigma_a_cond = calibrate_extra_hetero(
        rho, MU_ALPHA_MARG, SIGMA_ALPHA_MARG, TRUTH_BASE['z1_log_std'])
    truth_dict = dict(TRUTH_BASE)
    truth_dict['beta_a0']           = beta_a0
    truth_dict['beta_a1']           = beta_a1
    truth_dict['log_sigma_a_cond']  = log_sigma_a_cond
    truth_dict['rho_target']        = rho
    truth_dict['mu_alpha_marg']     = MU_ALPHA_MARG
    truth_dict['sigma_alpha_marg']  = SIGMA_ALPHA_MARG

    # --- simulate y from TRUTH ---
    torch.manual_seed(sim_seed); np.random.seed(sim_seed)
    truth_model = build_model(T, device, hidden_dim=hidden_dim)
    set_truth(truth_model, TRUTH_BASE, beta_a0, beta_a1, log_sigma_a_cond)
    y_obs, z_true, alpha_true = simulate_panel(truth_model, N, T,
                                                    sim_seed, device)
    print(f"  y range: [{float(y_obs.min()):.2f}, {float(y_obs.max()):.2f}]  "
          f"alpha emp mean={float(alpha_true.mean()):.3f}  "
          f"sd={float(alpha_true.std()):.3f}", flush=True)
    del truth_model

    # --- fresh model for the VI fit ---
    torch.manual_seed(vi_seed); np.random.seed(vi_seed)
    model = build_model(T, device, hidden_dim=hidden_dim)
    # init prior+decoder near truth structure but NOT at truth — let VI find it.
    # (we use the default init; encoder gets random init.)
    n_params_enc = sum(p.numel() for p in model.encoder.parameters())
    print(f"  |phi|={n_params_enc}  seed_dim={model.encoder.get_seed_dim()}",
          flush=True)

    # --- CRN tensor (constant across epochs, matches Phase-1 conv) ---
    if fix_noise:
        torch.manual_seed(noise_seed)
        seed_dim = model.encoder.get_seed_dim()
        u_crn = torch.randn(1, N, seed_dim, device=device, dtype=dtype)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    t0 = time.time()
    log_every = max(1, n_epochs // 10)
    elbo_history = []
    nan_streak = 0
    for epoch in range(n_epochs):
        opt.zero_grad()
        if fix_noise:
            loss = -model.elbo(y_obs, ndraws=1, eps_all=u_crn)
        else:
            loss = -model.elbo(y_obs, ndraws=1)
        if not torch.isfinite(loss):
            nan_streak += 1
            if nan_streak > 50 and epoch < 500:
                raise RuntimeError(f"NaN streak {nan_streak} at epoch {epoch}")
            continue
        nan_streak = 0
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if (epoch + 1) % log_every == 0 or epoch == 0:
            elbo = -loss.item()
            p = extract_params(model)
            print(f"  [VI rho={rho:.2f}] ep {epoch+1:5d}: "
                  f"ELBO={elbo:+.3f}  "
                  f"mu_a={p['_mu_alpha_marg_implied']:+.3f}  "
                  f"sd_a={p['_sigma_alpha_marg_implied']:.3f}  "
                  f"rho_imp={p['_rho_implied']:+.3f}  "
                  f"log_b={p['log_beta']:+.3f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
            elbo_history.append({'epoch': epoch + 1, 'elbo': elbo})
    wall = time.time() - t0
    final = extract_params(model)
    # Truth in same dict form
    truth_vec_dict = dict(TRUTH_BASE)
    truth_vec_dict.update({'beta_a0': beta_a0, 'beta_a1': beta_a1,
                            'log_sigma_a_cond': log_sigma_a_cond})
    # L2 over the 13-base + 3-hetero = 16 params (comparing fitted vs truth)
    common_keys = [k for k in final.keys() if k in truth_vec_dict]
    L2 = float(np.sqrt(sum((final[k] - truth_vec_dict[k]) ** 2
                              for k in common_keys)))
    return {
        'rho_target': rho,
        'truth':      truth_vec_dict,
        'fit':        final,
        'L2_truth':   L2,
        'wall_s':     wall,
        'elbo_history': elbo_history,
        'mu_alpha_marg_truth':    MU_ALPHA_MARG,
        'sigma_alpha_marg_truth': SIGMA_ALPHA_MARG,
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)

    N        = int(os.environ.get('N', 30000))
    T        = int(os.environ.get('T', 6))
    N_EP     = int(os.environ.get('N_EPOCHS', 20000))
    LR       = float(os.environ.get('LR', 1e-2))
    HIDDEN   = int(os.environ.get('HIDDEN_DIM', 64))
    FIX_NOISE= int(os.environ.get('FIX_NOISE', 1))
    NOISE_SEED=int(os.environ.get('NOISE_SEED', 12345))
    SIM_SEED = int(os.environ.get('SIM_SEED', 11))
    VI_SEED  = int(os.environ.get('VI_SEED', 11007))
    RHOS     = os.environ.get('RHOS', '0.0,0.5,0.9')
    rhos = [float(x) for x in RHOS.split(',')]

    print(f"Config: N={N} T={T} N_EP={N_EP} LR={LR} HIDDEN={HIDDEN} "
          f"FIX_NOISE={FIX_NOISE} NOISE_SEED={NOISE_SEED} SIM_SEED={SIM_SEED} "
          f"VI_SEED={VI_SEED} RHOS={rhos}", flush=True)

    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "output/bundles/cells/_scale_hetero_correlation_sweep")
    os.makedirs(out_dir, exist_ok=True)
    OUT_TAG = os.environ.get('OUT_TAG', 'sweep')
    out_path = os.path.join(out_dir, f"{OUT_TAG}.json")

    payload = {
        'config': {
            'N': N, 'T': T, 'n_epochs': N_EP, 'lr': LR, 'hidden_dim': HIDDEN,
            'fix_noise': bool(FIX_NOISE), 'noise_seed': NOISE_SEED,
            'sim_seed': SIM_SEED, 'vi_seed': VI_SEED, 'rhos': rhos,
            'truth_base': TRUTH_BASE,
            'mu_alpha_marg': MU_ALPHA_MARG,
            'sigma_alpha_marg': SIGMA_ALPHA_MARG,
        },
        'cells': []
    }
    for rho in rhos:
        cell = fit_one_cell(rho, N, T, N_EP, LR, HIDDEN, VI_SEED, SIM_SEED,
                              FIX_NOISE, NOISE_SEED, device)
        payload['cells'].append(cell)
        # save incrementally
        with open(out_path, 'w') as f:
            json.dump(payload, f, indent=2)
        print(f"  wrote {out_path}  cell L2_truth={cell['L2_truth']:.4f}",
              flush=True)
    print(f"\nDone. {out_path}", flush=True)


if __name__ == "__main__":
    main()
