r"""VI fit (mean-field-with-extra posterior) of the hetero-scale + sinh-MA(1)
model on the BPP PSID earnings panel.

Entry: psid-bpp-hetero-scale-vi-mf (see specs/compute-psid_bpp_hetero_scale.md).

Model class (specs/compute-psid_bpp_hetero_scale.md, models.md \S1.1, 2.2,
3.1, 4):
  z_t   ~ poly-2 mu + poly-2 sigma (no softplus kink, law_model='poly')
  z_1   ~ SinhArcsinh (all three z1 params free)
  alpha_i | z_1 ~ N(beta_a0 + beta_a1 z_1, exp(log_sigma_a_cond))   [iid]
  y_t   = z_t + exp(alpha_i) * MA(1)-SinhArcsinh(0, 1, 0, beta) noise
          (MASinhEmissionHetero, hetero_mode='scale', theta + log_beta free)

Encoder: JointNormalConfig(type='joint_normal_extra', dim=T, hidden_dim=64,
  extra_latents=1, regularize=1e-3, sd_clamp=3.0, diagonal=True) -- diagonal
  (mean-field) covariance over the joint (z_{1:T}, alpha) latent of dim T+1.
  The `diagonal=True` workaround forces the joint_normal_extra branch to
  build a diagonal posterior; the `type='normal_diagonal'` branch of
  `JointNormalConfig.build()` does not wire extra_latents through.
  Same pattern as `_simulation_hetero_scale_ivi_sweep._build_meanfield_extra`.

Estimator: single-sample ELBO maximisation (L=1), lr=1e-3, 20,000 epochs.

The 14 free parameters (in PARAM_NAMES order):
  mu0, mu1, mu2, sigma0, sigma1, sigma2,
  z1_log_std, z1_skew, z1_log_tail,
  log_beta, theta,
  beta_a0, beta_a1, log_sigma_a_cond
Note: under hetero_mode='scale' there is NO decoder.log_sigma.

Output: output/bundles/cells/employment-bpp-hetero-scale-vi-mf/bpp_hetero_scale_vi_mf.json
"""
import os
import sys
import time
import json

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from smc_truth_profile import _Bundle  # noqa: E402

from mlye.models.priors.markov import MarkovNormalConditionalPolyPrior  # noqa: E402
from mlye.models.decoders.ma import MASinhEmissionHetero  # noqa: E402
from mlye.models.encoders import JointNormalConfig  # noqa: E402
from mlye.models.full_model import FullModel  # noqa: E402
from bpp_hockey_common import compute_iwae_ref_log_p, compute_bhhh_se_fivo  # noqa: E402

import psid_smc_em_fivo_sinhz1 as base  # noqa: E402


METHOD_TAG = 'vi_mf'
ENCODER_LABEL = 'mean_field'
ENCODER_CONFIG_STR = (
    "JointNormalConfig(type='joint_normal_extra', dim=T, hidden_dim=64, "
    "extra_latents=1, regularize=1e-3, sd_clamp=3.0, diagonal=True)"
)
OUT_DIR = "output/bundles/cells/employment-bpp-hetero-scale-vi-mf"
OUT_JSON = os.path.join(OUT_DIR, "bpp_hetero_scale_vi_mf.json")
DATA_PATH = "output/data/bpp_y_matrix.npy"


# Canonical ordering for the 14-vec hetero-scale parameter vector.
# Mirrors _scale_hetero_correlation_ivi.PARAM_NAMES with `theta` inserted
# (the simulation pins theta=0; here it is free, per the BPP catalogue).
PARAM_NAMES = [
    'mu0', 'mu1', 'mu2',
    'sigma0', 'sigma1', 'sigma2',
    'z1_log_std', 'z1_skew', 'z1_log_tail',
    'log_beta', 'theta', 'alpha_eps',
    'beta_a0', 'beta_a1', 'log_sigma_a_cond',
]


def make_model(T, device, init_beta=1.0, seed=11):
    """Hetero-scale + sinh-z1 + sinh-MA(1) emission with scale heterogeneity.

    Prior: poly-2 mu + poly-2 sigma (no softplus kink); sinh-z1 with all
    three z1 params free; extra_heterogeneity=True, extra_prior_type='iid'.
    Decoder: MASinhEmissionHetero, hetero_mode='scale' (no log_sigma_eps),
    theta + log_beta both free.
    """
    torch.manual_seed(seed); np.random.seed(seed)
    prior = MarkovNormalConditionalPolyPrior(
        nt=T, hidden_dim=32, poly_degree=2,
        z1_distr='sinh', law_model='poly',
        extra_heterogeneity=True,
        extra_prior_type='iid',
        sigma_clamp=10.0, mu_clamp=10.0, skewed=False,
    ).to(device)
    with torch.no_grad():
        # mu(z) = 0 + 0.9 z + 0 z^2 ; sigma(z) underlay constant -1 (small sigma).
        prior.net_mu.coeffs.copy_(torch.tensor([0.0, 0.9, 0.0]))
        prior.net_sigma.coeffs.copy_(torch.tensor([-1.0, 0.0, 0.0]))
        prior.z1_log_std.fill_(0.0)
        prior.z1_skew.fill_(0.0)
        prior.z1_log_tail.fill_(0.0)
        # alpha | z_1 ~ N(beta_a0 + beta_a1 z_1, exp(log_sigma_a_cond)).
        # net_extra_mu is Polynomial(1) [2 coeffs: const + linear];
        # net_extra_logsigma is Polynomial(0) [1 coeff: const]. Default 0 inits.
    decoder = MASinhEmissionHetero(
        theta=0.0, fix_theta=False,
        sigma_eps=0.1,  # unused under hetero_mode='scale'
        hetero_mode='scale',
        alpha_eps=0.0, fix_skew=False,  # free residual skew, mean-preserving
    ).to(device)
    # log_beta init: nudge from log(1)=0 toward log(init_beta).
    with torch.no_grad():
        decoder.log_beta.fill_(float(np.log(init_beta)))
    return prior, decoder


def build_encoder(T, device, seed=11):
    """Mean-field (diagonal) normal encoder with one extra latent (alpha),
    hidden=64. `diagonal=True` forces the joint_normal_extra branch to build
    a diagonal-cov posterior over the joint (z_{1:T}, alpha) of dim T+1.

    The `type='normal_diagonal'` branch silently drops `extra_latents`,
    hence this workaround (matches `_simulation_hetero_scale_ivi_sweep.
    _build_meanfield_extra`).
    """
    torch.manual_seed(seed)
    cfg = JointNormalConfig(
        type='joint_normal_extra', dim=T, hidden_dim=64,
        extra_latents=1, regularize=1e-3, sd_clamp=3.0, diagonal=True,
    )
    return cfg.build().to(device)


def fit_vi(y, n_epochs=20000, lr=1e-3, log_every=500, clip=5.0, seed=11):
    T = y.shape[1]
    prior, decoder = make_model(T, y.device, seed=seed)
    encoder = build_encoder(T, y.device, seed=seed)
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
            print(
                f"  [VI-{METHOD_TAG}] epoch {epoch:5d}  "
                f"ELBO={-loss.item():.4f}  "
                f"beta={float(torch.exp(decoder.log_beta).item()):.3f}  "
                f"theta={float(decoder.theta.item()):+.4f}  "
                f"beta_a0={float(prior.net_extra_mu.coeffs[0].item()):+.3f}  "
                f"beta_a1={float(prior.net_extra_mu.coeffs[1].item()):+.3f}  "
                f"log_sig_a={float(prior.net_extra_logsigma.coeffs[0].item()):+.3f}  "
                f"z1_skew={float(prior.z1_skew.item()):+.3f}",
                flush=True,
            )
    print(f"  VI ({METHOD_TAG}) trained in {time.time()-t0:.1f}s", flush=True)
    return prior, decoder, encoder, history


# ----- parameter-table assembly (hetero-scale layout) ---------------------

def assemble_table_hetero_scale(prior, decoder, se):
    """Build the (raw, derived) parameter-table rows for a hetero-scale +
    MA(1) BPP fit. SE is the BHHH-derived vector in the order produced by
    `base.named_param_list(prior, decoder)`."""
    raw_vals = base.flatten_raw_params(prior, decoder)
    idx = base.flat_index_map(prior, decoder)
    rows = []
    for name in idx:
        s, e = idx[name]
        for k in range(s, e):
            sub = ""
            if e - s > 1:
                sub = f"[{k - s}]"
            rows.append((f"{name}{sub}", float(raw_vals[k]), float(se[k])))

    derived = []
    # beta = exp(log_beta).
    log_beta_idx = idx['decoder.log_beta'][0]
    beta = float(np.exp(raw_vals[log_beta_idx]))
    se_beta = beta * float(se[log_beta_idx])
    derived.append(("beta (=exp(log_beta))", beta, se_beta))

    # sigma_alpha_cond = exp(log_sigma_a_cond)  (from prior.net_extra_logsigma.coeffs[0]).
    extra_logsig_s, _ = idx['prior.net_extra_logsigma.coeffs']
    log_sig_a = float(raw_vals[extra_logsig_s])
    sig_a = float(np.exp(log_sig_a))
    se_sig_a = sig_a * float(se[extra_logsig_s])
    derived.append(("sigma_alpha_cond (=exp(log_sigma_a_cond))", sig_a, se_sig_a))

    # mu_1 (AR slope at z=0) = poly mu coeff [1].
    mu_s, _ = idx['prior.net_mu.coeffs']
    mu1 = float(raw_vals[mu_s + 1])
    se_mu1 = float(se[mu_s + 1])
    derived.append(("mu_1 (AR slope at z=0)", mu1, se_mu1))

    # sigma_z(0): under law_model='poly' the prior writes
    #   sigma(z) = softplus(net_sigma(z)) + regularize  (no shift / no kink),
    # so sigma_z(0) = softplus(sigma_0).
    sg_s, _ = idx['prior.net_sigma.coeffs']
    sg0 = float(raw_vals[sg_s])
    sigma_at_zero = float(np.log1p(np.exp(sg0))) if sg0 < 30 else sg0
    deriv_at_zero = float(1.0 / (1.0 + np.exp(-sg0)))
    se_sigma_at_zero = deriv_at_zero * float(se[sg_s])
    derived.append(("sigma_z(0) (=softplus(sigma_0))", sigma_at_zero, se_sigma_at_zero))

    # sigma_{z_1} = softplus(z1_log_std).
    z1_ls = idx['prior.z1_log_std'][0]
    z1_lt = idx['prior.z1_log_tail'][0]
    z1_sk = idx['prior.z1_skew'][0]
    raw_ls = float(raw_vals[z1_ls])
    sigma_z1 = float(np.log1p(np.exp(raw_ls))) if raw_ls < 30 else raw_ls
    deriv_ls = float(1.0 / (1.0 + np.exp(-raw_ls)))
    se_sigma_z1 = deriv_ls * float(se[z1_ls])
    derived.append(("sigma_{z_1} (=softplus(z1_log_std))", sigma_z1, se_sigma_z1))
    derived.append(("z1_skew",
                    float(raw_vals[z1_sk]), float(se[z1_sk])))
    raw_lt = float(raw_vals[z1_lt])
    tail_z1 = float(np.exp(raw_lt))
    se_tail_z1 = tail_z1 * float(se[z1_lt])
    derived.append(("z1_tail (=exp(z1_log_tail))", tail_z1, se_tail_z1))

    return rows, derived


def main():
    parser_epochs = int(os.environ.get('BPP_HETERO_SCALE_N_EPOCHS', 20000))
    seed_base = int(os.environ.get('BPP_HETERO_SCALE_SEED', 11))

    print(f"Loading BPP PSID earnings panel from {DATA_PATH}.", flush=True)
    y_np = np.load(DATA_PATH)
    y = torch.from_numpy(y_np).float()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    y = y.to(device)
    N, T = y.shape
    print(f"  shape: N={N}, T={T}, device={device}", flush=True)
    print(f"  empirical mean={y.mean().item():+.4f}, std={y.std().item():.4f}",
          flush=True)

    # Multi-draw ELBO: K draws per individual, implemented by tiling y K
    # times (one batched pass; identical mean/gradients to elbo(ndraws=K)
    # with fresh draws each epoch). The encoder is a function of y only,
    # so it still amortizes over the N distinct households. Diagnostics
    # (IWAE, BHHH) below use the actual y.
    ndraws = int(os.environ.get('BPP_HETERO_SCALE_NDRAWS', 1))
    y_fit = y.repeat(ndraws, 1) if ndraws > 1 else y
    if ndraws > 1:
        print(f"[multidraw] K={ndraws}: fit y tiled to "
              f"{tuple(y_fit.shape)}", flush=True)

    print(f"\n=== Fitting VI ({METHOD_TAG}: {ENCODER_LABEL}) "
          f"hidden=64 lr=1e-3 epochs={parser_epochs} ndraws={ndraws} ===",
          flush=True)
    prior_v, decoder_v, encoder_v, hist_v = fit_vi(
        y_fit, n_epochs=parser_epochs, lr=1e-3, seed=seed_base,
    )

    # Reference log p(y; theta_hat) via IWAE at converged theta.
    # (Bootstrap PF is biased under MA(1) AND not hetero-aware; see
    #  bpp_hockey_common.py and the spec.)
    iwae_logp, iwae_meta = compute_iwae_ref_log_p(
        prior_v, decoder_v, y, encoder=encoder_v, K=200,
    )

    print(f"\n=== BHHH SEs at VI ({METHOD_TAG}) fit "
          f"(per-individual FIVO scores, K=500, 3 reps) ===", flush=True)
    # NOTE: FIVO's bootstrap PF marginalises alpha away (it uses the prior
    # transition for z and the decoder's MARGINAL emission density). For
    # the hetero block (beta_a0, beta_a1, log_sigma_a_cond) the FIVO
    # gradient is zero, so the BHHH SE for those params is pinv-derived.
    # Same convention as bpp_hockey_vi_jn.py's `theta` row.
    se_v, _ = compute_bhhh_se_fivo(prior_v, decoder_v, y, K=500, n_reps=3,
                                   seed_base=seed_base)
    rows_v, derived_v = assemble_table_hetero_scale(prior_v, decoder_v, se_v)
    base.print_table(f"VI ({METHOD_TAG}) fit (hetero-scale BPP)", rows_v, derived_v)

    os.makedirs(OUT_DIR, exist_ok=True)
    payload = {
        'data': {
            'source': 'bpp',
            'path': DATA_PATH,
            'N': int(N), 'T': int(T),
            'mean_y': float(y.mean().item()),
            'std_y': float(y.std().item()),
        },
        'model': {
            'prior': 'MarkovNormalConditionalPolyPrior(poly_degree=2, '
                      'z1_distr=sinh, law_model=poly, extra_het=True, '
                      "extra_prior_type='iid')",
            'decoder': "MASinhEmissionHetero(hetero_mode='scale', "
                       'theta free, beta free)',
            'encoder': ENCODER_LABEL,
            'hetero': {
                'hetero_mode': 'scale',
                'extra_prior_type': 'iid',
                'extra_latents': 1,
            },
            'param_names': PARAM_NAMES,
            'note': 'BPP hetero-scale catalogue: theta NOT pinned '
                    "(fix_theta=False); no decoder.log_sigma under "
                    "hetero_mode='scale'.",
        },
        'training': {
            METHOD_TAG: {
                'encoder': ENCODER_LABEL,
                'encoder_config': ENCODER_CONFIG_STR,
                'hidden_dim': 64,
                'n_epochs': parser_epochs,
                'lr': 1e-3,
                'ndraws': ndraws,
                'seed_base': seed_base,
                'history': hist_v,
            },
        },
        'estimates': {
            METHOD_TAG: {
                'raw': [{'name': n, 'estimate': v, 'se': s}
                          for n, v, s in rows_v],
                'derived': [{'name': n, 'estimate': v, 'se': s}
                              for n, v, s in derived_v],
                'iwae_log_p': iwae_logp,
                'iwae_log_p_meta': iwae_meta,
            },
        },
        'se_method': {
            'method': 'BHHH (per-individual FIVO scores)',
            'K_scores': 500, 'n_reps_scores': 3, 'seed_base': seed_base,
            'note': 'FIVO bootstrap PF marginalises alpha; hetero block '
                    'SEs are pinv-derived from a near-singular information.',
        },
    }
    with open(OUT_JSON, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {OUT_JSON}")


if __name__ == "__main__":
    # Fix every RNG (python / numpy / torch CPU+CUDA) and request deterministic
    # kernels before any draw; the script's own explicit seeds still apply.
    from mlye.seeding import seed_everything, env_seed
    seed_everything(env_seed("BPP_HETERO_SCALE_SEED", 11))
    main()
