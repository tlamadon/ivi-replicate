r"""VI fit (joint-normal-extra posterior) of the hetero-MEAN (independent)
+ sinh-MA(1) model on the BPP PSID earnings panel.

Sister script to `bpp_hetero_scale_vi_jn.py` (the hetero-scale catalogue
entry). One-last-try variant of the abandoned hetero-mean line: the
per-individual mean shift alpha_i is imposed INDEPENDENT of z_1, which
removes the (c_1, z_1-scale) identification ridge that killed the
correlated version on PSID.

Model class:
  z_t   ~ poly-2 mu + poly-2 sigma (no softplus kink, law_model='poly')
  z_1   ~ SinhArcsinh (all three z1 params free)
  alpha_i ~ N(beta_a0, exp(2*log_sigma_a))          [INDEPENDENT of z_1]
  y_t   = z_t + alpha_i + sigma_eps * MA(1)-SinhArcsinh(0,1,alpha_eps,beta)
          (MASinhEmissionHetero, hetero_mode='mean';
           theta, log_beta, alpha_eps, log_sigma all free)

Independence is STRUCTURAL: `prior.net_extra_mu` is replaced by a
degree-0 Polynomial after construction, so E[alpha | z_1] has no slope
parameter at all (nothing to pin, nothing that can drift).

Encoder: JointNormalConfig(type='joint_normal_extra', dim=T, hidden_dim=64,
  extra_latents=1, regularize=1e-3, sd_clamp=3.0) -- covers the joint
  (z_{1:T}, alpha) latent of dim T+1.

Estimator: K-draw ELBO maximisation implemented by tiling y K times
(one batched pass; identical gradients to FullModel.elbo's ndraws loop).
Default K=40 (BPP_HETERO_MEAN_NDRAWS), lr=1e-3, 20,000 epochs.

The 15 free parameters (in PARAM_NAMES order):
  mu0, mu1, mu2, sigma0, sigma1, sigma2,
  z1_log_std, z1_skew, z1_log_tail,
  log_beta, theta, alpha_eps, log_sigma,
  beta_a0, log_sigma_a_cond
Note: under hetero_mode='mean' the decoder DOES carry log_sigma
(unlike hetero_mode='scale'); the hetero block has no beta_a1.

Output: output/bundles/cells/employment-bpp-hetero-mean-indep-vi-jn/bpp_hetero_mean_indep_vi_jn.json
"""
import os
import sys
import time
import json

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from smc_truth_profile import _Bundle  # noqa: E402,F401

from mlye.models.priors.markov import MarkovNormalConditionalPolyPrior  # noqa: E402
from mlye.models.priors.utils import Polynomial  # noqa: E402
from mlye.models.decoders.ma import MASinhEmissionHetero  # noqa: E402
from mlye.models.encoders import JointNormalConfig  # noqa: E402
from mlye.models.full_model import FullModel  # noqa: E402
from bpp_hockey_common import compute_iwae_ref_log_p, compute_bhhh_se_fivo  # noqa: E402

import psid_smc_em_fivo_sinhz1 as base  # noqa: E402


METHOD_TAG = 'vi_jn'
ENCODER_LABEL = 'joint_normal_extra'
ENCODER_CONFIG_STR = (
    "JointNormalConfig(type='joint_normal_extra', dim=T, hidden_dim=64, "
    "extra_latents=1, regularize=1e-3, sd_clamp=3.0)"
)
OUT_DIR = "output/bundles/cells/employment-bpp-hetero-mean-indep-vi-jn"
OUT_JSON = os.path.join(OUT_DIR, "bpp_hetero_mean_indep_vi_jn.json")
DATA_PATH = "output/data/bpp_y_matrix.npy"


# Canonical ordering for the 15-vec hetero-mean-independent parameter
# vector. Relative to the hetero-scale catalogue: beta_a1 is GONE
# (independence is structural) and decoder.log_sigma is ADDED
# (hetero_mode='mean' has a homogeneous emission scale).
PARAM_NAMES = [
    'mu0', 'mu1', 'mu2',
    'sigma0', 'sigma1', 'sigma2',
    'z1_log_std', 'z1_skew', 'z1_log_tail',
    'log_beta', 'theta', 'alpha_eps', 'log_sigma',
    'beta_a0', 'log_sigma_a_cond',
]


def make_model(T, device, init_beta=1.0, seed=11):
    """Hetero-mean (independent of z_1) + sinh-z1 + sinh-MA(1) emission.

    Prior: poly-2 mu + poly-2 sigma (no softplus kink); sinh-z1 with all
    three z1 params free; extra_heterogeneity=True, extra_prior_type='iid'
    with `net_extra_mu` REPLACED by a degree-0 Polynomial so
    alpha_i ~ N(beta_a0, sigma_a^2) has no z_1 slope.
    Decoder: MASinhEmissionHetero, hetero_mode='mean' (log_sigma free),
    theta + log_beta + alpha_eps free.
    """
    torch.manual_seed(seed); np.random.seed(seed)
    prior = MarkovNormalConditionalPolyPrior(
        nt=T, hidden_dim=32, poly_degree=2,
        z1_distr='sinh', law_model='poly',
        extra_heterogeneity=True,
        extra_prior_type='iid',
        sigma_clamp=10.0, mu_clamp=10.0, skewed=False,
    ).to(device)
    # Structural independence alpha _||_ z_1: degree-0 conditional mean.
    prior.net_extra_mu = Polynomial(0).to(device)
    with torch.no_grad():
        prior.net_mu.coeffs.copy_(torch.tensor([0.0, 0.9, 0.0]))
        prior.net_sigma.coeffs.copy_(torch.tensor([-1.0, 0.0, 0.0]))
        prior.z1_log_std.fill_(0.0)
        prior.z1_skew.fill_(0.0)
        prior.z1_log_tail.fill_(0.0)
        prior.net_extra_mu.coeffs.fill_(0.0)
        # sigma_a init: exp(-1) ~= 0.37, in the ballpark of a permanent
        # earnings component; the default exp(0)=1 overweights alpha
        # against the z process at init.
        prior.net_extra_logsigma.coeffs.fill_(-1.0)
    if os.environ.get('BPP_HETERO_MEAN_FREEZE_Z1_SKEW', '0') == '1':
        prior.z1_skew.requires_grad_(False)
    decoder = MASinhEmissionHetero(
        theta=0.0, fix_theta=False,
        sigma_eps=0.1,  # log_sigma init = log(0.1); FREE under 'mean'
        hetero_mode='mean',
        alpha_eps=0.0, fix_skew=False,  # free residual skew, mean-preserving
    ).to(device)
    with torch.no_grad():
        decoder.log_beta.fill_(float(np.log(init_beta)))
    return prior, decoder


def build_encoder(T, device, seed=11):
    """Joint-normal-extra encoder (lower-tri Cholesky over dim T+1), hidden=64."""
    torch.manual_seed(seed)
    cfg = JointNormalConfig(
        type='joint_normal_extra', dim=T, hidden_dim=64,
        extra_latents=1, regularize=1e-3, sd_clamp=3.0,
    )
    return cfg.build().to(device)


def fit_vi(y, n_epochs=20000, lr=1e-3, log_every=500, clip=5.0, seed=11,
           ndraws=1):
    """K-draw ELBO VI fit. ndraws > 1 tiles y K times (fresh eps per row
    per epoch), which is a K-draw ELBO estimate in one batched pass."""
    T = y.shape[1]
    prior, decoder = make_model(T, y.device, seed=seed)
    encoder = build_encoder(T, y.device, seed=seed)
    model = FullModel(encoder=encoder, decoder=decoder, prior=prior).to(y.device)
    y_fit = y.repeat(ndraws, 1) if ndraws > 1 else y
    if ndraws > 1:
        print(f"  [VI-{METHOD_TAG}] multidraw K={ndraws}: y tiled to "
              f"{tuple(y_fit.shape)}", flush=True)
    params = list(model.parameters())
    opt = torch.optim.AdamW(params, lr=lr)
    history = []
    t0 = time.time()
    for epoch in range(n_epochs):
        opt.zero_grad()
        loss = -model.elbo(y_fit, ndraws=1)
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
                f"sig_eps={float(torch.exp(decoder.log_sigma).item()):.3f}  "
                f"ba0={float(prior.net_extra_mu.coeffs[0].item()):+.3f}  "
                f"log_sig_a={float(prior.net_extra_logsigma.coeffs[0].item()):+.3f}  "
                f"z1_skew={float(prior.z1_skew.item()):+.3f}",
                flush=True,
            )
    print(f"  VI ({METHOD_TAG}) trained in {time.time()-t0:.1f}s", flush=True)
    return prior, decoder, encoder, history


# ----- parameter-table assembly (hetero-mean-independent layout) ----------

def assemble_table_hetero_mean(prior, decoder, se):
    """Build the (raw, derived) parameter-table rows for a hetero-mean
    (independent) + MA(1) BPP fit. SE is the BHHH-derived vector in the
    order produced by `base.named_param_list(prior, decoder)`."""
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

    # sigma_eps = exp(decoder.log_sigma) (homogeneous emission scale).
    log_sig_idx = idx['decoder.log_sigma'][0]
    sig_eps = float(np.exp(raw_vals[log_sig_idx]))
    se_sig_eps = sig_eps * float(se[log_sig_idx])
    derived.append(("sigma_eps (=exp(log_sigma))", sig_eps, se_sig_eps))

    # sigma_alpha = exp(log_sigma_a). Under independence the conditional
    # sd IS the marginal sd of the permanent component.
    extra_logsig_s, _ = idx['prior.net_extra_logsigma.coeffs']
    log_sig_a = float(raw_vals[extra_logsig_s])
    sig_a = float(np.exp(log_sig_a))
    se_sig_a = sig_a * float(se[extra_logsig_s])
    derived.append(("sigma_alpha (=exp(log_sigma_a); marginal)", sig_a, se_sig_a))

    # mu_1 (AR slope at z=0) = poly mu coeff [1].
    mu_s, _ = idx['prior.net_mu.coeffs']
    mu1 = float(raw_vals[mu_s + 1])
    se_mu1 = float(se[mu_s + 1])
    derived.append(("mu_1 (AR slope at z=0)", mu1, se_mu1))

    # sigma_z(0) = softplus(sigma_0).
    sg_s, _ = idx['prior.net_sigma.coeffs']
    sg0 = float(raw_vals[sg_s])
    sigma_at_zero = float(np.log1p(np.exp(sg0))) if sg0 < 30 else sg0
    deriv_at_zero = float(1.0 / (1.0 + np.exp(-sg0)))
    se_sigma_at_zero = deriv_at_zero * float(se[sg_s])
    derived.append(("sigma_z(0) (=softplus(sigma_0))", sigma_at_zero, se_sigma_at_zero))

    # sigma_{z_1} = softplus(z1_log_std) + z1 shape params.
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


def extract_theta(prior, decoder):
    """15-vec theta dict (PARAM_NAMES order) — the bootstrap-replicate
    payload; mirrors the IVI sibling's extract_theta_dict."""
    return {
        'mu0':          float(prior.net_mu.coeffs[0].item()),
        'mu1':          float(prior.net_mu.coeffs[1].item()),
        'mu2':          float(prior.net_mu.coeffs[2].item()),
        'sigma0':       float(prior.net_sigma.coeffs[0].item()),
        'sigma1':       float(prior.net_sigma.coeffs[1].item()),
        'sigma2':       float(prior.net_sigma.coeffs[2].item()),
        'z1_log_std':   float(prior.z1_log_std.item()),
        'z1_skew':      float(prior.z1_skew.item()),
        'z1_log_tail':  float(prior.z1_log_tail.item()),
        'log_beta':     float(decoder.log_beta.item()),
        'theta':        float(decoder.theta.item()),
        'alpha_eps':    float(decoder.alpha_eps.item()),
        'log_sigma':    float(decoder.log_sigma.item()),
        'beta_a0':      float(prior.net_extra_mu.coeffs[0].item()),
        'log_sigma_a_cond': float(prior.net_extra_logsigma.coeffs[0].item()),
    }


def main():
    parser_epochs = int(os.environ.get('BPP_HETERO_MEAN_N_EPOCHS', 20000))
    seed_base = int(os.environ.get('BPP_HETERO_MEAN_SEED', 11))
    ndraws = int(os.environ.get('BPP_HETERO_MEAN_NDRAWS', 40))
    # Bootstrap-replicate hooks (mirror the IVI sibling): skip the
    # expensive per-fit diagnostics (IWAE + BHHH) and write theta only.
    skip_diag = os.environ.get('BPP_HETERO_MEAN_SKIP_DIAG', '0') == '1'
    out_dir  = os.environ.get('BPP_HETERO_MEAN_OUT_DIR', OUT_DIR)
    out_json = os.environ.get('BPP_HETERO_MEAN_OUT_JSON', OUT_JSON)
    if 'BPP_HETERO_MEAN_OUT_DIR' in os.environ \
            and 'BPP_HETERO_MEAN_OUT_JSON' not in os.environ:
        out_json = os.path.join(out_dir, os.path.basename(OUT_JSON))

    print(f"Loading BPP PSID earnings panel from {DATA_PATH}.", flush=True)
    y_np = np.load(DATA_PATH)
    y = torch.from_numpy(y_np).float()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    y = y.to(device)
    N, T = y.shape
    # Nonparametric-bootstrap replicate: resample the N households with
    # replacement before anything else. Everything else (seeds, ndraws)
    # stays at production values so the numerical environment is common
    # across replicates and jitter cancels in the bootstrap spread.
    resample_seed = os.environ.get('BPP_HETERO_MEAN_RESAMPLE_SEED', '')
    if resample_seed:
        rng_rs = np.random.default_rng(int(resample_seed))
        idx_rs = torch.from_numpy(rng_rs.integers(0, N, N)).to(device)
        y = y[idx_rs]
        print(f"[bootstrap] household resample (with replacement, N={N}) "
              f"seed={resample_seed}", flush=True)
    print(f"  shape: N={N}, T={T}, device={device}", flush=True)
    print(f"  empirical mean={y.mean().item():+.4f}, std={y.std().item():.4f}",
          flush=True)

    print(f"\n=== Fitting VI ({METHOD_TAG}: {ENCODER_LABEL}) "
          f"hidden=64 lr=1e-3 epochs={parser_epochs} K={ndraws} ===", flush=True)
    prior_v, decoder_v, encoder_v, hist_v = fit_vi(
        y, n_epochs=parser_epochs, lr=1e-3, seed=seed_base, ndraws=ndraws,
    )

    if skip_diag:
        print("\n=== diagnostics skipped (BPP_HETERO_MEAN_SKIP_DIAG=1) ===",
              flush=True)
        rows_v, derived_v = [], []
        iwae_logp, iwae_meta = None, None
    else:
        # Reference log p(y; theta_hat) via IWAE at converged theta, on
        # the UNTILED panel. (Bootstrap PF is biased under MA(1) AND not
        # hetero-aware; see bpp_hockey_common.py and the hetero-scale spec.)
        iwae_logp, iwae_meta = compute_iwae_ref_log_p(
            prior_v, decoder_v, y, encoder=encoder_v, K=200,
        )

        print(f"\n=== BHHH SEs at VI ({METHOD_TAG}) fit "
              f"(per-individual FIVO scores, K=500, 3 reps) ===", flush=True)
        # NOTE: FIVO's bootstrap PF marginalises alpha away (it weights
        # with the decoder's unit-scale marginal emission density). The
        # hetero block (beta_a0, log_sigma_a) has zero FIVO gradient, so
        # its BHHH SEs are pinv-derived. Same convention as the
        # hetero-scale siblings.
        se_v, _ = compute_bhhh_se_fivo(prior_v, decoder_v, y, K=500,
                                       n_reps=3, seed_base=seed_base)
        rows_v, derived_v = assemble_table_hetero_mean(prior_v, decoder_v,
                                                       se_v)
        base.print_table(f"VI ({METHOD_TAG}) fit (hetero-mean-indep BPP)",
                         rows_v, derived_v)

    os.makedirs(out_dir, exist_ok=True)
    payload = {
        'data': {
            'source': 'bpp',
            'path': DATA_PATH,
            'N': int(N), 'T': int(T),
            'mean_y': float(y.mean().item()),
            'std_y': float(y.std().item()),
            'resample_seed': int(resample_seed) if resample_seed else None,
        },
        'model': {
            'prior': 'MarkovNormalConditionalPolyPrior(poly_degree=2, '
                      'z1_distr=sinh, law_model=poly, extra_het=True, '
                      "extra_prior_type='iid', net_extra_mu=Polynomial(0))",
            'decoder': "MASinhEmissionHetero(hetero_mode='mean', "
                       'theta free, beta free, alpha_eps free, log_sigma free)',
            'hetero': {
                'hetero_mode': 'mean',
                'extra_prior_type': 'iid',
                'extra_latents': 1,
                'alpha_independent_of_z1': True,
            },
            'param_names': PARAM_NAMES,
            'note': 'Hetero-mean with alpha_i STRUCTURALLY independent of '
                    'z_1 (net_extra_mu is degree-0). One-last-try variant '
                    'after the correlated hetero-mean line was dropped for '
                    'identification concerns.',
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
                'theta': extract_theta(prior_v, decoder_v),
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
    with open(out_json, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {out_json}")


if __name__ == "__main__":
    # Fix every RNG (python / numpy / torch CPU+CUDA) and request deterministic
    # kernels before any draw; the script's own explicit seeds still apply.
    from mlye.seeding import seed_everything, env_seed
    seed_everything(env_seed("BPP_HETERO_MEAN_SEED", 11))
    main()
