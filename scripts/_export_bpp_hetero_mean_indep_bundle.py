"""Build the BPP hetero-mean-independent bundle
(reference copy: bundles/2026-07-13-bpp-hetero-mean-indep-bundle.json).

Bundles the hetero-mean-independent (alpha _||_ z1) VI + IVI fits on the
BPP PSID panel together with the hetero-scale reference cells (it50
point estimate + bootstrap SEs) and the cross-model comparison,
so downstream analysis can run off one self-describing artefact.

Inputs:
  output/bundles/cells/employment-bpp-hetero-mean-indep-vi-jn/bpp_hetero_mean_indep_vi_jn.json
  output/bundles/cells/employment-bpp-hetero-mean-indep-ivi-jn/bpp_hetero_mean_indep_ivi_jn.json
  output/bundles/cells/employment-bpp-hetero-scale-ivi-jn-k40-picard-a06-it50/bpp_hetero_scale_ivi_jn.json
  output/bundles/cells/_bpp_hetero_scale_ivi_bootstrap_k40/bootstrap_theta.json
  output/data/bpp_y_matrix.npy

Output:
  $OUT_JSON (default output/bundles/2026-07-13-bpp-hetero-mean-indep-bundle.json)

Rerun after refitting any input cell. MC moments use a fixed seed.
"""
import json
import os
import math
import datetime

import numpy as np

HM_VI_JSON  = ("output/bundles/cells/employment-bpp-hetero-mean-indep-vi-jn/"
               "bpp_hetero_mean_indep_vi_jn.json")
HM_IVI_JSON = ("output/bundles/cells/employment-bpp-hetero-mean-indep-ivi-jn/"
               "bpp_hetero_mean_indep_ivi_jn.json")
HS_IT50_JSON = ("output/bundles/cells/employment-bpp-hetero-scale-ivi-jn-k40-picard-a06-it50/"
                "bpp_hetero_scale_ivi_jn.json")
HS_BOOT_JSON = "output/bundles/cells/_bpp_hetero_scale_ivi_bootstrap_k40/bootstrap_theta.json"
DATA_NPY = "output/data/bpp_y_matrix.npy"
OUT_JSON = os.environ.get(
    "OUT_JSON",
    "output/bundles/2026-07-13-bpp-hetero-mean-indep-bundle.json")

HM_PARAMS = ['mu0', 'mu1', 'mu2', 'sigma0', 'sigma1', 'sigma2',
             'z1_log_std', 'z1_skew', 'z1_log_tail',
             'log_beta', 'theta', 'alpha_eps', 'log_sigma',
             'beta_a0', 'log_sigma_a_cond']
HS_PARAMS = ['mu0', 'mu1', 'mu2', 'sigma0', 'sigma1', 'sigma2',
             'z1_log_std', 'z1_skew', 'z1_log_tail',
             'log_beta', 'theta', 'alpha_eps',
             'beta_a0', 'beta_a1', 'log_sigma_a_cond']
SHARED = ['mu0', 'mu1', 'mu2', 'sigma0', 'sigma1', 'sigma2',
          'z1_log_std', 'z1_skew', 'z1_log_tail',
          'log_beta', 'theta', 'alpha_eps']

N_MC = 4_000_000
softplus = lambda x: math.log1p(math.exp(x)) if x < 30 else x


def _moments(x):
    sd = float(x.std())
    xs = (x - x.mean()) / sd
    return sd, float((xs ** 3).mean()), float((xs ** 4).mean() - 3)


def residual_marginal_hm(theta, rng):
    """Per-period residual innovation (pre-MA) for the hetero-mean model:
    eps = exp(log_sigma) * W(alpha_eps, beta), homogeneous across i."""
    b = math.exp(theta['log_beta'])
    W = np.sinh((np.arcsinh(rng.standard_normal(N_MC))
                 + theta['alpha_eps']) * b)
    W = W - W.mean()
    return _moments(math.exp(theta['log_sigma']) * W)


def residual_marginal_hs(theta, rng):
    """Per-period residual innovation (pre-MA) for the hetero-scale
    model, MARGINAL over alpha_i (and over z1 through beta_a1):
    eps_i = exp(alpha_i) * W(alpha_eps, beta)."""
    z1s = softplus(theta['z1_log_std'])
    z1 = z1s * np.sinh((np.arcsinh(rng.standard_normal(N_MC))
                        + theta['z1_skew'])
                       * math.exp(theta['z1_log_tail']))
    z1 = z1 - z1.mean()
    alpha = (theta['beta_a0'] + theta['beta_a1'] * z1
             + math.exp(theta['log_sigma_a_cond'])
             * rng.standard_normal(N_MC))
    b = math.exp(theta['log_beta'])
    W = np.sinh((np.arcsinh(rng.standard_normal(N_MC))
                 + theta['alpha_eps']) * b)
    W = W - W.mean()
    return _moments(np.exp(alpha) * W)


def hm_derived(theta, resid):
    sd, g1, g2 = resid
    return {
        'beta': math.exp(theta['log_beta']),
        'sigma_eps_raw': math.exp(theta['log_sigma']),
        'sigma_eps_effective': sd,
        'resid_skewness_g1': g1,
        'resid_excess_kurtosis_g2': g2,
        'sigma_alpha_marginal': math.exp(theta['log_sigma_a_cond']),
        'mean_alpha': theta['beta_a0'],
        'sigma_z1': softplus(theta['z1_log_std']),
        'sigma_z_at_0': softplus(theta['sigma0']),
        'mu1_ar_slope': theta['mu1'],
        'z1_tail': math.exp(theta['z1_log_tail']),
    }


def hs_derived(theta, resid):
    sd, g1, g2 = resid
    return {
        'beta': math.exp(theta['log_beta']),
        'sigma_eps_raw': None,   # no homogeneous scale under hetero_mode='scale'
        'sigma_eps_effective': sd,   # marginal over alpha_i
        'resid_skewness_g1': g1,
        'resid_excess_kurtosis_g2': g2,
        'sigma_alpha_marginal': None,  # alpha is log-vol, not a mean shift
        'mean_log_vol': theta['beta_a0'],
        'sigma_log_vol_cond': math.exp(theta['log_sigma_a_cond']),
        'beta_a1_logvol_on_z1': theta['beta_a1'],
        'sigma_z1': softplus(theta['z1_log_std']),
        'sigma_z_at_0': softplus(theta['sigma0']),
        'mu1_ar_slope': theta['mu1'],
        'z1_tail': math.exp(theta['z1_log_tail']),
    }


def main():
    rng = np.random.default_rng(0)

    hm_vi_doc  = json.load(open(HM_VI_JSON))
    hm_ivi_doc = json.load(open(HM_IVI_JSON))
    hs_doc     = json.load(open(HS_IT50_JSON))
    hs_boot    = json.load(open(HS_BOOT_JSON))
    B_BOOT     = len(hs_boot['replicates'])

    # hetero-mean VI theta: the JSON's estimates.vi_jn.theta field exists
    # for fits made after the bootstrap-hook commit; reconstruct from the
    # raw rows for the original fit.
    evi = hm_vi_doc['estimates']['vi_jn']
    if 'theta' in evi:
        hm_vi = evi['theta']
    else:
        raw = {r['name']: r['estimate'] for r in evi['raw']}
        hm_vi = {
            'mu0': raw['prior.net_mu.coeffs[0]'],
            'mu1': raw['prior.net_mu.coeffs[1]'],
            'mu2': raw['prior.net_mu.coeffs[2]'],
            'sigma0': raw['prior.net_sigma.coeffs[0]'],
            'sigma1': raw['prior.net_sigma.coeffs[1]'],
            'sigma2': raw['prior.net_sigma.coeffs[2]'],
            'z1_log_std': raw['prior.z1_log_std'],
            'z1_skew': raw['prior.z1_skew'],
            'z1_log_tail': raw['prior.z1_log_tail'],
            'log_beta': raw['decoder.log_beta'],
            'theta': raw['decoder.theta'],
            'alpha_eps': raw['decoder.alpha_eps'],
            'log_sigma': raw['decoder.log_sigma'],
            'beta_a0': raw['prior.net_extra_mu.coeffs'],
            'log_sigma_a_cond': raw['prior.net_extra_logsigma.coeffs'],
        }
    hm_ivi_est = hm_ivi_doc['estimates']['ivi']
    hm_ivi = hm_ivi_est['theta_K']
    hm_obs = hm_ivi_est['theta_VI_obs']
    hs_ivi = hs_doc['estimates']['ivi']['theta_K']

    # --- residual marginal moments (per-period innovation, pre-MA) ---
    res_hm_vi  = residual_marginal_hm(hm_vi, rng)
    res_hm_ivi = residual_marginal_hm(hm_ivi, rng)
    res_hs     = residual_marginal_hs(hs_ivi, rng)

    # --- cross-model comparison on shared params ---
    comparison = []
    for p in SHARED:
        se = hs_boot['se'][p]
        diff = hm_ivi[p] - hs_ivi[p]
        comparison.append({
            'param': p,
            'hm_ivi': hm_ivi[p],
            'hs_ivi': hs_ivi[p],
            'hs_boot_se': se,
            'diff': diff,
            'diff_over_hs_se': diff / se if se > 0 else None,
        })

    y = np.load(DATA_NPY)

    cells = [
        {
            'tag': 'hm-vi-jn-k40',
            'model_class': 'hetero-mean-indep',
            'method': 'VI joint-normal-extra, K=40 ELBO draws (tiled), '
                      '20k ep, lr=1e-3',
            'n_iters_outer': None,
            'source_json': HM_VI_JSON,
            'run': 'a8c8f61b/fit-vi-jn',
            'status': 'completed',
        },
        {
            'tag': 'hm-ivi-picard06-k40-it10',
            'model_class': 'hetero-mean-indep',
            'method': 'IVI binding-equation, Picard alpha=0.6, K=40 CRN '
                      'draws both sides, n_sim=N=741, 10 outer x 16k '
                      'inner ep, lr=1e-2',
            'n_iters_outer': 10,
            'source_json': HM_IVI_JSON,
            'run': 'a8c8f61b/fit-ivi-jn',
            'status': 'completed',
            'caveat': 'trajectory NOT settled at iter 10: sigma_alpha '
                      'still rising, sigma_z1 still falling, ||g|| '
                      'plateaued ~0.15; log_beta/log_sigma sliding along '
                      'the (beta up, sigma down) ridge at ~constant '
                      'effective scale',
        },
        {
            'tag': 'hs-ivi-picard06-k40-it50',
            'model_class': 'hetero-scale (reference)',
            'method': 'IVI binding-equation, Picard alpha=0.6, K=40, 50 '
                      'outer iters -- the reportable hetero-scale point '
                      'estimate (journal 2026-07-10)',
            'n_iters_outer': 50,
            'source_json': HS_IT50_JSON,
            'run': 'e38ea352',
            'status': 'completed',
        },
        {
            'tag': 'hs-ivi-bootstrap-se',
            'model_class': 'hetero-scale (reference)',
            'method': f'full-IVI nonparametric bootstrap B={B_BOOT}, household '
                      'resample seeds 31000+b, 20 outer iters per rep',
            'n_iters_outer': 20,
            'source_json': HS_BOOT_JSON,
            'run': '823d93c3 + 53ef9b9d',
            'status': 'completed',
        },
    ]

    summary_table = [
        {'tag': 'hm-vi-jn-k40', 'model_class': 'hetero-mean-indep',
         'estimator': 'vi',
         **{p: hm_vi[p] for p in HM_PARAMS},
         'derived': hm_derived(hm_vi, res_hm_vi),
         'iwae_log_p': evi['iwae_log_p']},
        {'tag': 'hm-ivi-picard06-k40-it10', 'model_class': 'hetero-mean-indep',
         'estimator': 'ivi',
         **{p: hm_ivi[p] for p in HM_PARAMS},
         'derived': hm_derived(hm_ivi, res_hm_ivi),
         'iwae_log_p': hm_ivi_est['iwae_log_p']},
        {'tag': 'hs-ivi-picard06-k40-it50', 'model_class': 'hetero-scale',
         'estimator': 'ivi',
         **{p: hs_ivi[p] for p in HS_PARAMS},
         'boot_se': {p: hs_boot['se'][p] for p in HS_PARAMS},
         'derived': hs_derived(hs_ivi, res_hs),
         'iwae_log_p': hs_doc['estimates']['ivi']['iwae_log_p']},
    ]

    bundle = {
        'generated_at': datetime.datetime.now().astimezone().isoformat(),
        'session_journal': '2026-07-11-bpp-hetero-mean-indep-vi-ivi.md',
        'summary': (
            'One-last-try hetero-mean variant on the BPP PSID panel with '
            'alpha_i imposed STRUCTURALLY independent of z_1 '
            '(net_extra_mu = Polynomial(0)), fit by VI (joint-normal-'
            'extra, K=40) and IVI (Picard 0.6, K=40, no inflation), '
            'bundled with the reportable hetero-scale IVI point estimate '
            f'and its B={B_BOOT} bootstrap SEs for cross-model comparison. '
            'Headline: dynamics + MA(1) block agree almost exactly with '
            'hetero-scale (theta_MA 0.267 vs 0.264; mu/sigma poly within '
            '~1 SE except sigma0); the differences concentrate exactly '
            'where the two heterogeneity layers do different work -- the '
            'homogeneous hetero-mean residual mimics the hetero-scale '
            'MARGINAL (alpha-mixed) residual (sd 0.188 vs 0.199, g1 -1.5 '
            'vs -1.3, g2 +14 vs +20) by inflating beta (log_beta 0.89 vs '
            '0.47, an 8 SE gap in raw coordinates that mostly vanishes '
            'in the invariant residual moments); and the variance '
            'decomposition splits differently (h-mean adds a permanent '
            'sigma_alpha=0.16 in the mean while shrinking sigma_z1 to '
            '0.12 vs h-scale 0.18). h-mean IVI iter-10 trajectory is NOT '
            'settled -- treat its theta as in motion.'
        ),
        'description': (
            'Self-describing bundle. `cells_included` catalogues the '
            'four cells. `summary_table` has one row per FIT cell (the '
            'bootstrap-SE cell contributes the boot_se sub-dict on the '
            'hs row instead of its own row); raw params are top-level '
            'keys in each row (param_names differ by model_class -- '
            'hetero-mean-indep has log_sigma and no beta_a1), derived '
            'quantities live in the `derived` sub-dict, and residual '
            'moments there are MC (4M draws, seed 0) on the PER-PERIOD '
            'pre-MA innovation: for hetero-mean the homogeneous '
            'exp(log_sigma)*W; for hetero-scale the MARGINAL '
            'exp(alpha_i)*W mixing over alpha_i|z1 and z1 -- the '
            'apples-to-apples object. `comparison` is the shared-param '
            'table hm_ivi vs hs_ivi with diff/hs_boot_se. `trajectory` '
            'is the full hm IVI outer-loop trace (per-iter theta_k, '
            'theta_VI, per-component g). `runs` carries the full theta '
            'dicts incl. theta_VI_obs.'
        ),
        'context': {
            'data': {'path': DATA_NPY, 'N': int(y.shape[0]),
                     'T': int(y.shape[1]),
                     'mean_y': float(y.mean()), 'std_y': float(y.std()),
                     'std_y1': float(y[:, 0].std()),
                     'std_yT': float(y[:, -1].std())},
            'hetero_mean_indep_model': (
                'poly-2 LoM, sinh-arcsinh z1 (3 free), MA(1) sinh '
                'emission (theta, log_beta, alpha_eps free), homogeneous '
                'log_sigma; alpha_i ~ N(beta_a0, sigma_alpha^2) '
                'independent of z1 (structural: degree-0 net_extra_mu), '
                'y_t = z_t + alpha_i + eps_t. 15 params.'
            ),
            'hetero_scale_model': (
                'same LoM/z1/MA(1) but alpha_i is per-individual '
                'LOG-VOLATILITY: eps_{it} scale = exp(alpha_i), '
                'alpha_i | z1 ~ N(beta_a0 + beta_a1 z1, sigma_a_cond); '
                'no homogeneous log_sigma. 15 params.'
            ),
            'estimators': (
                'VI: joint-normal-extra encoder h=64, K=40 ELBO draws '
                '(tiled y), 20k ep lr=1e-3. IVI: binding-equation '
                'g_k = theta_VI_obs - b(theta_k), Picard alpha=0.6, '
                'K=40 CRN draws on both sides, n_sim = N = 741 (no '
                'inflation), cold-start inner fits 16k ep lr=1e-2.'
            ),
            'scripts': ['scripts/bpp_hetero_mean_indep_vi_jn.py',
                        'scripts/bpp_hetero_mean_indep_ivi_jn.py'],
            'workflow': 'hut.bpp_hetero_mean_indep.json (run a8c8f61b, '
                        'pythia-nb)',
        },
        'cells_included': cells,
        'schema': {
            'summary_table_row': {
                'tag': 'cell id (see cells_included)',
                'model_class': 'hetero-mean-indep | hetero-scale',
                'estimator': 'vi | ivi',
                '<param>': 'raw parameter estimate at the cell\'s final '
                           'theta; the param set differs by model_class '
                           '(hetero-mean-indep: log_sigma, no beta_a1; '
                           'hetero-scale: beta_a1, no log_sigma)',
                'boot_se': f'(hs row only) per-param bootstrap SE, B={B_BOOT} '
                           'household-resample full-IVI bootstrap',
                'derived': {
                    'beta': 'exp(log_beta), emission tailweight',
                    'sigma_eps_raw': 'exp(log_sigma); None for '
                                     'hetero-scale (no homogeneous scale)',
                    'sigma_eps_effective':
                        'sd of the per-period pre-MA residual innovation '
                        '(MC). For hetero-scale this is MARGINAL over '
                        'alpha_i -- directly comparable across models',
                    'resid_skewness_g1': 'Pearson g1 of the same object',
                    'resid_excess_kurtosis_g2': 'Pearson g2 of the same '
                                                'object',
                    'sigma_alpha_marginal': 'hetero-mean only: sd of the '
                                            'permanent mean shift',
                    'mean_log_vol / sigma_log_vol_cond / '
                    'beta_a1_logvol_on_z1': 'hetero-scale only',
                    'sigma_z1': 'softplus(z1_log_std)',
                    'sigma_z_at_0': 'softplus(sigma0)',
                    'mu1_ar_slope': 'LoM slope at z=0',
                    'z1_tail': 'exp(z1_log_tail)',
                },
                'iwae_log_p': 'IWAE K=200 reference log p(y) per '
                              'individual on the actual panel',
            },
            'comparison_row': {
                'param': 'shared-parameter name',
                'hm_ivi': 'hetero-mean-indep IVI theta_K (iter 10)',
                'hs_ivi': 'hetero-scale IVI theta_K (iter 50, reportable)',
                'hs_boot_se': f'hetero-scale bootstrap SE (B={B_BOOT})',
                'diff': 'hm_ivi - hs_ivi',
                'diff_over_hs_se': 'diff / hs_boot_se; |.| > 2 flags a '
                                   'real cross-model gap IF hm were '
                                   'converged (it is not -- see cell '
                                   'caveat)',
            },
            'trajectory_entry': 'as written by '
                                'bpp_hetero_mean_indep_ivi_jn.py: iter, '
                                'iter_wall_s, theta_k (post-update), '
                                'theta_VI (inner fit on y_sim), g '
                                '(per-component binding residual), '
                                'g_norm, restart, gamma, used_seed, '
                                'inner_history',
        },
        'summary_table': summary_table,
        'comparison': comparison,
        'runs': {
            'hm-vi-jn-k40': {
                'source_json': HM_VI_JSON,
                'theta': hm_vi,
                'iwae_log_p': evi['iwae_log_p'],
            },
            'hm-ivi-picard06-k40-it10': {
                'source_json': HM_IVI_JSON,
                'theta_VI_obs': hm_obs,
                'theta_K': hm_ivi,
                'iwae_log_p': hm_ivi_est['iwae_log_p'],
                'trajectory': [
                    {k: v for k, v in t.items() if k != 'inner_history'}
                    for t in hm_ivi_est['trajectory']
                ],
            },
            'hs-ivi-picard06-k40-it50': {
                'source_json': HS_IT50_JSON,
                'theta_K': hs_ivi,
                'boot_se': {p: hs_boot['se'][p] for p in HS_PARAMS},
                'boot_ci95': {p: hs_boot['ci95'][p] for p in HS_PARAMS},
                'iwae_log_p': hs_doc['estimates']['ivi']['iwae_log_p'],
            },
        },
    }

    os.makedirs(os.path.dirname(OUT_JSON) or '.', exist_ok=True)
    with open(OUT_JSON, 'w') as f:
        json.dump(bundle, f, indent=1)
    print(f"wrote {OUT_JSON}")
    print(f"cells: {[c['tag'] for c in cells]}")
    print(f"comparison rows: {len(comparison)}  "
          f"(|diff/se|>2: {[c['param'] for c in comparison if c['diff_over_hs_se'] and abs(c['diff_over_hs_se'])>2]})")


if __name__ == "__main__":
    main()
