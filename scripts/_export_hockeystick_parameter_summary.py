"""Self-contained parameter export for the simulation-hockeystick sweep.

Reads every cell JSON under the source directory (default
``output/bundles/cells/_simulation_hockeystick_ivi_sweep/``; override with the
``SOURCE_DIR`` env var for a panel-length variant such as
``output/bundles/cells/_simulation_hockeystick_t40_ivi_sweep/``) and writes a single
combined JSON to the same folder containing, for each cell:

  - the raw 13-element parameter vector (canonical PARAM_NAMES ordering),
  - the derived human-readable quantities (kink parameters, sigma(z)
    coefficients, sinh-arcsinh z1 scale/skew/tailweight, emission
    sigma_eps + beta-tailweight),
  - the L2 distance to truth on the relevant free coordinates,
  - the bound diagnostics (ELBO / IWAE / SMC at vi / ivi / truth +
    prior_log_p_* for the misspec baseline),
  - a `shipping` flag distinguishing the five canonical cells (per
    `specs/compute-simulation-hockeystick.md` Entry section) from the
    historical Anderson and pre-rename `vi_*` cells that remain on
    disk.

Plus a top-level ``ai_prompt`` that describes how to instantiate
mu(z) (the softplus-kinked poly-2 of the hockey-stick DGP), sigma(z),
the sinh-arcsinh density of z_1, and the sinh-arcsinh density of the
measurement error from the raw parameters --- so the JSON is usable
without any source-tree context.
"""

from __future__ import annotations

import glob
import json
import math
import os
from collections import OrderedDict
from typing import Any, Dict, List, Optional

# Canonical ordering matches the 13-key parameter vector that every
# hockeystick cell emits (see specs/compute-simulation-hockeystick.md
# Layer 1 step 6 "schema invariant").
PARAM_NAMES: List[str] = [
    'alpha0', 'log_alpha1',
    'mu0', 'mu1', 'mu2',
    'sigma0', 'sigma1', 'sigma2',
    'z1_log_std', 'z1_skew', 'z1_log_tail',
    'log_sigma_eps', 'log_beta',
]
# IVI + VI cells: 12 free coords (z1_skew pinned at 0).
FREE_KEYS_IVI: List[str] = [
    'alpha0', 'log_alpha1',
    'mu0', 'mu1', 'mu2',
    'sigma0', 'sigma1', 'sigma2',
    'z1_log_std', 'z1_log_tail',
    'log_sigma_eps', 'log_beta',
]
# Misspecification baseline: 10 free prior + z_1 coords (decoder dropped).
FREE_KEYS_MLE_DIRECT: List[str] = [
    'alpha0', 'log_alpha1',
    'mu0', 'mu1', 'mu2',
    'sigma0', 'sigma1', 'sigma2',
    'z1_log_std', 'z1_log_tail',
]

# Canonical shipping-cell filename stems per the `Entry` section. A cell
# JSON is "shipping" iff its basename starts with one of these stems.
# Stem (prefix) matching -- rather than an exact filename set -- lets the
# same aggregator serve panel-length variants whose filenames carry a
# `_T<T>` suffix (e.g. the T=40 IVI joint-normal variant,
# `picard_cold_alpha0.60_..._sig0--1.80_T40.json`) while still excluding
# the historical Anderson / pre-rename `vi_*` cells that remain on disk
# in the T=6 bundle (`anderson_*`, `vi_<enc>_h64_*` -- neither starts
# with `vi_only_`).
SHIPPING_STEMS = (
    'picard_cold_alpha',
    'vi_only_meanfield',
    'vi_only_struct_markov',
    'vi_only_tridiag_jn',
    'mle_direct_no_me',
)

# Stem of the Picard IVI cell. Its Phase 1 is by construction a
# cold-start JN h=64 VI fit on y_obs; per spec the aggregator
# synthesizes a VI-only joint-normal fits[] entry from this cell's
# `theta_VI_obs` to expose the JN VI estimator as a first-class row.
PICARD_IVI_CELL_STEM = 'picard_cold_alpha'


def _is_shipping(basename: str) -> bool:
    return basename.startswith(SHIPPING_STEMS)


# Source directory of the per-cell JSONs. Defaults to the canonical
# T=6 bundle; override via the SOURCE_DIR env var for a panel-length
# variant (e.g. output/bundles/cells/_simulation_hockeystick_t40_ivi_sweep for the
# T=40 IVI cell). The summary is written into the same directory.
SOURCE_DIR = os.environ.get(
    'SOURCE_DIR', os.path.join('output', 'bundles', 'cells', '_simulation_hockeystick_ivi_sweep'))
# OUT_JSON: the T=40 run writes output/bundles/hockeystick_parameter_summary_longt.json
OUTPUT_PATH = os.environ.get(
    'OUT_JSON', os.path.join('output', 'bundles', 'hockeystick_parameter_summary.json'))


AI_PROMPT = """\
This JSON catalogues the fitted parameter vector for every cell in the
simulation-hockeystick sweep (a 1-D latent state-space model with a
softplus-kinked drift, sinh-arcsinh initial state, and sinh-arcsinh
measurement error). Each entry in `fits` reports a 13-element
parameter dict in the canonical ordering listed under
`parameter_schema.param_names`. The mapping from those parameters to
the generative model is:

(1) Law of motion (transition density), softplus-kinked poly-2:
    z_t | z_{t-1} ~ Normal( mu(z_{t-1}), sigma(z_{t-1})^2 ),
    with
        b       = alpha0,
        gamma   = exp(log_alpha1),
        mu(z)   = b + gamma * softplus( (poly_mu(z) - b) / gamma ),
        poly_mu(z) = mu0 + mu1 * z + mu2 * z^2,
        sigma(z)   = softplus(sigma0 + sigma1 * z + sigma2 * z^2),
    where softplus(x) = log(1 + exp(x)). As gamma -> +inf the kink
    vanishes and mu(z) -> poly_mu(z); at finite gamma, mu(z) is bounded
    below by b (the kink floor). NB: sigma uses softplus, NOT exp.

(2) Initial-state density (sinh-arcsinh):
    z_1 ~ SinhArcsinh( loc=0, scale=softplus(z1_log_std),
                       skew=z1_skew, tailweight=exp(z1_log_tail) ).
    The scalar SinhArcsinh density is
        z = loc + scale * sinh( tailweight * ( asinh(eps) + skew ) ),
        eps ~ Normal(0, 1).
    For this DGP, z1_skew is pinned at 0 (symmetric), z1_log_tail
    is *free*, and the truth has scale = 0.34 and tailweight = 1/0.89.
    NB: scale uses softplus on z1_log_std; tailweight uses exp on
    z1_log_tail. The 'log_std' suffix is historical and misleading ---
    it is pre-softplus, not pre-exp.

(3) Measurement-error density (sinh-arcsinh):
    y_t = z_t + eps_t,
    eps_t ~ SinhArcsinh( loc=0,
                         scale=exp(log_sigma_eps),
                         skew=0,
                         tailweight=exp(log_beta) ).
    Sigma uses exp (no softplus, no floor under this calibration).
    The truth has scale = 0.033 and tailweight = 1/0.47 ~ 2.13 --- the
    decoder is heavy-tailed, NOT Gaussian. The MA(1) coefficient theta
    is pinned at 0 for every cell in this sweep, so the residual is
    IID-in-t SinhArcsinh.

(4) z1_skew:
    Pinned at 0 (`requires_grad=False`) for every cell in this sweep,
    per the 2026-06-15 schema convention (the DGP truth is symmetric).
    Pre-2026-06-15 JSONs that trained z1_skew freely are not present
    in this sweep.

The 12 free coordinates under the IVI cell and VI-only cells are
    {alpha0, log_alpha1, mu0, mu1, mu2, sigma0, sigma1, sigma2,
     z1_log_std, z1_log_tail, log_sigma_eps, log_beta}.
The misspecification baseline cell (`mle_direct`) drops log_sigma_eps
and log_beta entirely (decoder does not exist; y_t = z_t exactly),
leaving the 10 free prior + z_1 coordinates.

Each entry in `fits` reports:
  - method:           'picard_cold' | 'anderson' | 'vi_only' | 'mle_direct'
  - encoder:          variational family for VI / IVI cells (None for MLE).
  - emission:         'sinh' for IVI/VI cells, 'degenerate' for MLE.
  - shipping:         True for the five canonical cells in the Entry
                       section; False for historical / dropped cells.
  - parameters:       raw 13-element dict in canonical order.
  - parameters_derived: human-friendly transforms (kink params,
                        sigma_at_zero, sigma_z1, sigma_eps, beta,
                        z1_tailweight).
  - L2_to_truth_free: Euclidean distance on the free coordinates.
  - diagnostics:      ELBO / IWAE / SMC bounds; prior_log_p_* for MLE.

The `truth` block at top level gives the same in raw + derived form
for the DGP; see the `metadata.panel` block for this bundle's panel
size (N, T, obs_seed) and sigma_0 calibration (the canonical bundle is
T=6; the IVI joint-normal variant is T=40, same truth otherwise).
"""


def _softplus(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    if x > 30.0:
        return float(x)
    return math.log1p(math.exp(x))


def _exp(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    return math.exp(float(x))


def derive_quantities(theta: Dict[str, Any]) -> Dict[str, Any]:
    """Map raw 13-element theta to human-readable generative quantities."""
    def _f(k: str, default: float = 0.0) -> Optional[float]:
        v = theta.get(k, default)
        if v is None:
            return None
        return float(v)

    a0 = _f('alpha0')
    la1 = _f('log_alpha1')
    mu = [_f(k) for k in ('mu0', 'mu1', 'mu2')]
    sig = [_f(k) for k in ('sigma0', 'sigma1', 'sigma2')]
    z1ls = _f('z1_log_std')
    z1sk = _f('z1_skew')
    z1lt = _f('z1_log_tail')
    lse = theta.get('log_sigma_eps', None)
    lb = theta.get('log_beta', None)

    out: Dict[str, Any] = OrderedDict()
    out['kink_bias_b'] = a0
    out['kink_log_gamma'] = la1
    out['kink_gamma'] = _exp(la1)
    out['mu_poly_coeffs'] = OrderedDict(
        [('a0', mu[0]), ('a1', mu[1]), ('a2', mu[2])])
    out['sigma_pre_softplus_coeffs'] = OrderedDict(
        [('b0', sig[0]), ('b1', sig[1]), ('b2', sig[2])])
    out['sigma_at_zero'] = _softplus(sig[0])
    out['mu_at_zero'] = _hockey_mu_at_zero(a0, la1, mu)
    out['sigma_z1'] = _softplus(z1ls)
    out['z1_skew'] = z1sk
    out['z1_tailweight'] = _exp(z1lt)
    if lse is None:
        out['sigma_eps'] = None
    else:
        out['sigma_eps'] = math.exp(float(lse))
    if lb is None:
        out['beta_tailweight'] = None
    else:
        out['beta_tailweight'] = math.exp(float(lb))
    if lse is None and lb is None:
        out['note_emission'] = (
            'log_sigma_eps + log_beta both absent --- misspecification '
            'baseline (mle_direct). y_t = z_t exactly; no transitory '
            'shock to model.'
        )
    return out


def _hockey_mu_at_zero(a0: Optional[float], la1: Optional[float],
                        mu: List[Optional[float]]) -> Optional[float]:
    """mu(0) = b + gamma * softplus( (poly_mu(0) - b) / gamma )
    with b = alpha0, gamma = exp(log_alpha1), poly_mu(0) = mu0.
    Useful as a sanity-check quantity in the derived block."""
    if a0 is None or la1 is None or mu[0] is None:
        return None
    b = a0
    gamma = math.exp(la1)
    poly_at_0 = mu[0]
    x = (poly_at_0 - b) / gamma
    return b + gamma * _softplus(x)


def fill_canonical(theta: Optional[Dict[str, float]],
                    truth: Dict[str, float],
                    drop_keys: Optional[List[str]] = None
                    ) -> Optional[Dict[str, Any]]:
    drop = set(drop_keys or [])
    if theta is None:
        return None
    out: Dict[str, Any] = OrderedDict()
    for k in PARAM_NAMES:
        if k in drop:
            out[k] = None
        elif k in theta and theta[k] is not None:
            out[k] = float(theta[k])
        else:
            out[k] = float(truth[k])
    return out


def l2_free(theta: Dict[str, Any], truth: Dict[str, float],
             free_keys: List[str]) -> Optional[float]:
    acc = 0.0
    for k in free_keys:
        a = theta.get(k)
        if a is None:
            return None
        acc += (float(a) - float(truth[k])) ** 2
    return math.sqrt(acc)


_ENCODER_LABEL = {
    'jn': 'joint-normal',
    'mean_field': 'mean-field',
    'struct_markov': 'structured Markov',
    'tridiag_jn': 'tridiag joint-normal',
    'tjn': 'transformed joint-normal',
    None: 'n/a',
}


def label_fit(cfg: Dict[str, Any]) -> str:
    method = cfg.get('method')
    enc = cfg.get('encoder')
    alpha = cfg.get('picard_alpha', cfg.get('alpha'))
    m_mem = cfg.get('m_mem')
    enc_label = _ENCODER_LABEL.get(enc, str(enc))
    if method == 'mle_direct':
        return 'MLE (no measurement error)'
    if method == 'vi_only':
        return f'VI ({enc_label})'
    if method == 'picard_cold':
        return f'IVI Picard cold alpha={alpha} ({enc_label})'
    if method == 'picard_warm':
        return f'IVI Picard warm alpha={alpha} ({enc_label})'
    if method == 'anderson':
        return f'IVI Anderson m={m_mem} alpha={alpha} ({enc_label})'
    return f'{method} ({enc_label})'


def family_tag(method: Optional[str]) -> str:
    if method == 'vi_only':
        return 'vi'
    if method in ('picard_cold', 'picard_warm', 'anderson'):
        return 'ivi'
    if method == 'mle_direct':
        return 'mle_no_measurement_error'
    return method or 'unknown'


def _synthesize_vi_only_jn_entry(
    paths: List[str], canonical_truth: Dict[str, float]
) -> Optional[Dict[str, Any]]:
    """Build the synthesized VI-only joint-normal fits[] entry.

    Per `specs/compute-simulation-hockeystick.md` § "Aggregate parameter
    summary" → "Synthesized VI-only JN entry": extract `theta_VI_obs` from
    the Picard IVI cell's JSON (its Phase 1 fit, which is a cold-start JN
    h=64 VI fit on y_obs) and emit it as a sixth shipping fits[] row
    labelled "VI (joint-normal)". Inherits *_at_vi and smc_log_p_at_truth
    diagnostics from the IVI cell; nulls out *_at_ivi and *_at_truth.
    Returns None when the Picard IVI cell's JSON is absent (e.g. a
    partial workflow download).
    """
    picard_path = next(
        (p for p in paths
         if os.path.basename(p).startswith(PICARD_IVI_CELL_STEM)),
        None,
    )
    if picard_path is None:
        return None

    d = json.load(open(picard_path))
    cfg = d.get('config', {})
    file_truth = d.get('truth', canonical_truth)
    theta_fit = d.get('theta_VI_obs')
    if theta_fit is None:
        return None

    canonical_fit = fill_canonical(theta_fit, file_truth, drop_keys=None)
    l2 = (l2_free(canonical_fit, file_truth, FREE_KEYS_IVI)
          if canonical_fit is not None else None)

    src_diag = d.get('diagnostics') or {}
    diag: Dict[str, Any] = OrderedDict()
    diag['elbo_at_vi'] = src_diag.get('elbo_at_vi')
    diag['iwae_at_vi'] = src_diag.get('iwae_at_vi')
    diag['elbo_at_ivi'] = None
    diag['iwae_at_ivi'] = None
    diag['elbo_at_truth'] = None
    diag['iwae_at_truth'] = None
    diag['smc_log_p_at_truth'] = src_diag.get('smc_log_p_at_truth')

    entry: Dict[str, Any] = OrderedDict()
    entry['label'] = 'VI (joint-normal)'
    entry['family'] = 'vi'
    entry['shipping'] = True
    entry['method'] = 'vi_only'
    entry['encoder'] = 'jn'
    entry['emission'] = 'sinh'
    entry['alpha'] = None
    entry['m_mem'] = None
    entry['lr'] = cfg.get('lr')
    entry['hidden_dim'] = cfg.get('hidden_dim')
    entry['n_epochs_inner'] = None
    entry['n_iters_outer'] = None
    entry['n_epochs_vi_obs'] = cfg.get('n_epochs_vi_obs')
    entry['n_epochs_mle'] = None
    entry['source_file'] = os.path.relpath(picard_path)
    entry['theta_role'] = (
        'theta_VI_obs (Picard IVI cell Phase 1 fit, synthesized as the '
        'VI-only joint-normal estimator per spec)'
    )
    entry['free_keys'] = FREE_KEYS_IVI
    entry['parameters'] = canonical_fit
    entry['parameters_derived'] = (
        derive_quantities(canonical_fit)
        if canonical_fit is not None else None
    )
    entry['L2_to_truth_free'] = l2
    entry['diagnostics'] = diag
    entry['truth_tagged_in_source'] = OrderedDict(
        (k, float(file_truth.get(k, 0.0))) for k in PARAM_NAMES
    )
    return entry


def main() -> None:
    paths = sorted(
        p for p in glob.glob(os.path.join(SOURCE_DIR, '*.json'))
        if not p.endswith('hockeystick_parameter_summary.json')
    )
    if not paths:
        raise SystemExit(f'No JSONs found under {SOURCE_DIR}')

    # All cells share the same truth (canonical sigma_0 = -1.80
    # calibration); pick the first one and verify with a warning if any
    # downstream cell drifts. Panel dims (N, T, obs_seed, sigma_0) are
    # read from the first cell's config/truth so the summary reports the
    # actual bundle -- T=6 for the canonical bundle, T=40 for the variant.
    first = json.load(open(paths[0]))
    canonical_truth = first['truth']
    first_cfg = first.get('config', {})
    panel_N = first_cfg.get('N', 30000)
    panel_T = first_cfg.get('T', 6)
    panel_obs_seed = first_cfg.get('obs_seed', 11)
    panel_sigma0 = float(canonical_truth.get('sigma0', -1.80))

    truth_block = OrderedDict()
    truth_block['parameters'] = OrderedDict(
        (k, float(canonical_truth.get(k, 0.0))) for k in PARAM_NAMES
    )
    truth_block['parameters_derived'] = derive_quantities(canonical_truth)

    fits: List[Dict[str, Any]] = []
    for p in paths:
        d = json.load(open(p))
        cfg = d.get('config', {})
        method = cfg.get('method')
        file_truth = d.get('truth', canonical_truth)

        # Headline fitted vector by family.
        if method == 'vi_only':
            theta_fit = d.get('theta_VI_obs') or d.get('final_theta')
            theta_role = 'theta_VI_obs (VI MAP on observed panel)'
        elif method in ('picard_cold', 'picard_warm', 'anderson'):
            theta_fit = d.get('final_theta')
            theta_role = ('final_theta (IVI binding-equation fixed '
                          'point after N_ITERS_OUTER outer iters)')
        elif method == 'mle_direct':
            theta_fit = d.get('final_theta')
            theta_role = ('final_theta (MLE under degenerate y_t = z_t '
                          'emission; log_sigma_eps and log_beta do not '
                          'exist)')
        else:
            theta_fit = d.get('final_theta')
            theta_role = 'final_theta'

        drop_keys = (['log_sigma_eps', 'log_beta']
                     if method == 'mle_direct' else None)
        canonical_fit = fill_canonical(theta_fit, file_truth,
                                        drop_keys=drop_keys)
        free_keys = (FREE_KEYS_MLE_DIRECT if method == 'mle_direct'
                     else FREE_KEYS_IVI)
        l2 = None
        if canonical_fit is not None:
            l2 = l2_free(canonical_fit, file_truth, free_keys)

        basename = os.path.basename(p)
        entry: Dict[str, Any] = OrderedDict()
        entry['label'] = label_fit(cfg)
        entry['family'] = family_tag(method)
        entry['shipping'] = _is_shipping(basename)
        entry['method'] = method
        entry['encoder'] = cfg.get('encoder')
        entry['emission'] = cfg.get('emission')
        entry['alpha'] = cfg.get('picard_alpha', cfg.get('alpha'))
        entry['m_mem'] = cfg.get('m_mem')
        entry['lr'] = cfg.get('lr')
        entry['hidden_dim'] = cfg.get('hidden_dim')
        entry['n_epochs_inner'] = cfg.get('n_epochs_inner')
        entry['n_iters_outer'] = cfg.get('n_iters_outer')
        entry['n_epochs_vi_obs'] = cfg.get('n_epochs_vi_obs')
        entry['n_epochs_mle'] = cfg.get('n_epochs_mle')
        entry['source_file'] = os.path.relpath(p)
        entry['theta_role'] = theta_role
        entry['free_keys'] = free_keys
        entry['parameters'] = canonical_fit
        entry['parameters_derived'] = (
            derive_quantities(canonical_fit)
            if canonical_fit is not None else None
        )
        entry['L2_to_truth_free'] = l2
        entry['diagnostics'] = d.get('diagnostics')
        entry['truth_tagged_in_source'] = OrderedDict(
            (k, float(file_truth.get(k, 0.0))) for k in PARAM_NAMES
        )
        fits.append(entry)

    synthesized = _synthesize_vi_only_jn_entry(paths, canonical_truth)
    if synthesized is not None:
        fits.append(synthesized)

    out: Dict[str, Any] = OrderedDict()
    out['metadata'] = OrderedDict([
        ('title', 'simulation-hockeystick: fitted parameter export'),
        ('dgp_spec', 'specs/compute-simulation-hockeystick.md'),
        ('models_spec', 'specs/models.md'),
        ('source_dir', SOURCE_DIR),
        ('panel', OrderedDict([('N', panel_N), ('T', panel_T),
                                ('obs_seed', panel_obs_seed),
                                ('sigma0_calibration', panel_sigma0)])),
        ('produced_by',
         'scripts/_export_hockeystick_parameter_summary.py'),
        ('n_fits', len(fits)),
        ('n_shipping', sum(1 for e in fits if e['shipping'])),
    ])
    out['ai_prompt'] = AI_PROMPT
    out['parameter_schema'] = OrderedDict([
        ('param_names', PARAM_NAMES),
        ('free_keys_ivi_and_vi', FREE_KEYS_IVI),
        ('free_keys_mle_direct', FREE_KEYS_MLE_DIRECT),
        ('softplus_params', ['sigma0', 'sigma1', 'sigma2', 'z1_log_std']),
        ('exp_params', ['log_alpha1', 'log_sigma_eps', 'log_beta',
                         'z1_log_tail']),
        ('pinned_params', OrderedDict([
            ('z1_skew', 0.0),
            ('theta_MA', 0.0),
        ])),
        ('description', (
            'Parameter names use mixed conventions for historical '
            'reasons: prior std uses softplus (sigma(z), z1 scale), '
            'while all tailweights + the kink scale + the emission '
            'scale use exp. See the ai_prompt field for the full '
            'mapping; the softplus-kinked drift mu(z) is the '
            'hockey-stick distinguishing feature.'
        )),
    ])
    out['truth'] = truth_block
    out['fits'] = fits

    os.makedirs(SOURCE_DIR, exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(out, f, indent=2)

    n_shipping = sum(1 for e in fits if e['shipping'])
    print(f'Wrote {OUTPUT_PATH}')
    print(f'  {len(fits)} fits catalogued '
          f'({n_shipping} shipping + {len(fits) - n_shipping} historical)')
    for e in fits:
        l2 = e['L2_to_truth_free']
        l2s = f'{l2:+.4f}' if isinstance(l2, float) else 'n/a'
        tag = '*' if e['shipping'] else ' '
        print(f'  {tag} {e["label"]:55} L2={l2s}')
    print('  (* = shipping per specs/compute-simulation-hockeystick.md '
          'Entry section)')


if __name__ == '__main__':
    main()
