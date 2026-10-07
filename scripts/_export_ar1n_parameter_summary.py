"""Self-contained parameter export for the simulation-ar1n sweep.

Reads every fit JSON under ``output/bundles/cells/_simulation_ar1n_sweep/`` and writes
a single combined JSON to the same folder containing, for each fit:

  - the raw 13-element parameter vector (canonical PARAM_NAMES ordering),
  - the derived human-readable quantities (sigma_z1, sigma_eps, ...),
  - the L2 distance to truth on the eight free coordinates,
  - the bound diagnostics (ELBO / IWAE / SMC at vi / ivi / truth),

plus a top-level ``ai_prompt`` that describes how to instantiate
mu(z), sigma(z), the density of z_1, and the density of the
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

# Canonical ordering as defined in scripts/_ar1n_posterior_comparison.py
# (PARAM_NAMES = list(TRUTH.keys())) and re-exported by
# scripts/_simulation_ar1n_sweep.py.
PARAM_NAMES: List[str] = [
    'alpha0', 'log_alpha1',
    'mu0', 'mu1', 'mu2',
    'sigma0', 'sigma1', 'sigma2',
    'z1_log_std', 'z1_skew', 'z1_log_tail',
    'log_sigma_eps', 'log_beta',
]
FREE_KEYS_A: List[str] = [
    'mu0', 'mu1', 'mu2', 'sigma0', 'sigma1', 'sigma2',
    'z1_log_std', 'log_sigma_eps',
]
FREE_KEYS_B: List[str] = [
    'mu0', 'mu1', 'mu2', 'sigma0', 'sigma1', 'sigma2', 'z1_log_std',
]

SOURCE_DIR = os.path.join('output', 'bundles', 'cells', '_simulation_ar1n_sweep')
OUTPUT_PATH = os.path.join('output', 'bundles', 'ar1n_parameter_summary.json')


AI_PROMPT = """\
This JSON catalogues the fitted parameter vector for every model in the
simulation-ar1n sweep (a 1-D AR(1) latent state-space model with normal
emission). Each entry in `fits` reports a 13-element parameter dict in
the canonical ordering listed under `parameter_schema.param_names`.
The mapping from those parameters to the generative model is:

(1) Law of motion (transition density):
    z_t | z_{t-1} ~ Normal( mu(z_{t-1}), sigma(z_{t-1})^2 ),
    with
        mu(z)    = mu0 + mu1 * z + mu2 * z^2,
        sigma(z) = softplus(sigma0 + sigma1 * z + sigma2 * z^2),
    where softplus(x) = log(1 + exp(x)).
    NB: sigma uses softplus, NOT exp.

(2) Initial-state density:
    z_1 ~ SinhArcsinh( loc=0, scale=softplus(z1_log_std),
                       skew=z1_skew, tailweight=exp(z1_log_tail) ).
    The scalar SinhArcsinh density is
        z = loc + scale * sinh( tailweight * ( asinh(eps) + skew ) ),
        eps ~ Normal(0, 1).
    For the AR1+Normal sweep, z1_skew = 0 and z1_log_tail = 0 are pinned
    (exp(0) = 1), so z_1 reduces to Normal(0, softplus(z1_log_std)^2).
    NB: scale uses softplus on z1_log_std; tailweight uses exp on
    z1_log_tail. The 'log_std' suffix is historical and misleading ---
    it is pre-softplus, not pre-exp.

(3) Measurement-error density:
    y_t = z_t + eps_t,   eps_t ~ SinhArcsinh( loc=0,
                                              scale=exp(log_sigma_eps),
                                              skew=0,
                                              tailweight=exp(log_beta) ).
    For the AR1+Normal sweep, log_beta = 0 is pinned (exp(0) = 1), so
    eps_t reduces to Normal(0, exp(log_sigma_eps)^2). The 'log' prefix
    here IS log: sigma_eps = exp(log_sigma_eps) (no softplus).

(4) Heterogeneity parameters (alpha0, log_alpha1):
    Pinned at 0 for this DGP family --- ignore.

The 8 free coordinates under Model A (used by every VI and IVI fit) are
    {mu0, mu1, mu2, sigma0, sigma1, sigma2, z1_log_std, log_sigma_eps}.
The MLE no-error fit (Model B in `specs/compute-simulation-ar1n.md`)
drops log_sigma_eps and keeps only the 7 free prior coordinates --- the
emission collapses to y_t = z_t exactly (no measurement error).

Each entry in `fits` reports:
  - method:           'vi_only' | 'picard' | 'anderson' | 'mle_noerror'
  - encoder:          variational family for VI / IVI fits (None for MLE).
  - parameters:       raw 13-element dict in canonical order.
  - parameters_derived: human-friendly transforms (sigma_z1, sigma_eps,
                        beta, mu/sigma polynomial coefficients).
  - L2_to_truth:      Euclidean distance on the 8 free coordinates.
  - diagnostics:      ELBO / IWAE / SMC bounds at VI / IVI / truth.

The `truth` block at top level gives the same in raw + derived form for
the DGP that simulated the observed panel (N=30000, T=6, obs_seed=11).
"""


def _softplus(x: float) -> float:
    # Numerically-stable softplus.
    if x > 30.0:
        return x
    return math.log1p(math.exp(x))


def derive_quantities(theta: Dict[str, Any]) -> Dict[str, Any]:
    """Map raw 13-element theta to the human-readable generative-model
    quantities described in the AI prompt.

    Values stored as None (e.g. log_sigma_eps under Model B) propagate
    as None in the derived block rather than being treated as 0.
    """
    def _f(k: str, default: float = 0.0) -> Optional[float]:
        v = theta.get(k, default)
        if v is None:
            return None
        return float(v)

    a = _f('alpha0')
    la1 = _f('log_alpha1')
    mu = [_f(k) for k in ('mu0', 'mu1', 'mu2')]
    sig = [_f(k) for k in ('sigma0', 'sigma1', 'sigma2')]
    z1ls = _f('z1_log_std')
    z1sk = _f('z1_skew')
    z1lt = _f('z1_log_tail')
    lse = theta.get('log_sigma_eps', None)
    lb = _f('log_beta')

    out: Dict[str, Any] = OrderedDict()
    out['mu_coeffs'] = OrderedDict([('a0', mu[0]), ('a1', mu[1]), ('a2', mu[2])])
    out['sigma_pre_softplus_coeffs'] = OrderedDict(
        [('b0', sig[0]), ('b1', sig[1]), ('b2', sig[2])])
    out['sigma_at_zero'] = (_softplus(sig[0]) if sig[0] is not None else None)
    out['sigma_z1'] = (_softplus(z1ls) if z1ls is not None else None)
    out['z1_skew'] = z1sk
    out['z1_tailweight'] = (math.exp(z1lt) if z1lt is not None else None)
    if lse is None:
        out['sigma_eps'] = None
        out['note_emission'] = (
            'log_sigma_eps absent --- this is Model B (no measurement '
            'error). y_t = z_t exactly; no transitory shock to model.'
        )
    else:
        out['sigma_eps'] = math.exp(float(lse))
    out['beta_tailweight'] = (math.exp(lb) if lb is not None else None)
    out['alpha0'] = a
    out['log_alpha1'] = la1
    return out


def fill_canonical(theta: Optional[Dict[str, float]],
                    truth: Dict[str, float],
                    drop_keys: Optional[List[str]] = None
                    ) -> Optional[Dict[str, float]]:
    """Pad a partial theta dict up to the canonical 13-key ordering,
    using truth as the source for missing keys.

    ``drop_keys`` lists parameters that do not exist in this fit's
    parameter vector (e.g. log_sigma_eps under Model B): they are
    recorded as None in the output rather than being silently filled
    in from truth.

    Returns None unchanged.
    """
    drop = set(drop_keys or [])
    if theta is None:
        return None
    out: Dict[str, Any] = OrderedDict()
    for k in PARAM_NAMES:
        if k in drop:
            out[k] = None
        elif k in theta:
            out[k] = float(theta[k])
        else:
            out[k] = float(truth[k])
    return out


def l2_free(theta: Dict[str, float], truth: Dict[str, float],
             free_keys: List[str]) -> float:
    return math.sqrt(sum((theta[k] - truth[k]) ** 2 for k in free_keys))


def label_fit(filename: str, cfg: Dict[str, Any]) -> str:
    """Construct a short human-readable label for the fit."""
    method = cfg.get('method')
    enc = cfg.get('encoder')
    lr = cfg.get('lr')
    alpha = cfg.get('alpha')
    if method == 'mle_noerror':
        return 'MLE (no measurement error)'
    enc_label = {
        'mean_field': 'mean-field',
        'jn': 'joint-normal',
        'tridiag': 'tridiag joint-normal',
        'struct_markov': 'structured Markov',
        'tjn': 'transformed joint-normal',
    }.get(enc, str(enc))
    if method == 'vi_only':
        return f'VI ({enc_label}, lr={lr})'
    if method == 'picard':
        return f'IVI Picard alpha={alpha} ({enc_label}, lr={lr})'
    if method == 'anderson':
        return f'IVI Anderson alpha={alpha} ({enc_label}, lr={lr})'
    return f'{method} ({enc_label}, lr={lr})'


def family_tag(method: Optional[str]) -> str:
    if method == 'vi_only':
        return 'vi'
    if method in ('picard', 'anderson'):
        return 'ivi'
    if method == 'mle_noerror':
        return 'mle_no_measurement_error'
    return method or 'unknown'


def main() -> None:
    paths = sorted(
        p for p in glob.glob(os.path.join(SOURCE_DIR, '*.json'))
        if not p.endswith('ar1n_parameter_summary.json')
    )
    if not paths:
        raise SystemExit(f'No JSONs found under {SOURCE_DIR}')

    # Read the first file to fish out a canonical truth block. All sweep
    # files share the same DGP; per-file truth dicts may differ only by
    # whether the picard-side log_sigma_eps was overridden at the CLI,
    # so we report the truth that's tagged on each fit alongside the
    # canonical DGP-headline truth from the very first VI-only file.
    canonical_truth: Optional[Dict[str, float]] = None
    for p in paths:
        d = json.load(open(p))
        cfg = d.get('config', {})
        if cfg.get('method') == 'vi_only' and 'truth' in d:
            canonical_truth = d['truth']
            break
    if canonical_truth is None:
        canonical_truth = json.load(open(paths[0]))['truth']

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

        # Pick the headline fitted vector per family.
        if method == 'vi_only':
            theta_fit = d.get('theta_VI_obs')
            theta_role = 'theta_VI_obs (VI MAP on observed panel)'
        elif method in ('picard', 'anderson'):
            theta_fit = d.get('final_theta')
            theta_role = 'final_theta (IVI binding-equation fixed point)'
        elif method == 'mle_noerror':
            theta_fit = d.get('final_theta')
            theta_role = ('final_theta (MLE under degenerate y_t=z_t '
                          'emission; log_sigma_eps does not exist)')
        else:
            theta_fit = d.get('final_theta')
            theta_role = 'final_theta'

        drop_keys = ['log_sigma_eps'] if method == 'mle_noerror' else None
        canonical_fit = fill_canonical(theta_fit, file_truth,
                                        drop_keys=drop_keys)
        free_keys = FREE_KEYS_B if method == 'mle_noerror' else FREE_KEYS_A
        l2 = None
        if canonical_fit is not None:
            l2 = l2_free(canonical_fit, file_truth, free_keys)

        entry: Dict[str, Any] = OrderedDict()
        entry['label'] = label_fit(p, cfg)
        entry['family'] = family_tag(method)
        entry['method'] = method
        entry['encoder'] = cfg.get('encoder')
        entry['alpha'] = cfg.get('alpha')
        entry['lr'] = cfg.get('lr')
        entry['hidden_dim'] = cfg.get('hidden_dim')
        entry['n_epochs_inner'] = cfg.get('n_epochs_inner')
        entry['n_iters_outer'] = cfg.get('n_iters_outer')
        entry['n_epochs_vi_obs'] = cfg.get('n_epochs_vi_obs')
        entry['source_file'] = os.path.relpath(p)
        entry['theta_role'] = theta_role
        entry['free_keys'] = free_keys
        entry['parameters'] = canonical_fit
        entry['parameters_derived'] = (
            derive_quantities(canonical_fit) if canonical_fit is not None
            else None
        )
        entry['L2_to_truth_free'] = l2
        entry['diagnostics'] = d.get('diagnostics')
        # Convenience: store the file-tagged truth too, since the picard
        # files were re-run at a slightly different log_sigma_eps and
        # this can be confusing.
        entry['truth_tagged_in_source'] = OrderedDict(
            (k, float(file_truth.get(k, 0.0))) for k in PARAM_NAMES
        )
        fits.append(entry)

    out: Dict[str, Any] = OrderedDict()
    out['metadata'] = OrderedDict([
        ('title', 'simulation-ar1n: fitted parameter export'),
        ('dgp_spec', 'specs/compute-simulation-ar1n.md'),
        ('models_spec', 'specs/models.md'),
        ('source_dir', SOURCE_DIR),
        ('panel', OrderedDict([('N', 30000), ('T', 6),
                                ('obs_seed', 11)])),
        ('produced_by',
         'scripts/_export_ar1n_parameter_summary.py'),
        ('n_fits', len(fits)),
    ])
    out['ai_prompt'] = AI_PROMPT
    out['parameter_schema'] = OrderedDict([
        ('param_names', PARAM_NAMES),
        ('free_keys_model_a', FREE_KEYS_A),
        ('free_keys_model_b', FREE_KEYS_B),
        ('softplus_params', ['sigma0', 'sigma1', 'sigma2', 'z1_log_std']),
        ('exp_params', ['log_sigma_eps', 'log_beta', 'z1_log_tail',
                         'log_alpha1']),
        ('description', (
            'Parameter names use mixed conventions for historical '
            'reasons: prior std uses softplus (sigma(z), z1 scale), '
            'while emission scale + all tailweights use exp. See the '
            'ai_prompt field for the full mapping.'
        )),
    ])
    out['truth'] = truth_block
    out['fits'] = fits

    os.makedirs(SOURCE_DIR, exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(out, f, indent=2)

    print(f'Wrote {OUTPUT_PATH}')
    print(f'  {len(fits)} fits catalogued')
    for e in fits:
        l2 = e['L2_to_truth_free']
        l2s = f'{l2:+.4f}' if isinstance(l2, float) else 'n/a'
        print(f'  - {e["label"]:48} L2={l2s}')


if __name__ == '__main__':
    main()
