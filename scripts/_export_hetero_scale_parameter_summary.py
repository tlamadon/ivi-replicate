"""Self-contained parameter export for the simulation-hetero-scale sweep.

Reads every cell JSON under ``output/bundles/cells/_simulation_hetero_scale_ivi_sweep/``
and writes a single combined JSON to the same folder containing, for
each fit:

  - the raw 12-element parameter vector (canonical PARAM_NAMES ordering),
  - the derived human-readable quantities (sigma_z1, beta, alpha
    conditional law, implied marginal alpha + rho),
  - the L2 distance to truth on the free coordinates,
  - the bound diagnostics (ELBO / IWAE / SMC at vi / ivi / truth;
    plus the MLE-only prior_log_p quantities for the misspec cell),

plus a top-level ``ai_prompt`` that describes how to instantiate
mu(z), sigma(z), the density of z_1, the alpha conditional law, and
the per-individual emission density from the raw parameters --- so
the JSON is usable without any source-tree context.
"""

from __future__ import annotations

import glob
import json
import math
import os
from collections import OrderedDict
from typing import Any, Dict, List, Optional


# Canonical 12-tuple. `z1_skew` is pinned at 0 in the estimator
# (per specs/compute-simulation-hetero-scale.md) and is NOT part of
# the parameter vector. The emission has no `log_sigma_eps`: the
# per-individual residual log-scale is the extra latent `alpha_i`.
PARAM_NAMES: List[str] = [
    'mu0', 'mu1', 'mu2',
    'sigma0', 'sigma1', 'sigma2',
    'z1_log_std', 'z1_log_tail',
    'log_beta',
    'beta_a0', 'beta_a1', 'log_sigma_a_cond',
]

# Model A: well-specified prior + decoder (IVI/VI cells). All 12 free.
FREE_KEYS_A: List[str] = list(PARAM_NAMES)

# Model B: misspec baseline (mle_direct, degenerate emission y_t = z_t).
# The decoder is None and the extra-hetero branch also drops because
# alpha enters only through the emission. The 8 free coordinates are
# the homogeneous poly-2 prior + sinh-z_1 subset; `log_beta`,
# `beta_a0`, `beta_a1`, `log_sigma_a_cond` do not exist in the
# misspec model class.
FREE_KEYS_B: List[str] = [
    'mu0', 'mu1', 'mu2',
    'sigma0', 'sigma1', 'sigma2',
    'z1_log_std', 'z1_log_tail',
]
MISSPEC_ABSENT_KEYS: List[str] = [
    'log_beta', 'beta_a0', 'beta_a1', 'log_sigma_a_cond',
]

# Filenames the spec declares for the six canonical cells.
CANONICAL_FILES: List[str] = [
    'picard_cold_alpha0.60_ep16000_lr1e-2_crn_rho0.30.json',
    'anderson_alpha0.60_m4_ep16000_lr1e-2_crn_rho0.30.json',
    'vi_only_meanfield_h64_ep20000_lr1e-2_crn_rho0.30.json',
    'vi_only_struct_markov_h64_ep20000_lr1e-2_crn_rho0.30.json',
    'vi_only_tjn_h64_ep20000_lr1e-2_crn_rho0.30.json',
    'mle_direct_no_me_ep20000_lr1e-2_rho0.30.json',
]

SOURCE_DIR = os.path.join('output', 'bundles', 'cells', '_simulation_hetero_scale_ivi_sweep')
OUTPUT_PATH = os.path.join('output', 'bundles', 'hetero_scale_parameter_summary.json')


AI_PROMPT = """\
This JSON catalogues the fitted parameter vector for every cell in the
simulation-hetero-scale sweep --- a 1-D latent state-space model with
sinh-arcsinh emission noise that is rescaled by a per-individual scale
latent `alpha_i` correlated with the initial state `z_{i,1}`. Each
entry in `fits` reports a 12-element parameter dict in the canonical
ordering listed under `parameter_schema.param_names`. The mapping
from those parameters to the generative model is:

(1) Law of motion (transition density):
    z_t | z_{t-1} ~ Normal( mu(z_{t-1}), sigma(z_{t-1})^2 ),
    with
        mu(z)    = mu0 + mu1 * z + mu2 * z^2,
        sigma(z) = softplus(sigma0 + sigma1 * z + sigma2 * z^2),
    where softplus(x) = log(1 + exp(x)). The softplus kink of the
    hockey-stick sibling DGP is *removed* here --- only the poly-2
    mu and poly-2 sigma are estimated. NB: sigma uses softplus,
    NOT exp.

(2) Initial-state density:
    z_1 ~ SinhArcsinh( loc=0, scale=softplus(z1_log_std),
                       skew=0, tailweight=exp(z1_log_tail) ).
    The scalar SinhArcsinh density is
        z = loc + scale * sinh( tailweight * ( asinh(eps) + skew ) ),
        eps ~ Normal(0, 1).
    For this entry, `z1_skew = 0` is *pinned* (the truth has skew = 0
    and the estimator matches the hockey-stick sibling convention by
    fixing it) --- z1_skew is therefore absent from PARAM_NAMES and
    absent from the parameter vector. `z1_log_tail` is free; the
    truth has tailweight exp(0.117) approx 1.124. NB: scale uses
    softplus on `z1_log_std`; tailweight uses exp on `z1_log_tail`.

(3) Per-individual residual-scale heterogeneity:
    alpha_i | z_{i,1} ~ Normal( beta_a0 + beta_a1 * z_{i,1},
                                 exp(log_sigma_a_cond)^2 ).
    These three coefficients are the conditional-law parameters.
    Under the spec's marginal-preserving reparametrisation,
        beta_a0      = mu_alpha_marg,
        beta_a1      = rho * sigma_alpha_marg / sigma_z1,
        sigma_a_cond = sigma_alpha_marg * sqrt(1 - rho^2),
    where sigma_z1 = softplus(z1_log_std). The implied marginal
    parameters (mu_alpha_marg, sigma_alpha_marg) and the correlation
    `rho = corr(alpha_i, z_{i,1})` are reported in
    `parameters_derived` under `mu_alpha_marg_implied`,
    `sigma_alpha_marg_implied`, `rho_implied`.

(4) Emission density (per period, per individual):
    y_t = z_t + exp(alpha_i) * eps_t,
    eps_t ~ SinhArcsinh( loc=0, scale=1, skew=0,
                          tailweight=exp(log_beta) ).
    There is NO separate `log_sigma_eps`: the per-individual
    log-scale of the residual is the latent alpha_i. The emission
    skew theta is pinned at 0 (also fixed at construction). For the
    truth, exp(log_beta) approx 2.128.

The 12 free coordinates under Model A (every IVI and VI-only fit) are
    {mu0, mu1, mu2, sigma0, sigma1, sigma2,
     z1_log_std, z1_log_tail,
     log_beta, beta_a0, beta_a1, log_sigma_a_cond}.

The misspecification baseline (Model B in
`specs/compute-simulation-hetero-scale.md`, `mle_direct` with
`EMISSION=degenerate`) drops the four emission/extra-hetero
parameters (log_beta, beta_a0, beta_a1, log_sigma_a_cond) because
the inference model has no decoder and no alpha latent --- the
log-likelihood is the closed-form prior log p(z_{1:T}; theta)
evaluated at z = y_obs. The 8 free coordinates for Model B are the
homogeneous-prior subset {mu0..2, sigma0..2, z1_log_std,
z1_log_tail}; missing fields are recorded as None and the absent
parameters are listed under `parameter_schema.misspec_absent_keys`.

Each entry in `fits` reports:
  - family / method / encoder: how the cell was fit. method is one
    of 'picard_cold', 'anderson', 'vi_only', 'mle_direct'. encoder
    is one of 'jn' (joint-normal with one extra latent for alpha),
    'mean_field', 'struct_markov', 'tjn' (transformed joint-normal),
    or None (for mle_direct).
  - parameters:         raw 12-element dict; None for fields absent
                         in the misspec model class.
  - parameters_derived: human-friendly transforms (sigma_at_zero,
                         sigma_z1, beta_tailweight, sigma_a_cond,
                         and the implied alpha-marginal + rho).
  - L2_to_truth_free:   Euclidean distance on the free coordinates
                         (12 for Model A; 8 for Model B).
  - diagnostics:        ELBO / IWAE / SMC bounds at VI / IVI / truth
                         and (Model B only) prior_log_p_{at_mle,
                         at_truth}.

The `truth` block at top level gives the same in raw + derived form
for the DGP that simulated the observed panel (N=30000, T=6,
obs_seed=11, rho=0.30, mu_alpha_marg=log(0.10), sigma_alpha_marg=0.30).
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
    """Map raw 12-element theta to the human-readable generative-model
    quantities described in the AI prompt.

    Values stored as None (e.g. log_beta under Model B) propagate as
    None in the derived block rather than being treated as 0.
    """
    def _f(k: str) -> Optional[float]:
        v = theta.get(k, None)
        return None if v is None else float(v)

    mu = [_f('mu0'), _f('mu1'), _f('mu2')]
    sig = [_f('sigma0'), _f('sigma1'), _f('sigma2')]
    z1ls = _f('z1_log_std')
    z1lt = _f('z1_log_tail')
    lb = _f('log_beta')
    ba0 = _f('beta_a0')
    ba1 = _f('beta_a1')
    lsa = _f('log_sigma_a_cond')

    out: Dict[str, Any] = OrderedDict()
    out['mu_coeffs'] = OrderedDict(
        [('a0', mu[0]), ('a1', mu[1]), ('a2', mu[2])])
    out['sigma_pre_softplus_coeffs'] = OrderedDict(
        [('b0', sig[0]), ('b1', sig[1]), ('b2', sig[2])])
    out['sigma_at_zero'] = _softplus(sig[0])
    out['sigma_z1'] = _softplus(z1ls)
    out['z1_skew_pinned'] = 0.0
    out['z1_tailweight'] = _exp(z1lt)

    if lb is None:
        out['beta_tailweight'] = None
        out['note_emission'] = (
            'log_beta absent --- this is Model B (degenerate emission '
            'y_t = z_t; no decoder). The misspec inference model has '
            'no transitory shock and no alpha latent.'
        )
    else:
        out['beta_tailweight'] = _exp(lb)

    # Alpha conditional law (Model A only).
    out['alpha_conditional'] = OrderedDict([
        ('beta_a0', ba0),
        ('beta_a1', ba1),
        ('sigma_a_cond', _exp(lsa)),
    ])

    # Implied marginal-alpha + correlation. Requires beta_a1,
    # log_sigma_a_cond, and sigma_z1; if any is None, leave None.
    sigma_z1 = out['sigma_z1']
    sigma_a_cond = out['alpha_conditional']['sigma_a_cond']
    if (ba1 is not None and sigma_z1 is not None
            and sigma_a_cond is not None):
        var_alpha = (ba1 ** 2) * (sigma_z1 ** 2) + sigma_a_cond ** 2
        sigma_alpha_marg = math.sqrt(var_alpha)
        rho = (ba1 * sigma_z1 / sigma_alpha_marg
               if sigma_alpha_marg > 0 else 0.0)
        out['mu_alpha_marg_implied'] = ba0
        out['sigma_alpha_marg_implied'] = sigma_alpha_marg
        out['rho_implied'] = rho
    else:
        out['mu_alpha_marg_implied'] = None
        out['sigma_alpha_marg_implied'] = None
        out['rho_implied'] = None

    return out


def fill_canonical(theta: Optional[Dict[str, float]],
                   truth: Dict[str, float],
                   drop_keys: Optional[List[str]] = None
                   ) -> Optional[Dict[str, float]]:
    """Pad a partial theta dict up to the canonical 12-key ordering,
    using truth as the source for missing keys.

    `drop_keys` lists parameters that do not exist in this fit's
    parameter vector (Model B drops log_beta + the three extra-hetero
    parameters): they are recorded as None.
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
            out[k] = float(truth.get(k, 0.0))
    return out


def l2_free(theta: Dict[str, float], truth: Dict[str, float],
            free_keys: List[str]) -> float:
    return math.sqrt(sum((theta[k] - truth[k]) ** 2 for k in free_keys))


def label_fit(cfg: Dict[str, Any]) -> str:
    method = cfg.get('method')
    enc = cfg.get('encoder')
    lr = cfg.get('lr')
    alpha = cfg.get('alpha')
    m_mem = cfg.get('m_mem')
    if method == 'mle_direct':
        return 'MLE direct (no measurement error, degenerate emission)'
    enc_label = {
        'mean_field': 'mean-field',
        'jn': 'joint-normal-extra h=64',
        'struct_markov': 'structured Markov',
        'tjn': 'transformed joint-normal',
    }.get(enc, str(enc))
    if method == 'vi_only':
        return f'VI-only ({enc_label}, lr={lr})'
    if method == 'picard_cold':
        return f'IVI Picard cold alpha={alpha} ({enc_label}, lr={lr})'
    if method == 'anderson':
        return (f'IVI Anderson m={m_mem} beta={alpha} '
                f'({enc_label}, lr={lr})')
    return f'{method} ({enc_label}, lr={lr})'


def family_tag(method: Optional[str]) -> str:
    if method == 'vi_only':
        return 'vi'
    if method in ('picard_cold', 'anderson'):
        return 'ivi'
    if method == 'mle_direct':
        return 'mle_no_measurement_error'
    return method or 'unknown'


def main() -> None:
    available = sorted(
        fn for fn in os.listdir(SOURCE_DIR)
        if fn in set(CANONICAL_FILES)
    )
    if not available:
        raise SystemExit(
            f'No canonical cell JSONs found under {SOURCE_DIR}; expected '
            f'{CANONICAL_FILES}')

    # Read the first available file for the canonical DGP truth block.
    # All canonical cells share the same truth (same DGP, obs_seed).
    canonical_truth: Optional[Dict[str, float]] = None
    for fn in available:
        d = json.load(open(os.path.join(SOURCE_DIR, fn)))
        if 'truth' in d:
            canonical_truth = d['truth']
            break
    if canonical_truth is None:
        raise SystemExit('No truth block found in any cell JSON')

    truth_block = OrderedDict()
    truth_block['parameters'] = OrderedDict(
        (k, float(canonical_truth.get(k, 0.0))) for k in PARAM_NAMES
    )
    truth_block['z1_skew_pinned'] = 0.0
    truth_block['parameters_derived'] = derive_quantities(canonical_truth)

    fits: List[Dict[str, Any]] = []
    for fn in available:
        p = os.path.join(SOURCE_DIR, fn)
        d = json.load(open(p))
        cfg = d.get('config', {})
        method = cfg.get('method')
        file_truth = d.get('truth', canonical_truth)

        # Headline fitted vector per family.
        if method == 'vi_only':
            theta_fit = d.get('theta_VI_obs')
            theta_role = 'theta_VI_obs (VI MAP on observed panel)'
        elif method in ('picard_cold', 'anderson'):
            theta_fit = d.get('final_theta')
            theta_role = 'final_theta (IVI binding-equation fixed point)'
        elif method == 'mle_direct':
            theta_fit = d.get('final_theta')
            theta_role = ('final_theta (MLE under degenerate y_t=z_t '
                          'emission; log_beta and the three extra-'
                          'hetero parameters do not exist)')
        else:
            theta_fit = d.get('final_theta')
            theta_role = 'final_theta'

        drop_keys = (MISSPEC_ABSENT_KEYS if method == 'mle_direct'
                     else None)
        canonical_fit = fill_canonical(
            theta_fit, file_truth, drop_keys=drop_keys)
        free_keys = (FREE_KEYS_B if method == 'mle_direct'
                     else FREE_KEYS_A)
        l2 = None
        if canonical_fit is not None:
            l2 = l2_free(canonical_fit, file_truth, free_keys)

        entry: Dict[str, Any] = OrderedDict()
        entry['label'] = label_fit(cfg)
        entry['family'] = family_tag(method)
        entry['method'] = method
        entry['encoder'] = cfg.get('encoder')
        entry['alpha'] = cfg.get('alpha')
        entry['m_mem'] = cfg.get('m_mem')
        entry['lr'] = cfg.get('lr')
        entry['hidden_dim'] = cfg.get('hidden_dim')
        entry['n_epochs_inner'] = cfg.get('n_epochs_inner')
        entry['n_iters_outer'] = cfg.get('n_iters_outer')
        entry['n_epochs_vi_obs'] = cfg.get('n_epochs_vi_obs')
        entry['n_epochs_mle'] = cfg.get('n_epochs_mle')
        entry['rho_target'] = cfg.get('rho')
        entry['mu_alpha_marg_target'] = cfg.get('mu_alpha_marg')
        entry['sigma_alpha_marg_target'] = cfg.get('sigma_alpha_marg')
        entry['source_file'] = os.path.relpath(p)
        entry['theta_role'] = theta_role
        entry['free_keys'] = free_keys
        entry['parameters'] = canonical_fit
        entry['parameters_derived'] = (
            derive_quantities(canonical_fit)
            if canonical_fit is not None else None)
        entry['L2_to_truth_free'] = l2
        entry['L2_VI'] = d.get('L2_VI')
        entry['final_L2'] = d.get('final_L2')
        entry['diagnostics'] = d.get('diagnostics')
        # Convenience: include the file-tagged truth (with z1_skew=0
        # present for reproducibility of how the cell saw it).
        entry['truth_tagged_in_source'] = OrderedDict(
            (k, float(file_truth.get(k, 0.0)))
            for k in PARAM_NAMES + ['z1_skew']
        )
        fits.append(entry)

    # ------------------------------------------------------------------
    # Lifted seventh row: VI-only joint-normal Phase-1 reference.
    #
    # Per specs/compute-simulation-hetero-scale.md, the summary must
    # expose the Phase-1 VI MAP under the joint-normal-extra encoder
    # (the starting point of every IVI cell) as a standalone fit row.
    # Source: picard_cold_alpha0.60_* by convention; the IVI cells'
    # theta_VI_obs agree to 1e-6 across cells so the choice does not
    # change the reported numbers.
    # ------------------------------------------------------------------
    LIFT_SOURCE = 'picard_cold_alpha0.60_ep16000_lr1e-2_crn_rho0.30.json'
    if LIFT_SOURCE in available:
        p = os.path.join(SOURCE_DIR, LIFT_SOURCE)
        d = json.load(open(p))
        theta_vi = d.get('theta_VI_obs')
        if theta_vi is None:
            raise SystemExit(
                f'cannot lift VI(JN) Phase-1 row: '
                f'{LIFT_SOURCE} has no theta_VI_obs')
        cfg = d.get('config', {})
        file_truth = d.get('truth', canonical_truth)
        canonical_fit = fill_canonical(theta_vi, file_truth, drop_keys=None)
        l2 = l2_free(canonical_fit, file_truth, FREE_KEYS_A)

        # Tailor the diagnostics block: only *_at_vi and *_at_truth
        # carry over (they evaluate at theta_VI_obs and at truth
        # respectively); the source cell's *_at_ivi evaluates at the
        # IVI fixed point and does NOT apply to this row.
        diag_src = d.get('diagnostics') or {}
        diag = OrderedDict([
            ('elbo_at_vi', diag_src.get('elbo_at_vi')),
            ('iwae_at_vi', diag_src.get('iwae_at_vi')),
            ('elbo_at_ivi', None),
            ('iwae_at_ivi', None),
            ('elbo_at_truth', diag_src.get('elbo_at_truth')),
            ('iwae_at_truth', diag_src.get('iwae_at_truth')),
            ('smc_log_p_at_truth', diag_src.get('smc_log_p_at_truth')),
        ])

        # Build the label via the shared helper, then suffix [Phase-1
        # lift] so consumers can spot the row's provenance at a glance.
        lift_cfg = dict(cfg)
        lift_cfg['method'] = 'vi_only'
        lift_cfg['encoder'] = 'jn'

        entry = OrderedDict()
        entry['label'] = f'{label_fit(lift_cfg)} [Phase-1 lift]'
        entry['family'] = 'vi'
        entry['method'] = 'vi_only'
        entry['encoder'] = 'jn'
        entry['alpha'] = None
        entry['m_mem'] = None
        entry['lr'] = cfg.get('lr')
        entry['hidden_dim'] = cfg.get('hidden_dim')
        entry['n_epochs_inner'] = None
        entry['n_iters_outer'] = None
        entry['n_epochs_vi_obs'] = cfg.get('n_epochs_vi_obs')
        entry['n_epochs_mle'] = None
        entry['rho_target'] = cfg.get('rho')
        entry['mu_alpha_marg_target'] = cfg.get('mu_alpha_marg')
        entry['sigma_alpha_marg_target'] = cfg.get('sigma_alpha_marg')
        entry['source_file'] = os.path.relpath(p)
        entry['theta_role'] = (
            'theta_VI_obs (Phase 1 VI MAP, joint-normal encoder)'
        )
        entry['free_keys'] = FREE_KEYS_A
        entry['parameters'] = canonical_fit
        entry['parameters_derived'] = derive_quantities(canonical_fit)
        entry['L2_to_truth_free'] = l2
        entry['L2_VI'] = d.get('L2_VI')
        entry['final_L2'] = None
        entry['diagnostics'] = diag
        entry['truth_tagged_in_source'] = OrderedDict(
            (k, float(file_truth.get(k, 0.0)))
            for k in PARAM_NAMES + ['z1_skew']
        )
        fits.append(entry)

    out: Dict[str, Any] = OrderedDict()
    out['metadata'] = OrderedDict([
        ('title', 'simulation-hetero-scale: fitted parameter export'),
        ('dgp_spec', 'specs/compute-simulation-hetero-scale.md'),
        ('models_spec', 'specs/models.md'),
        ('source_dir', SOURCE_DIR),
        ('panel', OrderedDict([('N', 30000), ('T', 6),
                                ('obs_seed', 11)])),
        ('produced_by',
         'scripts/_export_hetero_scale_parameter_summary.py'),
        ('n_fits', len(fits)),
    ])
    out['ai_prompt'] = AI_PROMPT
    out['parameter_schema'] = OrderedDict([
        ('param_names', PARAM_NAMES),
        ('free_keys_model_a', FREE_KEYS_A),
        ('free_keys_model_b', FREE_KEYS_B),
        ('misspec_absent_keys', MISSPEC_ABSENT_KEYS),
        ('pinned_at_zero', ['z1_skew', 'emission_theta']),
        ('softplus_params', ['sigma0', 'sigma1', 'sigma2', 'z1_log_std']),
        ('exp_params', ['log_beta', 'log_sigma_a_cond', 'z1_log_tail']),
        ('description', (
            'Parameter names use mixed conventions for historical '
            'reasons: prior std uses softplus (sigma(z), z1 scale), '
            'while emission tailweight + alpha-conditional std use '
            'exp. `z1_skew` and the emission skew `theta` are both '
            'pinned at 0 in the estimator and are NOT part of the '
            'parameter vector. See the ai_prompt field for the full '
            'mapping.'
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
        print(f'  - {e["label"]:60s} L2={l2s}')


if __name__ == '__main__':
    main()
