r"""Self-contained bundle for the simulation-ar1n Layer 1b output.

Wraps `output/bundles/cells/_simulation_ar1n_mu1_profile/profile.json` --- the
ELBO + log-likelihood profile along the truth$\leftrightarrow$VI line
(mean-field IVI configuration only) --- in a thin layer that adds an
AI prompt and a parameter schema, plus per-theta derived quantities
($\mu(z), \sigma(z), \sigma_{z_1}, \sigma_\varepsilon$) at the three
reference points (truth / VI-obs / IVI) and at every profile cell.

The wrapping is pure-Python (no Torch, no GPU): every derived field is
a softplus or exp of an existing coordinate. The bundle is designed
to be consumable by downstream tooling that does not have the source
tree available --- e.g. analysis notebooks, AI agents asked to
explain the gap decomposition from the profile alone.

Output: `output/bundles/ar1n_mu1_profile_bundle.json`.
"""

from __future__ import annotations

import json
import math
import os
from collections import OrderedDict
from typing import Any, Dict, List, Optional

# Layer 1b emits the AR(1)-slice subset of the 13-key vector --- the
# four "extra" coordinates (mu0, mu2, sigma1, sigma2) are present in
# the 8-key dicts above but stay near 0 along the line by
# construction; the alpha0 / log_alpha1 / z1_skew / z1_log_tail /
# log_beta coordinates are pinned at 0 in the AR1+Normal DGP and
# omitted from the Layer 1b output entirely.
LAYER_1B_PARAM_NAMES: List[str] = [
    'mu0', 'mu1', 'mu2',
    'sigma0', 'sigma1', 'sigma2',
    'z1_log_std', 'log_sigma_eps',
]

SOURCE_PATH = os.path.join('output', 'bundles', 'cells', '_simulation_ar1n_mu1_profile', 'profile.json')
OUTPUT_PATH = os.path.join('output', 'bundles', 'ar1n_mu1_profile_bundle.json')


AI_PROMPT = """\
This JSON bundles a one-dimensional slice through the 8-D parameter
space of the simulation-ar1n model class (Model A: well-specified
prior + Gaussian decoder; 8 free coordinates). The slice is
parameterised by a scalar t and is defined by

    theta(t) = theta_truth + t * (theta_vi_obs - theta_truth),

so that t=0 corresponds to the DGP truth and t=1 corresponds to the
mean-field-VI MAP on the observed panel (theta_vi_obs). The scalar
t_ivi reported at top level is the projection of the IVI binding-
equation fixed point onto this line; on the well-specified DGP it
typically lies near 0 (truth).

At every grid cell `profile[i]`, three quantities are evaluated at
the *same* theta(t):

  (a) `elbo_mf_amort`     --- the converged amortized mean-field ELBO
                              (a single h=32 encoder NN trained to
                              optimum at theta(t));
  (b) `elbo_mf_nonamort`  --- the converged non-amortized mean-field
                              ELBO (per-individual variational
                              parameters {nu_i, log_tau_i}_i jointly
                              Adam-optimised to optimum at theta(t));
                              this is the best a mean-field family can
                              do at theta(t);
  (c) `logp_exact`        --- the closed-form marginal log-likelihood
                              log p(y | theta(t)) via an AR(1)
                              Cholesky, which is exact on the AR(1)
                              slice and approximately exact along
                              this line (deviation bounded by
                              `extra_coord_max`).

The two-step decomposition of the total ELBO gap at theta(t) is

    total_gap        = logp_exact - elbo_mf_amort
                     = family_gap + amortization_gap,
    family_gap       = logp_exact - elbo_mf_nonamort   (>= 0;
                       mean-field-family vs. true posterior --- the
                       gap NO mean-field encoder can close),
    amortization_gap = elbo_mf_nonamort - elbo_mf_amort (>= 0; the
                       cost of using a NN encoder vs. per-individual
                       variational parameters).

Both summands are non-negative by construction (the non-amortized
fit upper-envelopes the amortized fit; the closed-form is an upper
bound on any ELBO). The closed-form log p peaks at t=0 (truth); the
amortized mean-field ELBO peaks at t=1 (theta_vi_obs, by the VI fixed
point); the horizontal distance between the two argmaxes IS the
mean-field amortization bias on this DGP.

The 8-coordinate theta dict at each cell maps to the generative model
exactly as in the simulation-ar1n parameter summary:

  mu(z)         = mu0 + mu1 * z + mu2 * z^2
  sigma(z)      = softplus(sigma0 + sigma1 * z + sigma2 * z^2)
  sigma_z1      = softplus(z1_log_std)
  sigma_eps     = exp(log_sigma_eps)
  z_1 ~ Normal(0, sigma_z1**2)
  eps_t ~ Normal(0, sigma_eps**2)
  z_t | z_{t-1} ~ Normal(mu(z_{t-1}), sigma(z_{t-1})**2)
  y_t = z_t + eps_t

(z1_skew, z1_log_tail, log_beta, alpha0, log_alpha1 are all pinned at
0 in the Model A AR1+Normal DGP and omitted from the Layer 1b 8-key
vector; see `parameter_schema.pinned_params`.)

Derived quantities at the three reference points (truth / VI-obs /
IVI) and at every profile cell are surfaced under
`*.parameters_derived` for direct inspection. `direction` is the
8-coordinate (theta_vi_obs - theta_truth) vector; the L-inf norm of
its non-AR(1) coordinates along the line is bounded by
`extra_coord_max` (sanity diagnostic --- this scales linearly with
the back-computed |t| range, which itself depends on the chosen mu1
window: a tight mu1 window around truth keeps it small, a broad
window grows it. Cross-check against the O(0.01)-nat per-individual
tolerance for the closed-form approximation; if too large, fall back
to a bootstrap PF reference).
"""


def _softplus(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    if x > 30.0:
        return float(x)
    return math.log1p(math.exp(x))


def derive_quantities(theta: Dict[str, Any]) -> Dict[str, Any]:
    """Map the 8-key theta dict to human-readable generative quantities.

    Mirrors `scripts/_export_ar1n_parameter_summary.py:derive_quantities`
    restricted to the AR(1)-slice subset.
    """
    def _f(k: str, default: float = 0.0) -> Optional[float]:
        v = theta.get(k, default)
        if v is None:
            return None
        return float(v)

    mu = [_f(k) for k in ('mu0', 'mu1', 'mu2')]
    sig = [_f(k) for k in ('sigma0', 'sigma1', 'sigma2')]
    z1ls = _f('z1_log_std')
    lse = _f('log_sigma_eps')

    out: Dict[str, Any] = OrderedDict()
    out['mu_coeffs'] = OrderedDict(
        [('a0', mu[0]), ('a1', mu[1]), ('a2', mu[2])])
    out['sigma_pre_softplus_coeffs'] = OrderedDict(
        [('b0', sig[0]), ('b1', sig[1]), ('b2', sig[2])])
    out['sigma_at_zero'] = _softplus(sig[0])
    out['sigma_z1'] = _softplus(z1ls)
    out['sigma_eps'] = (math.exp(lse) if lse is not None else None)
    return out


def main() -> None:
    if not os.path.exists(SOURCE_PATH):
        raise SystemExit(f'Source profile JSON not found: {SOURCE_PATH}')

    src = json.load(open(SOURCE_PATH))

    # Reference points: derived blocks.
    truth = src['theta_truth']
    vi_obs = src['theta_vi_obs']
    ivi = src['theta_ivi']
    direction = src['direction']

    def _ref(name: str, theta: Dict[str, Any],
              t_value: Optional[float]) -> Dict[str, Any]:
        block: Dict[str, Any] = OrderedDict()
        block['name'] = name
        block['t'] = t_value
        block['parameters'] = OrderedDict(
            (k, float(theta[k])) for k in LAYER_1B_PARAM_NAMES)
        block['parameters_derived'] = derive_quantities(theta)
        return block

    # Per-cell: copy through the source's profile[i] and add derived.
    profile_out: List[Dict[str, Any]] = []
    for cell in src['profile']:
        c: Dict[str, Any] = OrderedDict()
        c['t'] = cell['t']
        c['parameters'] = OrderedDict(
            (k, float(cell['theta'][k])) for k in LAYER_1B_PARAM_NAMES)
        c['parameters_derived'] = derive_quantities(cell['theta'])
        c['elbo_mf_amort'] = cell['elbo_mf_amort']
        c['elbo_mf_nonamort'] = cell['elbo_mf_nonamort']
        c['logp_exact'] = cell['logp_exact']
        c['family_gap'] = cell['family_gap']
        c['amortization_gap'] = cell['amortization_gap']
        c['total_gap'] = cell['total_gap']
        c['wall_s'] = cell.get('wall_s')
        c['wall_s_amort'] = cell.get('wall_s_amort')
        c['wall_s_nonamort'] = cell.get('wall_s_nonamort')
        profile_out.append(c)

    out: Dict[str, Any] = OrderedDict()
    out['metadata'] = OrderedDict([
        ('title', 'simulation-ar1n: Layer 1b ELBO + log-p profile '
                  'along the truth<->VI line (bundle)'),
        ('dgp_spec', 'specs/compute-simulation-ar1n.md'),
        ('source_file', SOURCE_PATH),
        ('upstream_sweep_file', src['config'].get('upstream')),
        ('panel', OrderedDict([
            ('N', src['config'].get('N')),
            ('T', src['config'].get('T')),
            ('obs_seed', src['config'].get('obs_seed'))])),
        ('grid', OrderedDict([
            ('n_grid', src['config'].get('n_grid')),
            ('mu1_min', src['config'].get('mu1_min')),
            ('mu1_max', src['config'].get('mu1_max')),
            ('t_min', src['config'].get('t_min')),
            ('t_max', src['config'].get('t_max'))])),
        ('produced_by',
         'scripts/_export_ar1n_mu1_profile_bundle.py'),
    ])
    out['ai_prompt'] = AI_PROMPT
    out['parameter_schema'] = OrderedDict([
        ('param_names', LAYER_1B_PARAM_NAMES),
        ('softplus_params', ['sigma0', 'sigma1', 'sigma2', 'z1_log_std']),
        ('exp_params', ['log_sigma_eps']),
        ('pinned_params', OrderedDict([
            ('alpha0', 0.0),
            ('log_alpha1', 0.0),
            ('z1_skew', 0.0),
            ('z1_log_tail', 0.0),
            ('log_beta', 0.0),
        ])),
        ('description', (
            'Layer 1b operates on the AR(1) slice of the Model A '
            'parameter vector --- 8 free coords. The five pinned '
            'coords are omitted from each cell dict; see the '
            'simulation-ar1n parameter summary for the full 13-key '
            'vector. sigma uses softplus, sigma_eps uses exp.'
        )),
    ])
    out['reference_points'] = OrderedDict([
        ('truth', _ref('truth', truth, 0.0)),
        ('vi_obs', _ref('vi_obs', vi_obs, 1.0)),
        ('ivi', _ref('ivi', ivi, float(src['t_ivi']))),
    ])
    out['direction'] = OrderedDict(
        (k, float(direction[k])) for k in LAYER_1B_PARAM_NAMES)
    out['t_grid'] = list(src['t_grid'])
    out['extra_coord_max'] = float(src['extra_coord_max'])
    out['config'] = src['config']
    out['profile'] = profile_out

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(out, f, indent=2)

    print(f'Wrote {OUTPUT_PATH}')
    print(f'  source: {SOURCE_PATH}')
    print(f'  n_grid={len(profile_out)}, t_ivi={src["t_ivi"]:+.4f}, '
          f'extra_coord_max={src["extra_coord_max"]:.4f}')
    for c in profile_out:
        print(f'  t={c["t"]:+.2f}  elbo_amort={c["elbo_mf_amort"]:+.4f}  '
              f'elbo_nonamort={c["elbo_mf_nonamort"]:+.4f}  '
              f'logp_exact={c["logp_exact"]:+.4f}  '
              f'total_gap={c["total_gap"]:+.4f}')


if __name__ == '__main__':
    main()
