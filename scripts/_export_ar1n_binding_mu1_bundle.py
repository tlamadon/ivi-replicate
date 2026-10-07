r"""Self-contained bundle for the simulation-ar1n Layer 1d output.

Wraps `output/bundles/cells/_simulation_ar1n_binding_mu1/binding_mu1.json` --- the
1D $\mu_1$ binding-function post-step for the mean-field IVI
configuration --- in a thin layer that adds an AI prompt and a
parameter schema, plus per-reference-point derived quantities
($\mu(z)$ linear coefficient at truth / VI-obs / IVI / $b^\star$) and
a per-cell displacement array
$\Delta(\mu_1^{(0)}) = b_{\mu_1}(\mu_1^{(0)}) - \mu_1^{(0)}$.

The wrapping is pure-Python (no Torch, no GPU): every derived field
is an arithmetic difference of existing grid arrays. The bundle is
designed to be consumable by downstream tooling that does not have
the source tree available --- e.g. analysis notebooks, AI agents
asked to explain the mean-field binding-function geometry on the
$\mu_1$ slice alone.

Output: `output/bundles/ar1n_binding_mu1_bundle.json`.
"""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from typing import Any, Dict, List, Optional


SLICE_PARAM_NAMES: List[str] = ['mu1']

SOURCE_PATH = os.path.join('output', 'bundles', 'cells', '_simulation_ar1n_binding_mu1', 'binding_mu1.json')
OUTPUT_PATH = os.path.join('output', 'bundles', 'ar1n_binding_mu1_bundle.json')


AI_PROMPT = """\
This JSON bundles the 1D $\\mu_1$ binding-function post-step for the
simulation-ar1n model class (Model A: well-specified AR(1)+Normal
prior + Gaussian decoder; 8 free coordinates, 5 pinned). The slice is
1D in $\\mu_1$ --- the AR(1) coefficient --- with the other seven
free coordinates $(\\mu_0, \\mu_2, \\sigma_0, \\sigma_1, \\sigma_2,
\\texttt{z1\\_log\\_std}, \\log\\sigma_\\varepsilon)$ initialised at
the DGP truth (which lies on the AR(1) slice $\\mu_0 = \\mu_2 =
\\sigma_1 = \\sigma_2 = 0$ and at specific values for $\\sigma_0$ and
$\\texttt{z1\\_log\\_std}$ --- see the sibling Layer 1a-summary bundle
at `output/bundles/cells/_simulation_ar1n_sweep/ar1n_parameter_summary.json` for
the numerical values), and the four shape coordinates $(\\log\\beta,
\\theta_{\\rm MA}, z_1\\,\\text{skew}, z_1\\,\\log\\,\\text{tail})$
pinned at 0 (Model A defaults; $\\alpha_0$ and $\\log\\alpha_1$ are
not registered under the poly law model).

Grid orientation. `mu1_grid` (length `config.n_grid`) is a 1D array
covering $[\\texttt{config.grid\\_min}, \\texttt{config.grid\\_max}]$.
At each grid cell $i$ the source records one binding-function value
evaluated at $\\theta_0$ (the 8-coord vector with the slice coord set
to $\\mu_1^{(0)} = \\texttt{mu1\\_grid}[i]$ and the other seven
initialised at truth):

  `b_mu1[i]` --- the mean-field $h=32$ binding map with ALL EIGHT free
                 parameters trainable jointly (the same parameter set
                 the upstream Layer 1a mean-field VI fit trains; the
                 four shape coordinates above remain pinned at zero).
                 $\\theta_0$ is used to simulate
                 $y_{\\rm sim}(\\theta_0)$ via strong CRN (z_noise,
                 y_noise drawn once at `obs_seed` and reused at every
                 cell), then the mean-field VI is fit on $y_{\\rm sim}$
                 for `n_epochs_vi` epochs at `lr` with AdamW. The
                 converged $\\hat\\mu_1$ is read off as the binding-map
                 output; the other seven converged components are NOT
                 recorded here.

Fixed-point reference. `b_mu1_star` is the same all-eight-free-params
mean-field VI fit applied to $y_{\\rm obs}$ directly (rather than to a
simulated $y_{\\rm sim}$). Under strong CRN on a well-specified DGP
it equals $b_{\\mu_1}(\\mu_1^{(0)})$ at $\\mu_1^{(0)} = \\mu_1^{\\rm
truth}$, AND it equals $\\mu_1^{\\rm VI}$ (the upstream Layer 1a VI
estimate on $y_{\\rm obs}$) by construction --- the binding function
passes through the upstream VI estimate when evaluated at the truth.

Picard fixed-point picture. The displacement array
$\\Delta(\\mu_1^{(0)}) = b_{\\mu_1}(\\mu_1^{(0)}) - \\mu_1^{(0)}$ is
pre-computed in this bundle as `b_mu1_minus_identity` --- handy for
one-shot plotting (zero crossings = Picard fixed points) without
re-broadcasting `mu1_grid` against the source array. Two distinct
iterations consume this field:

  - Standard Picard $\\mu_1^{(k+1)} = b(\\mu_1^{(k)})$: fixed point at
    $\\Delta(\\mu_1^{(0)}) = 0$, i.e. wherever $b(\\mu_1^{(0)}) =
    \\mu_1^{(0)}$. Generally NOT at $\\mu_1^{\\rm truth}$ on this DGP
    --- mean-field VI is biased downward (`vi_mu1` $\\approx 0.755 <
    \\mu_1^{\\rm truth} = 0.9$), so the standard-Picard attractor
    sits below truth at the binding's identity crossing.
  - Binding-equation Picard $\\mu_1^{(k+1)} = \\mu_1^{(k)} +
    (\\theta_{\\rm VI}^{\\rm obs} - b(\\mu_1^{(k)}))$: fixed point at
    $b(\\mu_1^{(0)}) = b^\\star = \\theta_{\\rm VI}^{\\rm obs}$. Under
    strong CRN on a well-specified DGP this lands at $\\mu_1^{(0)} =
    \\mu_1^{\\rm truth}$. This is the IVI iteration whose converged
    value is exposed as the `ivi` reference point.

Reference points (`reference_points`):

  - `truth` --- the DGP truth $\\mu_1^{\\rm truth} = 0.9$. Under
    strong CRN at $\\mu_1^{(0)} = \\mu_1^{\\rm truth}$ the binding
    map satisfies $b_{\\mu_1}(\\mu_1^{\\rm truth}) = b^\\star =
    \\mu_1^{\\rm VI}$, so truth is the binding-equation Picard fixed
    point on this DGP (the AR(1)+Normal model class is
    well-specified).
  - `vi_obs` --- the mean-field VI fit on $y_{\\rm obs}$ (the upstream
    Layer 1a `picard_meanfield` configuration's starting point,
    equivalently the IVI iteration's $\\mu_1$ at outer step $k=0$).
  - `ivi` --- the mean-field damped-Picard IVI's converged $\\hat
    \\mu_1$ (the upstream Layer 1a `picard_meanfield` configuration's
    final $\\theta_K$). Should be close to `truth` under strong CRN.
  - `bstar` --- the all-eight-free-params mean-field VI fit applied to
    $y_{\\rm obs}$ directly. Equals $b_{\\mu_1}(\\mu_1^{\\rm truth})$
    on the binding curve and equals $\\mu_1^{\\rm VI}$ by construction
    under strong CRN.

Each reference point carries the 1-key slice coord dict `parameters`
(`mu1`) and a `parameters_derived` block. On the AR(1) slice the
generative drift is exactly $\\mu(z) = \\mu_1 \\cdot z$ (since $\\mu_0
= \\mu_2 = 0$ at truth), so the derived block reports
`mu_z_linear_coef = mu1` directly --- no softplus/exp needed for the
1D slice.

Parameter mapping. The full 13-coordinate Model A parameter dict
maps to the generative model as

  mu(z)         = mu0 + mu1 * z + mu2 * z^2
  sigma(z)      = softplus(sigma0 + sigma1 * z + sigma2 * z^2)
  sigma_z1      = softplus(z1_log_std)
  sigma_eps     = exp(log_sigma_eps)
  z_1 ~ Normal(0, sigma_z1**2)
  eps_t ~ Normal(0, sigma_eps**2)
  z_t | z_{t-1} ~ Normal(mu(z_{t-1}), sigma(z_{t-1})**2)
  y_t = z_t + eps_t

The 1D slice plotted here exposes only `mu1`; see
`parameter_schema.initialised_at_truth` /
`parameter_schema.pinned_at_zero` for the other 12 coordinates' values
along this slice. The full 13-key vector and the relationships
between coordinates are documented in the sibling Layer 1a-summary
bundle at
`output/bundles/cells/_simulation_ar1n_sweep/ar1n_parameter_summary.json`.
"""


def derive_slice_quantities(slice_pt: Dict[str, Any]) -> Dict[str, Any]:
    """Map the 1-key slice point to human-readable generative quantities.

    On the AR(1) slice (mu0 = mu2 = 0 at truth) the drift simplifies to
    mu(z) = mu1 * z, so the linear coefficient is just mu1. No
    softplus or exp transforms apply to the 1D slice coord.
    """
    out: Dict[str, Any] = OrderedDict()
    m1 = slice_pt.get('mu1')
    out['mu_z_linear_coef'] = (float(m1) if m1 is not None else None)
    return out


def _array_subtract(
        values: List[Optional[float]],
        grid: List[float],
) -> List[Optional[float]]:
    """Compute b[i] - mu1_grid[i] elementwise; preserves None entries."""
    if len(values) != len(grid):
        raise SystemExit(
            f'Length mismatch: len(values)={len(values)} '
            f'vs. len(grid)={len(grid)}')
    out: List[Optional[float]] = []
    for v, g in zip(values, grid):
        if v is None:
            out.append(None)
        else:
            out.append(float(v) - float(g))
    return out


def main() -> None:
    if not os.path.exists(SOURCE_PATH):
        raise SystemExit(f'Source binding-mu1 JSON not found: {SOURCE_PATH}')

    src = json.load(open(SOURCE_PATH))

    mu1_grid = list(src['mu1_grid'])
    n_grid = len(mu1_grid)

    b_mu1 = list(src['b_mu1'])
    b_star = src.get('b_mu1_star')

    delta = _array_subtract(b_mu1, mu1_grid)

    def _ref(name: str, mu1_value: Optional[float]) -> Dict[str, Any]:
        block: Dict[str, Any] = OrderedDict()
        block['name'] = name
        block['parameters'] = OrderedDict([
            ('mu1', (float(mu1_value) if mu1_value is not None else None)),
        ])
        block['parameters_derived'] = derive_slice_quantities(
            {'mu1': mu1_value})
        return block

    out: Dict[str, Any] = OrderedDict()
    out['metadata'] = OrderedDict([
        ('title', 'simulation-ar1n: Layer 1d mu_1 binding-function '
                  'post-step (bundle)'),
        ('dgp_spec', 'specs/compute-simulation-ar1n.md'),
        ('source_file', SOURCE_PATH),
        ('upstream_sweep_file', src['config'].get('upstream')),
        ('panel', OrderedDict([
            ('N', src['config'].get('N')),
            ('T', src['config'].get('T')),
            ('obs_seed', src['config'].get('obs_seed'))])),
        ('grid', OrderedDict([
            ('n_grid', src['config'].get('n_grid')),
            ('grid_min', src['config'].get('grid_min')),
            ('grid_max', src['config'].get('grid_max'))])),
        ('vi_fit', OrderedDict([
            ('encoder', 'mean-field (normal_diagonal)'),
            ('hidden_dim', src['config'].get('hidden_dim')),
            ('n_epochs_vi', src['config'].get('n_epochs_vi')),
            ('lr', src['config'].get('lr')),
            ('fix_noise', src['config'].get('fix_noise')),
            ('noise_seed', src['config'].get('noise_seed'))])),
        ('produced_by',
         'scripts/_export_ar1n_binding_mu1_bundle.py'),
    ])
    out['ai_prompt'] = AI_PROMPT
    out['parameter_schema'] = OrderedDict([
        ('slice_param_names', SLICE_PARAM_NAMES),
        ('initialised_at_truth', [
            'mu0', 'mu2', 'sigma0', 'sigma1', 'sigma2', 'z1_log_std',
            'log_sigma_eps',
        ]),
        ('pinned_at_zero', OrderedDict([
            ('log_beta', 0.0),
            ('theta_MA', 0.0),
            ('z1_skew', 0.0),
            ('z1_log_tail', 0.0),
        ])),
        ('truth_values_reference', (
            'output/bundles/ar1n_parameter_summary.json '
            '(reference_points.truth.parameters)'
        )),
        ('description', (
            'Layer 1d plots the 1D mu1 slice of the Model A parameter '
            'vector. At each grid cell, a mean-field h=32 VI is fit on '
            'y_sim(theta_0) with ALL EIGHT free parameters trainable '
            '(mu0, mu1, mu2, sigma0, sigma1, sigma2, z1_log_std, '
            'log_sigma_eps) --- the same parameter set as the upstream '
            'Layer 1a VI fit. The four shape parameters (log_beta, '
            'theta_MA, z1_skew, z1_log_tail) remain pinned at zero. '
            'The non-slice free parameters are initialised at truth at '
            'every cell; mu1 is initialised at mu1^(0). The fixed-point '
            'reference `b_mu1_star` is the same all-eight-free-params '
            'fit applied directly to y_obs --- under strong CRN it '
            'equals b_mu1 evaluated at truth, AND it equals mu1^VI by '
            'construction (the binding function passes through the '
            'upstream VI estimate when evaluated at the truth).'
        )),
    ])
    out['reference_points'] = OrderedDict([
        ('truth', _ref('truth', src.get('truth_mu1'))),
        ('vi_obs', _ref('vi_obs', src.get('vi_mu1'))),
        ('ivi', _ref('ivi', src.get('ivi_mu1'))),
        ('bstar', _ref('bstar', b_star)),
    ])
    out['mu1_grid'] = mu1_grid
    out['b_mu1'] = b_mu1
    out['b_mu1_minus_identity'] = delta
    out['config'] = src['config']

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(out, f, indent=2)

    print(f'Wrote {OUTPUT_PATH}')
    print(f'  source: {SOURCE_PATH}')
    print(f'  n_grid={n_grid}  '
          f'mu1 in [{mu1_grid[0]:+.4f}, {mu1_grid[-1]:+.4f}]')
    tm = src.get('truth_mu1')
    vm = src.get('vi_mu1')
    im = src.get('ivi_mu1')
    bs = b_star
    fmt = lambda x: ('MISSING' if x is None else f'{float(x):+.4f}')
    print(f'  truth={fmt(tm)}  vi_obs={fmt(vm)}  '
          f'ivi={fmt(im)}  bstar={fmt(bs)}')
    finite_deltas = [d for d in delta if d is not None]
    if finite_deltas:
        print(f'  delta[0]={delta[0]}  delta[-1]={delta[-1]}')
    print('OK')


if __name__ == '__main__':
    main()
