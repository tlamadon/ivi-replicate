r"""Self-contained bundle for the simulation-ar1n Layer 1c output.

Wraps `output/bundles/cells/_simulation_ar1n_quiver/quiver.json` --- the
$(\mu_1, \log\sigma_\varepsilon)$ Picard-quiver post-step for the
mean-field IVI configuration --- in a thin layer that adds an AI
prompt and a parameter schema, plus derived quantities
($\sigma_\varepsilon = \exp(\log\sigma_\varepsilon)$) at the
reference points (truth / VI-obs / IVI / b*) and a per-cell
displacement field $\Delta = b(\theta_0) - \theta_0$ projected onto
the 2D slice.

The wrapping is pure-Python (no Torch, no GPU): every derived field
is an `exp` of an existing coordinate or an arithmetic difference of
existing grid arrays. The bundle is designed to be consumable by
downstream tooling that does not have the source tree available
--- e.g. analysis notebooks, AI agents asked to explain the
binding-map geometry or the Picard fixed-point picture from the
quiver JSON alone.

Output: `output/bundles/ar1n_quiver_bundle.json`.
"""

from __future__ import annotations

import json
import math
import os
from collections import OrderedDict
from typing import Any, Dict, List, Optional


SLICE_PARAM_NAMES: List[str] = ['mu1', 'log_sigma_eps']

SOURCE_PATH = os.path.join('output', 'bundles', 'cells', '_simulation_ar1n_quiver', 'quiver.json')
OUTPUT_PATH = os.path.join('output', 'bundles', 'ar1n_quiver_bundle.json')


AI_PROMPT = """\
This JSON bundles the 2D Picard-quiver post-step for the
simulation-ar1n model class (Model A: well-specified AR(1)+Normal
prior + Gaussian decoder; 8 free coordinates, 5 pinned). The slice is
$(\\mu_1, \\log\\sigma_\\varepsilon)$ --- the AR(1) coefficient and the
emission log-std --- with the other six free coordinates
$(\\mu_0, \\mu_2, \\sigma_0, \\sigma_1, \\sigma_2,
\\texttt{z1\\_log\\_std})$ pinned at the DGP truth (which lies on the
AR(1) slice $\\mu_0 = \\mu_2 = \\sigma_1 = \\sigma_2 = 0$), and the
five shape coordinates $(\\alpha_0, \\log\\alpha_1, z_1\\,\\text{skew},
z_1\\,\\log\\,\\text{tail}, \\log\\beta)$ pinned at 0 (Model A defaults).

Grid orientation. Both `m1_grid` (length `config.n_grid`) and
`lse_grid` (length `config.n_grid`) are 1D arrays. The 2D arrays
`logp_grid`, `b_m1`, `b_lse` are indexed as

    array[i][j]   with   i = lse row,   j = m1 col,

so the grid cell at row `i` column `j` corresponds to
$(\\mu_1, \\log\\sigma_\\varepsilon) = (m1\\_grid[j], lse\\_grid[i])$.

At every grid cell (i, j) two quantities are evaluated at the *same*
$\\theta_0$ (the 8-coord vector with the two slice coords set to
$(m1\\_grid[j], lse\\_grid[i])$ and the other six pinned at truth):

  (a) `logp_grid[i][j]`   --- the closed-form per-individual marginal
                              log-likelihood $\\log p(y_{\\rm obs} \\mid
                              \\theta_0)$ via an AR(1) Cholesky
                              (`M.ar1n_marginal_logp`). Exact on this
                              slice because the non-target coords are
                              0 at truth, so $\\theta_0$ is on the
                              AR(1) slice exactly. Cross-individual
                              mean is what is stored.
  (b) `b_m1[i][j]`,       --- the projection onto the
      `b_lse[i][j]`           $(\\mu_1, \\log\\sigma_\\varepsilon)$
                              slice of the full mean-field VI
                              binding map: $\\theta_0$ is used to
                              simulate $y_{\\rm sim}(\\theta_0)$ via
                              strong CRN (z_noise, y_noise drawn
                              once at obs_seed and reused at every
                              cell), then a full mean-field $h=32$
                              encoder + all 8 free parameters are
                              jointly fit on $y_{\\rm sim}$ for
                              `n_epochs_vi` epochs at `lr` with AdamW.
                              The converged $\\hat\\mu_1$ and
                              $\\widehat{\\log\\sigma_\\varepsilon}$
                              are read off as the binding-map
                              output; the other six converged
                              components are NOT recorded here (the
                              raw Layer 1c JSON drops them; this
                              bundle inherits the projection).

Picard fixed-point picture. The displacement field
$\\Delta(\\theta_0) = b(\\theta_0) - \\theta_0$ is pre-computed in
this bundle as `delta_m1[i][j]` and `delta_lse[i][j]` --- handy for
quiver / streamline visualisation without having to broadcast
`m1_grid`/`lse_grid` against the 2D binding output. Two distinct
iterations consume this field:

  - Standard Picard $\\theta_{k+1} = b(\\theta_k)$: fixed point at
    $\\Delta(\\theta_0) = 0$, i.e. wherever $b(\\theta_0) = \\theta_0$.
    Generally NOT at truth on this DGP (mean-field VI is biased).
  - Binding-equation Picard
    $\\theta_{k+1} = \\theta_k + (\\theta_{\\rm VI}^{\\rm obs} - b(\\theta_k))$:
    fixed point at $b(\\theta_0) = b^\\star = \\theta_{\\rm VI}^{\\rm obs}$
    (under strong CRN this lands at $\\theta_0 = \\theta_{\\rm truth}$
    on a well-specified DGP). This is the IVI iteration that
    produced the `trajectory_*` curve and the `ivi` reference point.

Reference points (`reference_points`):

  - `truth` --- the DGP truth $(\\mu_1^{\\rm truth},
    \\log\\sigma_\\varepsilon^{\\rm truth})$. Under strong CRN at
    $\\theta_0 = \\theta_{\\rm truth}$ the binding map satisfies
    $b(\\theta_{\\rm truth}) = b^\\star$, so the truth IS the Picard
    fixed point on this DGP (the AR(1)+Normal model class is
    well-specified).
  - `vi_obs` --- the mean-field VI fit on $y_{\\rm obs}$
    (the upstream Layer 1a `picard_meanfield` configuration's
    starting point, equivalently $\\theta_K$ at $k=0$ of the IVI
    iteration).
  - `ivi` --- the mean-field damped-Picard IVI's converged estimate
    (the upstream Layer 1a `picard_meanfield` configuration's final
    $\\theta_K$). Should be close to `truth` under strong CRN.
  - `bstar` --- the binding map applied to $y_{\\rm obs}$ directly
    (rather than to a simulated $y_{\\rm sim}$). Equals $b(\\theta_0)$
    at $\\theta_0 = \\theta_{\\rm truth}$ under strong CRN.

Each reference point carries the 2-key slice coord dict
`parameters` (mu1 + log_sigma_eps) and a `parameters_derived` block
(sigma_eps = exp(log_sigma_eps)).

Trajectory. `trajectory_m1` and `trajectory_lse` are the
damped-Picard / Anderson Picard IVI trajectory in
$(\\mu_1, \\log\\sigma_\\varepsilon)$ space, one entry per outer
iteration $k$. Starts at $\\theta_K^{(0)} = \\theta_{\\rm VI}^{\\rm
obs}$ (or `truth`, depending on the IVI configuration's warm start
--- see the upstream JSON) and converges toward the fixed point.

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

The 2D slice plotted here exposes only `mu1` and `log_sigma_eps`;
see `parameter_schema.pinned_at_truth` / `parameter_schema.pinned_at_zero`
for the other 11 coordinates' values along this slice. The full
13-key vector and the relationships between coordinates are
documented in the sibling Layer 1a-summary bundle
(`output/bundles/cells/_simulation_ar1n_sweep/ar1n_parameter_summary.json`).
"""


def _exp_of(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    return math.exp(float(x))


def derive_slice_quantities(slice_pt: Dict[str, Any]) -> Dict[str, Any]:
    """Map the 2-key slice point to human-readable generative quantities."""
    out: Dict[str, Any] = OrderedDict()
    lse = slice_pt.get('log_sigma_eps')
    out['sigma_eps'] = _exp_of(lse)
    return out


def _matrix_subtract_axis(
        grid_2d: List[List[Optional[float]]],
        axis_values: List[float],
        axis: str,
) -> List[List[Optional[float]]]:
    """Compute b - theta_0 on the 2D grid, broadcasting the axis vector.

    axis = 'm1':  subtract m1_grid[j]  along axis=1.
    axis = 'lse': subtract lse_grid[i] along axis=0.
    """
    out: List[List[Optional[float]]] = []
    for i, row in enumerate(grid_2d):
        new_row: List[Optional[float]] = []
        for j, v in enumerate(row):
            if v is None:
                new_row.append(None)
                continue
            ref = axis_values[j] if axis == 'm1' else axis_values[i]
            new_row.append(float(v) - float(ref))
        out.append(new_row)
    return out


def main() -> None:
    if not os.path.exists(SOURCE_PATH):
        raise SystemExit(f'Source quiver JSON not found: {SOURCE_PATH}')

    src = json.load(open(SOURCE_PATH))

    m1_grid = list(src['m1_grid'])
    lse_grid = list(src['lse_grid'])
    n_grid = len(m1_grid)
    if len(lse_grid) != n_grid:
        raise SystemExit(
            f'Inconsistent grid: m1_grid len={n_grid} but '
            f'lse_grid len={len(lse_grid)}')

    delta_m1 = _matrix_subtract_axis(src['b_m1'], m1_grid, 'm1')
    delta_lse = _matrix_subtract_axis(src['b_lse'], lse_grid, 'lse')

    def _ref(name: str, slice_pt: Dict[str, Any]) -> Dict[str, Any]:
        block: Dict[str, Any] = OrderedDict()
        block['name'] = name
        block['parameters'] = OrderedDict(
            (k, (float(slice_pt[k]) if slice_pt.get(k) is not None else None))
            for k in SLICE_PARAM_NAMES)
        block['parameters_derived'] = derive_slice_quantities(slice_pt)
        return block

    bstar_pt = {
        'mu1': src.get('b_star_m1'),
        'log_sigma_eps': src.get('b_star_lse'),
    }

    out: Dict[str, Any] = OrderedDict()
    out['metadata'] = OrderedDict([
        ('title', 'simulation-ar1n: Layer 1c (mu_1, log_sigma_eps) Picard '
                  'quiver post-step for the mean-field IVI (bundle)'),
        ('dgp_spec', 'specs/compute-simulation-ar1n.md'),
        ('source_file', SOURCE_PATH),
        ('upstream_sweep_file', src['config'].get('upstream')),
        ('panel', OrderedDict([
            ('N', src['config'].get('N')),
            ('T', src['config'].get('T')),
            ('obs_seed', src['config'].get('obs_seed'))])),
        ('grid', OrderedDict([
            ('n_grid', src['config'].get('n_grid')),
            ('grid_pad', src['config'].get('grid_pad')),
            ('m1_min', float(m1_grid[0])),
            ('m1_max', float(m1_grid[-1])),
            ('lse_min', float(lse_grid[0])),
            ('lse_max', float(lse_grid[-1]))])),
        ('vi_fit', OrderedDict([
            ('encoder', 'mean-field (normal_diagonal)'),
            ('hidden_dim', src['config'].get('hidden_dim')),
            ('n_epochs_vi', src['config'].get('n_epochs_vi')),
            ('lr', src['config'].get('lr')),
            ('fix_noise', src['config'].get('fix_noise')),
            ('noise_seed', src['config'].get('noise_seed'))])),
        ('produced_by',
         'scripts/_export_ar1n_quiver_bundle.py'),
    ])
    out['ai_prompt'] = AI_PROMPT
    out['parameter_schema'] = OrderedDict([
        ('slice_param_names', SLICE_PARAM_NAMES),
        ('grid_indexing', 'array[i][j] with i = lse row, j = m1 col'),
        ('exp_params', ['log_sigma_eps']),
        ('pinned_at_truth', [
            'mu0', 'mu2', 'sigma0', 'sigma1', 'sigma2', 'z1_log_std',
        ]),
        ('pinned_at_zero', OrderedDict([
            ('alpha0', 0.0),
            ('log_alpha1', 0.0),
            ('z1_skew', 0.0),
            ('z1_log_tail', 0.0),
            ('log_beta', 0.0),
        ])),
        ('truth_values_reference', (
            'output/bundles/ar1n_parameter_summary.json '
            '(reference_points.truth.parameters)'
        )),
        ('description', (
            'Layer 1c plots the 2D (mu1, log_sigma_eps) slice of the '
            'Model A parameter vector. The other 6 free coords are '
            'pinned at the DGP truth (which lies on the AR(1) slice: '
            'mu0 = mu2 = sigma1 = sigma2 = 0; sigma0 and z1_log_std at '
            'their truth values --- see the sibling Layer 1a-summary '
            'bundle for the numerical values); the 5 shape coords are '
            'pinned at 0 (Model A defaults). The closed-form log p is '
            'exact on the AR(1) slice; the binding map fits all 8 free '
            'coords on the simulated y, then projects to the 2D slice '
            'for the figure.'
        )),
    ])
    out['reference_points'] = OrderedDict([
        ('truth', _ref('truth', src['truth'])),
        ('vi_obs', _ref('vi_obs', src['vi'])),
        ('ivi', _ref('ivi', src['ivi'])),
        ('bstar', _ref('bstar', bstar_pt)),
    ])
    out['m1_grid'] = m1_grid
    out['lse_grid'] = lse_grid
    out['logp_grid'] = src['logp_grid']
    out['b_m1'] = src['b_m1']
    out['b_lse'] = src['b_lse']
    out['delta_m1'] = delta_m1
    out['delta_lse'] = delta_lse
    out['trajectory_m1'] = list(src['trajectory_m1'])
    out['trajectory_lse'] = list(src['trajectory_lse'])
    out['config'] = src['config']

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(out, f, indent=2)

    print(f'Wrote {OUTPUT_PATH}')
    print(f'  source: {SOURCE_PATH}')
    print(f'  n_grid={n_grid}  '
          f'm1 in [{m1_grid[0]:+.4f}, {m1_grid[-1]:+.4f}]  '
          f'lse in [{lse_grid[0]:+.4f}, {lse_grid[-1]:+.4f}]')
    print(f'  truth=({src["truth"]["mu1"]:+.4f}, '
          f'{src["truth"]["log_sigma_eps"]:+.4f})  '
          f'vi_obs=({src["vi"]["mu1"]:+.4f}, '
          f'{src["vi"]["log_sigma_eps"]:+.4f})  '
          f'ivi=({src["ivi"]["mu1"]:+.4f}, '
          f'{src["ivi"]["log_sigma_eps"]:+.4f})')
    bsm = src.get('b_star_m1'); bsl = src.get('b_star_lse')
    if bsm is not None and bsl is not None:
        print(f'  bstar=({bsm:+.4f}, {bsl:+.4f})  '
              f'trajectory length={len(src["trajectory_m1"])}')
    else:
        print('  bstar=MISSING  '
              f'trajectory length={len(src["trajectory_m1"])}')


if __name__ == '__main__':
    main()
