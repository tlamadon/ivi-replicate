r"""Table 2: Parameter Estimates in the Nonlinear Model.

Input: output/bundles/hockeystick_parameter_summary.json
The T=40 panel is read from hockeystick_parameter_summary_longt.json
in the same directory.
Usage: python scripts/fig_hockeystick_parameter_table.py output/bundles/hockeystick_parameter_summary.json -o output/figures/tex/hockeystick_parameter_table.tex
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

CAPTION = r"Parameter Estimates in the Nonlinear Model"
LABEL = "hockeystick_parameter_table"

# T=40 summary, loaded from the same directory as the T=6 summary.
LONGT_FILENAME = "hockeystick_parameter_summary_longt.json"

# Width of the bold panel label box; also the indent of rows under it.
PANEL_INDENT = "4.0em"

FIGURENOTE = (
    r"\\ \vspace{0.2cm} \figurenote{The table reports the true parameter values of the simulated nonlinear DGP alongside estimates at two panel lengths, $T=6$ (top panel) and $T=40$ (bottom panel), for different variational-posterior specifications, the IVI correction of the unrestricted-Gaussian posterior, and a VI specification that ignores transitory shocks (only for $T=6$). The estimation model follows equations (\ref{eq:mod_quad1})--(\ref{eq:mod_quad2}), applied to data generated with sinh-arcsinh innovations for $z_1$ and $e_t$. Nonlinear DGP, $N=30{,}000$.}"
)

COLUMN_SYMBOLS: List[str] = [
    r'$\eta_0$', r'$\eta_1$',
    r'$\mu_0$', r'$\mu_1$', r'$\mu_2$',
    r'$\sigma_0$', r'$\sigma_1$', r'$\sigma_2$',
    r'$\alpha_1$', r'$\alpha_2$',
    r'$\gamma_1$', r'$\gamma_2$',
]
N_COLS = len(COLUMN_SYMBOLS)

# Group header above the parameter symbols: (label, column span).
COLUMN_GROUPS: List[Tuple[str, int]] = [
    (r'$\mu(z_{t-1})$', 5),
    (r'$\sigma(z_{t-1})$', 3),
    (r'$f_\alpha(z_1)$', 2),
    (r'$\psi_\gamma(e_t)$', 2),
]

# Rows are (label, fit selector); "src" disambiguates duplicate encoders.
T6_SECTIONS: List[Tuple[str, List[Tuple[str, dict]]]] = [
    ("Variational posterior", [
        ("unrestr.\\ Gaussian",
         {"sel": {"family": "vi", "method": "vi_only", "encoder": "jn"}}),
        ("tridiagonal",
         {"sel": {"family": "vi", "method": "vi_only", "encoder": "tridiag_jn"}}),
        ("diagonal",
         {"sel": {"family": "vi", "method": "vi_only", "encoder": "mean_field"},
          "src": "vi_only_meanfield_h64"}),
        ("hidden Markov",
         {"sel": {"family": "vi", "method": "vi_only", "encoder": "struct_markov"},
          "src": "vi_only_struct_markov_h64"}),
    ]),
    ("Indirect Variational Inference", [
        ("unrestr.\\ Gaussian (IVI)",
         {"sel": {"family": "ivi", "method": "picard_cold", "encoder": "jn"}}),
    ]),
    ("Ignoring transitory shocks", [
        ("",
         {"sel": {"family": "mle_no_measurement_error"}}),
    ]),
]

T40_SECTIONS: List[Tuple[str, List[Tuple[str, dict]]]] = [
    ("Variational posterior", [
        ("unrestr.\\ Gaussian",
         {"sel": {"family": "vi", "method": "vi_only", "encoder": "jn"}}),
    ]),
    ("Indirect Variational Inference", [
        ("unrestr.\\ Gaussian (IVI)",
         {"sel": {"family": "ivi", "method": "picard_cold", "encoder": "jn"}}),
    ]),
]

# (panel label, summary tag, sections)
PANELS: List[Tuple[str, str, list]] = [
    (r"$T = 6$",  "t6",  T6_SECTIONS),
    (r"$T = 40$", "t40", T40_SECTIONS),
]


def _softplus(x: float) -> float:
    """Numerically stable softplus."""
    if x > 30.0:
        return float(x)
    return math.log1p(math.exp(float(x)))


# None -> NaN, rendered as "--".
def _f(v):           return float('nan') if v is None else float(v)
def _exp(v):         return float('nan') if v is None else math.exp(float(v))
def _softplus_n(v):  return float('nan') if v is None else _softplus(float(v))
def _inv_exp(v):     return float('nan') if v is None else 1.0 / math.exp(float(v))


def transform_theta(theta: Dict[str, float]) -> List[float]:
    """Raw-key -> displayed-value mapping; 12 floats in column order."""
    return [
        _f(theta.get('alpha0')),              # eta_0
        _exp(theta.get('log_alpha1')),        # eta_1
        _f(theta.get('mu0')),                 # mu_0
        _f(theta.get('mu1')),                 # mu_1
        _f(theta.get('mu2')),                 # mu_2
        _f(theta.get('sigma0')),              # sigma_0
        _f(theta.get('sigma1')),              # sigma_1
        _f(theta.get('sigma2')),              # sigma_2
        _softplus_n(theta.get('z1_log_std')), # alpha_1
        _inv_exp(theta.get('z1_log_tail')),   # alpha_2
        _exp(theta.get('log_sigma_eps')),     # gamma_1
        _inv_exp(theta.get('log_beta')),      # gamma_2
    ]


def fmt_cell(x: Optional[float]) -> str:
    """2 decimals, sign-aligned with \\phantom{-}; None/NaN -> ``--``."""
    if x is None:
        return '--'
    fx = float(x)
    if fx != fx:  # NaN
        return '--'
    rounded = round(fx, 2)
    if rounded == 0.0:
        rounded = 0.0  # canonicalise away negative zero
    if rounded < 0.0:
        return f'$-{abs(rounded):.2f}$'
    return rf'$\phantom{{-}}{rounded:.2f}$'


def fmt_row_cells(vals: Optional[List[float]]) -> str:
    if vals is None:
        return ' & '.join([''] * N_COLS)
    assert len(vals) == N_COLS, f'expected {N_COLS} values, got {len(vals)}'
    return ' & '.join(fmt_cell(v) for v in vals)


def die(msg: str) -> None:
    sys.stderr.write("fig_hockeystick_parameter_table.py: ERROR: " + msg + "\n")
    sys.exit(1)


def load_summary(path: str) -> dict:
    try:
        with open(path) as f:
            doc = json.load(f)
    except OSError as e:
        die(f"could not read summary {path!r}: {e}")
    if not isinstance(doc.get("truth"), dict) or "parameters" not in doc["truth"]:
        die(f"summary {path!r} missing truth.parameters")
    if not isinstance(doc.get("fits"), list) or not doc["fits"]:
        die(f"summary {path!r} missing non-empty 'fits' list")
    return doc


def select_theta(summary: dict, cell: dict) -> Dict[str, float]:
    sel = cell["sel"]
    desc = ", ".join(f"{k}={v}" for k, v in sorted(sel.items()))
    matches = [f for f in summary["fits"]
               if isinstance(f, dict) and all(f.get(k) == v for k, v in sel.items())]
    if "src" in cell:
        matches = [f for f in matches if cell["src"] in (f.get("source_file") or "")]
        desc += f", src~{cell['src']}"
    if len(matches) != 1:
        die(f"selector [{desc}] matched {len(matches)} fits (expected 1): "
            f"{[m.get('label') for m in matches]}")
    params = matches[0].get("parameters")
    if not isinstance(params, dict):
        die(f"selector [{desc}] fit has no 'parameters' object")
    return params


def build_rows(t6: dict, t40: dict):
    """DGP row (shared) + a list of (panel label, sections-with-values)."""
    src = {"t6": t6, "t40": t40}
    dgp_vals = transform_theta(t6["truth"]["parameters"])
    panels = []
    for panel_label, tag, sections in PANELS:
        summary = src[tag]
        built = [(name, [(label, transform_theta(select_theta(summary, cell)))
                         for label, cell in rows])
                 for name, rows in sections]
        panels.append((panel_label, built))
    return dgp_vals, panels


def render_table(dgp_vals, panels) -> str:
    colspec = 'l' + 'c' * N_COLS
    header = 'Parameter & ' + ' & '.join(COLUMN_SYMBOLS) + r' \\'

    group_cells, group_rules, col = [], [], 2
    for label, span in COLUMN_GROUPS:
        group_cells.append(r'\multicolumn{' + str(span) + r'}{c}{' + label + r'}')
        group_rules.append(r'\cmidrule(l{2pt}r{2pt}){' + f'{col}-{col + span - 1}' + r'}')
        col += span
    group_header = ' & ' + ' & '.join(group_cells) + r' \\'
    group_cmid = ''.join(group_rules)

    span_all = str(1 + N_COLS)

    def mc(body: str) -> str:                 # full-width left-aligned header row
        return r'\multicolumn{' + span_all + r'}{l}{' + body + r'} \\'

    lines: List[str] = [
        r'\begin{table}[!t]',
        r'\centering',
        rf'\caption{{{CAPTION}}}',
        rf'\label{{{LABEL}}}',
        r'\setlength{\tabcolsep}{4pt}%',
        r'\resizebox{\textwidth}{!}{%',
        r'\begin{tabular}{' + colspec + '}',
        r'\toprule',
        group_header,
        group_cmid,
        header,
        r'\midrule',
        f'DGP & {fmt_row_cells(dgp_vals)} ' + r'\\',
    ]

    indent = r'\hspace*{' + PANEL_INDENT + r'}'

    row_n = 0
    for p_idx, (panel_label, sections) in enumerate(panels):
        lines.append(r'\addlinespace')        # gap between DGP/panels (no rule)
        for s_idx, (section_name, rows) in enumerate(sections):
            if s_idx == 0:
                lines.append(mc(r'\makebox[' + PANEL_INDENT + r'][l]{\textbf{'
                                + panel_label + r'}}\textit{' + section_name + r'}'))
            else:
                lines.append(r'\addlinespace')
                lines.append(mc(indent + r'\textit{' + section_name + r'}'))
            for label, vals in rows:
                row_n += 1
                tag = f'({row_n})'
                full_label = tag if label == '' else f'{tag} {label}'
                lines.append(indent + full_label + f' & {fmt_row_cells(vals)} ' + r'\\')

    lines.extend([
        r'\bottomrule',
        r'\end{tabular}%',
        r'}',
        FIGURENOTE,
        r'\end{table}',
    ])
    return '\n'.join(lines) + '\n'


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("summary", help="path to the T=6 hockeystick_parameter_summary.json")
    ap.add_argument("-o", dest="out", required=True, help="output .tex path")
    args = ap.parse_args(argv)

    t6 = load_summary(args.summary)
    t40 = load_summary(os.path.join(os.path.dirname(args.summary), LONGT_FILENAME))
    dgp_vals, panels = build_rows(t6, t40)
    body = render_table(dgp_vals, panels)

    with open(args.out, 'w') as f:
        f.write(body)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
