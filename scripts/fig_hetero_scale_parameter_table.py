#!/usr/bin/env python3
r"""Table 3: Parameter Estimates in the Heterogeneity Model.

Input: output/bundles/hetero_scale_parameter_summary.json
Usage: python scripts/fig_hetero_scale_parameter_table.py --summary output/bundles/hetero_scale_parameter_summary.json --output output/figures/tex/hetero_scale_parameter_table.tex
"""

import argparse
import json
import math
import sys
from decimal import Decimal, ROUND_HALF_UP


def identity(x):
    return x


def softplus(x):
    """log(1 + e^x), numerically stable."""
    if x > 30.0:
        return x
    return math.log1p(math.exp(max(-50.0, min(30.0, x))))


def exp_(x):
    return math.exp(x)


def exp_neg(x):
    return math.exp(-x)


# (header LaTeX, JSON raw key, transform), in printed order
COLUMNS = [
    (r"$\mu_0$",      "mu0",              identity),
    (r"$\mu_1$",      "mu1",              identity),
    (r"$\mu_2$",      "mu2",              identity),
    (r"$\sigma_0$",   "sigma0",           identity),
    (r"$\sigma_1$",   "sigma1",           identity),
    (r"$\sigma_2$",   "sigma2",           identity),
    (r"$\alpha_1$",   "z1_log_std",       softplus),
    (r"$\alpha_2$",   "z1_log_tail",      exp_neg),
    (r"$\lambda_0$",  "beta_a0",          identity),
    (r"$\lambda_1$",  "beta_a1",          identity),
    (r"$\sigma_a$",   "log_sigma_a_cond", exp_),
    (r"$\gamma_2$",   "log_beta",         exp_neg),
]
N_COL = len(COLUMNS)
N_SPAN = N_COL + 1

# (group header LaTeX, span in parameter columns)
COLUMN_GROUPS = [
    (r"$\mu(z_{t-1})$",     3),
    (r"$\sigma(z_{t-1})$",  3),
    (r"$f_\alpha(z_1)$",    2),
    (r"$p(a\,|\,z_1)$",     3),
    (r"$\psi_\gamma(e_t)$", 1),
]

# ("rule", None) | ("group", title) | ("data", (label, fit selector, theta_role prefix))
ROWS = [
    ("data",  ("DGP",                           {"_dgp": True},                                                    "truth")),
    ("rule",  None),
    ("group", "Variational posterior"),
    ("data",  (r"(1) unrestr.\ Gaussian",       {"family": "vi",  "method": "vi_only", "encoder": "jn"},           "theta_VI_obs")),
    ("data",  (r"(2) diagonal",                 {"family": "vi",  "method": "vi_only", "encoder": "mean_field"},   "theta_VI_obs")),
    ("rule",  None),
    ("group", "Indirect Variational Inference"),
    ("data",  (r"(3) unrestr.\ Gaussian (IVI)", {"family": "ivi", "method": "picard_cold"},                        "final_theta")),
]

_COLSPEC = ("l" + "c" * N_COL)


def _group_header_lines():
    """The group-header row and its segmented \\cmidrule."""
    cells, rules, col = [], [], 2
    for label, span in COLUMN_GROUPS:
        cells.append(r"\multicolumn{" + str(span) + r"}{c}{" + label + r"}")
        rules.append(r"\cmidrule(l{2pt}r{2pt}){" + f"{col}-{col + span - 1}" + r"}")
        col += span
    return [" & " + " & ".join(cells) + r" \\", "".join(rules)]


_PREAMBLE_LINES = [
    r"\begin{table}[!t]",
    r"\centering",
    r"\caption{Parameter Estimates in the Heterogeneity Model}",
    r"\label{hetero_scale_parameter_table}",
    r"\setlength{\tabcolsep}{4pt}%",
    r"\resizebox{\textwidth}{!}{%",
    r"\begin{tabular}{" + _COLSPEC + r"}",
    r"\toprule",
    *_group_header_lines(),
    "Parameter & " + " & ".join(h for h, _, _ in COLUMNS) + r" \\",
    r"\midrule",
]
FIGURENOTE = (
    r"\\ \vspace{0.2cm} \figurenote{True parameter values of the simulated heterogeneity-model DGP and the estimates under two variational posteriors and the IVI correction of the unrestricted Gaussian posterior. The transitory shock has a fixed baseline dispersion ($\gamma_1=1$) that each individual rescales by $e^{a}$, with $a|z_1\sim\mathcal N(\lambda_0+\lambda_1 z_1,\sigma_a^2)$. Nonlinear DGP with heterogeneity, $N=30{,}000$, $T=6$.}"
)

_POSTAMBLE_LINES = [
    r"\bottomrule",
    r"\end{tabular}%",
    r"}",
    FIGURENOTE,
    r"\end{table}",
]


def fmt_cell(value):
    """2 dp round-half-up; non-negatives get a leading \\phantom{-}."""
    d = Decimal(repr(float(value))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    if d == 0:
        d = Decimal("0.00")
    if d < 0:
        body = "-" + f"{-d:.2f}"
    else:
        body = r"\phantom{-}" + f"{d:.2f}"
    return "$" + body + "$"


def die(msg):
    sys.stderr.write("fig_hetero_scale_parameter_table.py: ERROR: " + msg + "\n")
    sys.exit(1)


def select_params(summary, selector, role):
    """Return the `parameters` dict of the truth or of the unique matching fit."""
    if selector.get("_dgp"):
        return summary["truth"]["parameters"]

    desc = ", ".join("{}={}".format(k, v) for k, v in sorted(selector.items()))
    matches = [
        f for f in summary["fits"]
        if isinstance(f, dict) and all(f.get(k) == v for k, v in selector.items())
    ]
    if len(matches) != 1:
        die("row selector [{}] matched {} fits (expected exactly 1)".format(desc, len(matches)))
    fit = matches[0]
    theta_role = fit.get("theta_role", "")
    if not isinstance(theta_role, str) or not theta_role.startswith(role):
        die("fit [{}] has theta_role {!r} but this row expects a '{}' vector"
            .format(desc, theta_role, role))
    return fit["parameters"]


def data_row_cells(params):
    return [fmt_cell(transform(float(params[key]))) for _, key, transform in COLUMNS]


def build_table(summary):
    lines = list(_PREAMBLE_LINES)
    for kind, payload in ROWS:
        if kind == "rule":
            lines.append(r"\addlinespace")
        elif kind == "group":
            lines.append(
                r"\multicolumn{" + str(N_SPAN) + r"}{l}{\textit{" + payload + r"}} \\"
            )
        else:
            label, selector, role = payload
            params = select_params(summary, selector, role)
            cells = data_row_cells(params)
            lines.append(label + " & " + " & ".join(cells) + r" \\")
    lines.extend(_POSTAMBLE_LINES)
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    with open(args.summary, "r", encoding="utf-8") as fh:
        summary = json.load(fh)
    with open(args.output, "w", encoding="utf-8") as fh:
        fh.write(build_table(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
