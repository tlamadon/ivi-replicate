#!/usr/bin/env python3
r"""Table 1: Parameter Estimates in the Linear Gaussian Model.

Input: output/bundles/ar1n_parameter_summary.json
Usage: python scripts/fig_ar1n_parameter_table.py output/bundles/ar1n_parameter_summary.json -o output/figures/tex/ar1n_parameter_table.tex
"""
import argparse
import json
import sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

CAPTION = "Parameter Estimates in the Linear Gaussian Model"
LABEL = "ar1n_parameter_table"

FIGURENOTE = (
    r"\\ \vspace{0.2cm} \figurenote{The table reports the true parameter values of the simulated DGP alongside the estimates obtained under four variational-posterior specifications and under Indirect Variational Inference (IVI) applied to the diagonal posterior. The estimation model allows for quadratic terms in both the conditional mean $\mu(z_{t-1})$ and the conditional volatility $\sigma(z_{t-1})$. The final VI specification abstracts from transitory shocks. Linear Gaussian DGP, $N = 30{,}000$, $T = 6$.}"
)

# Column order. Each entry: (LaTeX header, source, key).
#   source "raw"     -> fit["parameters"][key]
#   source "derived" -> fit["parameters_derived"][key]
COLUMNS = [
    (r"$\mu_0$",       "raw",     "mu0"),
    (r"$\mu_1$",       "raw",     "mu1"),
    (r"$\mu_2$",       "raw",     "mu2"),
    (r"$\sigma_0$",    "raw",     "sigma0"),
    (r"$\sigma_1$",    "raw",     "sigma1"),
    (r"$\sigma_2$",    "raw",     "sigma2"),
    (r"$\sigma_{z_1}$", "derived", "sigma_z1"),
    (r"$\sigma_e$",    "derived", "sigma_eps"),
]
DECIMALS = 2

# Group headers are italic rows; each data row's selector must match exactly one fit.
ROW_PLAN = [
    {"kind": "data",   "label": "DGP", "selector": "truth"},
    {"kind": "header", "label": "Variational posterior"},
    {"kind": "data",   "label": r"(1) unrestr.\ Gaussian",
     "selector": {"family": "vi",  "encoder": "jn",            "lr": 0.01}},
    {"kind": "data",   "label": "(2) tridiagonal",
     "selector": {"family": "vi",  "encoder": "tridiag",       "lr": 0.01}},
    {"kind": "data",   "label": "(3) hidden Markov",
     "selector": {"family": "vi",  "encoder": "struct_markov", "lr": 0.01}},
    {"kind": "data",   "label": "(4) diagonal",
     "selector": {"family": "vi",  "encoder": "mean_field",    "lr": 0.01}},
    {"kind": "header", "label": "Indirect Variational Inference"},
    {"kind": "data",   "label": "(5) diagonal (IVI)",
     "selector": {"family": "ivi", "method": "picard", "encoder": "mean_field",
                  "alpha": 0.6, "lr": 0.01}},
    {"kind": "header", "label": "Ignoring transitory shocks"},
    {"kind": "data",   "label": "(6)",
     "selector": {"method": "mle_noerror"}},
]


def fmt_cell(x):
    """2dp, round-half-up, no negative zero, sign-aligned with \\phantom; None -> "--"."""
    if x is None:
        return "--"
    q = Decimal(10) ** -DECIMALS
    r = Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP)
    if r == 0:
        r = abs(r)  # kill negative zero
    s = f"{abs(r):.{DECIMALS}f}"
    return rf"$-{s}$" if r < 0 else rf"$\phantom{{-}}{s}$"


def select_fit(data, selector):
    if selector == "truth":
        t = data["truth"]
        return t["parameters"], t["parameters_derived"]
    matches = [
        f for f in data["fits"]
        if all(f.get(k) == v for k, v in selector.items())
    ]
    if len(matches) != 1:
        raise SystemExit(
            f"selector {selector} matched {len(matches)} fits (expected 1). "
            f"Matched labels: {[m.get('label') for m in matches]}"
        )
    f = matches[0]
    return f["parameters"], f["parameters_derived"]


def data_row(label, params, derived):
    cells = []
    for _, source, key in COLUMNS:
        src = params if source == "raw" else derived
        cells.append(fmt_cell(src.get(key)))
    return label + " & " + " & ".join(cells) + r" \\"


def build_table(data):
    ncol = len(COLUMNS) + 1
    colspec = "l" + "c" * len(COLUMNS)
    header = "Parameter & " + " & ".join(h for h, _, _ in COLUMNS) + r" \\"

    group_header = (
        r" & \multicolumn{3}{c}{$\mu(z_{t-1})$}"
        r" & \multicolumn{3}{c}{$\sigma(z_{t-1})$}"
        r" & $f_\alpha(z_1)$ & $\psi_\gamma(e_t)$ \\"
    )
    group_cmid = (r"\cmidrule(l{2pt}r{2pt}){2-4}\cmidrule(l{2pt}r{2pt}){5-7}"
                  r"\cmidrule(l{2pt}r{2pt}){8-8}\cmidrule(l{2pt}r{2pt}){9-9}")

    body = []
    for row in ROW_PLAN:
        if row["kind"] == "header":
            body.append(r"\addlinespace")
            body.append(rf"\multicolumn{{{ncol}}}{{l}}{{\textit{{{row['label']}}}}} \\")
        else:
            params, derived = select_fit(data, row["selector"])
            body.append(data_row(row["label"], params, derived))

    return "\n".join([
        r"\begin{table}[!t]",
        r"\centering",
        rf"\caption{{{CAPTION}}}",
        rf"\label{{{LABEL}}}",
        r"\setlength{\tabcolsep}{5pt}%",
        r"\resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}{%",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
        group_header,
        group_cmid,
        header,
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        FIGURENOTE,
        r"\end{table}",
    ]) + "\n"


def main():
    ap = argparse.ArgumentParser(
        description="Generate the linear Gaussian parameter-estimates table.")
    ap.add_argument("json", help="path to ar1n_parameter_summary.json")
    ap.add_argument("--output", "-o", required=True, help="output .tex path")
    args = ap.parse_args()

    with open(args.json) as fh:
        data = json.load(fh)
    body = build_table(data)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body)
    print(f"Wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
