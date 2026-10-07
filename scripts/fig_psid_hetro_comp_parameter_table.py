#!/usr/bin/env python3
r"""Table H1 (web appendix): Estimates on the PSID: Heterogeneous Mean vs. Heterogeneous Variance.

Input: output/bundles/2026-07-13-bpp-hetero-mean-indep-bundle.json
Also reads bundle_psid_heteroscale.json from the same directory.
Usage: python scripts/fig_psid_hetro_comp_parameter_table.py output/bundles/2026-07-13-bpp-hetero-mean-indep-bundle.json -o output/figures/tex/psid_hetro_comp_parameter_table.tex
"""

import argparse
import json
import math
import os
import sys
from decimal import Decimal, ROUND_HALF_UP

CAPTION = r"Estimates on the PSID: Heterogeneous Mean vs.\ Heterogeneous Variance"
LABEL = "psid_hetro_comp_parameter_table"

# scale-heterogeneity bundle, read from the same directory as the positional input
HS_FILENAME = "bundle_psid_heteroscale.json"

FIGURENOTE = (
    r"\\ \vspace{0.2cm} \figurenote{The table compares two specifications of individual heterogeneity in the nonlinear earnings model on the PSID, 1980--1989 ($N=741$, $T=10$), each under the IVI-corrected unrestricted-Gaussian variational posterior. In the heterogeneous mean model the individual effect enters additively, $y_t = z_t + a + \varepsilon_t$, with $a \sim \mathcal{N}(\lambda_0, \sigma_a^2)$ independent of $z_1$; in the heterogeneous variance model it enters as a log-scale of the transitory component, $e^{a}\varepsilon_t$, with $a\,|\,z_1 \sim \mathcal{N}(\lambda_0+\lambda_1 z_1, \sigma_a^2)$. Here $\mu(z_{t-1})$ and $\sigma(z_{t-1})$ are the conditional mean and volatility of the persistent state, $f_\alpha(z_1)$ the initial-state law, $\psi_\gamma(\varepsilon)$ the transitory law (scale $\gamma_1$, tail $\gamma_2$, skewness $\gamma_3$), and $\zeta$ the MA(1) coefficient. The transitory scale $\gamma_1$ equals one by construction in the heterogeneous variance model.}"
)

# Union of both models' columns: (header, raw key, transform kind)
COLUMNS = [
    (r"$\mu_0$",    "mu0",              "id"),
    (r"$\mu_1$",    "mu1",              "id"),
    (r"$\mu_2$",    "mu2",              "id"),
    (r"$\sigma_0$", "sigma0",           "id"),
    (r"$\sigma_1$", "sigma1",           "id"),
    (r"$\sigma_2$", "sigma2",           "id"),
    (r"$\alpha_1$", "z1_log_std",       "softplus"),
    (r"$\alpha_2$", "z1_log_tail",      "expneg"),
    (r"$\alpha_3$", "z1_skew",          "id"),
    (r"$\lambda_0$","beta_a0",          "id"),
    (r"$\lambda_1$","beta_a1",          "id"),
    (r"$\sigma_a$", "log_sigma_a_cond", "exp"),
    (r"$\gamma_1$", "log_sigma",        "exp"),
    (r"$\gamma_2$", "log_beta",         "expneg"),
    (r"$\gamma_3$", "alpha_eps",        "id"),
    (r"$\zeta$",    "theta",            "id"),
]
N_COL = len(COLUMNS)
N_SPAN = N_COL + 1

# (group header LaTeX, span in parameter columns)
COLUMN_GROUPS = [
    (r"$\mu(z_{t-1})$",     3),
    (r"$\sigma(z_{t-1})$",  3),
    (r"$f_\alpha(z_1)$",    3),
    (r"$p(a)$",             3),
    (r"$\psi_\gamma(\varepsilon)$", 3),
    (r"$\mathrm{MA}(1)$",   1),
]

# (section label, source, [(row label, selector)]); "mi" = summary_table row, "hs" = fit key
SECTIONS = [
    ("Heterogeneity in the mean", "mi", [
        (r"(1) unrestr.\ Gaussian (IVI)", {"model_class": "hetero-mean-indep", "estimator": "ivi"}),
    ]),
    ("Heterogeneity in the variance", "hs", [
        (r"(2) unrestr.\ Gaussian (IVI)", "ivi_jn"),
    ]),
]

# Map the scale bundle's structured parameter names to the flat keys used here.
_STRUCT_TO_FLAT = {
    "prior.net_mu.coeffs[0]": "mu0", "prior.net_mu.coeffs[1]": "mu1",
    "prior.net_mu.coeffs[2]": "mu2", "prior.net_sigma.coeffs[0]": "sigma0",
    "prior.net_sigma.coeffs[1]": "sigma1", "prior.net_sigma.coeffs[2]": "sigma2",
    "prior.z1_log_std": "z1_log_std", "prior.z1_skew": "z1_skew",
    "prior.z1_log_tail": "z1_log_tail", "prior.net_extra_mu.coeffs[0]": "beta_a0",
    "prior.net_extra_mu.coeffs[1]": "beta_a1",
    "prior.net_extra_logsigma.coeffs": "log_sigma_a_cond",
    "decoder.log_beta": "log_beta", "decoder.theta": "theta",
    "decoder.alpha_eps": "alpha_eps",
}


def _softplus(x):
    if x > 30.0:
        return float(x)
    return math.log1p(math.exp(x))


def transform(kind, est):
    """Displayed value for a raw estimate under the column's transform."""
    x = float(est)
    if kind == "id":
        return x
    if kind == "softplus":
        return _softplus(x)
    if kind == "expneg":
        return math.exp(-x)
    if kind == "exp":
        return math.exp(x)
    raise ValueError(f"unknown transform kind {kind!r}")


# 2 dp, round-half-up; sign-aligned estimates
def _round2(v):
    d = Decimal(repr(float(v))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return Decimal("0.00") if d == 0 else d


def fmt_est(v):
    d = _round2(v)
    if d < 0:
        return "$-" + f"{-d:.2f}" + "$"
    return r"$\phantom{-}" + f"{d:.2f}" + "$"


def die(msg):
    sys.stderr.write("fig_psid_hetro_comp_parameter_table.py: ERROR: " + msg + "\n")
    sys.exit(1)


def load_meanindep(path):
    """Return the mean-independent bundle's summary_table (list of flat dicts)."""
    with open(path) as f:
        doc = json.load(f)
    table = doc.get("summary_table")
    if not isinstance(table, list) or not table:
        die("mean-independent bundle missing non-empty 'summary_table'")
    return table


def load_heteroscale(path):
    """Return {fit key: flat param dict} from the scale bundle's 'fits' block."""
    with open(path) as f:
        doc = json.load(f)
    fits = doc.get("fits")
    if not isinstance(fits, dict) or not fits:
        die(f"scale bundle {path!r} missing non-empty 'fits'")
    by = {}
    for fitkey, fit in fits.items():
        raw = fit.get("raw") if isinstance(fit, dict) else None
        if not isinstance(raw, list):
            continue
        by[fitkey] = {_STRUCT_TO_FLAT.get(r["name"], r["name"]): r["estimate"]
                      for r in raw}
        # scale model fixes gamma_1 = 1 (log_sigma = 0)
        by[fitkey]["log_sigma"] = 0.0
    return by


def select_mi_row(table, sel):
    matches = [r for r in table if all(r.get(k) == v for k, v in sel.items())]
    if len(matches) != 1:
        desc = ", ".join(f"{k}={v}" for k, v in sel.items())
        die(f"selector [{desc}] matched {len(matches)} rows (expected 1)")
    return matches[0]


def method_cells(params):
    """Formatted estimate cells for one row; '--' where the parameter is absent."""
    cells = []
    for header, key, kind in COLUMNS:
        if key in params and params[key] is not None:
            cells.append(fmt_est(transform(kind, params[key])))
        else:
            cells.append("--")
    return cells


def build_table(mi_table, hs_by_method):
    colspec = "l" + "c" * N_COL
    header = "Parameter & " + " & ".join(h for h, _, _ in COLUMNS) + r" \\"

    g_cells, g_rules, col = [], [], 2
    for label, span in COLUMN_GROUPS:
        g_cells.append(r"\multicolumn{" + str(span) + r"}{c}{" + label + r"}")
        g_rules.append(r"\cmidrule(l{2pt}r{2pt}){" + f"{col}-{col + span - 1}" + r"}")
        col += span
    group_header = " & " + " & ".join(g_cells) + r" \\"
    group_cmid = "".join(g_rules)

    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        rf"\caption{{{CAPTION}}}",
        rf"\label{{{LABEL}}}",
        r"\setlength{\tabcolsep}{4pt}%",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{" + colspec + r"}",
        r"\toprule",
        group_header,
        group_cmid,
        header,
        r"\midrule",
    ]

    for s_idx, (section_name, src, rows) in enumerate(SECTIONS):
        if s_idx > 0:
            lines.append(r"\addlinespace")
        lines.append(
            r"\multicolumn{" + str(N_SPAN) + r"}{l}{\textit{" + section_name + r"}} \\"
        )
        for label, sel in rows:
            if src == "mi":
                params = select_mi_row(mi_table, sel)
            else:
                params = hs_by_method.get(sel)
                if params is None:
                    die(f"scale bundle has no fit {sel!r}")
            lines.append(label + " & " + " & ".join(method_cells(params)) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        FIGURENOTE,
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("summary")
    ap.add_argument("-o", dest="out", required=True)
    args = ap.parse_args(argv)

    hs_path = os.path.join(os.path.dirname(args.summary), HS_FILENAME)
    mi_table = load_meanindep(args.summary)
    hs_by_method = load_heteroscale(hs_path)
    tex = build_table(mi_table, hs_by_method)
    with open(args.out, "w") as f:
        f.write(tex)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
