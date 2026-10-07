#!/usr/bin/env python3
r"""Table 4: Estimates on the PSID.

Input: output/bundles/bundle_psid_heteroscale.json
IVI standard errors: std of the bootstrap replicates after the column transform.
Usage: python scripts/fig_psid_parameter_table.py output/bundles/bundle_psid_heteroscale.json -o output/figures/tex/psid_parameter_table.tex
"""

import argparse
import json
import math
import statistics
import sys
from decimal import Decimal, ROUND_HALF_UP

CAPTION = "Estimates on the PSID"
LABEL = "psid_parameter_table"

BOOTSTRAP_METHOD = "ivi"   # row that gets the bootstrap SE line

SE_FONT = r"\footnotesize "
SE_ROW_SKIP = r"[-0.6ex]"

FIGURENOTE = (
    r"\\ \vspace{0.2cm} \figurenote{The table reports parameter estimates for the nonlinear earnings model on the PSID, 1980--1989 ($N=741$, $T=10$), under two variational-posterior specifications (unrestricted Gaussian, diagonal) and the IVI correction of the unrestricted-Gaussian posterior. Reported parameters are those of the conditional mean $\mu(z_{t-1})$, conditional volatility $\sigma(z_{t-1})$, the initial state $z_1$, the individual scale heterogeneity $p(a\,|\,z_1)$, and the transitory shock $\varepsilon_t$ with its MA(1) coefficient $\zeta$. Standard errors in parentheses are from a nonparametric bootstrap that resamples households (100 replications).}"
)

# (header symbol, raw param name, transform kind)
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
    (r"$p(a\,|\,z_1)$",     3),
    (r"$\psi_\gamma(\varepsilon)$", 2),
    (r"$\mathrm{MA}(1)$",   1),
]

# Row layout: (label, method-key)
SECTIONS = [
    ("Variational posterior", [
        (r"(1) unrestr.\ Gaussian", "vi_jn"),
        (r"(2) diagonal",           "vi_mf"),
    ]),
    ("Indirect Variational Inference", [
        (r"(3) unrestr.\ Gaussian (IVI)", "ivi"),
    ]),
]


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


def fmt_se(v):
    """Standard error in parentheses at 2 dp, smaller font."""
    d = Decimal(repr(float(v))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    if d == 0:
        d = Decimal("0.00")
    return r"{" + SE_FONT + r"$\phantom{-}(" + f"{d:.2f}" + r")$}"


def die(msg):
    sys.stderr.write("fig_psid_parameter_table.py: ERROR: " + msg + "\n")
    sys.exit(1)


# Bundle parameter names -> flat keys; bundle fit key -> SECTIONS method key.
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
_FIT_ALIAS = {"ivi_jn": "ivi"}


def load_methods(path):
    """method -> {flat param name: estimate} from the bundle's 'fits' block."""
    with open(path) as f:
        doc = json.load(f)
    fits = doc.get("fits")
    if not isinstance(fits, dict) or not fits:
        die("bundle JSON missing non-empty 'fits'")
    by_method = {}
    for fitkey, fit in fits.items():
        raw = fit.get("raw") if isinstance(fit, dict) else None
        if not isinstance(raw, list):
            continue
        mkey = _FIT_ALIAS.get(fitkey, fitkey)
        by_method[mkey] = {_STRUCT_TO_FLAT.get(r["name"], r["name"]): r["estimate"]
                           for r in raw}
    return by_method


def method_cells(params, method):
    """Return the list of formatted estimate cells for one method."""
    cells = []
    for header, key, kind in COLUMNS:
        if key not in params:
            die(f"method {method!r} missing raw parameter '{key}' (column {header})")
        cells.append(fmt_est(transform(kind, params[key])))
    return cells


def load_bootstrap_se(path):
    """{raw key -> SE}: sample std (ddof=1) of the transformed replicates."""
    with open(path) as f:
        doc = json.load(f)
    b = doc["bootstrap"]["unconditional"]
    idx = {n: i for i, n in enumerate(b["param_names"])}
    reps = b["replicates"]
    se = {}
    for header, key, kind in COLUMNS:
        j = idx[key]
        tvals = [transform(kind, r[j]) for r in reps]
        finite = [v for v in tvals if math.isfinite(v)]
        if len(finite) < len(tvals):  # only seen with smoke-test bootstraps
            sys.stderr.write(f"fig_psid_parameter_table.py: WARNING: {len(tvals) - len(finite)} "
                             f"non-finite bootstrap replicate(s) for {key} skipped\n")
        se[key] = statistics.stdev(finite) if len(finite) > 1 else float("nan")
    return se


def build_table(by_method, boot_se):
    colspec = "l" + "c" * N_COL
    header = "Parameter & " + " & ".join(h for h, _, _ in COLUMNS) + r" \\"

    # group-header row and segmented \cmidrule (col 1 = label)
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
        r"\setlength{\tabcolsep}{3pt}%",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{" + colspec + r"}",
        r"\toprule",
        group_header,
        group_cmid,
        header,
        r"\midrule",
    ]

    for s_idx, (section_name, rows) in enumerate(SECTIONS):
        if s_idx > 0:
            lines.append(r"\addlinespace")
        lines.append(
            r"\multicolumn{" + str(N_SPAN) + r"}{l}{\textit{" + section_name + r"}} \\"
        )
        for label, method in rows:
            if method not in by_method:
                die(f"summary JSON has no method '{method}'")
            cells = method_cells(by_method[method], method)
            has_se = method == BOOTSTRAP_METHOD
            term = r" \\" + SE_ROW_SKIP if has_se else r" \\"
            lines.append(label + " & " + " & ".join(cells) + term)
            if has_se:
                se_cells = [fmt_se(boot_se[key]) for _, key, _ in COLUMNS]
                lines.append(" & " + " & ".join(se_cells) + r" \\")

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

    by_method = load_methods(args.summary)
    boot_se = load_bootstrap_se(args.summary)
    table = build_table(by_method, boot_se)
    with open(args.out, "w") as f:
        f.write(table)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()