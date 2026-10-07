#!/usr/bin/env python3
r"""Table B1: PSID Earnings Sample: BPP, Our Replication, and Our Balanced Panel.

Renders tab:bpp-selection from the layer-1 data-summary bundle
bpp_selection_table.json (written by scripts/psid_bpp_selection_table.py):

    column 1  BPP (published)       -- Blundell, Pistaferri and Preston (2008),
                                       Table 1 and Data Appendix A, stored in
                                       the bundle's `bpp_published` block
    column 2  our replication       -- our port of their sample selection,
                                       1978--1992, unbalanced
    column 3  our balanced panel    -- households observed every year
                                       1980--1989: the estimation sample
                                       (N=741, T=10)

Means refer to the 1980 cross-section. Self-contained: Python stdlib only.

Usage:
    python fig_psid_sample_table.py bpp_selection_table.json -o out.tex
"""
import argparse
import json
import math
import sys

DEMO_YEAR = "1980"

# (bundle key, row label, format)
MEANS = [
    ("age",               "Age",                    "mean"),
    ("fsize",             "Family size",            "mean"),
    ("kids",              r"No.\ of children",      "mean"),
    ("white",             "White",                  "share"),
    ("hs_dropout",        "HS dropout",             "share"),
    ("hs_graduate",       "HS graduate",            "share"),
    ("college",           "College dropout",        "share"),
    ("northeast",         "Northeast",              "share"),
    ("midwest",           "Midwest",                "share"),
    ("south",             "South",                  "share"),
    ("west",              "West",                   "share"),
    ("husband_working",   "Husband working",        "share"),
    ("wife_working",      "Wife working",           "share"),
    ("disposable_income", r"Disposable income (\$)", "money"),
]

NOTE = (r"The first column reports the sample statistics published by "
        r"\citet{BlundellEtAl2008} (BPP). The second column applies our "
        r"implementation of their sample selection to the PSID over the same "
        r"period; the third column restricts it to households observed in every "
        r"year from 1980 to 1989, the balanced panel used for estimation "
        r"($N=741$, $T=10$). Means refer to the 1980 cross-section. Residual log "
        r"earnings are log household earnings net of the first-step regression "
        r"on demographic variables. A dash indicates that the statistic is not "
        r"reported.")


def fmt(v, kind):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "--"
    if kind == "money":
        return f"{v:,.0f}"
    if kind == "share":
        return f"{v:.3f}"
    if kind == "sd":
        return f"{v:.4f}"
    if kind == "int":
        return f"{v:,d}"
    return f"{v:.2f}"


def render(doc):
    bpp, sel, bal = doc["bpp_published"], doc["our_selection"], doc["our_balanced"]
    bc, sc, kc = bpp["counts"], sel["counts"], bal["counts"]

    def span(c, lo, hi):
        return f"{c[lo]}--{c[hi]}" if lo in c and hi in c else "--"

    rows = [
        ("Households", fmt(bc["households"], "int"), fmt(sc["households"], "int"),
         fmt(kc["households"], "int")),
        ("Household-year observations", fmt(bc["observations"], "int"),
         fmt(sc["observations"], "int"), fmt(kc["observations"], "int")),
        ("Years per household (mean)", fmt(bc.get("years_per_hh_mean"), "mean"),
         fmt(sc["years_per_hh_mean"], "mean"), fmt(kc["years_per_hh_mean"], "mean")),
        ("Years per household (min--max)", span(bc, "years_per_hh_min", "years_per_hh_max"),
         span(sc, "years_per_hh_min", "years_per_hh_max"),
         span(kc, "years_per_hh_min", "years_per_hh_max")),
        ("Balanced", *("yes" if c["balanced"] else "no" for c in (bc, sc, kc))),
    ]
    means = [(label, fmt(bpp["table1"][k].get(DEMO_YEAR), kind),
              fmt(sel["table1"][k].get(DEMO_YEAR), kind),
              fmt(bal["table1"][k].get(DEMO_YEAR), kind)) for k, label, kind in MEANS]
    sd = ("Std.\\ dev.", "--", fmt(sel["levels"]["resid_sd"], "sd"),
          fmt(bal["levels"]["resid_sd"], "sd"))

    def line(r):
        return f"{r[0]} & {r[1]} & {r[2]} & {r[3]} \\\\"

    out = [
        r"\begin{table}[!h]",
        r"\centering",
        r"\caption{PSID Earnings Sample: BPP, Our Replication, and Our Balanced Panel}",
        r"\label{tab:bpp-selection}",
        r"\setlength{\tabcolsep}{8pt}%",
        r"\resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}{%",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r" & BPP & Our & Our balanced \\",
        r" & (published) & replication & panel \\",
        r"\cmidrule(l{2pt}r{2pt}){2-2}\cmidrule(l{2pt}r{2pt}){3-3}\cmidrule(l{2pt}r{2pt}){4-4}",
        line(("Years covered", span(bc, "first_year", "last_year"),
              span(sc, "first_year", "last_year"), span(kc, "first_year", "last_year"))),
        r"\midrule",
        r"\multicolumn{4}{l}{\textit{Panel structure}} \\",
        *map(line, rows),
        r"\addlinespace",
        r"\multicolumn{4}{l}{\textit{Means in " + DEMO_YEAR + r"}} \\",
        *map(line, means),
        r"\addlinespace",
        r"\multicolumn{4}{l}{\textit{Residual log earnings}} \\",
        line(sd),
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\\ \vspace{0.2cm} \figurenote{" + NOTE + "}",
        r"\end{table}",
    ]
    return "\n".join(out) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("summary", help="path to bpp_selection_table.json")
    ap.add_argument("-o", "--out", "--output", dest="out", default=None,
                    help="output .tex (default: stdout)")
    args = ap.parse_args(argv)
    with open(args.summary) as f:
        tex = render(json.load(f))
    if args.out:
        with open(args.out, "w") as f:
            f.write(tex)
    else:
        sys.stdout.write(tex)


if __name__ == "__main__":
    main()
