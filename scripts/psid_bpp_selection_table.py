"""Layer 1: sample statistics for the PSID data table (BPP vs ours vs balanced).

Column 1 -- **BPP published**. Hard-coded from Blundell, Pistaferri and
Preston, "Consumption Inequality and Partial Insurance", AER 98(5),
2008: Table 1 (PSID column, p. 1893), Table 3 (p. 1902), and the
sample counts in Data Appendix A (p. 1915).

Column 2 -- **our reconstruction**, i.e. the output of
``psid_bpp_build.select_sample`` + ``merge_taxes`` on the replication
package's ``data.dta``. This is the sample the port is *supposed* to
match: unbalanced, 1978--1992, one row per household-year.

Column 3 -- **our balanced panel**, the subset that survives
``psid_bpp_build.to_wide_matrix`` at the requested window (default
1980--1989) and is what ``output/data/bpp_y_matrix.npy`` actually contains.
BPP never balance their panel -- Appendix A is explicit that heads
appear "from a minimum of one year to a maximum of fifteen years" --
so the gap between columns 2 and 3 is the cost of the rectangular
(N, T) array our estimator needs, not a discrepancy in the port.

Writes output/data/bpp_selection_table.json (the data-summary bundle;
the published copy is tracked at that path), which
scripts/fig_psid_sample_table.py renders as the paper's sample table. Also
prints a longer comparison (every published year, growth moments, why
households drop at the balancing step) to the console.

Run:
    python replicate.py run --only bpp-sample-stats
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from psid_bpp_build import (  # noqa: E402
    EXT,
    build_log_residual_earnings,
    merge_taxes,
    select_sample,
    to_wide_matrix,
)

OUT_DIR = Path("output") / "data"


# ---------------------------------------------------------------------------
# BPP as published
# ---------------------------------------------------------------------------
# Table 1, PSID column (AER 2008 p. 1893).
BPP_TABLE1 = {
    "age":               {1980: 42.94, 1983: 43.43, 1986: 43.86, 1989: 44.03, 1992: 45.95},
    "fsize":             {1980: 3.61,  1983: 3.52,  1986: 3.48,  1989: 3.44,  1992: 3.42},
    "kids":              {1980: 1.32,  1983: 1.25,  1986: 1.21,  1989: 1.18,  1992: 1.14},
    "white":             {1980: 0.91,  1983: 0.92,  1986: 0.93,  1989: 0.94,  1992: 0.94},
    "hs_dropout":        {1980: 0.21,  1983: 0.18,  1986: 0.16,  1989: 0.14,  1992: 0.13},
    "hs_graduate":       {1980: 0.30,  1983: 0.31,  1986: 0.32,  1989: 0.32,  1992: 0.32},
    "college":           {1980: 0.49,  1983: 0.51,  1986: 0.53,  1989: 0.54,  1992: 0.55},
    "northeast":         {1980: 0.21,  1983: 0.21,  1986: 0.22,  1989: 0.22,  1992: 0.22},
    "midwest":           {1980: 0.33,  1983: 0.31,  1986: 0.30,  1989: 0.30,  1992: 0.31},
    "south":             {1980: 0.31,  1983: 0.31,  1986: 0.30,  1989: 0.30,  1992: 0.30},
    "west":              {1980: 0.15,  1983: 0.17,  1986: 0.18,  1989: 0.18,  1992: 0.18},
    "husband_working":   {1980: 0.96,  1983: 0.94,  1986: 0.93,  1989: 0.94,  1992: 0.93},
    "wife_working":      {1980: 0.69,  1983: 0.71,  1986: 0.74,  1989: 0.78,  1992: 0.77},
    "disposable_income": {1980: 29333, 1983: 35427, 1986: 42374, 1989: 50684, 1992: 58841},
    "food_total":        {1980: 4447,  1983: 4868,  1986: 5294,  1989: 5872,  1992: 6604},
}

# Table 3, "The Autocovariance Matrix of Income Growth" (AER 2008 p. 1902).
# var(dy_t) and cov(dy_{t+1}, dy_t) of *residual* log income growth.
BPP_TABLE3_VAR = {
    1980: 0.0832, 1981: 0.0717, 1982: 0.0718, 1983: 0.0783, 1984: 0.0805,
    1985: 0.1090, 1986: 0.1023, 1987: 0.1116, 1988: 0.0925, 1989: 0.0883,
    1990: 0.0924, 1991: 0.0818, 1992: 0.1177,
}
BPP_TABLE3_COV1 = {
    1980: -0.0196, 1981: -0.0220, 1982: -0.0226, 1983: -0.0209, 1984: -0.0288,
    1985: -0.0379, 1986: -0.0354, 1987: -0.0375, 1988: -0.0313, 1989: -0.0280,
    1990: -0.0296, 1991: -0.0299,
}

# Data Appendix A (p. 1915).
BPP_COUNTS = {
    "households": 1765,
    "observations": 17604,
    "first_year": 1978,
    "last_year": 1992,
    "balanced": False,
    "years_per_hh_min": 1,
    "years_per_hh_max": 15,
    "note": "Heads appear from a minimum of one to a maximum of fifteen years.",
}

LABELS = [
    ("age",               "Age",                       "mean"),
    ("fsize",             "Family size",               "mean"),
    ("kids",              "No. of children",           "mean"),
    ("white",             "White",                     "share"),
    ("hs_dropout",        "HS dropout",                "share"),
    ("hs_graduate",       "HS graduate",               "share"),
    ("college",           "College dropout",           "share"),
    ("northeast",         "Northeast",                 "share"),
    ("midwest",           "Midwest",                   "share"),
    ("south",             "South",                     "share"),
    ("west",              "West",                      "share"),
    ("husband_working",   "Husband working",           "share"),
    ("wife_working",      "Wife working",              "share"),
    ("disposable_income", "Disposable income (\\$)",   "money"),
    ("food_total",        "Food expenditure (\\$)",    "money"),
]


def add_table1_vars(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the Table 1 indicator columns to a selected sample."""
    df = df.copy()
    df["white"] = (df["race"] == 1).astype(float)
    df["hs_dropout"] = (df["educ"] == 1).astype(float)
    df["hs_graduate"] = (df["educ"] == 2).astype(float)
    df["college"] = (df["educ"] == 3).astype(float)
    df["northeast"] = (df["region"] == 1).astype(float)
    df["midwest"] = (df["region"] == 2).astype(float)
    df["south"] = (df["region"] == 3).astype(float)
    df["west"] = (df["region"] == 4).astype(float)
    df["husband_working"] = (df["hours"].fillna(0) > 0).astype(float)
    df["wife_working"] = (df["hourw"].fillna(0) > 0).astype(float)
    df["disposable_income"] = df["y"] - df["ftax"]
    df["food_total"] = df["food"].fillna(0) + df["fout"].fillna(0)
    return df


def panel_counts(df: pd.DataFrame) -> dict:
    per_hh = df.groupby("person")["year"].nunique()
    return {
        "households": int(df["person"].nunique()),
        "observations": int(len(df)),
        "first_year": int(df["year"].min()),
        "last_year": int(df["year"].max()),
        "years_per_hh_mean": float(per_hh.mean()),
        "years_per_hh_min": int(per_hh.min()),
        "years_per_hh_max": int(per_hh.max()),
        "balanced": bool(per_hh.nunique() == 1),
    }


def year_means(df: pd.DataFrame, years: list[int]) -> dict:
    out = {}
    for key, _, _ in LABELS:
        out[key] = {}
        for y in years:
            sub = df[df["year"] == y]
            out[key][y] = float(sub[key].mean()) if len(sub) else float("nan")
    return out


def growth_moments(df: pd.DataFrame, years: list[int]) -> dict:
    """var(dy_t) and cov(dy_{t+1}, dy_t) of residual log earnings growth.

    Computed year by year on whoever is present in both adjacent years,
    which is how BPP's unrestricted minimum-distance estimates in
    Table 3 are formed (their diagonal-by-diagonal moments).
    """
    wide = df.pivot_table(index="person", columns="year", values="resid_log_y")
    dy = {}
    for t in sorted(wide.columns):
        if (t - 1) in wide.columns:
            dy[t] = wide[t] - wide[t - 1]
    var = {}
    cov1 = {}
    for t in years:
        if t in dy:
            v = dy[t].dropna()
            var[t] = float(v.var(ddof=1)) if len(v) > 1 else float("nan")
        if t in dy and (t + 1) in dy:
            pair = pd.concat([dy[t + 1], dy[t]], axis=1).dropna()
            cov1[t] = float(pair.cov().iloc[0, 1]) if len(pair) > 1 else float("nan")
    return {"var_dy": var, "cov1_dy": cov1}


def coverage_breakdown(df: pd.DataFrame, start: int, end: int) -> dict:
    """Why households fail the balanced-window cut.

    Separates "the spell is too short" from "the spell is long enough
    but sits in the wrong decade" -- the second group is the one that
    makes the balancing a selection on *timing* rather than on panel
    length, and it is invisible in the household count alone.
    """
    T = end - start + 1
    per_hh = df.groupby("person")["year"]
    span = per_hh.nunique()
    in_win = df[df["year"].between(start, end)].groupby("person")["year"].nunique()
    in_win = in_win.reindex(span.index, fill_value=0)
    long_enough = span >= T
    covers = in_win == T
    return {
        "households_total": int(len(span)),
        "covers_window": int(covers.sum()),
        "spell_shorter_than_T": int((~long_enough).sum()),
        "spell_ge_T_but_misses_window": int((long_enough & ~covers).sum()),
        "mean_spell_all": float(span.mean()),
        "mean_spell_dropped": float(span[~covers].mean()) if (~covers).any() else float("nan"),
    }


def level_moments(df: pd.DataFrame) -> dict:
    r = df["resid_log_y"].astype(float)
    return {
        "resid_mean": float(r.mean()),
        "resid_sd": float(r.std(ddof=1)),
        "resid_var": float(r.var(ddof=1)),
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def fmt(v, kind):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "--"
    if kind == "money":
        return f"{v:,.0f}"
    if kind == "share":
        return f"{v:.3f}"
    if kind == "moment":
        return f"{v:.4f}"
    if kind == "int":
        return f"{v:,d}"
    return f"{v:.2f}"


def render_console(rows: list[tuple[str, str, str, str]]) -> str:
    w0 = max(len(r[0]) for r in rows)
    head = f"{'':<{w0}}  {'BPP published':>15}  {'Our selection':>15}  {'Our balanced':>15}"
    out = [head, "-" * len(head)]
    for label, a, b, c in rows:
        if a is None:  # section header
            out.append("")
            out.append(label)
            continue
        out.append(f"{label:<{w0}}  {a:>15}  {b:>15}  {c:>15}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-year", type=int, default=1980)
    ap.add_argument("--end-year", type=int, default=1989)
    ap.add_argument("--ext", type=Path, default=EXT)
    ap.add_argument("--keep-seo", action="store_true")
    ap.add_argument("--demo-year", type=int, default=1980,
                    help="Which Table 1 year to put in the compact table.")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    window = (args.start_year, args.end_year)

    data = pd.read_stata(args.ext / "data.dta", convert_categoricals=False)
    tax = pd.read_stata(args.ext / "tax9192.dta", convert_categoricals=False)
    natpr = pd.read_stata(args.ext / "natpr.dta", convert_categoricals=False)
    for d in (data, tax, natpr):
        if "year" in d.columns:
            d["year"] = d["year"].astype(int)

    # Column 2: the selection port, exactly as psid_bpp_build runs it.
    sel, _audit = select_sample(data, drop_seo=not args.keep_seo)
    sel = merge_taxes(sel, tax)
    sel = add_table1_vars(sel)

    # Residualized panel + the balanced subset that becomes the .npy.
    resid = build_log_residual_earnings(sel, natpr)
    _M, persons, _years = to_wide_matrix(resid, args.start_year, args.end_year)
    persons = set(persons)

    # Column 3: same rows, restricted to balanced heads and the window.
    in_window = sel["year"].between(args.start_year, args.end_year)
    bal = sel[sel["person"].isin(persons) & in_window].copy()
    resid_bal = resid[resid["person"].isin(persons)
                      & resid["year"].between(args.start_year, args.end_year)].copy()

    pub_years = [1980, 1983, 1986, 1989, 1992]
    win_years = list(range(args.start_year, args.end_year + 1))

    summary = {
        "window": {"start_year": args.start_year, "end_year": args.end_year},
        "drop_seo": not args.keep_seo,
        "bpp_published": {
            "counts": BPP_COUNTS,
            "table1": BPP_TABLE1,
            "table3_var_dy": BPP_TABLE3_VAR,
            "table3_cov1_dy": BPP_TABLE3_COV1,
        },
        "our_selection": {
            "counts": panel_counts(sel),
            "table1": year_means(sel, pub_years),
            "growth": growth_moments(resid, win_years),
            "levels": level_moments(resid),
        },
        "coverage": coverage_breakdown(sel, args.start_year, args.end_year),
        "our_balanced": {
            "counts": panel_counts(bal),
            "table1": year_means(bal, pub_years),
            "growth": growth_moments(resid_bal, win_years),
            "levels": level_moments(resid_bal),
        },
    }

    # ---- compact three-column table -------------------------------------
    sel_c, bal_c = summary["our_selection"], summary["our_balanced"]
    cov = summary["coverage"]
    dy = args.demo_year
    mid_years = [y for y in (1981, 1985, 1989) if y in win_years]

    rows: list[tuple] = []
    rows.append(("Panel structure", None, None, None))
    rows.append(("Households", fmt(BPP_COUNTS["households"], "int"),
                 fmt(sel_c["counts"]["households"], "int"),
                 fmt(bal_c["counts"]["households"], "int")))
    rows.append(("Household-year observations", fmt(BPP_COUNTS["observations"], "int"),
                 fmt(sel_c["counts"]["observations"], "int"),
                 fmt(bal_c["counts"]["observations"], "int")))
    rows.append(("Years covered", "1978--1992",
                 f"{sel_c['counts']['first_year']}--{sel_c['counts']['last_year']}",
                 f"{bal_c['counts']['first_year']}--{bal_c['counts']['last_year']}"))
    rows.append(("Years per household (mean)", "--",
                 fmt(sel_c["counts"]["years_per_hh_mean"], "mean"),
                 fmt(bal_c["counts"]["years_per_hh_mean"], "mean")))
    rows.append(("Years per household (min--max)",
                 f"{BPP_COUNTS['years_per_hh_min']}--{BPP_COUNTS['years_per_hh_max']}",
                 f"{sel_c['counts']['years_per_hh_min']}--{sel_c['counts']['years_per_hh_max']}",
                 f"{bal_c['counts']['years_per_hh_min']}--{bal_c['counts']['years_per_hh_max']}"))
    rows.append(("Balanced", "no",
                 "yes" if sel_c["counts"]["balanced"] else "no",
                 "yes" if bal_c["counts"]["balanced"] else "no"))

    rows.append((f"Means in {dy}", None, None, None))
    for key, label, kind in LABELS:
        rows.append((label,
                     fmt(BPP_TABLE1[key].get(dy), kind),
                     fmt(sel_c["table1"][key].get(dy), kind),
                     fmt(bal_c["table1"][key].get(dy), kind)))

    rows.append(("Residual log earnings", None, None, None))
    rows.append(("Std. dev.", "--",
                 fmt(sel_c["levels"]["resid_sd"], "moment"),
                 fmt(bal_c["levels"]["resid_sd"], "moment")))

    rows.append(("Growth moments", None, None, None))
    for t in mid_years:
        rows.append((f"$\\mathrm{{E}}[\\Delta y_{{{t}}}^2]$",
                     fmt(BPP_TABLE3_VAR.get(t), "moment"),
                     fmt(sel_c["growth"]["var_dy"].get(t), "moment"),
                     fmt(bal_c["growth"]["var_dy"].get(t), "moment")))
    for t in mid_years[:-1]:
        rows.append((f"$\\mathrm{{E}}[\\Delta y_{{{t+1}}}\\Delta y_{{{t}}}]$",
                     fmt(BPP_TABLE3_COV1.get(t), "moment"),
                     fmt(sel_c["growth"]["cov1_dy"].get(t), "moment"),
                     fmt(bal_c["growth"]["cov1_dy"].get(t), "moment")))

    # Ratio block: the decomposition of the variance shortfall.
    def _ratios(key, bpp_ref):
        ys = [t for t in win_years
              if t in bpp_ref
              and np.isfinite(sel_c["growth"][key].get(t, float("nan")))
              and np.isfinite(bal_c["growth"][key].get(t, float("nan")))]
        if not ys:
            return None, None, None, 0
        sp = float(np.mean([sel_c["growth"][key][t] / bpp_ref[t] for t in ys]))
        bs = float(np.mean([bal_c["growth"][key][t] / sel_c["growth"][key][t] for t in ys]))
        bp = float(np.mean([bal_c["growth"][key][t] / bpp_ref[t] for t in ys]))
        return sp, bs, bp, len(ys)

    v_sp, v_bs, v_bp, n_v = _ratios("var_dy", BPP_TABLE3_VAR)
    c_sp, _c_bs, c_bp, _n_c = _ratios("cov1_dy", BPP_TABLE3_COV1)

    rows.append(("Mean ratio to BPP", None, None, None))
    if n_v:
        rows.append(("$\\mathrm{E}[\\Delta y^2]$, vs.\\ BPP", "1.000",
                     f"{v_sp:.3f}", f"{v_bp:.3f}"))
        rows.append(("$\\mathrm{E}[\\Delta y_{t+1}\\Delta y_t]$, vs.\\ BPP", "1.000",
                     f"{c_sp:.3f}", f"{c_bp:.3f}"))
        rows.append(("$\\mathrm{E}[\\Delta y^2]$, balanced vs.\\ selection",
                     "--", "--", f"{v_bs:.3f}"))

    rows.append(("Households lost to the balanced cut", None, None, None))
    rows.append(("Cover the window in full", "--", "--",
                 fmt(cov["covers_window"], "int")))
    rows.append((f"Spell shorter than $T={len(win_years)}$", "--",
                 fmt(cov["spell_shorter_than_T"], "int"), "--"))
    rows.append(("Spell $\\geq T$ but misaligned", "--",
                 fmt(cov["spell_ge_T_but_misses_window"], "int"), "--"))

    print(render_console([(r[0], r[1], r[2], r[3]) for r in rows]))

    # ---- full year-by-year dump -----------------------------------------
    print("\n\nTable 1 comparison, every published year")
    print(f"{'Variable':<22}" + "".join(f"{y:>26d}" for y in pub_years))
    print(f"{'':<22}" + "".join(f"{'pub / sel / bal':>26}" for _ in pub_years))
    for key, label, kind in LABELS:
        line = f"{label.replace(chr(92) + '$', '$'):<22}"
        for y in pub_years:
            a = fmt(BPP_TABLE1[key].get(y), kind)
            b = fmt(sel_c["table1"][key].get(y), kind)
            c = fmt(bal_c["table1"][key].get(y), kind)
            line += f"{a + ' / ' + b + ' / ' + c:>26}"
        print(line)

    print("\n\nWhy households drop at the balancing step")
    print(f"  households in the selection sample      : {cov['households_total']:,d}")
    print(f"  cover {args.start_year}-{args.end_year} in full (kept)          : {cov['covers_window']:,d}")
    print(f"  spell shorter than T={args.end_year - args.start_year + 1}                  : {cov['spell_shorter_than_T']:,d}")
    print(f"  spell >= T but misaligned with window   : {cov['spell_ge_T_but_misses_window']:,d}")
    print(f"  mean spell, all / dropped               : "
          f"{cov['mean_spell_all']:.2f} / {cov['mean_spell_dropped']:.2f}")

    print("\n\nGrowth moments, every window year")
    print(f"{'Year':<8}{'var pub':>10}{'var sel':>10}{'var bal':>10}"
          f"{'cov1 pub':>11}{'cov1 sel':>11}{'cov1 bal':>11}")
    for t in win_years:
        print(f"{t:<8}"
              f"{fmt(BPP_TABLE3_VAR.get(t), 'moment'):>10}"
              f"{fmt(sel_c['growth']['var_dy'].get(t), 'moment'):>10}"
              f"{fmt(bal_c['growth']['var_dy'].get(t), 'moment'):>10}"
              f"{fmt(BPP_TABLE3_COV1.get(t), 'moment'):>11}"
              f"{fmt(sel_c['growth']['cov1_dy'].get(t), 'moment'):>11}"
              f"{fmt(bal_c['growth']['cov1_dy'].get(t), 'moment'):>11}")

    # ---- artefacts -------------------------------------------------------
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / "bpp_selection_table.json"
    with out_json.open("w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    print(f"\nwrote {out_json}")


if __name__ == "__main__":
    main()
