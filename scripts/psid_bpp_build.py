"""Build the BPP (Blundell-Pistaferri-Preston, AER 2008) PSID log-earnings panel.

Inputs: the AER 2008 replication package (ICPSR 210782, V2.1) extracted under
``ext/bpp/`` (override with ``--ext`` or ``$BPP_EXT``). We use only the PSID side:

- ``data.dta``     -- pre-built household-year panel (from create1/create2)
- ``tax9192.dta``  -- TAXSIM federal income tax for 1991--92 (PSID stopped
                      computing federal taxes after 1990)
- ``natpr.dta``    -- national price index (we use ``cpi``)

Pipeline:
1. Port ``adjust_AER.do`` line-by-line: BPP's PSID sample selection.
2. Merge TAXSIM taxes for 1991--92; impute the few residual pre-1991 missings
   via a polynomial regression on income, exactly as ``impute_AER.do`` does.
3. Build real after-tax non-financial family income (AER 2008 Appendix A;
   mindist_AER.do incdef=0, aftertax=1):
   ``((y - asset) - ratio_ya * ftax) / (cpi/100)`` with
   ``ratio_ya = (y - asset)/y``. Then log.
4. Residualize log earnings per AER 2008 Sec III.A: a single pooled OLS with
   year FE, year-of-birth FE, education, race, family size, # kids, region,
   employment (head's hours > 0), residence in a large city (smsa==1),
   outside-dependent (outkid==1), other-income-recipients (tyoth>0), and
   year x education interactions.
5. Keep individuals observed in every calendar year of the requested window
   (default 1980--1989 inclusive --> T=10) and emit ``output/data/bpp_y_matrix.npy``
   plus a sibling JSON with selection counts mirroring Table 1 of the paper.

Run:
    uv run python scripts/psid_bpp_build.py

See ``specs/psid_bpp_data.md`` for the full code contract.
"""

from __future__ import annotations

import argparse
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


REPO = Path(__file__).resolve().parent.parent
EXT = Path(os.environ.get("BPP_EXT", REPO / "ext" / "bpp"))
OUT_MATRIX = Path("output") / "data" / "bpp_y_matrix.npy"
OUT_META = Path("output") / "data" / "bpp_y_matrix.meta.json"


# ---------------------------------------------------------------------------
# Step 1: port adjust_AER.do
# ---------------------------------------------------------------------------
def select_sample(data: pd.DataFrame, drop_seo: bool = True) -> tuple[pd.DataFrame, list[dict]]:
    """Apply the BPP sample selection from adjust_AER.do.

    Returns the trimmed dataframe and a step-by-step audit (list of dicts).
    """
    audit: list[dict] = []

    def record(label: str, df: pd.DataFrame, before_obs: int, before_hh: int):
        audit.append(
            {
                "step": label,
                "dropped_obs": before_obs - len(df),
                "remaining_obs": len(df),
                "remaining_hh": df["person"].nunique(),
                "remaining_hh_dropped": before_hh - df["person"].nunique(),
            }
        )

    df = data.copy()
    before_obs, before_hh = len(df), df["person"].nunique()
    audit.append(
        {
            "step": "initial (post create1/create2 merge)",
            "dropped_obs": 0,
            "remaining_obs": before_obs,
            "remaining_hh": before_hh,
            "remaining_hh_dropped": 0,
        }
    )

    # ``drop pid; sort person year``
    df = df.drop(columns=["pid"]).sort_values(["person", "year"]).reset_index(drop=True)

    # ``qby person: gen dyear=year-year[_n-1]``
    # ``egen todrop=sum(dyear>1 & dyear!=.),by(person)``  -- intermittent
    # ``egen n=sum(person!=.),by(person)``                -- appears once
    # ``replace todrop=1 if n==1``
    # ``drop if todrop>0``
    df["dyear"] = df.groupby("person")["year"].diff()
    bad_gap = df.groupby("person")["dyear"].transform(
        lambda s: ((s > 1) & s.notna()).sum()
    )
    n_per_person = df.groupby("person")["year"].transform("size")
    keep = (bad_gap == 0) & (n_per_person > 1)
    before_o, before_h = len(df), df["person"].nunique()
    df = df[keep].drop(columns=["dyear"]).copy()
    record("drop intermittent / single-obs heads", df, before_o, before_h)

    # ``drop if year<1978``
    before_o, before_h = len(df), df["person"].nunique()
    df = df[df["year"] >= 1978].copy()
    record("drop year < 1978", df, before_o, before_h)

    # ``replace age=age-1``  (retrospective income data)
    df["age"] = df["age"] - 1

    # Race recode: 1=white, 2=black, 3=others
    rc = df["race"].copy()
    rc = rc.where(rc <= 2, other=np.nan)
    rc = rc.fillna(df["race"].where((df["race"] >= 3) & (df["race"] <= 7), other=np.nan).map(lambda _: 3.0))
    # Simpler: explicit recode
    rc = pd.Series(np.nan, index=df.index)
    rc = rc.mask((df["race"] >= 1) & (df["race"] <= 2), df["race"])
    rc = rc.mask((df["race"] >= 3) & (df["race"] <= 7), 3.0)
    df["race"] = rc

    # Education recode (consistent across PSID waves)
    # pre-1990: educ values 1..8 -> grouped as 1/2/3
    # post-1989: educ stores grade counts 0..17 -> grouped 0-11/12/13-17
    sc = pd.Series(np.nan, index=df.index)
    pre = df["year"] <= 1989
    e = df["educ"]
    sc = sc.mask(pre & (e <= 3), 1.0)
    sc = sc.mask(pre & ((e == 4) | (e == 5)), 2.0)
    sc = sc.mask(pre & (e >= 6) & (e <= 8), 3.0)
    post = df["year"] > 1989
    sc = sc.mask(post & (e >= 0) & (e <= 11), 1.0)
    sc = sc.mask(post & (e == 12), 2.0)
    sc = sc.mask(post & (e >= 13) & (e <= 17), 3.0)
    df["educ"] = sc

    # Manual fill of education from adjacent records, then max-grade-achieved
    df = df.sort_values(["person", "year"])
    df["educ"] = df.groupby("person")["educ"].ffill().bfill()
    df["educ"] = df.groupby("person")["educ"].transform(
        lambda s: s.max(skipna=True)
    )

    # Demographically stable households (fchg<=1 except in first observed year)
    # ``egen miny=min(year),by(person)``
    # ``gen todrop1=fchg>1``
    # ``replace todrop1=0 if year==miny & todrop1==1``
    # ``egen todrop2=sum(todrop1),by(person)``
    # ``drop if todrop2!=0``
    df["_miny"] = df.groupby("person")["year"].transform("min")
    bad = (df["fchg"] > 1).astype(int)
    bad = bad.where(~((df["year"] == df["_miny"]) & (bad == 1)), other=0)
    n_bad = bad.groupby(df["person"]).transform("sum")
    before_o, before_h = len(df), df["person"].nunique()
    df = df[n_bad == 0].drop(columns=["_miny"]).copy()
    record("drop demographically unstable hh (fchg>1)", df, before_o, before_h)

    # Female head
    before_o, before_h = len(df), df["person"].nunique()
    df = df[df["sex"] != 2].copy()
    record("drop female head", df, before_o, before_h)

    # Missing race
    before_o, before_h = len(df), df["person"].nunique()
    df = df[df["race"].notna()].copy()
    record("drop missing race", df, before_o, before_h)

    # Missing education on any obs of person
    n_miss = df.groupby("person")["educ"].transform(lambda s: s.isna().sum())
    before_o, before_h = len(df), df["person"].nunique()
    df = df[n_miss == 0].copy()
    record("drop hh with any missing education", df, before_o, before_h)

    # Manual age fixes from adjust_AER.do
    df.loc[(df["person"] == 5084) & (df["age"] == 98), "age"] = 30
    df.loc[(df["person"] == 5084) & (df["age"].isna()), "age"] = 31
    df.loc[(df["person"] == 6927) & (df["agew"].isna()), "agew"] = 77

    # Recode age so it has no gaps/jumps: anchor on the last observed
    # (year, age) pair per person, exactly like the do-file's
    # ``yb = lasty - lastage`` followed by ``age = year - yb``.
    df = df.sort_values(["person", "year"])
    last_year_p = df.groupby("person")["year"].transform("last")
    last_age_p = df.groupby("person")["age"].transform("last")
    yb = last_year_p - last_age_p
    df["age"] = df["year"] - yb

    # Region from state (BPP 4-region grouping)
    df["region"] = np.nan
    NE = [6, 18, 20, 28, 29, 31, 37, 38, 44]
    MW = [12, 13, 14, 15, 21, 22, 24, 26, 33, 34, 40, 48]
    SO = [1, 3, 7, 8, 9, 10, 16, 17, 19, 23, 32, 35, 39, 41, 42, 45, 47]
    WE = [2, 4, 5, 11, 25, 27, 30, 36, 43, 46, 49, 50, 51]
    df.loc[df["state"].isin(NE), "region"] = 1
    df.loc[df["state"].isin(MW), "region"] = 2
    df.loc[df["state"].isin(SO), "region"] = 3
    df.loc[df["state"].isin(WE), "region"] = 4

    # Drop hh with any missing/zero/99 state
    bad_state = df["state"].isna() | (df["state"] == 0) | (df["state"] == 99)
    n_bad_state = bad_state.groupby(df["person"]).transform("sum")
    before_o, before_h = len(df), df["person"].nunique()
    df = df[n_bad_state == 0].copy()
    record("drop hh with any missing state", df, before_o, before_h)

    # Continuously married: every observation has marit==1.
    n_married = (df["marit"] == 1).groupby(df["person"]).transform("sum")
    n_total = df.groupby("person")["year"].transform("size")
    before_o, before_h = len(df), df["person"].nunique()
    df = df[n_married == n_total].copy()
    record("drop hh not continuously married", df, before_o, before_h)

    # Income outliers (growth > 500%, or < -80%, or level <= 100)
    df = df.sort_values(["person", "year"])
    gy = df.groupby("person")["y"].transform(lambda s: (s - s.shift(1)) / s.shift(1))
    bad = ((gy > 5) & gy.notna()) | ((gy < -0.8) & gy.notna()) | (df["y"] <= 100)
    n_bad = bad.groupby(df["person"]).transform("sum")
    before_o, before_h = len(df), df["person"].nunique()
    df = df[n_bad == 0].copy()
    record("drop income outliers", df, before_o, before_h)

    # Year of birth, then keep only 1920 <= yb <= 1959
    df["yb"] = df["year"] - df["age"]
    before_o, before_h = len(df), df["person"].nunique()
    df = df[(df["yb"] >= 1920) & (df["yb"] <= 1959)].copy()
    record("drop hh outside birth cohort 1920-1959", df, before_o, before_h)

    # Adjust_AER.do does NOT drop the SEO sample explicitly; that happens in
    # create2_AER.do via ``drop if v30001>=7001 & v30001<=9308`` (Latino) and
    # is then optionally excluded downstream. The paper's Table 1 reports
    # counts AFTER dropping SEO. Default: drop. Pass --keep-seo to keep them.
    if drop_seo and "seo" in df.columns:
        before_o, before_h = len(df), df["person"].nunique()
        df = df[df["seo"] != 1].copy()
        record("drop SEO (poverty) subsample", df, before_o, before_h)

    # Age 30-65 (BPP final cut)
    before_o, before_h = len(df), df["person"].nunique()
    df = df[(df["age"] >= 30) & (df["age"] <= 65)].copy()
    record("drop age < 30 or > 65", df, before_o, before_h)

    return df, audit


# ---------------------------------------------------------------------------
# Step 2: federal taxes from TAXSIM + polynomial imputation
# ---------------------------------------------------------------------------
def merge_taxes(df: pd.DataFrame, tax: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(tax, on=["id", "year"], how="left")
    df["ftax"] = df["ftax"].astype("float64")
    # Replace ftax with TAXSIM for years >1990
    post = df["year"] > 1990
    df.loc[post, "ftax"] = df.loc[post, "fiitax"].astype("float64")
    df = df.drop(columns=["fiitax"])

    # Impute residual missing ftax via the same poly regression as impute_AER.do
    Xcols = ["y", "y2", "y3", "y4", "y5", "ncomp", "ncomp2", "post86", "kids", "self"]

    fit = df[df["year"] < 1991].copy()
    fit["y2"] = fit["y"] ** 2
    fit["y3"] = fit["y"] ** 3
    fit["y4"] = fit["y"] ** 4
    fit["y5"] = fit["y"] ** 5
    fit["ncomp"] = fit["fsize"]
    fit["ncomp2"] = fit["ncomp"] ** 2
    fit["post86"] = (fit["year"] > 1986).astype(int)
    fit = fit.dropna(subset=Xcols + ["ftax"])
    X = sm.add_constant(fit[Xcols].astype(float))
    model = sm.OLS(fit["ftax"].astype(float), X).fit()

    need = df["ftax"].isna() & (df["year"] >= 1991)
    if need.any():
        tmp = df.loc[need].copy()
        tmp["y2"] = tmp["y"] ** 2
        tmp["y3"] = tmp["y"] ** 3
        tmp["y4"] = tmp["y"] ** 4
        tmp["y5"] = tmp["y"] ** 5
        tmp["ncomp"] = tmp["fsize"]
        tmp["ncomp2"] = tmp["ncomp"] ** 2
        tmp["post86"] = (tmp["year"] > 1986).astype(int)
        Xn = sm.add_constant(tmp[Xcols].astype(float), has_constant="add")
        # align columns
        Xn = Xn[X.columns]
        df.loc[need, "ftax"] = model.predict(Xn).values
    return df


# ---------------------------------------------------------------------------
# Step 3 + 4: real after-tax non-financial income, log, residualize
# ---------------------------------------------------------------------------
def build_log_residual_earnings(df: pd.DataFrame, natpr: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(natpr[["year", "cpi"]], on="year", how="left")
    df["price"] = df["cpi"] / 100.0

    # BPP main income (AER 2008 Appendix A; mindist_AER.do incdef=0, aftertax=1):
    #   y_nonfin    = y - asset                       (gross non-financial)
    #   ratio_ya    = (y - asset) / y                 (non-financial share)
    #   y_real_aft  = y_nonfin/price - ratio_ya * ftax/price
    # i.e. real after-tax family income, scaling the federal tax by the
    # non-financial share. Mirror the do-file's trunca/truncy NaNing.
    df.loc[df.get("trunca", 0) == 1, "asset"] = np.nan
    df["asset"] = df["asset"].fillna(0.0)
    df["y_nonfin"] = df["y"] - df["asset"]
    df["ratio_ya"] = df["y_nonfin"] / df["y"]
    df["y_real"] = (df["y_nonfin"] - df["ratio_ya"] * df["ftax"]) / df["price"]
    df = df[df["y_real"] > 0].copy()  # log requires positive
    df["log_y"] = np.log(df["y_real"])

    # Cohort bands (used downstream for indexing, not in the regression).
    df["coh"] = np.nan
    df.loc[(df["yb"] >= 1950) & (df["yb"] <= 1959), "coh"] = 1
    df.loc[(df["yb"] >= 1940) & (df["yb"] <= 1949), "coh"] = 2
    df.loc[(df["yb"] >= 1930) & (df["yb"] <= 1939), "coh"] = 3
    df.loc[(df["yb"] >= 1920) & (df["yb"] <= 1929), "coh"] = 4

    # Controls per AER 2008 \S III.A (single pooled regression, not per cell):
    #   year FE, year-of-birth FE, education, race, family size, # kids,
    #   region, employment status, residence in a large city, outside
    #   dependent, presence of income recipients other than husband/wife.
    df["white"] = (df["race"] == 1).astype(int)
    df["parth"] = (df["hours"].fillna(0) > 0).astype(int)
    df["large_city"] = (df["smsa"] == 1).astype(int)  # PSID smsa==1 = 500k+
    df["outside_dep"] = (df["outkid"] == 1).astype(int)  # PSID outkid: 1=yes
    df["other_inc_rec"] = (df["tyoth"] > 0).astype(int)

    def dummies(s, prefix):
        return pd.get_dummies(
            s.astype("Int64").astype(str), prefix=prefix, drop_first=True
        ).astype(float)

    # Build the design matrix.
    df["_idx"] = np.arange(len(df))
    parts = [
        pd.DataFrame({"intercept": np.ones(len(df))}, index=df.index),
        df[["fsize", "kids", "white", "parth",
            "large_city", "outside_dep", "other_inc_rec"]].astype(float),
        dummies(df["year"], "yr"),
        dummies(df["yb"].astype(int), "yb"),
        dummies(df["educ"], "ed"),
        dummies(df["region"], "rg"),
    ]
    # "We allow for the effect of most of these characteristics to vary with
    # calendar time" (AER 2008 \S III.A). Add year x education interactions
    # as the most empirically important time-varying control (the literature
    # treats the education premium as the canonical driver of the inequality
    # trend in this period).
    yr_d = dummies(df["year"], "yr")
    ed_d = dummies(df["educ"], "ed")
    inter = pd.DataFrame(
        {
            f"{yc}_x_{ec}": yr_d[yc].values * ed_d[ec].values
            for yc in yr_d.columns
            for ec in ed_d.columns
        },
        index=df.index,
    )
    parts.append(inter)
    X = pd.concat(parts, axis=1).astype(float).values
    # Drop columns that are constant or all-zero (degenerate after sample cuts)
    col_std = X.std(axis=0)
    keep_cols = col_std > 1e-12
    keep_cols[0] = True  # always keep intercept
    X = X[:, keep_cols]

    yv = df["log_y"].astype(float).values
    beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
    df["resid_log_y"] = yv - X @ beta
    df = df.drop(columns=["_idx"])
    return df


# ---------------------------------------------------------------------------
# Step 5: pivot to balanced (N, T) panel for a requested year window
# ---------------------------------------------------------------------------
def to_wide_matrix(df: pd.DataFrame, start_year: int, end_year: int):
    years = list(range(start_year, end_year + 1))
    T = len(years)

    sub = df[df["year"].isin(years)].copy()
    counts = sub.groupby("person")["year"].nunique()
    keep_persons = counts[counts == T].index
    sub = sub[sub["person"].isin(keep_persons)]

    wide = (
        sub.pivot_table(index="person", columns="year", values="resid_log_y")
        .reindex(columns=years)
        .sort_index()
    )
    return wide.values.astype(np.float32), wide.index.tolist(), years


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-year", type=int, default=1980)
    ap.add_argument("--end-year", type=int, default=1989)
    ap.add_argument(
        "--ext", type=Path, default=EXT,
        help="Directory holding data.dta / tax9192.dta / natpr.dta from the ICPSR 210782 package (default ext/bpp, or $BPP_EXT)",
    )
    ap.add_argument(
        "--out", type=Path, default=OUT_MATRIX,
        help="Output .npy path (N, T) of residualized log earnings",
    )
    ap.add_argument(
        "--meta", type=Path, default=OUT_META,
        help="Output JSON path for selection counts + metadata",
    )
    ap.add_argument(
        "--keep-seo", action="store_true",
        help="Keep the SEO (poverty) subsample. Default: drop, matching the "
             "BPP AER 2008 main sample in Table 1.",
    )
    args = ap.parse_args()

    print(f"Loading data.dta from {args.ext}/data.dta ...")
    data = pd.read_stata(args.ext / "data.dta", convert_categoricals=False)
    tax = pd.read_stata(args.ext / "tax9192.dta", convert_categoricals=False)
    natpr = pd.read_stata(args.ext / "natpr.dta", convert_categoricals=False)

    # Cast year to int to avoid float-key headaches
    for d in (data, tax, natpr):
        if "year" in d.columns:
            d["year"] = d["year"].astype(int)

    print("Step 1: sample selection (adjust_AER.do port)")
    df, audit = select_sample(data, drop_seo=not args.keep_seo)

    print("Step 2: merge TAXSIM 1991-92 + impute residual missing ftax")
    df = merge_taxes(df, tax)

    print("Step 3-4: real after-tax non-financial income, log, residualize")
    df = build_log_residual_earnings(df, natpr)

    print(f"Step 5: balanced window [{args.start_year}, {args.end_year}] -> wide matrix")
    M, persons, years = to_wide_matrix(df, args.start_year, args.end_year)
    print(f"  matrix shape: {M.shape}  (N={M.shape[0]}, T={M.shape[1]})")
    print(f"  mean: {M.mean():.4g}  std: {M.std():.4g}  min: {M.min():.3g}  max: {M.max():.3g}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, M)
    print(f"  wrote {args.out}")

    meta = {
        "source_replication": "ICPSR 210782 V2.1 (BPP AER 2008)",
        "source_files": ["data.dta", "tax9192.dta", "natpr.dta"],
        "window": {"start_year": args.start_year, "end_year": args.end_year, "T": M.shape[1]},
        "panel_balance": "balanced (every person observed in every year of window)",
        "drop_seo": not args.keep_seo,
        "N": int(M.shape[0]),
        "income_definition": (
            "BPP main income (AER 2008 Appendix A; mindist_AER.do incdef=0, "
            "aftertax=1): y_real_aftertax = ((y - asset) - ratio_ya * ftax) "
            "/ (cpi/100), where ratio_ya = (y - asset)/y. Then log. "
            "Residualized per AER 2008 sec III.A: single pooled OLS with "
            "year FE, year-of-birth FE, education, race, family size, "
            "# kids, region, employment, large-city (smsa), outside-dependent "
            "(outkid), other-income-recipients (tyoth>0), and year x "
            "education interactions."
        ),
        "selection_audit": audit,
        "person_ids": persons,
        "years": years,
    }
    with args.meta.open("w") as fh:
        json.dump(meta, fh, indent=2, default=float)
    print(f"  wrote {args.meta}")


if __name__ == "__main__":
    main()
