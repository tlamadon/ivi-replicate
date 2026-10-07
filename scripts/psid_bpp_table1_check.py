"""Reproduce BPP AER 2008 Table 1 (PSID columns) and compare to the paper."""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from psid_bpp_build import select_sample, merge_taxes
import os
EXT = os.environ.get("BPP_EXT", "ext/bpp")

# Load
data = pd.read_stata(EXT + "/data.dta", convert_categoricals=False)
tax = pd.read_stata(EXT + "/tax9192.dta", convert_categoricals=False)
for d in (data, tax):
    if "year" in d.columns:
        d["year"] = d["year"].astype(int)

# Selection (default: drop SEO, age 30-65, etc.)
df, _ = select_sample(data)
df = merge_taxes(df, tax)
# Wife participation: PSID variable `hourw` (annual hours of wife). 0 = not working.
df["wife_working"] = (df["hourw"].fillna(0) > 0).astype(float)
df["husband_working"] = (df["hours"].fillna(0) > 0).astype(float)
df["white"] = (df["race"] == 1).astype(float)
df["hs_dropout"]  = (df["educ"] == 1).astype(float)
df["hs_graduate"] = (df["educ"] == 2).astype(float)
df["college_dropout"] = (df["educ"] == 3).astype(float)
df["northeast"] = (df["region"] == 1).astype(float)
df["midwest"]   = (df["region"] == 2).astype(float)
df["south"]     = (df["region"] == 3).astype(float)
df["west"]      = (df["region"] == 4).astype(float)
df["disposable_income"] = df["y"] - df["ftax"]  # nominal disposable (not deflated)
df["food_total"] = df["food"].fillna(0) + df["fout"].fillna(0)

years = [1980, 1983, 1986, 1989, 1992]
rows = [
    ("Age",               "age"),
    ("Family size",       "fsize"),
    ("No. of children",   "kids"),
    ("White",             "white"),
    ("HS dropout",        "hs_dropout"),
    ("HS graduate",       "hs_graduate"),
    ("College dropout",   "college_dropout"),
    ("Northeast",         "northeast"),
    ("Midwest",           "midwest"),
    ("South",             "south"),
    ("West",              "west"),
    ("Husband working",   "husband_working"),
    ("Wife working",      "wife_working"),
    ("Disposable income", "disposable_income"),
    ("Food expenditure",  "food_total"),
]

# AER 2008 Table 1, PSID columns
paper = {
    "Age":               {1980:42.94, 1983:43.43, 1986:43.86, 1989:44.03, 1992:45.95},
    "Family size":       {1980:3.61,  1983:3.52,  1986:3.48,  1989:3.44,  1992:3.42},
    "No. of children":   {1980:1.32,  1983:1.25,  1986:1.21,  1989:1.18,  1992:1.14},
    "White":             {1980:0.91,  1983:0.92,  1986:0.93,  1989:0.94,  1992:0.94},
    "HS dropout":        {1980:0.21,  1983:0.18,  1986:0.16,  1989:0.14,  1992:0.13},
    "HS graduate":       {1980:0.30,  1983:0.31,  1986:0.32,  1989:0.32,  1992:0.32},
    "College dropout":   {1980:0.49,  1983:0.51,  1986:0.53,  1989:0.54,  1992:0.55},
    "Northeast":         {1980:0.21,  1983:0.21,  1986:0.22,  1989:0.22,  1992:0.22},
    "Midwest":           {1980:0.33,  1983:0.31,  1986:0.30,  1989:0.30,  1992:0.31},
    "South":             {1980:0.31,  1983:0.31,  1986:0.30,  1989:0.30,  1992:0.30},
    "West":              {1980:0.15,  1983:0.17,  1986:0.18,  1989:0.18,  1992:0.18},
    "Husband working":   {1980:0.96,  1983:0.94,  1986:0.93,  1989:0.94,  1992:0.93},
    "Wife working":      {1980:0.69,  1983:0.71,  1986:0.74,  1989:0.78,  1992:0.77},
    "Disposable income": {1980:29333, 1983:35427, 1986:42374, 1989:50684, 1992:58841},
    "Food expenditure":  {1980:4447,  1983:4868,  1986:5294,  1989:5872,  1992:6604},
}

def fmt(x, is_money):
    if is_money:
        return f"{x:>7,.0f}"
    if abs(x) >= 10:
        return f"{x:>7.2f}"
    return f"{x:>7.3f}"

print(f"{'Variable':<20s}", end="")
for y in years:
    print(f"|   {y} paper   |    {y} ours   ", end="")
print()
print("-" * (20 + 5 * (len("|   YYYY paper   |    YYYY ours   "))))
for label, col in rows:
    is_money = col in ("disposable_income", "food_total")
    print(f"{label:<20s}", end="")
    for y in years:
        sub = df[df["year"] == y]
        ours = sub[col].mean()
        pap = paper[label][y]
        print(f"| {fmt(pap, is_money)}        | {fmt(ours, is_money)}       ", end="")
    print()

# Per-year sample sizes
print()
print("Per-year N (households observed in that year, post-selection):")
for y in years:
    n = (df["year"] == y).sum()
    print(f"  {y}: {n} household-years")
