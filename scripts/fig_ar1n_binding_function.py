#!/usr/bin/env python3
r"""Figure 5: Slice of the Binding Function for rho.

Input: output/bundles/ar1n_binding_mu1_bundle.json
Usage: python scripts/fig_ar1n_binding_function.py output/bundles/ar1n_binding_mu1_bundle.json -o output/figures/tex/ar1n_binding_function.tex
"""

import argparse
import json
import sys
from pathlib import Path

# Layout / style choices (data-independent).
CONFIG = {
    # bundle keys
    "binding_key":        "b_mu1",     # array of binding-map outputs on the grid
    "grid_key":           "mu1_grid",  # array of rho_0 grid values
    "vi_ref":             "vi_obs",    # reference_points key -> VI estimate
    "truth_ref":          "truth",     # reference_points key -> DGP truth

    # canvas
    "width_cm":           8.0,
    "height_cm":          5.0,

    # axis frame: fixed upper bounds; None lower bounds are derived from the first point
    "xmax":               1.0,
    "ymax":               1.0,
    "xmin":               None,        # auto: round(grid_min  - pad_x, 2)
    "ymin":               None,        # auto: round(curve_min - pad_y, 2)
    "pad_x":              0.01,        # how far below the first x the left edge sits
    "pad_y":              0.015,       # how far below the first b the bottom edge sits
    "tick_distance":      0.1,

    # axis labels
    "xlabel":             r"$\rho_\star$",
    "ylabel":             r"$\hat\rho$",

    # binding curve
    "curve_color":        "teal!90!black",
    "curve_mark_size_pt": 1.6,

    # 45-degree line
    "diag_color":         "gray",
    "diag_style":         "dotted",
    "diag_label":         r"$45^\circ$",
    "diag_label_pos":     0.93,        # fraction along the line (0 = bottom-left)
    "diag_label_font":    r"\small",

    # VI estimate (horizontal line)
    "vi_color":           "camo",
    "vi_style":           "dashed",
    "vi_label_prefix":    "VI",
    "vi_label_pos":       0.2,         # fraction along the horizontal line
    "vi_label_anchor":    "above",

    # truth (vertical line)
    "truth_color":        "blue!55!black",
    "truth_style":        "dashed",
    "truth_label_prefix": "truth",
    "truth_label_pos":    0.07,        # fraction up the vertical line

    # label formatting
    "label_decimals":     2,           # decimals shown in the VI/truth labels
    "ref_label_font":     r"\normalsize\bfseries",

    # caption + label
    "caption":            r"Slice of the Binding Function for $\rho$",
    "label":              "ar1n_binding_function",
}

FIGURENOTE = (
    r"\vspace{0.2cm} \figurenote{Binding function $\widehat b_\rho(\vartheta_{\star})$ for the AR(1) coefficient (solid teal): the mean-field VI estimate of $\rho$ on data $\widetilde y_{1:T}(\vartheta_{\star})$ drawn from $\mathcal P_{\vartheta_{\star}}(y_{1:T})$. Along the horizontal axis only the candidate $\rho_\star$ varies, while the other coordinates of $\vartheta_{\star}$ stay at their true values. At $\rho_0=0.90$ (vertical dashed blue) the binding function returns $\widehat\rho^{\rm VI}=0.76$ (horizontal dashed orange). Linear Gaussian DGP, $N = 30{,}000$, $T = 6$.}"
)


def fmt(v):
    """Format a number for LaTeX, preserving the JSON value verbatim."""
    return repr(float(v))


def dim(v):
    """Format a length: integers drop the trailing .0 (8.0 -> '8')."""
    v = float(v)
    return str(int(v)) if v.is_integer() else repr(v)


def load_series(bundle, cfg):
    """Return (slice_name, sorted [(x, b), ...], vi_value, truth_value)."""
    slice_name = bundle["parameter_schema"]["slice_param_names"][0]
    grid = bundle[cfg["grid_key"]]
    curve = bundle[cfg["binding_key"]]
    if len(grid) != len(curve):
        sys.exit("ERROR: grid and binding arrays have different lengths.")
    pairs = sorted(zip(grid, curve), key=lambda p: p[0])
    rp = bundle["reference_points"]
    vi_value = rp[cfg["vi_ref"]]["parameters"][slice_name]
    truth_value = rp[cfg["truth_ref"]]["parameters"][slice_name]
    return slice_name, pairs, vi_value, truth_value


def resolve_limits(pairs, cfg):
    """Compute xmin/xmax/ymin/ymax. Auto lower bounds put the first dot on the edge."""
    xmax, ymax = cfg["xmax"], cfg["ymax"]
    x0, b0 = pairs[0]                      # first (lower-left) plotted dot
    xmin = cfg["xmin"] if cfg["xmin"] is not None else round(x0 - cfg["pad_x"], 2)
    ymin = cfg["ymin"] if cfg["ymin"] is not None else round(b0 - cfg["pad_y"], 2)
    return xmin, xmax, ymin, ymax


# @TOKENS@ are substituted in build_tex.
TEX_TEMPLATE = r"""\begin{figure}[!t]
\centering
\begin{tikzpicture}
\begin{axis}[
    width=@WIDTH@cm, height=@HEIGHT@cm,
    scale only axis=true,
    xmin=@XMIN@, xmax=@XMAX@,
    ymin=@YMIN@, ymax=@YMAX@,
    xtick distance=@TICK@,
    ytick distance=@TICK@,
    xlabel={@XLABEL@},
    ylabel={@YLABEL@},
    label style={font=\normalsize},
    tick label style={font=\normalsize},
    axis lines=left,
    enlargelimits=false,
    grid=major,
    grid style={gray!20},
    tick align=outside,
]

\addplot[domain=@XMIN@:@XMAX@, samples=2, @DIAGSTYLE@, @DIAGCOLOR@, thick] {x}
    node[sloped, pos=@DIAGPOS@, fill=white, inner sep=1.5pt, font=@DIAGFONT@, text=black] {@DIAGLABEL@};

\addplot[domain=@XMIN@:@XMAX@, samples=2, @VICOLOR@, thick, @VISTYLE@] {@VIVAL@}
    node[pos=@VIPOS@, @VIANCHOR@, text=@VICOLOR@, font=@REFFONT@] {@VILABEL@};

\addplot[@TRUTHCOLOR@, thick, @TRUTHSTYLE@] coordinates {(@TRUTHVAL@,@YMIN@) (@TRUTHVAL@,@YMAX@)}
    node[pos=@TRUTHPOS@, fill=white, inner sep=2pt, text=@TRUTHCOLOR@, font=@REFFONT@] {@TRUTHLABEL@};

\addplot[
    @CURVECOLOR@, thick, mark=*, mark size=@MARKSIZE@pt,
] coordinates {
@COORDS@
};

\end{axis}
\end{tikzpicture}
\caption{@CAPTION@}
\label{@LABEL@}
@FIGURENOTE@
\end{figure}
"""


def build_tex(bundle, cfg):
    _, pairs, vi_value, truth_value = load_series(bundle, cfg)
    xmin, xmax, ymin, ymax = resolve_limits(pairs, cfg)

    # keep only points inside the visible x-window [xmin, xmax]
    eps = 1e-9
    visible = [(x, b) for (x, b) in pairs if xmin - eps <= x <= xmax + eps]
    coords = "\n".join(f"    ({fmt(x)},{fmt(b)})" for (x, b) in visible)

    dec = cfg["label_decimals"]
    vi_label = f'{cfg["vi_label_prefix"]} ({vi_value:.{dec}f})'
    truth_label = f'{cfg["truth_label_prefix"]} ({truth_value:.{dec}f})'

    subs = {
        "@CAPTION@":    cfg["caption"],
        "@LABEL@":      cfg["label"],
        "@WIDTH@":      dim(cfg["width_cm"]),
        "@HEIGHT@":     dim(cfg["height_cm"]),
        "@XMIN@":       fmt(xmin),
        "@XMAX@":       fmt(xmax),
        "@YMIN@":       fmt(ymin),
        "@YMAX@":       fmt(ymax),
        "@TICK@":       fmt(cfg["tick_distance"]),
        "@XLABEL@":     cfg["xlabel"],
        "@YLABEL@":     cfg["ylabel"],
        "@DIAGSTYLE@":  cfg["diag_style"],
        "@DIAGCOLOR@":  cfg["diag_color"],
        "@DIAGPOS@":    fmt(cfg["diag_label_pos"]),
        "@DIAGFONT@":   cfg["diag_label_font"],
        "@DIAGLABEL@":  cfg["diag_label"],
        "@VISTYLE@":    cfg["vi_style"],
        "@VICOLOR@":    cfg["vi_color"],
        "@VIVAL@":      fmt(vi_value),
        "@VIPOS@":      fmt(cfg["vi_label_pos"]),
        "@VIANCHOR@":   cfg["vi_label_anchor"],
        "@VILABEL@":    vi_label,
        "@TRUTHSTYLE@": cfg["truth_style"],
        "@TRUTHCOLOR@": cfg["truth_color"],
        "@TRUTHVAL@":   fmt(truth_value),
        "@TRUTHPOS@":   fmt(cfg["truth_label_pos"]),
        "@TRUTHLABEL@": truth_label,
        "@CURVECOLOR@": cfg["curve_color"],
        "@MARKSIZE@":   fmt(cfg["curve_mark_size_pt"]),
        "@REFFONT@":    cfg["ref_label_font"],
        "@COORDS@":     coords,
        "@FIGURENOTE@": FIGURENOTE,
    }
    tex = TEX_TEMPLATE
    for token, value in subs.items():
        tex = tex.replace(token, value)
    return tex


def main():
    ap = argparse.ArgumentParser(description="Build the mu1 binding-function TikZ figure.")
    ap.add_argument("json", help="path to the ar1n binding-mu1 bundle JSON")
    ap.add_argument("-o", "--out", required=True, help="output .tex path")
    args = ap.parse_args()

    bundle = json.loads(Path(args.json).read_text())
    tex = build_tex(bundle, CONFIG)
    out = Path(args.out)
    out.write_text(tex)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
