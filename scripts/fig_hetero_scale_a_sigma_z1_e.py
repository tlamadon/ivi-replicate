#!/usr/bin/env python3
r"""Figure 8: Simulation Results in the Heterogeneity Model.

Input: output/bundles/hetero_scale_parameter_summary.json
Usage: python scripts/fig_hetero_scale_a_sigma_z1_e.py output/bundles/hetero_scale_parameter_summary.json -o output/figures/tex/hetero_scale_a_sigma_z1_e.tex
"""
import argparse
import json
import sys
from statistics import NormalDist

import numpy as np

CAPTION = "Simulation Results in the Heterogeneity Model"
LABEL = "hetero_scale_a_sigma_z1_e"

FIGURENOTE = (
    r"\vspace{0.2cm} \figurenote{The figure plots the components of the simulated heterogeneity-model DGP (solid black) together with their estimated counterparts under two variational-posterior specifications and the IVI correction of the unrestricted-Gaussian posterior. Top row: the conditional density of the individual scale $p(a\,|\,z_1{=}0)$ and the conditional volatility $\sigma(z_{t-1})$; bottom row: the densities of the initial state $z_1$ and the transitory shock $e_t$. Nonlinear DGP with heterogeneity, $N=30{,}000$, $T=6$.}"
)

NPTS       = 400                      # points per x-grid
Z_GRID     = (-1.3, 1.3)             # sigma(z) x-grid (axis clips to [-1,1])
XZ1_GRID   = (-1.6, 1.6)             # p(z1)   x-grid (axis clips to [-1.3,1.3])
XE_GRID    = (-4.5, 4.5)             # p(e)    x-grid (axis clips to [-4,4])
ALPHA_PAD_SD = 4.0                    # alpha grid half-width in conditional SDs
SIGMA_FLOOR  = 1e-3                   # sigma(z) = softplus(...) + 1e-3

# Overlay order after the DGP; each selector must match exactly one fit.
CELLS = [
    {"key": "jn",
     "label": r"unrestr.\ Gaussian",
     "style": "camo, dashed, very thick",
     "sel": {"family": "vi", "method": "vi_only", "encoder": "jn"}},
    {"key": "ivijn",
     "label": r"unrestr.\ Gaussian (IVI)",
     "style": "teal!90!black, dashdotted, very thick",
     "sel": {"family": "ivi", "method": "picard_cold", "encoder": "jn"}},
    {"key": "mf",
     "label": r"diagonal",
     "style": "blue!55!black, densely dotted, very thick",
     "sel": {"family": "vi", "method": "vi_only", "encoder": "mean_field"}},
]


def softplus(x):
    x = np.asarray(x, float)
    return np.where(x > 30, x, np.log1p(np.exp(np.clip(x, -50, 30))))


def sinharcsinh_pdf(x, loc, scale, skew, tail):
    """SinhArcsinh density."""
    x = np.asarray(x, float)
    w = (x - loc) / scale
    z = np.sinh(np.arcsinh(w) / tail - skew)
    h = (np.arcsinh(z) + skew) * tail
    log_base = -0.5 * z**2 - 0.5 * np.log(2 * np.pi)
    log_det  = -np.log(scale) - np.log(tail) - np.log(np.cosh(h)) + 0.5 * np.log1p(z**2)
    return np.exp(log_base + log_det)


def sinharcsinh_quantile(p, loc, scale, skew, tail):
    """Inverse CDF: standard-normal quantile pushed through the forward map."""
    znorm = NormalDist().inv_cdf(p)
    return loc + scale * np.sinh(tail * (np.arcsinh(znorm) + skew))


def sigma_fn(z, th):
    q = th["sigma0"] + th["sigma1"] * z + th["sigma2"] * z**2
    return softplus(q) + SIGMA_FLOOR


def z1_params(th):
    return dict(loc=0.0,
                scale=float(softplus(th["z1_log_std"])),
                skew=float(th.get("z1_skew", 0.0)),   # pinned at 0; absent from theta
                tail=float(np.exp(th["z1_log_tail"])))


def e_tail(th):
    return float(np.exp(th["log_beta"]))


def alpha_cond(alpha, th, z1):
    mu = th["beta_a0"] + th["beta_a1"] * z1
    sd = float(np.exp(th["log_sigma_a_cond"]))
    return np.exp(-0.5 * ((alpha - mu) / sd)**2) / (sd * np.sqrt(2 * np.pi))


def die(msg):
    sys.stderr.write("fig_hetero_scale_a_sigma_z1_e.py: ERROR: " + msg + "\n")
    sys.exit(1)


def load_summary(path):
    with open(path) as f:
        doc = json.load(f)
    if not isinstance(doc.get("truth"), dict) or "parameters" not in doc["truth"]:
        die("summary JSON missing truth.parameters")
    if not isinstance(doc.get("fits"), list) or not doc["fits"]:
        die("summary JSON missing non-empty 'fits' list")
    return doc


def select_theta(summary, cell):
    sel = cell["sel"]
    desc = ", ".join(f"{k}={v}" for k, v in sorted(sel.items()))
    matches = [f for f in summary["fits"]
               if isinstance(f, dict) and all(f.get(k) == v for k, v in sel.items())]
    if len(matches) != 1:
        die(f"cell '{cell['key']}' selector [{desc}] matched {len(matches)} fits "
            f"(expected 1): {[m.get('label') for m in matches]}")
    params = matches[0].get("parameters")
    if not isinstance(params, dict):
        die(f"cell '{cell['key']}' fit has no 'parameters' object")
    return params


def _coords(xs, ys):
    return " ".join(f"({x:.6f},{y:.6f})" for x, y in zip(xs, ys))


def build_tex(summary):
    truth = summary["truth"]["parameters"]
    curves = [("DGP", "black, solid, very thick", truth)]
    for cell in CELLS:
        curves.append((cell["label"], cell["style"], select_theta(summary, cell)))

    # z1 conditioning point = DGP z1-density median (q50); title says z1=0.
    zp = z1_params(truth)
    z1_med = float(sinharcsinh_quantile(0.50, **zp))

    z_grid   = np.linspace(*Z_GRID,   NPTS)
    xz1_grid = np.linspace(*XZ1_GRID, NPTS)
    xe_grid  = np.linspace(*XE_GRID,  NPTS)

    # alpha grid spans all cells' conditional means +/- ALPHA_PAD_SD * max sd.
    means, sds = [], []
    for _, _, th in curves:
        sds.append(float(np.exp(th["log_sigma_a_cond"])))
        means.append(th["beta_a0"] + th["beta_a1"] * z1_med)
    xa_grid = np.linspace(min(means) - ALPHA_PAD_SD * max(sds),
                          max(means) + ALPHA_PAD_SD * max(sds), NPTS)

    def panel(header, fn, xs, with_legend):
        lines = [header]
        for label, style, th in curves:
            line = rf"  \addplot[{style}] coordinates {{{_coords(xs, fn(th, xs))}}};"
            if with_legend:
                line += rf" \addlegendentry{{{label}}}"
            lines.append(line)
        return lines

    tex = [
        r"\begin{figure}[!t]",
        r"\centering",
        r"\begin{tikzpicture}",
        r"\begin{groupplot}[",
        r"  group style={group size=2 by 2, horizontal sep=0.9cm, vertical sep=1.6cm},",
        r"  width=0.53\linewidth, height=5.4cm,",
        r"  no marks,",
        r"  enlarge x limits=false,",
        r"  title style={yshift=-1.6ex, font=\normalsize},",
        r"  xlabel style={yshift=3pt},",
        r"  legend cell align=left,",
        r"  legend columns=-1,",
        r"  legend style={font=\scriptsize, draw=black, fill=white,"
        r" /tikz/every even column/.append style={column sep=8pt}},",
        r"]",
    ]
    tex += panel(
        r"\nextgroupplot[title={$p(a\,|\,z_1{=}0)$}, xlabel={$a$},"
        r" xtick={-3.5,-3,-2.5,-2,-1.5,-1}, ymin=0,"
        r" legend to name=grouplegend]",
        lambda th, xs: alpha_cond(xs, th, z1_med), xa_grid, with_legend=True)
    tex += panel(
        r"\nextgroupplot[title={$\sigma(z_{t-1})$}, xlabel={$z_{t-1}$},"
        r" xmin=-1.0, xmax=1.0, xtick={-1,-0.5,0,0.5,1}, ymin=0]",
        lambda th, xs: sigma_fn(xs, th), z_grid, with_legend=False)
    tex += panel(
        r"\nextgroupplot[title={density of $z_1$}, xlabel={$z_1$},"
        r" xmin=-1.3, xmax=1.3, xtick={-1,-0.5,0,0.5,1}, ymin=0]",
        lambda th, xs: sinharcsinh_pdf(xs, **z1_params(th)), xz1_grid, with_legend=False)
    tex += panel(
        r"\nextgroupplot[title={density of $e_t$}, xlabel={$e_t$},"
        r" xmin=-4.0, xmax=4.0, xtick={-4,-2,0,2,4}, ymin=0]",
        lambda th, xs: sinharcsinh_pdf(xs, 0.0, 1.0, 0.0, e_tail(th)), xe_grid,
        with_legend=False)
    tex += [
        r"\end{groupplot}",
        r"\end{tikzpicture}",
        r"",
        r"\vspace{1mm}",
        r"\pgfplotslegendfromname{grouplegend}",
        rf"\caption{{{CAPTION}}}",
        rf"\label{{{LABEL}}}",
        FIGURENOTE,
        r"\end{figure}",
    ]
    return "\n".join(tex) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("summary")
    ap.add_argument("-o", dest="out", required=True)
    args = ap.parse_args(argv)

    summary = load_summary(args.summary)
    tex = build_tex(summary)
    with open(args.out, "w") as f:
        f.write(tex)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()