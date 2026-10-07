#!/usr/bin/env python3
r"""Figure 11: Discounted Mixture Density in the PSID.

Input: output/bundles/bundle_psid_heteroscale.json
ftilde(z | z_1) = (1 - beta) sum_{t>=1} beta^{t-1} f_t(z | z_1), at the
first decile, median and last decile of z_1.
Usage: python scripts/fig_psid_mixture_density.py output/bundles/bundle_psid_heteroscale.json -o output/figures/tex/psid_mixture_density.tex
"""
import argparse
import json
import sys
from math import asinh, sinh
from statistics import NormalDist

import numpy as np

_N = NormalDist()
_trap = getattr(np, "trapezoid", None) or np.trapz

CAPTION = r"Discounted Mixture Density $\widetilde{f}$ in the PSID"
LABEL = "psid_mixture_density"

FIGURENOTE = (
    r"\vspace{0.2cm} \figurenote{The figure plots the discounted mixture density $\widetilde{f}(z\,|\,z_1)$ defined in equation~(\ref{eq:ftilde}) at the IVI estimate on the PSID (1980--1989), for $\beta=0.9$. Each curve conditions on a different initial persistent state $z_1$ of the estimated initial-state distribution $f_\alpha(z_1)$. The horizontal axis is the latent persistent component $z$; the vertical axis the discounted density. PSID, $N=741$, $T=10$.}"
)

BETA = 0.9                      # discount factor in the mixture weights
DECILES = (0.10, 0.50, 0.90)    # z_1 conditioning quantiles (first/median/last)
Z_LO, Z_HI, NZ = -3.5, 3.5, 2801   # propagation grid (wide; leakage negligible)
KMAX = 250                      # horizon truncation (beta^250 ~ 1e-11)
SIGMA_FLOOR = 1e-3
NPLOT = 361                     # points emitted per curve (subsampled to axis)

XLIM = (-1.5, 1.5)

# One entry per conditioning decile: (quantile, legend label, TikZ style).
CURVES = [
    (0.10, r"$z_1$: first decile", "blue!65!black, dotted, very thick"),
    (0.50, r"$z_1$: median",       "black, solid, very thick"),
    (0.90, r"$z_1$: last decile",  "orange!85!black, dashed, very thick"),
]


def softplus(x):
    x = np.asarray(x, float)
    return np.where(x > 30, x, np.log1p(np.exp(np.clip(x, -50, 30))))


def mu_fn(z, th):
    return th["mu0"] + th["mu1"] * z + th["mu2"] * np.asarray(z) ** 2


def sigma_fn(z, th):
    q = th["sigma0"] + th["sigma1"] * z + th["sigma2"] * np.asarray(z) ** 2
    return softplus(q) + SIGMA_FLOOR


def z1_quantile(p, th):
    """Inverse CDF of the z_1 sinh-arcsinh (loc=0):
    z = scale * sinh(tail * (asinh(U) + skew)), U = Phi^{-1}(p)."""
    scale = float(softplus(th["z1_log_std"]))
    skew = float(th["z1_skew"])
    tail = float(np.exp(th["z1_log_tail"]))
    u = _N.inv_cdf(p)
    return scale * sinh(tail * (asinh(u) + skew))


def _gauss(x, m, s):
    return np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))


def mixture_density(th):
    """Return (grid, {p: ftilde_on_grid}) for each conditioning decile p."""
    g = np.linspace(Z_LO, Z_HI, NZ)
    dz = g[1] - g[0]
    mu_j, sig_j = mu_fn(g, th), sigma_fn(g, th)
    # Grid transition kernel; off-grid leakage is fixed by renormalising the mixture.
    K = _gauss(g[:, None], mu_j[None, :], sig_j[None, :]) * dz

    out = {}
    for p in DECILES:
        z1 = z1_quantile(p, th)
        # f_1 = Normal(mu(z_1), sigma(z_1)^2)  (one step ahead of z_1)
        cur = _gauss(g, mu_fn(z1, th), sigma_fn(z1, th))
        cur = cur / _trap(cur, g)
        mix = np.zeros_like(g)
        w = 1.0 - BETA                       # weight (1-beta) beta^{t-1}, t=1,2,...
        for _ in range(KMAX):
            mix += w * cur
            cur = K @ cur
            w *= BETA
        mix /= _trap(mix, g)                 # renormalise to a proper density
        out[p] = mix
    return g, out


def die(msg):
    sys.stderr.write("fig_psid_mixture_density.py: ERROR: " + msg + "\n")
    sys.exit(1)


_STRUCT_TO_FLAT = {
    "prior.net_mu.coeffs[0]": "mu0", "prior.net_mu.coeffs[1]": "mu1",
    "prior.net_mu.coeffs[2]": "mu2", "prior.net_sigma.coeffs[0]": "sigma0",
    "prior.net_sigma.coeffs[1]": "sigma1", "prior.net_sigma.coeffs[2]": "sigma2",
    "prior.z1_log_std": "z1_log_std", "prior.z1_skew": "z1_skew",
    "prior.z1_log_tail": "z1_log_tail",
}


def load_ivi(path):
    """Flat {param: estimate} for the IVI cell ('ivi_jn') of the PSID bundle."""
    with open(path) as f:
        doc = json.load(f)
    fit = (doc.get("fits") or {}).get("ivi_jn")
    if not isinstance(fit, dict) or not isinstance(fit.get("raw"), list):
        die("bundle JSON missing fits.ivi_jn.raw")
    th = {_STRUCT_TO_FLAT.get(r["name"], r["name"]): r["estimate"] for r in fit["raw"]}
    if not th:
        die("bundle JSON ivi_jn has no raw parameters")
    return th


def _coords(xs, ys):
    return " ".join(f"({x:.5f},{y:.5f})" for x, y in zip(xs, ys))


def build_tex(th):
    g, mixes = mixture_density(th)

    xlo, xhi = XLIM
    xs = np.linspace(xlo, xhi, NPLOT)

    body = []
    for p, label, style in CURVES:
        ys = np.interp(xs, g, mixes[p])
        body.append(rf"  \addplot[{style}] coordinates {{{_coords(xs, ys)}}};")
        body.append(rf"  \addlegendentry{{{label}}}")

    tex = [
        r"\begin{figure}[!t]",
        r"\centering",
        r"\begin{tikzpicture}",
        r"\begin{axis}[",
        r"  width=0.72\linewidth, height=6.6cm,",
        r"  no marks, enlarge x limits=false,",
        rf"  xmin={xlo:g}, xmax={xhi:g}, ymin=0,",
        r"  xlabel={$z$}, ylabel={$\widetilde{f}(z\,|\,z_1)$},",
        r"  xlabel style={yshift=3pt}, ylabel style={yshift=-3pt},",
        r"  legend cell align=left,",
        r"  legend style={font=\small, draw=black, fill=white, at={(0.97,0.97)},"
        r" anchor=north east},",
        r"]",
    ]
    tex += body
    tex += [
        r"\end{axis}",
        r"\end{tikzpicture}",
        rf"\caption{{{CAPTION}}}",
        rf"\label{{{LABEL}}}",
        FIGURENOTE,
        r"\end{figure}",
    ]
    return "\n".join(tex) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("summary", help="path to bundle_psid_heteroscale.json")
    ap.add_argument("-o", dest="out", required=True, help="output .tex path")
    args = ap.parse_args(argv)

    tex = build_tex(load_ivi(args.summary))
    with open(args.out, "w") as f:
        f.write(tex)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
