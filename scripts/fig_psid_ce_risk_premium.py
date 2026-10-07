#!/usr/bin/env python3
r"""Figure 12: Certainty Equivalent and Risk Premium in the PSID.

Input: output/bundles/bundle_psid_heteroscale.json
c^CE = U^{-1}(E_ftilde[U(e^z)]),  pi = 1 - c^CE / E_ftilde[e^z],
ftilde = (1-beta)(I - beta K)^{-1} f_1;  quadratic, log and CRRA utility, c = e^z.
Usage: python scripts/fig_psid_ce_risk_premium.py output/bundles/bundle_psid_heteroscale.json -o output/figures/tex/psid_ce_risk_premium.tex
"""
import argparse
import json
import sys
from math import asinh, sinh
from statistics import NormalDist

import numpy as np

_N = NormalDist()
_trap = getattr(np, "trapezoid", None) or np.trapz

CAPTION = r"Certainty Equivalent and Risk Premium in the PSID"
LABEL = "psid_ce_risk_premium"

FIGURENOTE = (
    r"\vspace{0.2cm} \figurenote{The figure displays certainty equivalents (left) and risk premia (right) computed from PSID estimates of the nonlinear earnings model for $\beta=0.9$, using three alternative utility specifications: quadratic, logarithmic, and CRRA. Each panel reports values as a function of the initial latent state $z_1$, based on the estimated discounted mixture density $\widetilde{f}(z\,|\,z_1)$. PSID, $N=741$, $T=10$.}"
)

BETA = 0.9                      # discount factor in the mixture weights
QUAD_A = 0.1                    # quadratic utility U(c) = c - (QUAD_A/2) c^2
Z_LO, Z_HI, NZ = -5.0, 5.0, 2201   # propagation grid (wide; captures e^z tail)
P_LO, P_HI = 0.01, 0.99         # z_1 range = 1st--99th percentile
N_Z1 = 121                      # z_1 sample points along the x-axis

# One entry per utility: (key, legend label, TikZ style).
UTILITIES = [
    ("quad", r"quadratic",         "blue!65!black, densely dotted, very thick"),
    ("log",  r"logarithmic",       "black, solid, very thick"),
    ("crra", r"CRRA ($\gamma=2$)", "orange!85!black, dashed, very thick"),
]


def softplus(x):
    x = np.asarray(x, float)
    return np.where(x > 30, x, np.log1p(np.exp(np.clip(x, -50, 30))))


def mu_fn(z, th):
    return th["mu0"] + th["mu1"] * z + th["mu2"] * np.asarray(z) ** 2


def sigma_fn(z, th):
    q = th["sigma0"] + th["sigma1"] * z + th["sigma2"] * np.asarray(z) ** 2
    return softplus(q) + 1e-3


def z1_quantile(p, th):
    scale = float(softplus(th["z1_log_std"]))
    skew = float(th["z1_skew"])
    tail = float(np.exp(th["z1_log_tail"]))
    return scale * sinh(tail * (asinh(_N.inv_cdf(p)) + skew))


def _gauss(x, m, s):
    return np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))


def mixtures(th, z1s):
    """Return (grid, F) where F[:,j] = ftilde(. | z1s[j]) on the grid."""
    g = np.linspace(Z_LO, Z_HI, NZ)
    dz = g[1] - g[0]
    K = _gauss(g[:, None], mu_fn(g, th)[None, :], sigma_fn(g, th)[None, :]) * dz

    # f_1(z_1) = Normal(mu(z_1), sigma(z_1)^2) as columns, each normalised.
    F1 = _gauss(g[:, None], mu_fn(z1s, th)[None, :], sigma_fn(z1s, th)[None, :])
    F1 /= _trap(F1, g, axis=0)[None, :]

    # ftilde = (1-beta)(I - beta K)^{-1} f_1  -- one factorisation, all columns.
    X = np.linalg.solve(np.eye(NZ) - BETA * K, (1.0 - BETA) * F1)
    X /= _trap(X, g, axis=0)[None, :]        # renormalise (grid-leakage safety)
    return g, X


def ce_and_premium(th, z1s):
    """Return dict key -> (ce[array], premium_pct[array]) for each utility."""
    g, F = mixtures(th, z1s)
    ez, enz = np.exp(g), np.exp(-g)
    Ez = _trap(F * g[:, None], g, axis=0)
    Ec = _trap(F * ez[:, None], g, axis=0)
    Enz = _trap(F * enz[:, None], g, axis=0)
    Ec2 = _trap(F * (ez ** 2)[:, None], g, axis=0)
    Var = Ec2 - Ec ** 2

    # Quadratic U(c)=c-(a/2)c^2: c^CE solves c-(a/2)c^2 = E[U]=E[c]-(a/2)E[c^2];
    # increasing-branch (c<1/a) root. Discriminant = (1-a E[c])^2 + a^2 Var >= 0.
    a = QUAD_A
    disc = (1.0 - a * Ec) ** 2 + a ** 2 * Var
    ce = {
        "log":  np.exp(Ez),
        "crra": Enz ** (-1.0),                                   # gamma = 2
        "quad": (1.0 - np.sqrt(disc)) / a,
    }
    out = {}
    for key in ("quad", "log", "crra"):
        out[key] = (ce[key], 100.0 * (1.0 - ce[key] / Ec))
    return out


def die(msg):
    sys.stderr.write("fig_psid_ce_risk_premium.py: ERROR: " + msg + "\n")
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
    z0, z1max = z1_quantile(P_LO, th), z1_quantile(P_HI, th)
    z1s = np.linspace(z0, z1max, N_Z1)
    res = ce_and_premium(th, z1s)
    xlo, xhi = float(z1s[0]), float(z1s[-1])

    def panel_curves(idx, with_legend):
        lines = []
        for key, label, style in UTILITIES:
            ys = res[key][idx]
            lines.append(rf"  \addplot[{style}] coordinates {{{_coords(z1s, ys)}}};")
            if with_legend:
                lines.append(rf"  \addlegendentry{{{label}}}")
        return lines

    tex = [
        r"\begin{figure}[!t]",
        r"\centering",
        r"\begin{tikzpicture}",
        r"\begin{groupplot}[",
        r"  group style={group size=2 by 1, horizontal sep=1.1cm},",
        r"  width=0.5\linewidth, height=5.3cm,",
        r"  no marks, enlarge x limits=false,",
        rf"  xmin={xlo:.3f}, xmax={xhi:.3f},",
        r"  xlabel={$z_1$}, xlabel style={yshift=3pt},",
        r"  title style={font=\normalsize},",
        r"  legend cell align=left,",
        r"  legend style={at={(0.03,0.97)}, anchor=north west, font=\small,"
        r" draw=black, fill=white, row sep=-2pt, inner ysep=1pt},",
        r"]",
        r"\nextgroupplot[title={Certainty equivalent}, ymin=0]",
    ]
    tex += panel_curves(0, with_legend=True)
    tex += [
        r"\nextgroupplot[title={Risk premium (\%)}, ymin=0]",
    ]
    tex += panel_curves(1, with_legend=False)
    tex += [
        r"\end{groupplot}",
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
