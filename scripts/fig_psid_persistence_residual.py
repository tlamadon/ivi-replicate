#!/usr/bin/env python3
r"""Figure 10: Estimated Persistent and Transitory Components of Earnings in the PSID.

Input: output/bundles/bundle_psid_heteroscale.json
Left: rho(z_{t-1}, tau) = mu'(z) + sigma'(z) Phi^{-1}(tau);
right: simulated densities of E[e^a] e_t and e^a e_t.
Usage: python scripts/fig_psid_persistence_residual.py output/bundles/bundle_psid_heteroscale.json -o output/figures/tex/psid_persistence_residual.tex
"""
import argparse
import json
import sys
from math import asinh, sinh, exp
from statistics import NormalDist

import numpy as np

CAPTION = r"Estimated Persistent and Transitory Components of Earnings in the PSID"
LABEL = "psid_persistence_residual"

FIGURENOTE = (
    r"\vspace{0.2cm} \figurenote{Both panels are evaluated at the IVI estimate on the PSID. Panel (a) displays estimates of the state- and shock-dependent persistence measure $\rho(z_{t-1},\tau)$ defined in equation (\ref{eq:rho}); the horizontal axes indicate the percentile ranks of $z_{t-1}$ and of the innovation $u_t$, and the vertical axis reports the implied persistence $\rho(z_{t-1},\tau)$. Panel (b) plots the density of the individual-scaled transitory residual $e^{a}e_t$, and of $\mathbb{E}[e^{a}]e_t$ (deterministic scaling), against a variance-matched normal $\mathcal N(0,\mathrm{Var}(e^{a}e_t))$; the multiplicative scale heterogeneity makes $e^{a}e_t$ markedly leptokurtic. PSID, $N=741$, $T=10$.}"
)

P_LO, P_HI = 0.01, 0.99  # percentile window on both surface axes
N_Z, N_TAU = 11, 11      # grid resolution (state-percentile x tau)

NPTS = 500               # KDE grid points
N_SIM = 120000           # simulation draws
SIM_SEED = 20240701      # base RNG seed

# The two overlaid densities: (label, style, sampler-name).
CURVES = [
    (r"$\mathbb{E}[e^{a}]e_t$",  "blue!55!black, very thick, densely dotted", "et_scaled"),
    (r"$e^{a}e_t$",              "camo, very thick, densely dashed",          "resid"),
]
NORMAL_STYLE = "black!70, very thick"
NORMAL_LABEL = r"$\mathcal{N}(0,\mathrm{Var}(e^{a}e_t))$"

_N = NormalDist()        # standard normal


def softplus(x):
    x = np.asarray(x, float)
    return np.where(x > 30, x, np.log1p(np.exp(np.clip(x, -50, 30))))


def logistic(x):
    if x >= 0:
        return 1.0 / (1.0 + exp(-x))
    e = exp(x)
    return e / (1.0 + e)


# Persistence surface
def dmu(z, th):
    """mu'(z) = mu1 + 2 mu2 z."""
    return th["mu1"] + 2.0 * th["mu2"] * z


def dsigma(z, th):
    """sigma'(z) = logistic(q(z)) * (sigma1 + 2 sigma2 z), q = sigma0+sigma1 z+sigma2 z^2."""
    q = th["sigma0"] + th["sigma1"] * z + th["sigma2"] * z * z
    return logistic(q) * (th["sigma1"] + 2.0 * th["sigma2"] * z)


def persistence(z, tau, th):
    return dmu(z, th) + dsigma(z, th) * _N.inv_cdf(tau)


def z1_params(th):
    return (float(softplus(th["z1_log_std"])),   # scale
            float(th["z1_skew"]),                 # skew
            exp(th["z1_log_tail"]))               # tail


def z1_quantile(p, scale, skew, tail):
    """Inverse CDF of the z_1 sinh-arcsinh (loc=0): z = scale*sinh(tail*(asinh(U)+skew))."""
    u = _N.inv_cdf(p)
    return scale * sinh(tail * (asinh(u) + skew))


def z1_mean(scale, skew, tail):
    """E[z_1] by Gaussian-weighted quadrature over U ~ N(0,1)."""
    n, lo, hi = 4001, -10.0, 10.0
    step = (hi - lo) / (n - 1)
    s = w_tot = 0.0
    for i in range(n):
        u = lo + i * step
        w = exp(-0.5 * u * u)
        s += w * scale * sinh(tail * (asinh(u) + skew))
        w_tot += w
    return s / w_tot


def ivi_surface(th, p_grid, tau_grid):
    """rho on the (z-percentile, tau) grid; z-percentile maps to the demeaned z_1 quantile."""
    scale, skew, tail = z1_params(th)
    mean_z1 = z1_mean(scale, skew, tail)
    z_of_p = [z1_quantile(p, scale, skew, tail) - mean_z1 for p in p_grid]
    return [[persistence(z, tau, th) for tau in tau_grid] for z in z_of_p]


# Residual densities
def sinharcsinh_mean(scale, skew, tail):
    """E[X] for the sinh-arcsinh (loc=0), X = scale*sinh(tail(asinh(U)+skew))."""
    u = np.linspace(-10.0, 10.0, 8001)
    w = np.exp(-0.5 * u**2) / np.sqrt(2 * np.pi)
    x = scale * np.sinh(tail * (np.arcsinh(u) + skew))
    return float((getattr(np, "trapezoid", None) or np.trapz)(x * w, u))


def _sample_eps_centered(n, th, rng):
    """iid demeaned emission shocks eps = tilde eps - E[tilde eps]."""
    scale, skew, tail = 1.0, float(th["alpha_eps"]), float(np.exp(th["log_beta"]))
    u = rng.standard_normal(n)
    eps = scale * np.sinh(tail * (np.arcsinh(u) + skew))
    return eps - sinharcsinh_mean(scale, skew, tail)


def _sample_et(n, th, rng):
    """MA(1) composite e_t = eps_t + zeta eps_{t-1}, zeta = theta."""
    zeta = float(th["theta"])
    return _sample_eps_centered(n, th, rng) + zeta * _sample_eps_centered(n, th, rng)


def _sample_z1_centered(n, th, rng):
    """Demeaned latent state z1 (sinh-arcsinh)."""
    scale = float(softplus(th["z1_log_std"]))
    skew, tail = float(th["z1_skew"]), float(np.exp(th["z1_log_tail"]))
    v = rng.standard_normal(n)
    z = scale * np.sinh(tail * (np.arcsinh(v) + skew))
    return z - sinharcsinh_mean(scale, skew, tail)


def _sample_resid(n, th, rng):
    """exp(a) e_t, a ~ N(beta_a0+beta_a1 z1, sigma_a), z1 marginalized; e_t independent."""
    z1 = _sample_z1_centered(n, th, rng)
    sd = float(np.exp(th["log_sigma_a_cond"]))
    a = th["beta_a0"] + th["beta_a1"] * z1 + sd * rng.standard_normal(n)
    return np.exp(a) * _sample_et(n, th, rng)


def exp_a_mean(th):
    """E[e^a] = exp(beta_a0 + 0.5 sigma_a^2) * E_{z1}[exp(beta_a1 z1)] (z1 demeaned)."""
    scale = float(softplus(th["z1_log_std"]))
    skew, tail = float(th["z1_skew"]), float(np.exp(th["z1_log_tail"]))
    mean_z1 = sinharcsinh_mean(scale, skew, tail)
    b0, b1 = float(th["beta_a0"]), float(th["beta_a1"])
    sd = float(np.exp(th["log_sigma_a_cond"]))
    u = np.linspace(-10.0, 10.0, 8001)
    w = np.exp(-0.5 * u**2) / np.sqrt(2 * np.pi)
    z1 = scale * np.sinh(tail * (np.arcsinh(u) + skew)) - mean_z1
    e_z1 = float((getattr(np, "trapezoid", None) or np.trapz)(np.exp(b1 * z1) * w, u))
    return np.exp(b0 + 0.5 * sd * sd) * e_z1


def _sample_et_scaled(n, th, rng):
    """E[exp(a)] * e_t."""
    return exp_a_mean(th) * _sample_et(n, th, rng)


SAMPLERS = {"resid": _sample_resid, "et_scaled": _sample_et_scaled}


def _kde(samples, xs):
    """Gaussian-kernel density estimate (Silverman bandwidth) on grid xs."""
    s = np.asarray(samples, float)
    h = 1.06 * s.std() * s.size ** (-0.2)
    inv = 1.0 / (h * np.sqrt(2 * np.pi))
    xs = np.asarray(xs, float)
    out = np.empty(xs.size)
    for i in range(xs.size):
        out[i] = np.exp(-0.5 * ((xs[i] - s) / h) ** 2).mean() * inv
    return out


def die(msg):
    sys.stderr.write("fig_psid_persistence_residual.py: ERROR: " + msg + "\n")
    sys.exit(1)


_STRUCT_TO_FLAT = {
    "prior.net_mu.coeffs[1]": "mu1",
    "prior.net_mu.coeffs[2]": "mu2", "prior.net_sigma.coeffs[0]": "sigma0",
    "prior.net_sigma.coeffs[1]": "sigma1", "prior.net_sigma.coeffs[2]": "sigma2",
    "prior.z1_log_std": "z1_log_std", "prior.z1_skew": "z1_skew",
    "prior.z1_log_tail": "z1_log_tail", "prior.net_extra_mu.coeffs[0]": "beta_a0",
    "prior.net_extra_mu.coeffs[1]": "beta_a1",
    "prior.net_extra_logsigma.coeffs": "log_sigma_a_cond",
    "decoder.log_beta": "log_beta", "decoder.theta": "theta",
    "decoder.alpha_eps": "alpha_eps",
}


def load_ivi(path):
    with open(path) as f:
        doc = json.load(f)
    fit = (doc.get("fits") or {}).get("ivi_jn")
    if not isinstance(fit, dict) or not isinstance(fit.get("raw"), list):
        die("bundle JSON missing fits.ivi_jn.raw")
    th = {_STRUCT_TO_FLAT.get(r["name"], r["name"]): r["estimate"] for r in fit["raw"]}
    missing = [k for k in _STRUCT_TO_FLAT.values() if k not in th]
    if missing:
        die(f"fit 'ivi_jn' missing params {missing}")
    return th


def linspace(a, b, n):
    return [a + (b - a) * i / (n - 1) for i in range(n)]


def patch_means(R):
    """Mean of each faceted cell's 4 corners (a facet is coloured by that mean)."""
    return [0.25 * (R[i][j] + R[i + 1][j] + R[i][j + 1] + R[i + 1][j + 1])
            for i in range(len(R) - 1) for j in range(len(R[0]) - 1)]


def surf_coords(p_grid, tau_grid, R):
    rows = []
    for i, p in enumerate(p_grid):
        cells = [f"({p:.4f},{tau_grid[j]:.4f},{R[i][j]:.5f})"
                 for j in range(len(tau_grid))]
        rows.append("  " + " ".join(cells))
    return "\n\n".join(rows)


def _coords(xs, ys):
    return " ".join(f"({x:.6f},{y:.6f})" for x, y in zip(xs, ys))


def persistence_panel(th):
    p_grid = linspace(P_LO, P_HI, N_Z)
    tau_grid = linspace(P_LO, P_HI, N_TAU)
    R = ivi_surface(th, p_grid, tau_grid)
    # Colormap range = range of the facet (4-corner) means.
    cm = patch_means(R)
    cmin, cmax = min(cm), max(cm)
    coords = surf_coords(p_grid, tau_grid, R)
    return [
        r"\begin{tikzpicture}[baseline=(current bounding box.north)]",
        r"\begin{axis}[",
        r"  width=0.52\linewidth, height=8.6cm,",
        r"  title={(a) Nonlinear Persistence of $z_t$},",
        r"  title style={font=\normalsize},",
        r"  view={56}{30},",
        r"  enlargelimits=false,",
        r"  x dir=reverse,",
        r"  zmin=0, zmax=1.4, ztick={0,0.2,0.4,0.6,0.8,1.0,1.2,1.4},",
        r"  grid=both, grid style={gray!25},",
        r"  xlabel={percentile of $z_{t-1}$}, ylabel={percentile of shock $\tau$},",
        r"  zlabel={persistence $\rho$},",
        r"  xlabel style={sloped}, ylabel style={sloped},",
        r"  label style={font=\small}, tick label style={font=\footnotesize},",
        r"  xtick={0.1,0.3,0.5,0.7,0.9}, ytick={0.1,0.3,0.5,0.7,0.9},",
        r"  colormap name=rhomap,",
        rf"  point meta min={cmin:.4f}, point meta max={cmax:.4f},",
        r"]",
        rf"\addplot3[surf, shader=faceted, mesh/rows={N_Z}] coordinates {{",
        coords,
        r"};",
        r"\end{axis}",
        r"\end{tikzpicture}%",
    ]


def residual_panel(th):
    samples = {name: SAMPLERS[name](N_SIM, th, np.random.default_rng(SIM_SEED + i))
               for i, (_, _, name) in enumerate(CURVES)}
    xs = np.linspace(-1.05, 1.05, NPTS)

    plots = []
    for label, style, name in CURVES:
        plots.append(rf"  \addplot[{style}] coordinates {{{_coords(xs, _kde(samples[name], xs))}}};")
        plots.append(rf"  \addlegendentry{{{label}}}")
    # Mean-0 normal with the same variance as e^a e_t.
    sd = float(np.std(samples["resid"]))
    ys_norm = np.exp(-0.5 * (xs / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
    plots.append(rf"  \addplot[{NORMAL_STYLE}] coordinates {{{_coords(xs, ys_norm)}}};")
    plots.append(rf"  \addlegendentry{{{NORMAL_LABEL}}}")

    head = [
        r"\begin{tikzpicture}[baseline=(current bounding box.north)]",
        r"\begin{axis}[",
        r"  width=0.52\linewidth, height=7.2cm,",
        r"  title={(b) Residual Density vs.\ Normal},",
        r"  title style={font=\normalsize, yshift=18pt},",
        r"  no marks,",
        r"  enlarge x limits=false,",
        r"  xmin=-0.7, xmax=0.7, xtick={-0.5,0,0.5}, ymin=0, ymax=4.2,",
        r"  xlabel={residual},",
        r"  label style={font=\small}, tick label style={font=\footnotesize},",
        r"  legend cell align=left,",
        r"  legend style={at={(0.97,0.97)}, anchor=north east, font=\scriptsize,"
        r" draw=black, fill=white, row sep=-2pt, inner sep=2pt},",
        r"]",
    ]
    tail = [r"\end{axis}", r"\end{tikzpicture}%"]
    return head + plots + tail


def build_tex(th):
    tex = [
        r"\begin{figure}[!t]",
        r"\centering",
        # Turbo colormap (11 anchors, low -> high).
        r"\pgfplotsset{colormap={rhomap}{rgb255=(48,18,59) rgb255=(70,81,188)"
        r" rgb255=(84,134,234) rgb255=(62,186,226) rgb255=(40,224,168)"
        r" rgb255=(96,247,99) rgb255=(164,252,59) rgb255=(221,222,47)"
        r" rgb255=(253,165,49) rgb255=(235,84,24) rgb255=(122,4,3)}}",
        r"\resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}{%",
    ]
    tex += persistence_panel(th)
    tex += [r"\hspace{0.02\linewidth}%"]
    tex += residual_panel(th)
    tex += [
        r"}",
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
