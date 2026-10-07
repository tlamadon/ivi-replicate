#!/usr/bin/env python3
r"""Figure 9: Results on the PSID.

Input: output/bundles/bundle_psid_heteroscale.json
z_1 and eps are the sinh-arcsinh variables shifted to mean zero.
Usage: python scripts/fig_psid_a_sigma_z1_eps.py output/bundles/bundle_psid_heteroscale.json -o output/figures/tex/psid_a_sigma_z1_eps.tex
"""
import argparse
import json
import sys

import numpy as np

CAPTION = "Results on the PSID"
LABEL = "psid_a_sigma_z1_eps"

CI_LO, CI_HI = 2.5, 97.5                 # pointwise 95% band
BAND_FILL = r"fill=teal!70!black, draw=none, fill opacity=0.15"

FIGURENOTE = (
    r"\vspace{0.2cm} \figurenote{Estimated components of the nonlinear earnings model on the PSID (1980--1989) under two variational posteriors (unrestricted Gaussian, diagonal) and the IVI correction of the unrestricted-Gaussian posterior. Top row: the conditional scale density $p(a\,|\,z_1{=}0)$ and the volatility $\sigma(z_{t-1})$; bottom row: the densities of $z_1$ and the transitory shock $\varepsilon_t$. The shaded band around the IVI curve is a pointwise 95\% confidence interval from a nonparametric bootstrap that resamples households (100 replications). PSID, $N=741$, $T=10$.}"
)

NPTS         = 400
Z_GRID       = (-1.3, 1.3)    # sigma(z) grid; axis clips to [-1, 1]
Z1_GRID      = (-1.3, 1.3)    # z_1 (demeaned) grid; axis clips to [-1, 1]
EPS_GRID     = (-5.0, 5.0)    # eps (demeaned) grid; axis clips to [-4, 4]
ALPHA_PAD_SD = 4.0            # alpha grid half-width in conditional SDs
SIGMA_FLOOR  = 1e-3
Z1_COND      = 0.0            # condition the alpha density on z1 = 0

CELLS = [
    {"key": "vi_jn", "label": r"unrestr.\ Gaussian",
     "style": "camo, dashed, very thick"},
    {"key": "ivi",   "label": r"unrestr.\ Gaussian (IVI)",
     "style": "teal!90!black, dashdotted, very thick"},
    {"key": "vi_mf", "label": r"diagonal",
     "style": "blue!55!black, densely dotted, very thick"},
]


def softplus(x):
    x = np.asarray(x, float)
    return np.where(x > 30, x, np.log1p(np.exp(np.clip(x, -50, 30))))


def sinharcsinh_pdf(x, loc, scale, skew, tail):
    """SinhArcsinh density: pdf = phi(S) * S', S = sinh((1/tail) asinh((x-loc)/scale) - skew)."""
    x = np.asarray(x, float)
    w = (x - loc) / scale
    z = np.sinh(np.arcsinh(w) / tail - skew)
    h = (np.arcsinh(z) + skew) * tail
    log_base = -0.5 * z**2 - 0.5 * np.log(2 * np.pi)
    log_det  = -np.log(scale) - np.log(tail) - np.log(np.cosh(h)) + 0.5 * np.log1p(z**2)
    return np.exp(log_base + log_det)


def sigma_fn(z, th):
    q = th["sigma0"] + th["sigma1"] * z + th["sigma2"] * z**2
    return softplus(q) + SIGMA_FLOOR


def alpha_cond(alpha, th, z1):
    mu = th["beta_a0"] + th["beta_a1"] * z1
    sd = float(np.exp(th["log_sigma_a_cond"]))
    return np.exp(-0.5 * ((alpha - mu) / sd)**2) / (sd * np.sqrt(2 * np.pi))


def sinharcsinh_mean(scale, skew, tail):
    """E[scale * sinh(tail*(asinh(U)+skew))], U~N(0,1), by quadrature over U."""
    u = np.linspace(-10.0, 10.0, 8001)
    w = np.exp(-0.5 * u**2) / np.sqrt(2 * np.pi)
    x = scale * np.sinh(tail * (np.arcsinh(u) + skew))
    return float((getattr(np, "trapezoid", None) or np.trapz)(x * w, u))


def tilde_z1_pdf(x, th):
    return sinharcsinh_pdf(x, loc=0.0,
                           scale=float(softplus(th["z1_log_std"])),
                           skew=float(th["z1_skew"]),
                           tail=float(np.exp(th["z1_log_tail"])))


def tilde_eps_pdf(x, th):
    return sinharcsinh_pdf(x, loc=0.0,
                           scale=1.0,
                           skew=float(th["alpha_eps"]),
                           tail=float(np.exp(th["log_beta"])))


def z1_pdf(x, th):
    mu = sinharcsinh_mean(scale=float(softplus(th["z1_log_std"])),
                          skew=float(th["z1_skew"]),
                          tail=float(np.exp(th["z1_log_tail"])))
    return tilde_z1_pdf(np.asarray(x, float) + mu, th)


def eps_pdf(x, th):
    mu = sinharcsinh_mean(scale=1.0,
                          skew=float(th["alpha_eps"]),
                          tail=float(np.exp(th["log_beta"])))
    return tilde_eps_pdf(np.asarray(x, float) + mu, th)


def die(msg):
    sys.stderr.write("fig_psid_a_sigma_z1_eps.py: ERROR: " + msg + "\n")
    sys.exit(1)


_STRUCT_TO_FLAT = {
    "prior.net_sigma.coeffs[0]": "sigma0",
    "prior.net_sigma.coeffs[1]": "sigma1", "prior.net_sigma.coeffs[2]": "sigma2",
    "prior.z1_log_std": "z1_log_std", "prior.z1_skew": "z1_skew",
    "prior.z1_log_tail": "z1_log_tail", "prior.net_extra_mu.coeffs[0]": "beta_a0",
    "prior.net_extra_mu.coeffs[1]": "beta_a1",
    "prior.net_extra_logsigma.coeffs": "log_sigma_a_cond",
    "decoder.log_beta": "log_beta", "decoder.alpha_eps": "alpha_eps",
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


def load_bootstrap(path):
    """Bootstrap replicates (list of {param name: value}) from bootstrap.unconditional."""
    with open(path) as f:
        doc = json.load(f)
    b = (doc.get("bootstrap") or {}).get("unconditional")
    if not isinstance(b, dict):
        die(f"bundle {path!r} has no bootstrap.unconditional block")
    names = b.get("param_names")
    reps = b.get("replicates")
    if not (isinstance(names, list) and isinstance(reps, list) and reps):
        die(f"bundle {path!r} bootstrap.unconditional missing param_names / replicates")
    return [dict(zip(names, r)) for r in reps]


def _coords(xs, ys):
    return " ".join(f"({x:.6f},{y:.6f})" for x, y in zip(xs, ys))


def build_tex(by_method, reps):
    curves = []
    for cell in CELLS:
        if cell["key"] not in by_method:
            die(f"summary JSON has no method '{cell['key']}'")
        curves.append((cell["label"], cell["style"], by_method[cell["key"]]))

    def ivi_band(fn, xs):
        """Pointwise percentiles of the replicate curves; lower edge clipped at 0."""
        mat = np.array([np.asarray(fn(th, xs), float) for th in reps])
        ok = np.isfinite(mat).all(axis=1)
        if not ok.all():  # only seen with smoke-test bootstraps
            sys.stderr.write(f"fig_psid_a_sigma_z1_eps.py: WARNING: {int((~ok).sum())} "
                             "non-finite bootstrap replicate curve(s) skipped\n")
            mat = mat[ok]
        lo_q = np.percentile(mat, CI_LO, axis=0)
        hi_q = np.percentile(mat, CI_HI, axis=0)
        return np.clip(lo_q, 0.0, None), hi_q

    z_grid    = np.linspace(*Z_GRID,   NPTS)
    z1_grid   = np.linspace(*Z1_GRID,  NPTS)
    eps_grid  = np.linspace(*EPS_GRID, NPTS)

    # alpha grid spans all cells' conditional means (at z1=0) +/- ALPHA_PAD_SD*max sd.
    means, sds = [], []
    for _, _, th in curves:
        sds.append(float(np.exp(th["log_sigma_a_cond"])))
        means.append(th["beta_a0"] + th["beta_a1"] * Z1_COND)
    xa_grid = np.linspace(min(means) - ALPHA_PAD_SD * max(sds),
                          max(means) + ALPHA_PAD_SD * max(sds), NPTS)

    def panel(header, fn, xs, with_legend, band_id):
        lines = [header]
        # IVI band first so the posterior curves overlay it.
        lo, hi = ivi_band(fn, xs)
        lines.append(rf"  \addplot[draw=none, forget plot, name path={band_id}lo]"
                     rf" coordinates {{{_coords(xs, lo)}}};")
        lines.append(rf"  \addplot[draw=none, forget plot, name path={band_id}hi]"
                     rf" coordinates {{{_coords(xs, hi)}}};")
        lines.append(rf"  \addplot[{BAND_FILL}, forget plot]"
                     rf" fill between[of={band_id}lo and {band_id}hi];")
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
        r" xmin=-4, xmax=-1, xtick={-4,-3,-2,-1}, ymin=0, legend to name=grouplegend]",
        lambda th, xs: alpha_cond(xs, th, Z1_COND), xa_grid, with_legend=True,
        band_id="bandA")
    tex += panel(
        r"\nextgroupplot[title={$\sigma(z_{t-1})$}, xlabel={$z_{t-1}$},"
        r" xmin=-1.0, xmax=1.0, xtick={-1,-0.5,0,0.5,1}, ymin=0]",
        lambda th, xs: sigma_fn(xs, th), z_grid, with_legend=False,
        band_id="bandB")
    tex += panel(
        r"\nextgroupplot[title={density of $z_1$}, xlabel={$z_1$},"
        r" xmin=-1.0, xmax=1.0, xtick={-1,-0.5,0,0.5,1}, ymin=0]",
        lambda th, xs: z1_pdf(xs, th), z1_grid, with_legend=False,
        band_id="bandC")
    tex += panel(
        r"\nextgroupplot[title={density of $\varepsilon_t$}, xlabel={$\varepsilon_t$},"
        r" xmin=-4.0, xmax=4.0, xtick={-4,-2,0,2,4}, ymin=0]",
        lambda th, xs: eps_pdf(xs, th), eps_grid, with_legend=False,
        band_id="bandD")
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
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("summary", help="path to bundle_psid_heteroscale.json")
    ap.add_argument("-o", dest="out", required=True, help="output .tex path")
    args = ap.parse_args(argv)

    tex = build_tex(load_methods(args.summary), load_bootstrap(args.summary))
    with open(args.out, "w") as f:
        f.write(tex)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
