r"""Figure 7: Simulation Results in the Nonlinear Model.

Input: output/bundles/hockeystick_parameter_summary.json
Usage: python scripts/fig_hockeystick_mu_sigma_z1_e.py output/bundles/hockeystick_parameter_summary.json -o output/figures/tex/hockeystick_mu_sigma_z1_e.tex
"""
import argparse
import json
import math
import sys

import numpy as np

CAPTION = "Simulation Results in the Nonlinear Model"
LABEL = "hockeystick_mu_sigma_z1_e"

FIGURENOTE = (
    r"\vspace{0.2cm} \figurenote{The figure plots the components of the simulated nonlinear DGP (solid black) together with their estimated counterparts under three variational-posterior specifications and the IVI correction of the unrestricted-Gaussian posterior. The top row shows the conditional mean $\mu(z_{t-1})$ and volatility $\sigma(z_{t-1})$ of the law of motion, as specified in (\ref{eq:nonlinear_mu})--(\ref{eq:nonlinear_sigma}); the bottom row shows the sinh-arcsinh densities of the initial state $z_1$ and the transitory shock $e_t$. Nonlinear DGP, $N=30{,}000$, $T=6$.}"
)

# Each cell selects exactly one fit by `sel` (+ `src` substring of source_file).
VI_CELLS = [
    {"key": "jn",
     "label": r"unrestr.\ Gaussian",
     "style": "camo, dashed",
     "sel": {"family": "vi", "method": "vi_only", "encoder": "jn"}},
    {"key": "ivijn",
     "label": r"unrestr.\ Gaussian (IVI)",
     "style": "teal!90!black, dashdotted",
     "sel": {"family": "ivi", "method": "picard_cold", "encoder": "jn"}},
    {"key": "mf",
     "label": r"diagonal",
     "style": "blue!55!black, densely dotted",
     "sel": {"family": "vi", "method": "vi_only", "encoder": "mean_field"},
     "src": "vi_meanfield_h64"},
    {"key": "sm",
     "label": r"hidden Markov",
     "style": "violet!55!white, densely dashed",
     "sel": {"family": "vi", "method": "vi_only", "encoder": "struct_markov"},
     "src": "vi_struct_markov_h64"},
]

Z_MIN, Z_MAX = -1.0, 1.0
Z1_MIN, Z1_MAX = -1.3, 1.3
E_MIN, E_MAX = -0.4, 0.4
N_GRID = 240


def _softplus(x):
    return np.where(x > 30, x, np.log1p(np.exp(np.clip(x, -50, 30))))


def _mu_fn(t, z):
    a0 = t["alpha0"]
    a1 = math.exp(t["log_alpha1"])
    mu_raw = t["mu0"] + t["mu1"] * z + t["mu2"] * z ** 2
    u = (mu_raw - a0) / a1
    return a0 + a1 * _softplus(u)


def _sigma_fn(t, z):
    s = t["sigma0"] + t["sigma1"] * z + t["sigma2"] * z ** 2
    return _softplus(s) + 1e-3


def _log_cosh(h):
    a = np.abs(h)
    return a + np.log1p(np.exp(-2 * a)) - np.log(2.0)


def sinharcsinh_pdf(x, loc, scale, skew, tail):
    """Sinh-arcsinh density."""
    w = (x - loc) / scale
    z = np.sinh(np.arcsinh(w) / tail - skew)          # inverse: y -> N(0,1)
    h = (np.arcsinh(z) + skew) * tail
    log_base = -0.5 * z ** 2 - 0.5 * np.log(2 * np.pi)
    log_det = -np.log(scale) - np.log(tail) - _log_cosh(h) + 0.5 * np.log1p(z ** 2)
    return np.exp(log_base + log_det)


def _f_z1(t, x):
    """Density of z_1: sinh-arcsinh, scale = softplus(z1_log_std)."""
    return sinharcsinh_pdf(x, loc=0.0,
                           scale=math.log1p(math.exp(t["z1_log_std"])),
                           skew=t["z1_skew"],
                           tail=math.exp(t["z1_log_tail"]))


def _f_e(t, x):
    """Density of e: sinh-arcsinh, scale = exp(log_sigma_eps), skew = 0."""
    return sinharcsinh_pdf(x, loc=0.0,
                           scale=math.exp(t["log_sigma_eps"]),
                           skew=0.0,
                           tail=math.exp(t["log_beta"]))


def die(msg):
    sys.stderr.write("fig_hockeystick_mu_sigma_z1_e.py: ERROR: " + msg + "\n")
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
    """The unique fit's `parameters` dict for `cell` (selector + optional src)."""
    sel = cell["sel"]
    desc = ", ".join(f"{k}={v}" for k, v in sorted(sel.items()))
    matches = [f for f in summary["fits"]
               if isinstance(f, dict) and all(f.get(k) == v for k, v in sel.items())]
    if "src" in cell:
        matches = [f for f in matches if cell["src"] in (f.get("source_file") or "")]
        desc += f", src~{cell['src']}"
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
    curves = [("DGP", "black, solid", truth)]
    for cell in VI_CELLS:
        curves.append((cell["label"], cell["style"], select_theta(summary, cell)))

    z = np.linspace(Z_MIN, Z_MAX, N_GRID)
    x_z1 = np.linspace(Z1_MIN, Z1_MAX, N_GRID)
    x_e = np.linspace(E_MIN, E_MAX, N_GRID)

    def panel(header, fn, xs, with_legend):
        lines = [header]
        for label, style, theta in curves:
            lines.append(rf"  \addplot[{style}, very thick] coordinates "
                         rf"{{{_coords(xs, fn(theta, xs))}}};")
            if with_legend:
                lines.append(rf"  \addlegendentry{{{label}}}")
        return lines

    tex = [
        r"\begin{figure}[!t]",
        r"\centering",
        r"\begin{tikzpicture}",
        r"\begin{groupplot}[",
        r"  group style={group size=2 by 2, horizontal sep=0.9cm,"
        r" vertical sep=1.6cm},",
        r"  width=0.53\linewidth, height=5.4cm,",
        r"  no marks,",
        r"  enlarge x limits=false,",
        r"  xtick distance=0.5,",
        r"  title style={yshift=-1.6ex, font=\normalsize},",
        r"  xlabel style={yshift=3pt},",
        r"  legend cell align=left,",
        r"  legend columns=1,",
        r"  legend style={font=\scriptsize, draw=black, fill=white,"
        r" at={(0.03,0.97)}, anchor=north west, row sep=-2pt,"
        r" inner sep=2pt},",
        r"]",
    ]
    tex += panel(
        rf"\nextgroupplot[title={{$\mu(z_{{t-1}})$}}, xlabel={{$z_{{t-1}}$}},"
        rf" xmin={Z_MIN}, xmax={Z_MAX}, ymax=1.2]",
        _mu_fn, z, with_legend=True)
    tex += panel(
        rf"\nextgroupplot[title={{$\sigma(z_{{t-1}})$}}, xlabel={{$z_{{t-1}}$}},"
        rf" xmin={Z_MIN}, xmax={Z_MAX}, ymin=0]",
        _sigma_fn, z, with_legend=False)
    tex += panel(
        rf"\nextgroupplot[title={{density of $z_1$}}, xlabel={{$z_1$}},"
        rf" xmin={Z1_MIN}, xmax={Z1_MAX}, ymin=0]",
        _f_z1, x_z1, with_legend=False)
    tex += panel(
        rf"\nextgroupplot[title={{density of $e_t$}}, xlabel={{$e_t$}},"
        rf" xmin={E_MIN}, xmax={E_MAX}, ymin=0, xtick distance=0.2]",
        _f_e, x_e, with_legend=False)
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
    ap.add_argument("summary", help="path to hockeystick_parameter_summary.json")
    ap.add_argument("-o", dest="out", required=True, help="output .tex path")
    args = ap.parse_args(argv)

    summary = load_summary(args.summary)
    tex = build_tex(summary)
    with open(args.out, "w") as f:
        f.write(tex)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()