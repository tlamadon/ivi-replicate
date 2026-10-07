#!/usr/bin/env python3
r"""Figure 4: Log-Likelihood, Amortization Gap and Family (Mean-Field) Gap.

Input: output/bundles/ar1n_mu1_profile_bundle.json
Usage: python scripts/fig_ar1n_likelihood_elbo.py --json output/bundles/ar1n_mu1_profile_bundle.json --out output/figures/tex/ar1n_likelihood_elbo.tex
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Bundle:
    t: np.ndarray
    rho: np.ndarray
    logp: np.ndarray
    amort: np.ndarray
    nonamort: np.ndarray
    rho_truth: float
    rho_vi: float
    t_truth: float
    t_vi: float


def load_bundle(path: str) -> Bundle:
    obj = json.loads(Path(path).read_text())
    prof = obj["profile"]
    t = np.asarray([c["t"] for c in prof], float)
    order = np.argsort(t)                       # ensure ascending t for interp
    t = t[order]

    def col(key):
        return np.asarray([prof[i][key] for i in order], float)

    def mu1():
        return np.asarray([prof[i]["parameters"]["mu1"] for i in order], float)

    b = Bundle(
        t=t,
        rho=mu1(),
        logp=col("logp_exact"),
        amort=col("elbo_mf_amort"),
        nonamort=col("elbo_mf_nonamort"),
        rho_truth=float(obj["reference_points"]["truth"]["parameters"]["mu1"]),
        rho_vi=float(obj["reference_points"]["vi_obs"]["parameters"]["mu1"]),
        t_truth=float(obj["reference_points"]["truth"]["t"]),
        t_vi=float(obj["reference_points"]["vi_obs"]["t"]),
    )
    _validate(b)
    return b


def _validate(b: Bundle) -> None:
    n = len(b.t)
    for name, arr in [("rho", b.rho), ("logp", b.logp),
                      ("amort", b.amort), ("nonamort", b.nonamort)]:
        if len(arr) != n:
            raise ValueError(f"profile column '{name}' length {len(arr)} != {n}")



@dataclass(frozen=True)
class Style:
    # colours (clogp, cnon, camo, cfam, camg, cgrid) are defined in the preamble
    # axis box
    width_cm:  float = 12.6
    height_cm: float = 7.5

    # line widths / dash patterns
    lw_logp: float = 2.2
    lw_elbo: float = 1.7
    dash_amo:  str = "on 3pt off 1.5pt on 1pt off 1.5pt"   # dash-dot
    dash_non:  str = "on 5pt off 2pt"                       # dashed
    lw_vline:  float = 0.9
    dash_vline: str = "on 4pt off 3pt"

    # extremum markers
    mk_truth: str = "triangle*"
    mk_truth_size: float = 4.0
    mk_vi: str = "diamond*"
    mk_vi_size: float = 4.5
    mk_edge_lw: float = 0.7

    # gap-band opacities
    fam_opacity: float = 0.50
    amg_opacity: float = 0.85

    # axis ranges (None -> data-driven)
    x_lower: float | None = 0.6
    x_upper: float | None = None
    x_pad:   float = 0.012        # padding used when a bound is data-driven
    y_pad_lo: float = 0.06        # bottom headroom as fraction of curve span
    y_pad_hi: float = 0.18        # top headroom (room for the inside labels)
    vtop_frac: float = 0.085      # vertical-line top above yhi (frac of span)
    laby_frac: float = 0.105      # label baseline above yhi (frac of span)
    xtick_distance: float = 0.05
    ytick_distance: float = 0.5

    # text
    title: str = ""
    caption: str = ("Log-Likelihood, Amortization Gap and "
                    "Family (Mean-Field) Gap")
    label: str = "ar1n_likelihood_elbo"
    font_axis:  str = r"\normalsize"   # title, labels, ticks
    font_small: str = r"\small"        # legend + truth/VI labels

    # legend
    legend_at: str = "(0.015,0.015)"
    legend_anchor: str = "south west"

    # labels on the reference verticals
    label_truth: str = "truth"
    label_vi: str = "VI"
    # curve legend entries
    leg_logp: str = r"$\log p(y|\vartheta)$"
    leg_amo:  str = r"amortized ELBO"
    leg_non:  str = r"non-amortized ELBO"
    leg_fam:  str = r"family (mean-field) gap"
    leg_amg:  str = r"amortization gap"


@dataclass
class Layout:
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    vtop: float
    laby: float
    y_truth: float   # marker y on the log p curve at rho_truth
    y_vi: float      # marker y on the amortized ELBO at rho_vi


def compute_layout(b: Bundle, s: Style) -> Layout:
    ylo = float(min(b.logp.min(), b.amort.min(), b.nonamort.min()))
    yhi = float(max(b.logp.max(), b.amort.max(), b.nonamort.max()))
    span = yhi - ylo

    lo, hi = s.x_lower, s.x_upper
    xmin_v = (b.rho.min() - s.x_pad) if lo is None else lo
    xmax_v = (b.rho.max() + s.x_pad) if hi is None else hi

    # markers: interpolate the curve at the reference t (= exact polyline value)
    y_truth = float(np.interp(b.t_truth, b.t, b.logp))
    y_vi = float(np.interp(b.t_vi, b.t, b.amort))

    return Layout(
        xmin=xmin_v, xmax=xmax_v,
        ymin=ylo - s.y_pad_lo * span,
        ymax=yhi + s.y_pad_hi * span,
        vtop=yhi + s.vtop_frac * span,
        laby=yhi + s.laby_frac * span,
        y_truth=y_truth, y_vi=y_vi,
    )


def _coords(x, y) -> str:
    return " ".join(f"({xi:.6f},{yi:.6f})" for xi, yi in zip(x, y))


def build_body(b: Bundle, s: Style, L: Layout) -> str:
    rt, rv = b.rho_truth, b.rho_vi
    C = _coords
    return rf"""\begin{{tikzpicture}}
\begin{{axis}}[
    scale only axis,
    width={s.width_cm}cm, height={s.height_cm}cm,
    axis lines=left,
    axis line style={{draw=black!55}},
    title={{{s.title}}},
    title style={{font={s.font_axis}, yshift=2pt}},
    label style={{font={s.font_axis}}},
    tick label style={{font={s.font_axis}}},
    xlabel={{$\rho$}},
    xlabel style={{yshift=2pt}},
    xmin={L.xmin:.4f}, xmax={L.xmax:.4f},
    ymin={L.ymin:.4f}, ymax={L.ymax:.4f},
    xtick distance={s.xtick_distance},
    ytick distance={s.ytick_distance},
    tick align=outside, tick style={{draw=black!45}},
    grid=both,
    major grid style={{draw=cgrid, line width=0.6pt}},
    minor tick num=0,
    clip=true,
    legend cell align=left,
    legend style={{
        at={{{s.legend_at}}}, anchor={s.legend_anchor},
        draw=black!40, fill=white, fill opacity=0.95, text opacity=1,
        line width=0.6pt, font={s.font_small}, row sep=1pt,
    }},
]

\addplot[name path=plogp, draw=none, forget plot] coordinates {{{C(b.rho, b.logp)}}};
\addplot[name path=pnon,  draw=none, forget plot] coordinates {{{C(b.rho, b.nonamort)}}};
\addplot[name path=pamo,  draw=none, forget plot] coordinates {{{C(b.rho, b.amort)}}};
\addplot[cfam, opacity={s.fam_opacity}, forget plot] fill between[of=plogp and pnon];
\addplot[camg, opacity={s.amg_opacity}, forget plot] fill between[of=pnon and pamo];

\draw[clogp, dash pattern={s.dash_vline}, line width={s.lw_vline}pt]
    (axis cs:{rt:.6f},{L.ymin:.4f}) -- (axis cs:{rt:.6f},{L.vtop:.4f});
\draw[camo, dash pattern={s.dash_vline}, line width={s.lw_vline}pt]
    (axis cs:{rv:.6f},{L.ymin:.4f}) -- (axis cs:{rv:.6f},{L.vtop:.4f});
\node[clogp, font={s.font_small}\bfseries, anchor=south]
    at (axis cs:{rt:.6f},{L.laby:.4f}) {{{s.label_truth}}};
\node[camo, font={s.font_small}\bfseries, anchor=south]
    at (axis cs:{rv:.6f},{L.laby:.4f}) {{{s.label_vi}}};

\addplot[clogp, line width={s.lw_logp}pt, solid, forget plot]
    coordinates {{{C(b.rho, b.logp)}}};
\addplot[cnon, line width={s.lw_elbo}pt, dash pattern={s.dash_non}, forget plot]
    coordinates {{{C(b.rho, b.nonamort)}}};
\addplot[camo, line width={s.lw_elbo}pt, dash pattern={s.dash_amo}, forget plot]
    coordinates {{{C(b.rho, b.amort)}}};

\addplot[only marks, mark={s.mk_truth}, mark size={s.mk_truth_size}pt,
    mark options={{fill=clogp, draw=clogp, line width={s.mk_edge_lw}pt}}, forget plot]
    coordinates {{({rt:.6f},{L.y_truth:.6f})}};
\addplot[only marks, mark={s.mk_vi}, mark size={s.mk_vi_size}pt,
    mark options={{fill=camo, draw=camo, line width={s.mk_edge_lw}pt}}, forget plot]
    coordinates {{({rv:.6f},{L.y_vi:.6f})}};

\addlegendimage{{clogp, line width={s.lw_logp}pt, solid}}
\addlegendentry{{{s.leg_logp}}}
\addlegendimage{{camo, line width={s.lw_elbo}pt, dash pattern={s.dash_amo}}}
\addlegendentry{{{s.leg_amo}}}
\addlegendimage{{cnon, line width={s.lw_elbo}pt, dash pattern={s.dash_non}}}
\addlegendentry{{{s.leg_non}}}
\addlegendimage{{area legend, fill=cfam, fill opacity={s.fam_opacity}, draw=none}}
\addlegendentry{{{s.leg_fam}}}
\addlegendimage{{area legend, fill=camg, fill opacity={s.amg_opacity}, draw=none}}
\addlegendentry{{{s.leg_amg}}}

\end{{axis}}
\end{{tikzpicture}}"""


FIGURENOTE = (
    r"\vspace{0.2cm} \figurenote{Log-likelihood (solid blue), amortized (orange dash-dotted) and non-amortized (teal dashed) mean-field ELBO along the line $\vartheta(\iota)=\vartheta_0+\iota(\widehat\vartheta^{\rm VI}-\vartheta_0)$ from the DGP truth to the VI optimum, plotted against $\rho$. The log-likelihood peaks near the truth $\rho_0=0.90$, the amortized ELBO near the VI pseudo-true value $\overline{\rho}_0=0.76$. Purple: amortization gap (non-amortized minus amortized ELBO); gray: family gap (log-likelihood minus non-amortized ELBO). Linear Gaussian DGP, $N = 30{,}000$, $T = 6$.}"
)


def wrap_fragment(body: str, s: Style) -> str:
    return rf"""\begin{{figure}}[!t]
\centering
{body}
\caption{{{s.caption}}}
\label{{{s.label}}}
{FIGURENOTE}
\end{{figure}}
"""


def main():
    ap = argparse.ArgumentParser(description="Log-likelihood / ELBO profile figure.")
    ap.add_argument("--json", required=True, help="path to the profile bundle JSON")
    ap.add_argument("--out", required=True, help="output .tex path")
    a = ap.parse_args()

    s = Style()
    b = load_bundle(a.json)
    L = compute_layout(b, s)
    tex = wrap_fragment(build_body(b, s, L), s)

    out = Path(a.out)
    out.write_text(tex)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
