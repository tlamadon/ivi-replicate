#!/usr/bin/env python3
r"""Figure 6: Variational Approximation to the True Posterior Density.

Input: output/bundles/contour_bundle.json
Usage: python scripts/fig_hockeystick_posteriors.py output/bundles/contour_bundle.json -o output/figures/tex/hockeystick_posteriors.tex
"""
import argparse
import json
import sys

CAPTION = r"Variational Approximation to the True Posterior Density $p_{\vartheta}(z_1,z_2|y_{1:6})$"
LABEL = "hockeystick_posteriors"

FIGURENOTE = (
    r"\vspace{0.2cm} \figurenote{Posterior over the first two latent states $(z_1,z_2)$ given $y_{1:6}=(-0.1,0.1,0.1,0.1,0.1,0.1)$. The true posterior (left) versus the unrestricted-Gaussian (center) and diagonal mean-field (right) variational approximations $q$. Contours are density levels; gray lines show each covariance's principal axes and the black \textbf{+} is the observation $(y_1,y_2)$. Nonlinear DGP, $T = 6$.}"
)

LEVELS = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]

# Visible axis half-window (square); contours are clipped to this box.
AXLIM = 0.4

# viridis colour anchors (position in [0,1] -> RGB 0-255); colour = viridis(level)
VIRIDIS_ANCHORS = [
    (0.00, (68, 1, 84)),
    (0.25, (59, 82, 139)),
    (0.50, (33, 144, 140)),
    (0.75, (93, 201, 99)),
    (1.00, (253, 231, 37)),
]


def viridis(t):
    """RGB (0-255 ints) at position t in [0,1] along the viridis ramp."""
    t = max(0.0, min(1.0, t))
    for k in range(len(VIRIDIS_ANCHORS) - 1):
        t0, c0 = VIRIDIS_ANCHORS[k]
        t1, c1 = VIRIDIS_ANCHORS[k + 1]
        if t <= t1:
            f = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
            return tuple(round(c0[i] + f * (c1[i] - c0[i])) for i in range(3))
    return VIRIDIS_ANCHORS[-1][1]


def die(msg):
    sys.stderr.write("fig_hockeystick_posteriors.py: ERROR: " + msg + "\n")
    sys.exit(1)


# Marching squares: per-cell case -> list of (edgeA, edgeB) crossings to connect. Edges:
# 0=bottom, 1=right, 2=top, 3=left. Corner bits: 1=v00, 2=v10, 4=v11, 8=v01.
_SEG = {
    0: [], 1: [(3, 0)], 2: [(0, 1)], 3: [(3, 1)], 4: [(1, 2)],
    5: [(3, 0), (1, 2)], 6: [(0, 2)], 7: [(3, 2)], 8: [(2, 3)], 9: [(0, 2)],
    10: [(0, 1), (2, 3)], 11: [(1, 2)], 12: [(3, 1)], 13: [(0, 1)],
    14: [(3, 0)], 15: [],
}


def contour_segments(xs, ys, Z, level):
    """Line segments of the `level` isocontour of Z, where Z[i][j] sits at
    (xs[i], ys[j]). Returns a list of ((x0,y0),(x1,y1)) tuples."""
    n, m = len(xs), len(ys)
    segs = []
    for i in range(n - 1):
        x0, x1 = xs[i], xs[i + 1]
        Zi, Zi1 = Z[i], Z[i + 1]
        for j in range(m - 1):
            y0, y1 = ys[j], ys[j + 1]
            v00, v10 = Zi[j], Zi1[j]
            v01, v11 = Zi[j + 1], Zi1[j + 1]
            case = ((1 if v00 >= level else 0) | (2 if v10 >= level else 0)
                    | (4 if v11 >= level else 0) | (8 if v01 >= level else 0))
            pairs = _SEG[case]
            if not pairs:
                continue

            def edgept(e):
                if e == 0:    # bottom: v00 -> v10 along y0
                    t = (level - v00) / (v10 - v00) if v10 != v00 else 0.5
                    return (x0 + t * (x1 - x0), y0)
                if e == 1:    # right: v10 -> v11 along x1
                    t = (level - v10) / (v11 - v10) if v11 != v10 else 0.5
                    return (x1, y0 + t * (y1 - y0))
                if e == 2:    # top: v01 -> v11 along y1
                    t = (level - v01) / (v11 - v01) if v11 != v01 else 0.5
                    return (x0 + t * (x1 - x0), y1)
                # e == 3, left: v00 -> v01 along x0
                t = (level - v00) / (v01 - v00) if v01 != v00 else 0.5
                return (x0, y0 + t * (y1 - y0))

            for ea, eb in pairs:
                pa, pb = edgept(ea), edgept(eb)
                if pa != pb:
                    segs.append((pa, pb))
    return segs


def load_bundle(path):
    with open(path) as f:
        d = json.load(f)
    for k in ("grid", "log_p_true", "log_p_jn", "log_p_mf", "markers"):
        if k not in d:
            die(f"bundle missing top-level key '{k}'")
    return d


def to_P(log_p):
    """exp() each entry; the grids are already normalised so max log_p = 0."""
    from math import exp
    return [[exp(v) for v in row] for row in log_p]


def _c(v):
    return f"{v:.5f}"


def color_opt(rgb):
    r, g, b = rgb
    return f"{{rgb,255:red,{r};green,{g};blue,{b}}}"


def contour_lines(xs, ys, P):
    """Per-panel TikZ: one \\draw per level (the contour) plus a small level
    number placed just inside each contour, in the level's colour."""
    out = []
    for lev in LEVELS:
        segs = contour_segments(xs, ys, P, lev)
        if not segs:
            continue
        col = color_opt(viridis(lev))
        parts = [rf"\draw[color={col}, line width=0.7pt]"]
        for (x0, y0), (x1, y1) in segs:
            parts.append(
                f"  (axis cs:{_c(x0)},{_c(y0)})--(axis cs:{_c(x1)},{_c(y1)})")
        parts[-1] += ";"
        out.append("\n".join(parts))

        # Level label on the topmost in-window contour point.
        pts = [p for seg in segs for p in seg
               if abs(p[0]) <= AXLIM and abs(p[1]) <= AXLIM]
        if pts:
            lx, ly = max(pts, key=lambda p: p[1])
            out.append(
                rf"\node[font=\tiny, text={col}, fill=white, inner sep=0.5pt] "
                rf"at (axis cs:{_c(lx)},{_c(ly)}) {{${lev:g}$}};")
    return out


def markers(yobs):
    """Observation marker (black +)."""
    return [
        rf"\addplot[only marks, mark=+, mark size=4pt, black, very thick] "
        rf"coordinates {{({_c(yobs[0])},{_c(yobs[1])})}};",
    ]


def grid_moments(xs, ys, P):
    """Posterior mean and 2x2 covariance of (z1,z2) from the density grid,
    using P[i][j] (at (xs[i],ys[j])) as unnormalised weights."""
    S = mx = my = 0.0
    for i, x in enumerate(xs):
        Pi = P[i]
        for j, y in enumerate(ys):
            w = Pi[j]
            S += w; mx += w * x; my += w * y
    mx /= S; my /= S
    cxx = cyy = cxy = 0.0
    for i, x in enumerate(xs):
        Pi = P[i]; dx = x - mx
        for j, y in enumerate(ys):
            w = Pi[j]; dy = y - my
            cxx += w * dx * dx; cyy += w * dy * dy; cxy += w * dx * dy
    return [mx, my], [[cxx / S, cxy / S], [cxy / S, cyy / S]]


def principal_axes(mean, Sigma, span=4 * AXLIM):
    """Lines along the eigenvectors (principal directions) of the 2x2 covariance
    `Sigma`, through `mean`. Drawn long (half-length `span`, well past the box)
    so that with clip=true they reach the panel edges; orientation, not length,
    is what they convey."""
    from math import sqrt, hypot
    a, b, d = Sigma[0][0], Sigma[0][1], Sigma[1][1]
    tr, det = a + d, a * d - b * b
    s = sqrt(max(0.0, tr * tr - 4 * det))
    l1, l2 = (tr + s) / 2.0, (tr - s) / 2.0
    if abs(b) > 1e-12:
        vecs = [(l1 - d, b), (l2 - d, b)]
    elif a >= d:
        vecs = [(1.0, 0.0), (0.0, 1.0)]
    else:
        vecs = [(0.0, 1.0), (1.0, 0.0)]
    lines = []
    for v in vecs:
        n = hypot(v[0], v[1])
        if n == 0.0:
            continue
        ux, uy = v[0] / n, v[1] / n
        x0, y0 = mean[0] - span * ux, mean[1] - span * uy
        x1, y1 = mean[0] + span * ux, mean[1] + span * uy
        lines.append(
            rf"\draw[black!60, line width=0.5pt] "
            rf"(axis cs:{_c(x0)},{_c(y0)})--(axis cs:{_c(x1)},{_c(y1)});")
    return lines


def build_tex(bundle):
    g = bundle["grid"]
    xs, ys = g["z1_grid"], g["z2_grid"]
    yobs = bundle["markers"]["y_marker"]

    enc = bundle.get("encoders", {})
    try:
        jn, mf = enc["joint_normal"], enc["normal_diagonal"]
        jn_axes = (jn["mu"], jn["Sigma"])
        mf_axes = (mf["mu"], mf["Sigma"])
    except (KeyError, TypeError):
        die("bundle missing encoders.joint_normal/normal_diagonal mu+Sigma")

    P_true = to_P(bundle["log_p_true"])
    # (title, P-grid, (mean, Sigma) for the principal axes)
    panels = [
        (r"true posterior $p_{\vartheta}(z_1,z_2|y_{1:6})$", P_true,
         grid_moments(xs, ys, P_true)),
        (r"unrestr.\ Gaussian $q_{\phi}(z_1,z_2|y_{1:6})$", to_P(bundle["log_p_jn"]),
         jn_axes),
        (r"diagonal $q_{\phi}(z_1,z_2|y_{1:6})$", to_P(bundle["log_p_mf"]),
         mf_axes),
    ]

    tex = [
        r"\begin{figure}[!t]",
        r"\centering",
        r"\resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}{%",
        r"\begin{tikzpicture}",
        r"\begin{groupplot}[",
        r"  group style={group size=3 by 1, horizontal sep=0.7cm,"
        r" ylabels at=edge left, yticklabels at=edge left},",
        r"  width=5.0cm, height=5.0cm, scale only axis,",
        rf"  xmin={-AXLIM:g}, xmax={AXLIM:g}, ymin={-AXLIM:g}, ymax={AXLIM:g},",
        r"  xtick={-0.4,-0.2,0,0.2,0.4}, ytick={-0.4,-0.2,0,0.2,0.4},",
        r"  xlabel={$z_1$}, ylabel={$z_2$},",
        r"  title style={font=\footnotesize, yshift=-1ex},",
        r"  tick label style={font=\footnotesize},",
        r"  label style={font=\small},",
        r"  enlargelimits=false, clip=true,",
        r"]",
    ]

    for title, P, (mean, Sigma) in panels:
        tex.append(rf"\nextgroupplot[title={{{title}}}]")
        tex += principal_axes(mean, Sigma)   # under the contours
        tex += contour_lines(xs, ys, P)
        tex += markers(yobs)

    tex += [
        r"\end{groupplot}",
        r"\end{tikzpicture}%",
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
    ap.add_argument("bundle", help="path to contour_bundle.json")
    ap.add_argument("-o", dest="out", required=True, help="output .tex path")
    args = ap.parse_args(argv)

    bundle = load_bundle(args.bundle)
    tex = build_tex(bundle)
    with open(args.out, "w") as f:
        f.write(tex)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()