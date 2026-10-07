#!/usr/bin/env python3
r"""Figure 3: Trajectory of the IVI Bias Correction in the Linear Gaussian Model.

Input: output/bundles/ar1n_quiver_bundle.json
Quiver arrows show u(theta) = theta_VI_obs - b(theta).
Usage: python scripts/fig_ar1n_trajectory_fixed_point.py output/bundles/ar1n_quiver_bundle.json -o output/figures/tex/ar1n_trajectory_fixed_point.tex
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

XMIN, XMAX = 0.60, 0.98          # rho-axis range; y-range comes from the data
AX_W_CM, AX_H_CM = 11.0, 7.0     # plot-area size

CONTOUR_LEVELS = [-5.0, -4.5, -4.0, -3.6, -3.2, -2.9, -2.7, -2.55]
CONTOUR_FINE = 240               # fine-grid resolution for smoothing

# Contour-label placement: "auto" = midpoint of longest segment;
# ("target", x, y) = in-box contour vertex nearest (x, y). Unlisted levels are unlabelled.
LABEL_CFG = {
    -4.00: "auto",
    -3.60: "auto",
    -3.20: ("target", 0.892, 0.124),
    -2.90: "auto",
    -2.70: "auto",
    -2.55: ("target", 0.89, 0.195),
}

QUIVER_SCALE = 0.12              # uniform arrow shrink
NQ_Y = 10                        # number of arrow rows, equi-spaced in sigma_e

QUIVER_COLOR = "black!30"
TRAJ_COLOR = "teal!90!black"
VI_COLOR = "camo"                # defined in preamble
TRUTH_COLOR = "blue!60!black"
CONTOUR_STYLE = "densely dotted, mark=none, smooth, line width=0.8pt"
# Viridis sub-range for contours (stops short of the near-white top).
CONTOUR_T_LO, CONTOUR_T_HI = 0.0, 0.82

# Viridis ramp anchors (position, RGB 0-255).
_VIRIDIS = [
    (0.00, (68, 1, 84)),
    (0.25, (59, 82, 139)),
    (0.50, (33, 144, 140)),
    (0.75, (93, 201, 99)),
    (1.00, (253, 231, 37)),
]


def _viridis(t):
    """RGB (0-255) at position t in [0,1] along the viridis ramp."""
    t = max(0.0, min(1.0, t))
    for k in range(len(_VIRIDIS) - 1):
        t0, c0 = _VIRIDIS[k]
        t1, c1 = _VIRIDIS[k + 1]
        if t <= t1:
            f = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
            return tuple(round(c0[i] + f * (c1[i] - c0[i])) for i in range(3))
    return _VIRIDIS[-1][1]


def _contour_color(lev, lo, hi):
    """pgfplots colour for a contour at log-likelihood ``lev``, viridis mapped
    across the drawn range [lo, hi] and remapped into [CONTOUR_T_LO, CONTOUR_T_HI]."""
    t = (lev - lo) / (hi - lo) if hi > lo else 0.5
    t = CONTOUR_T_LO + t * (CONTOUR_T_HI - CONTOUR_T_LO)
    r, g, b = _viridis(t)
    return "{rgb,255:red,%d;green,%d;blue,%d}" % (r, g, b)

CAPTION = ("Trajectory of the IVI Bias Correction in the "
           "Linear Gaussian Model")
LABEL = "ar1n_trajectory_fixed_point"

FIGURENOTE = (
    r"\vspace{0.2cm} \figurenote{Fixed-point update directions $\widehat{\vartheta}^{\rm VI}-\widehat b(\vartheta_\star)$, projected onto the displayed plane and overlaid on log-likelihood contours; arrow length shows relative step size. $\widehat b(\vartheta_\star)$ is the mean-field VI estimate on a panel $\widetilde y_{1:T}(\vartheta_{\star})$ drawn from $\mathcal P_{\vartheta_{\star}}(y_{1:T})$. The teal line is the fixed-point trajectory $\vartheta_{\star}^{(k+1)}=\vartheta_{\star}^{(k)}-\kappa(\widehat b(\vartheta_{\star}^{(k)})-\widehat{\vartheta}^{\rm VI})$ over 10 iterations from the VI estimate (orange diamond) to a neighborhood of the truth $\vartheta_0$ (blue triangle). Linear Gaussian DGP, $N=30{,}000$, $T=6$.}"
)

# Label offsets from their markers (data units).
VI_LABEL_DY = -0.010
TRUTH_LABEL_DY = +0.009

REQUIRED_KEYS = ["m1_grid", "lse_grid", "logp_grid", "b_m1", "b_lse",
                 "trajectory_m1", "trajectory_lse", "reference_points"]
REQUIRED_REFS = ["vi_obs", "truth"]


# numpy-only interpolation + contouring
def _cubic_kernel(s):
    """Keys cubic-convolution kernel (a = -0.5), vectorised over |s|."""
    a = -0.5
    s = np.abs(s)
    out = np.zeros_like(s)
    m1 = s <= 1.0
    m2 = (s > 1.0) & (s < 2.0)
    out[m1] = (a + 2) * s[m1] ** 3 - (a + 3) * s[m1] ** 2 + 1
    out[m2] = a * s[m2] ** 3 - 5 * a * s[m2] ** 2 + 8 * a * s[m2] - 4 * a
    return out


def _bicubic(xs, ys, Z, xq, yq):
    """Bicubic (cubic-convolution) interpolation of Z on grid (ys, xs).

    Z has shape (len(ys), len(xs)); Z[i, j] sits at (ys[i], xs[j]). xq, yq are
    arrays of query coordinates (same shape). Query coordinates are mapped to
    fractional grid indices via np.interp, so non-uniform grids are handled;
    edges are clamped. Returns interpolated values with xq's shape.
    """
    ny, nx = Z.shape
    xq = np.asarray(xq, float)
    yq = np.asarray(yq, float)
    gx = np.interp(xq, xs, np.arange(nx))
    gy = np.interp(yq, ys, np.arange(ny))
    x0 = np.floor(gx).astype(int)
    y0 = np.floor(gy).astype(int)
    fx = gx - x0
    fy = gy - y0
    # weights for the four neighbours at index offsets -1, 0, 1, 2
    wx = [_cubic_kernel(1 + fx), _cubic_kernel(fx),
          _cubic_kernel(1 - fx), _cubic_kernel(2 - fx)]
    wy = [_cubic_kernel(1 + fy), _cubic_kernel(fy),
          _cubic_kernel(1 - fy), _cubic_kernel(2 - fy)]
    out = np.zeros_like(gx, float)
    for m in range(4):
        iy = np.clip(y0 - 1 + m, 0, ny - 1)
        for n in range(4):
            ix = np.clip(x0 - 1 + n, 0, nx - 1)
            out += wy[m] * wx[n] * Z[iy, ix]
    return out


# Marching-squares segment table, keyed by the 4-bit corner code
#   bit0 = bottom-left, bit1 = bottom-right, bit2 = top-right, bit3 = top-left
# Each entry lists the edge pairs the iso-line crosses; edges are
#   B(ottom), R(ight), T(op), L(eft).
_MS_TABLE = {
    0: [], 1: [("L", "B")], 2: [("B", "R")], 3: [("L", "R")],
    4: [("R", "T")], 5: [("L", "B"), ("R", "T")], 6: [("B", "T")],
    7: [("L", "T")], 8: [("T", "L")], 9: [("T", "B")],
    10: [("B", "R"), ("T", "L")], 11: [("T", "R")], 12: [("R", "L")],
    13: [("B", "R")], 14: [("L", "B")], 15: [],
}


def _chain(segs):
    """Join unordered segments (pairs of points) into ordered polylines.

    Crossing points on a shared cell edge are computed identically from the two
    shared corner values, so endpoints match exactly; we key on rounded points
    and walk the adjacency both ways from each unused segment.
    """
    def key(p):
        return (round(p[0], 7), round(p[1], 7))

    adj = {}
    for idx, (a, b) in enumerate(segs):
        adj.setdefault(key(a), []).append(idx)
        adj.setdefault(key(b), []).append(idx)

    used = [False] * len(segs)
    polylines = []
    for start in range(len(segs)):
        if used[start]:
            continue
        used[start] = True
        a, b = segs[start]
        poly = [a, b]
        for extend_tail in (True, False):
            changed = True
            while changed:
                changed = False
                tip = poly[-1] if extend_tail else poly[0]
                for si in adj.get(key(tip), []):
                    if used[si]:
                        continue
                    a2, b2 = segs[si]
                    nxt = b2 if key(a2) == key(tip) else (
                        a2 if key(b2) == key(tip) else None)
                    if nxt is None:
                        continue
                    used[si] = True
                    if extend_tail:
                        poly.append(nxt)
                    else:
                        poly.insert(0, nxt)
                    changed = True
        polylines.append(np.asarray(poly, float))
    return polylines


def _marching_squares(x, y, Z, level):
    """Iso-contours of Z at `level` as ordered polylines in (x, y) coords.

    x, y are 1-D coordinate vectors (len nx, ny); Z has shape (ny, nx). Only
    cells straddling the level are visited.
    """
    f = Z - level
    f00 = f[:-1, :-1]; f10 = f[:-1, 1:]; f11 = f[1:, 1:]; f01 = f[1:, :-1]
    code = ((f00 >= 0).astype(int) + (f10 >= 0).astype(int) * 2
            + (f11 >= 0).astype(int) * 4 + (f01 >= 0).astype(int) * 8)
    crossing = (code != 0) & (code != 15)
    segs = []
    for i, j in np.argwhere(crossing):
        i, j = int(i), int(j)
        v00, v10 = f[i, j], f[i, j + 1]
        v11, v01 = f[i + 1, j + 1], f[i + 1, j]

        def frac(fa, fb):
            d = fb - fa
            return 0.5 if d == 0 else (0.0 - fa) / d

        def pt(edge):
            if edge == "B":
                return (x[j] + frac(v00, v10) * (x[j + 1] - x[j]), y[i])
            if edge == "R":
                return (x[j + 1], y[i] + frac(v10, v11) * (y[i + 1] - y[i]))
            if edge == "T":
                return (x[j] + frac(v01, v11) * (x[j + 1] - x[j]), y[i + 1])
            return (x[j], y[i] + frac(v00, v01) * (y[i + 1] - y[i]))  # "L"

        for ea, eb in _MS_TABLE[int(code[i, j])]:
            segs.append((pt(ea), pt(eb)))
    return _chain(segs)


def _fmt_coords(xy) -> str:
    return " ".join(f"({x:.6f},{y:.6f})" for x, y in xy)


def _check_contract(d: dict) -> None:
    """Fail loudly with a clear message if the JSON is missing what we read."""
    missing = [k for k in REQUIRED_KEYS if k not in d]
    if missing:
        raise KeyError(f"bundle missing required keys: {missing}")
    refs = d["reference_points"]
    miss_ref = [r for r in REQUIRED_REFS if r not in refs
                or "parameters" not in refs[r]]
    if miss_ref:
        raise KeyError(f"reference_points missing/invalid: {miss_ref}")
    for r in REQUIRED_REFS:
        p = refs[r]["parameters"]
        if "mu1" not in p or "log_sigma_eps" not in p:
            raise KeyError(f"reference_points['{r}'].parameters needs "
                           f"mu1 and log_sigma_eps")


def _ref_point(d: dict, name: str):
    """(rho, sigma_e) for a named reference point."""
    p = d["reference_points"][name]["parameters"]
    return p["mu1"], float(np.exp(p["log_sigma_eps"]))


def _contours(m1, lse, lp, xunit, yunit, ymin, ymax):
    """Smooth contour paths + clustered, rotated labels in (rho, sigma_e).

    Contours are computed on the NATIVE uniform (m1, lse) grid (bicubic
    upsample -> fine mesh) and then mapped y -> exp(lse), so the smoothing is
    done where the grid is regular and undistorted.
    """
    m1_f = np.linspace(m1.min(), m1.max(), CONTOUR_FINE)
    lse_f = np.linspace(lse.min(), lse.max(), CONTOUR_FINE)
    MX, MY = np.meshgrid(m1_f, lse_f)
    Zf = _bicubic(m1, lse, lp, MX.ravel(), MY.ravel()).reshape(MX.shape)

    paths, labels = [], []
    for lev in CONTOUR_LEVELS:
        polys = _marching_squares(m1_f, lse_f, Zf, lev)
        xy_segs = [np.column_stack([p[:, 0], np.exp(p[:, 1])])
                   for p in polys if len(p) >= 2]
        paths.extend((lev, seg) for seg in xy_segs)
        key = round(float(lev), 2)
        if key not in LABEL_CFG or not xy_segs:
            continue
        cfg = LABEL_CFG[key]
        if cfg == "auto":
            seg = max(xy_segs, key=len)
            k = len(seg) // 2
        else:                                    # snap to nearest in-box vertex
            _, tx, ty = cfg
            best = None
            for s in xy_segs:
                inb = ((s[:, 0] >= XMIN) & (s[:, 0] <= XMAX)
                       & (s[:, 1] >= ymin) & (s[:, 1] <= ymax))
                if not inb.any():
                    continue
                idx = np.where(inb)[0]
                d2 = (((s[idx, 0] - tx) * xunit) ** 2
                      + ((s[idx, 1] - ty) * yunit) ** 2)
                if best is None or d2.min() < best[0]:
                    best = (d2.min(), s, idx[int(np.argmin(d2))])
            if best is None:
                continue
            _, seg, k = best
        px, py = seg[k]
        a, b = seg[max(k - 1, 0)], seg[min(k + 1, len(seg) - 1)]
        # rotate label to the contour tangent, measured in canvas units
        dx, dy = (b[0] - a[0]) * xunit, (b[1] - a[1]) * yunit
        ang = np.degrees(np.arctan2(dy, dx))
        ang = ang - 180 if ang > 90 else (ang + 180 if ang < -90 else ang)
        labels.append((px, py, ang, float(lev)))
    return paths, labels


def _quiver(m1, lse, bm, bl, vi, ymin, ymax):
    """Binding-equation update arrows on a sigma_e-equispaced grid.

    u = theta_VI_obs - b(theta); b is interpolated bicubically onto the
    sigma_e rows.
    """
    se_t = np.linspace(ymin, ymax, NQ_Y)         # equi-spaced in sigma_e
    rows = []
    for se_i in se_t:
        lse_i = float(np.log(se_i))
        for x in m1:
            bmi = float(_bicubic(m1, lse, bm, [x], [lse_i])[0])
            bli = float(_bicubic(m1, lse, bl, [x], [lse_i])[0])
            u = (vi["mu1"] - bmi) * QUIVER_SCALE
            v = (se_i * np.exp(vi["log_sigma_eps"] - bli) - se_i) * QUIVER_SCALE
            rows.append((x, se_i, u, v))
    return rows


def _trajectory(d):
    """Trajectory points and per-segment arrow vectors in (rho, sigma_e)."""
    tm = np.asarray(d["trajectory_m1"], float)
    tse = np.exp(np.asarray(d["trajectory_lse"], float))
    traj = np.column_stack([tm, tse])
    segs = [(a[0], a[1], b[0] - a[0], b[1] - a[1])     # skip zero-length steps
            for a, b in zip(traj[:-1], traj[1:]) if not np.allclose(a, b)]
    return traj, segs


def build(bundle_path: Path) -> str:
    d = json.loads(Path(bundle_path).read_text())
    _check_contract(d)

    m1 = np.asarray(d["m1_grid"], float)
    lse = np.asarray(d["lse_grid"], float)
    bm = np.asarray(d["b_m1"], float)
    bl = np.asarray(d["b_lse"], float)
    lp = np.asarray(d["logp_grid"], float)

    ymin = float(np.exp(lse).min())
    ymax = float(np.exp(lse).max())
    xunit = AX_W_CM / (XMAX - XMIN)
    yunit = AX_H_CM / (ymax - ymin)

    vi = d["reference_points"]["vi_obs"]["parameters"]

    paths, labels = _contours(m1, lse, lp, xunit, yunit, ymin, ymax)
    quiver_rows = _quiver(m1, lse, bm, bl, vi, ymin, ymax)
    traj, traj_seg = _trajectory(d)
    vi_pt, tr_pt = _ref_point(d, "vi_obs"), _ref_point(d, "truth")

    L: list[str] = []
    A = L.append
    A(r"\begin{figure}[!t]")
    A(r"\centering")
    A(r"\begin{tikzpicture}")
    A(r"\begin{axis}[")
    A(f"  scale only axis, width={AX_W_CM}cm, height={AX_H_CM}cm,")
    A(f"  xmin={XMIN}, xmax={XMAX}, ymin={ymin:.6f}, ymax={ymax:.6f},")
    A(r"  xlabel={$\rho_\star$}, ylabel={$\sigma_{e\star}$},")
    A(r"  xlabel style={font=\normalsize}, ylabel style={font=\normalsize},")
    A(r"  tick label style={font=\small}, scaled y ticks=false,")
    A(r"  yticklabel style={/pgf/number format/fixed, "
      r"/pgf/number format/precision=2},")
    A(r"  axis line style={gray!60},")
    A(r"  grid=major, grid style={gray!15, very thin},")
    A(r"  enlargelimits=false, clip=true,")
    A(r"]")

    # Normalise the colour ramp over the levels visible in the plot box.
    def _in_box(seg):
        return bool(np.any((seg[:, 0] >= XMIN) & (seg[:, 0] <= XMAX)
                           & (seg[:, 1] >= ymin) & (seg[:, 1] <= ymax)))
    _vis = sorted({lev for lev, seg in paths if _in_box(seg)})
    lvl_lo, lvl_hi = ((_vis[0], _vis[-1]) if _vis else
                      (min(l for l, _ in paths), max(l for l, _ in paths)))
    for lev, seg in paths:
        A(r"\addplot[color=%s, %s] coordinates {%s};"
          % (_contour_color(lev, lvl_lo, lvl_hi), CONTOUR_STYLE, _fmt_coords(seg)))

    A(r"\addplot[%s, quiver={u=\thisrow{u}, v=\thisrow{v}, "
      r"scale arrows=1.0}, -{Stealth[length=3pt]}, thin]" % QUIVER_COLOR)
    A(r"  table[x=x, y=y, meta=u] {")
    A(r"  x y u v")
    for x, y, u, v in quiver_rows:
        A(f"  {x:.6f} {y:.6f} {u:.6f} {v:.6f}")
    A(r"  };")

    for x, y, dx, dy in traj_seg:
        A(r"\draw[%s, line width=1.1pt, -{Stealth[length=6pt]}] (axis cs:%.6f,%.6f) -- "
          r"(axis cs:%.6f,%.6f);"
          % (TRAJ_COLOR, x, y, x + dx, y + dy))
    A(r"\addplot[only marks, mark=*, mark size=2.2pt, "
      r"mark options={fill=white, draw=%s, line width=0.6pt}] "
      r"coordinates {%s};" % (TRAJ_COLOR, _fmt_coords(traj)))

    for px, py, ang, lev in labels:
        A(r"\node[rotate=%.1f, fill=white, inner sep=1pt, "
          r"font=\footnotesize, text=%s] at (axis cs:%.6f,%.6f) {$%.1f$};"
          % (ang, _contour_color(lev, lvl_lo, lvl_hi), px, py, lev))

    box = (r"fill=white, draw=none, rounded corners=2pt, inner sep=2.5pt, "
           r"font=\small\bfseries, text=%s")
    A(r"\addplot[only marks, mark=diamond*, mark size=4.5pt, "
      r"mark options={fill=%s, draw=%s, line width=0.5pt}] "
      r"coordinates {(%.6f,%.6f)};" % (VI_COLOR, VI_COLOR, *vi_pt))
    A(r"\node[anchor=north, %s] at (axis cs:%.6f,%.6f) {VI (%.2f,%.2f)};"
      % (box % VI_COLOR, vi_pt[0], vi_pt[1] + VI_LABEL_DY, vi_pt[0], vi_pt[1]))
    A(r"\addplot[only marks, mark=triangle*, mark size=4.5pt, "
      r"mark options={fill=%s, draw=%s, line width=0.5pt}] "
      r"coordinates {(%.6f,%.6f)};" % (TRUTH_COLOR, TRUTH_COLOR, *tr_pt))
    A(r"\node[anchor=south, %s] at (axis cs:%.6f,%.6f) {truth (%.2f,%.2f)};"
      % (box % TRUTH_COLOR, tr_pt[0], tr_pt[1] + TRUTH_LABEL_DY,
         tr_pt[0], tr_pt[1]))

    A(r"\end{axis}")
    A(r"\end{tikzpicture}")
    A(r"\caption{%s}" % CAPTION)
    A(r"\label{%s}" % LABEL)
    A(FIGURENOTE)
    A(r"\end{figure}")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="IVI trajectory figure.")
    ap.add_argument("bundle", type=Path, help="ar1n_quiver_bundle JSON file")
    ap.add_argument("-o", "--out", type=Path, required=True,
                    help="output .tex path")
    args = ap.parse_args(argv)

    tex = build(args.bundle)
    args.out.write_text(tex)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
