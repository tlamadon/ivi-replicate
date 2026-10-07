#!/usr/bin/env python3
r"""Table E1 (web appendix): Resource Use for Variational Inference and Sequential Monte Carlo.

Input: output/bundles/time-to-convergence-hockey-bundle.json
Metrics: converged_wall_s/60 (min), total_wall_s/converged_at*1000
(ms/iter; last logged epoch if not converged), peak_gpu_mem_mb.
Usage: python scripts/fig_hockeystick_resource_table.py output/bundles/time-to-convergence-hockey-bundle.json -o output/figures/tex/hockeystick_resource_table.tex
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Dict, List, Tuple

CAPTION = r"Resource Use for Variational Inference and Sequential Monte Carlo"
LABEL = "hockeystick_resource_table"

FIGURENOTE = (
    r"\\ \vspace{0.2cm} \figurenote{Resource cost of estimating the nonlinear DGP by variational inference (VI) and by sequential Monte Carlo (SMC), across four data sizes $(N,T)$. VI is shown for two hidden widths of the neural network and SMC for two particle counts. The convergence criterion is reached when the difference in the (running-averaged) parameter vector stays below $0.02$. Runs share a common seed and a budget of $25{,}000$ epochs. NVIDIA L40S GPU.}"
)

# Corners (data columns), in display order: (N, T).
CORNER_ORDER: List[Tuple[int, int]] = [
    (1000, 6),
    (1000, 40),
    (30000, 6),
    (30000, 40),
]

# (section label, [(row stub, method, H or K), ...])
METHOD_SECTIONS: List[Tuple[str, List[Tuple[str, str, int]]]] = [
    ("Variational Inference", [(r"$32$ hidden nodes", "vi",  32),
                               (r"$64$ hidden nodes", "vi",  64)]),
    ("Sequential Monte Carlo", [(r"$50$ particles",   "smc", 50),
                                (r"$200$ particles",  "smc", 200)]),
]

METRICS: List[Tuple[str, str]] = [
    ("conv", r"Time to convergence (min)"),
    ("msit", r"Time per iteration (ms)"),
    ("mem",  r"Peak GPU memory (MB)"),
]

N_CORNER = len(CORNER_ORDER)
N_METRIC = len(METRICS)
N_LABEL = 1
N_DATACOL = N_METRIC * N_CORNER

CELL_W = r"2.5em"                       # fixed width of every data cell
PANEL_GAP = r"@{\hspace{16pt}}"         # gap between metric panels
TABCOLSEP = r"2pt"


def die(msg: str) -> None:
    sys.stderr.write("fig_hockeystick_resource_table.py: ERROR: " + msg + "\n")
    sys.exit(1)


def index_cells(doc: dict) -> Dict[Tuple, dict]:
    """Key every cell by (N, T, method, config-value)."""
    if not isinstance(doc.get("cells"), list) or not doc["cells"]:
        die("bundle JSON missing non-empty 'cells' list")
    out: Dict[Tuple, dict] = {}
    for c in doc["cells"]:
        p = c.get("param", {})
        cfg = p.get("H") if c["method"] == "vi" else p.get("K")
        out[(c["N"], c["T"], c["method"], cfg)] = c
    return out


def cell_metrics(c: dict) -> Dict[str, str]:
    """{conv, msit, mem} as formatted LaTeX strings for one cell."""
    conv_wall = c.get("converged_wall_s")
    converged_at = c.get("converged_at")
    total_wall = c.get("total_wall_s")
    mem = c.get("peak_gpu_mem_mb")

    conv_str = "--" if conv_wall is None else f"{conv_wall / 60.0:.1f}"

    # Not converged: use the last logged epoch as denominator.
    denom = converged_at
    if denom is None:
        snaps = c.get("snapshots") or []
        denom = snaps[-1]["epoch"] if snaps else None
    ms_str = "--" if (denom in (None, 0) or total_wall is None) \
        else f"{total_wall / denom * 1000.0:.1f}"

    mem_str = "--" if mem is None else f"{mem:.0f}"
    return {"conv": conv_str, "msit": ms_str, "mem": mem_str}


def _n_short(N: int) -> str:
    """N in thousands."""
    return f"${N / 1000:g}$"


def _box(s: str) -> str:
    """Centre a cell in a fixed-width box."""
    return r"\makebox[" + CELL_W + r"][c]{" + s + r"}"


def render_table(cells: Dict[Tuple, dict]) -> str:
    colspec = "l" + PANEL_GAP.join([""] + ["c" * N_CORNER] * N_METRIC)

    g1_cells, g1_rules, col = [], [], N_LABEL + 1
    for _key, header in METRICS:
        g1_cells.append(r"\multicolumn{" + str(N_CORNER) + r"}{c}{" + header + r"}")
        g1_rules.append(r"\cmidrule(l{3pt}r{3pt}){" + f"{col}-{col + N_CORNER - 1}" + r"}")
        col += N_CORNER
    group1 = " & " + " & ".join(g1_cells) + r" \\"
    group1_cmid = "".join(g1_rules)

    n_vals = " & ".join(_n_short(N) for (N, _T) in CORNER_ORDER * N_METRIC)
    t_vals = " & ".join(f"${T}$" for (_N, T) in CORNER_ORDER * N_METRIC)
    n_row = r"$N$ (thousands) & " + n_vals + r" \\"
    t_row = r"$T$ & " + t_vals + r" \\"

    lines: List[str] = [
        r"\begin{table}[!t]",
        r"\centering",
        rf"\caption{{{CAPTION}}}",
        rf"\label{{{LABEL}}}",
        r"\setlength{\tabcolsep}{" + TABCOLSEP + r"}%",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{" + colspec + r"}",
        r"\toprule",
        group1,
        group1_cmid,
        n_row,
        t_row,
        r"\midrule",
    ]

    empty_data = " & ".join([""] * N_DATACOL)
    for s_idx, (section, rows) in enumerate(METHOD_SECTIONS):
        if s_idx > 0:
            lines.append(r"\addlinespace")
        lines.append(r"\textit{" + section + r"} & " + empty_data + r" \\")
        for stub, method, val in rows:
            row_cells = []
            for key, _header in METRICS:
                for (N, T) in CORNER_ORDER:
                    c = cells.get((N, T, method, val))
                    if c is None:
                        die(f"bundle missing cell N={N}, T={T}, {method}={val}")
                    row_cells.append(_box(cell_metrics(c)[key]))
            lines.append(r"\quad " + stub + " & " + " & ".join(row_cells) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        FIGURENOTE,
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bundle", help="path to time-to-convergence-hockey-bundle.json")
    ap.add_argument("-o", dest="out", required=True, help="output .tex path")
    args = ap.parse_args(argv)

    with open(args.bundle) as f:
        doc = json.load(f)
    cells = index_cells(doc)
    body = render_table(cells)

    with open(args.out, "w") as f:
        f.write(body)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()