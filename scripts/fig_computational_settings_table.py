#!/usr/bin/env python3
r"""Table F1 (web appendix): Estimation Settings for the Results.

Input: none (fixed content).
Usage: python scripts/fig_computational_settings_table.py -o output/figures/tex/computational_settings_table.tex
"""
import argparse
from pathlib import Path

TEX = r"""\begin{table}[!t]
\centering
\caption{Estimation Settings for the Results}
\label{tab:computational-appendix}
\setlength{\tabcolsep}{4pt}%
\resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}{%
\begin{tabular}{lccccc}
\toprule
 & {AR(1)} & {Nonlinear ($T{=}6$)} & {Nonlinear ($T{=}40$)} & {Heterogeneity} & {PSID} \\
\midrule
\multicolumn{6}{l}{\textit{Data and model}} \\
Panel $N \times T$ & $30{,}000 \times 6$ & $30{,}000 \times 6$ & $30{,}000 \times 40$ & $30{,}000 \times 6$ & $741 \times 10$ \\
Data & simulated & simulated & simulated & simulated & 1980--1989 \\
Free model parameters & 8 & 12 & 12 & 12 & 15 \\
\addlinespace
\multicolumn{6}{l}{\textit{Variational posterior}} \\
Family & mean-field & joint normal & joint normal & joint normal $+\,a$ & joint normal $+\,a$ \\
Hidden width $d$ & 32 & 64 & 64 & 64 & 64 \\
Latent dimension & 6 & 6 & 40 & 7 & 11 \\
Var.\ posterior parameters & 620 & 2,203 & 58,524 & 2,723 & 5,709 \\
\addlinespace
\multicolumn{6}{l}{\textit{Inner VI (ELBO maximization)}} \\
ELBO draws & 1 & 1 & 1 & 1 & 40 \\
Learning rate & $10^{-2}$ & $10^{-2}$ & $10^{-2}$ & $10^{-2}$ & $10^{-2}$ \\
Epochs, fit on data & 10,000 & 20,000 & 20,000 & 20,000 & 20,000 \\
Epochs, per outer iter. & 8,000 & 16,000 & 16,000 & 16,000 & 16,000 \\
\addlinespace
\multicolumn{6}{l}{\textit{Outer IVI loop}} \\
Outer iterations & 10 & 25 & 50 & 10 & 50 \\
Damping $\kappa$ & 0.6 & 0.6 & 0.6 & 0.6 & 0.6 \\
\bottomrule
\end{tabular}%
}
\\ \vspace{0.2cm} \figurenote{This table summarizes the estimation settings for the AR(1)-normal simulation (Section \ref{sec:benchmark}), the nonlinear model simulation at $T = 6$ and at $T = 40$ (Section \ref{sec:nonlinear-model}), the permanent heterogeneity simulation (Subsection \ref{sec:hetero-model}), and the model estimated on the PSID panel (Section \ref{sec:PSID}).}
\end{table}
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--output", required=True, help="output .tex path")
    args = ap.parse_args()
    Path(args.output).write_text(TEX, encoding="utf-8")


if __name__ == "__main__":
    main()
