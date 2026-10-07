#!/usr/bin/env python3
r"""Figure 1: VI Estimation and the Pseudo-True Parameter.

Input: none (fixed content).
Usage: python scripts/fig_sample_vs_population.py -o output/figures/tex/sample_vs_population.tex
"""
import argparse
from pathlib import Path

TEX = r"""\begin{figure}[!t]
\centering
\begin{tikzpicture}[
  >=Stealth,
  box/.style={draw=black!40,rounded corners=2pt,align=center,
              text width=4.25cm,minimum height=1.1cm,inner sep=5pt},
  every node/.style={font=\small}]
\node[box] (sample) {Sample objective\\[2pt] $\widehat{\mathcal E}_{\vartheta}\text{ with }y_{1:T} {\sim} \mathcal P_{\vartheta_0}$ };
\node[box,right=2.4cm of sample] (pop)
  {Population objective\\[2pt] $\overline{\mathcal E}_{\vartheta,\vartheta_0}$};
\node[box,below=0.85cm of sample] (estimate)
  {VI estimator\\[2pt] $\widehat{\vartheta}^{\rm VI}$};
\node[box,below=0.85cm of pop] (pseudo)
  {Pseudo-true parameter\\[2pt] $\overline{\vartheta}_0$};
\draw[->] (sample) -- node[above] {$N\to\infty$} (pop);
\draw[->] (sample) -- node[left] {$\arg\max_{\vartheta}$} (estimate);
\draw[->] (pop) -- node[right] {$\arg\max_{\vartheta}$} (pseudo);
\draw[->] (estimate) -- node[above] {$N\to\infty$} (pseudo);
\node[below=0.22cm of pseudo,align=center]
  {may differ from the true parameter $\vartheta_0$};
\end{tikzpicture}
\caption{VI Estimation and the Pseudo-True Parameter}
\label{sample_vs_population}
\vspace{0.2cm} \figurenote{The figure summarizes the relationship between the sample VI objective $\widehat{\mathcal E}_{\vartheta}$, its population counterpart $\overline{\mathcal E}_{\vartheta,\vartheta_0}$, the VI estimator $\widehat{\vartheta}^{\rm VI}$, and its pseudo-true probability limit $\overline{\vartheta}_0$. As the number of individuals $N$ increases, the sample objective converges to the population objective and, under standard regularity conditions, $\widehat{\vartheta}^{\rm VI}$ converges in probability to $\overline{\vartheta}_0$. Because the variational penalty depends on $\vartheta$, the pseudo-true parameter $\overline{\vartheta}_0$ need not coincide with the true parameter $\vartheta_0$.}
\end{figure}
"""


def main():
    ap = argparse.ArgumentParser(
        description="Emit the sample-vs-population VI diagram as a LaTeX fragment.")
    ap.add_argument("-o", "--output", required=True, help="output .tex path")
    args = ap.parse_args()
    Path(args.output).write_text(TEX, encoding="utf-8")


if __name__ == "__main__":
    main()
