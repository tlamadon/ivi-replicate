#!/usr/bin/env python3
r"""Figure 2: Population Binding Function and IVI Matching.

Input: none (fixed content).
Usage: python scripts/fig_ivi_binding_function.py -o output/figures/tex/ivi_binding_function.tex
"""
import argparse
from pathlib import Path

TEX = r"""\begin{figure}[!t]
\centering
\begin{tikzpicture}[
  >=Stealth,
  every node/.style={font=\small\linespread{1}\selectfont},
  box/.style={draw=black!50,rounded corners=2pt,align=center,
    minimum height=1.65cm,inner xsep=7pt,inner ysep=6pt},
  flow/.style={->,semithick},
  arrowlabel/.style={align=center,font=\footnotesize\linespread{1}\selectfont}
]
\node[box,text width=4cm] (target)
  {Candidate VI target\\[4pt] $b(\vartheta_{\star})$};
\node[box,text width=4cm,below=1.5cm of target] (trueTarget)
  {Observed VI target\\[4pt] $\overline\vartheta_0=b(\vartheta_0)$};
\node[box,text width=2.15cm,left=1.7cm of target] (dist)
  {Distribution\\[4pt] $\mathcal P_{\vartheta_{\star}}$};
\node[box,text width=2.15cm,left=1.7cm of trueTarget] (trueDist)
  {Distribution\\[4pt] $\mathcal P_{\vartheta_0}$};
\node[box,text width=1.85cm,left=0.85cm of dist] (theta)
  {Candidate\\[4pt] $\vartheta_{\star}$};
\node[box,text width=1.85cm,left=0.85cm of trueDist] (truth)
  {Truth\\[4pt] $\vartheta_0$};
\draw[flow] (theta) -- (dist);
\draw[flow] (truth) -- (trueDist);
\draw[flow] (dist) -- node[above=3pt,arrowlabel] {VI \\[2pt]$N\to\infty$} (target);
\draw[flow] (trueDist) -- node[above=3pt,arrowlabel] {VI\\[2pt]$N\to\infty$} (trueTarget);
\draw[<->,semithick] (target.south) --
  node[left=5pt,arrowlabel]
  {match}
  (trueTarget.north);
\draw[<->,semithick] (target.south) --
  node[right=5pt,arrowlabel]
  {$b(\vartheta_{\star})=\overline\vartheta_0 \quad
   \xRightarrow{\substack{\text{if } b \text{ is}\\ \text{one-to-one}}}
   \quad \vartheta_{\star}=\vartheta_0$}
  (trueTarget.north);
\end{tikzpicture}
\caption{Population Binding Function and IVI Matching}
\label{ivi_binding_function}
\vspace{0.2cm}  \figurenote{The binding function $b(\vartheta_{\star})$ maps each candidate data-generating parameter $\vartheta_{\star}$ into the corresponding population VI target. Under the true data-generating parameter $\vartheta_0$, this target is $\overline{\vartheta}_0=b(\vartheta_0)$, the pseudo-true parameter of the VI estimator. IVI varies $\vartheta_{\star}$ until $b(\vartheta_{\star})$ matches $\overline{\vartheta}_0$. If the binding function is one-to-one, this match uniquely recovers the true parameter $\vartheta_0$.}
\end{figure}
"""


def main():
    ap = argparse.ArgumentParser(
        description="Emit the IVI binding-function diagram as a LaTeX fragment.")
    ap.add_argument("-o", "--output", required=True, help="output .tex path")
    args = ap.parse_args()
    Path(args.output).write_text(TEX, encoding="utf-8")


if __name__ == "__main__":
    main()
