"""Layer 3: export the paper's figures and tables as EPS for the journal.

For every artifact not in the web appendix, writes
output/figures/eps/<number>_<name>.eps (e.g. fig3_ar1n_trajectory_fixed_point.eps):

  figures  the plot only: caption and note removed
  tables   the full float, caption and note included, numbered as in the paper

Each one is compiled alone on an empty page with Tectonic, then converted with
Ghostscript's eps2write, which writes a tight bounding box (no pdfcrop needed).
Requires `tectonic` and Ghostscript (`gs`) on PATH.
"""
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig_compile import FIGURES, TEX_DIR, check_fragments, compile_tex, head, kind  # noqa: E402

EPS_DIR = os.path.join("output", "figures", "eps")
WORK_DIR = os.path.join("output", "figures", "eps-work")


def strip_for(fig):
    if kind(fig) == "figure":
        return ("\\AtBeginDocument{\\renewcommand{\\caption}[2][]{}}\n"
                "\\renewcommand{\\figurenote}[1]{}\n")
    return "\\renewcommand{\\thetable}{" + fig["number"][3:] + "}\n"


def main():
    gs = shutil.which("gs")
    if not gs:
        sys.exit("Ghostscript (`gs`) not found on PATH: install it (e.g. `apt install ghostscript`, "
                 "`brew install ghostscript`) to export EPS")
    check_fragments()
    os.makedirs(EPS_DIR, exist_ok=True)
    os.makedirs(WORK_DIR, exist_ok=True)
    preamble = head()
    for fig in FIGURES:
        if fig.get("online"):
            continue
        base = f"{fig['name']}.eps-art"
        # compiled next to the fragments so \input finds them
        with open(os.path.join(TEX_DIR, base + ".tex"), "w", encoding="utf-8") as f:
            f.write(preamble + "\\pagestyle{empty}\n" + strip_for(fig) + "\\begin{document}\n"
                    f"\\input{{{fig['name']}.tex}}\n\\end{{document}}\n")
        pdf = compile_tex(base, pdf_dir=WORK_DIR)
        eps = os.path.join(EPS_DIR, f"{fig['number']}_{fig['name']}.eps")
        r = subprocess.run([gs, "-q", "-dNOPAUSE", "-dBATCH", "-dSAFER", "-r600",
                            "-sDEVICE=eps2write", "-o", eps, pdf],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if r.returncode != 0 or not os.path.exists(eps):
            sys.stdout.write(r.stdout[-2500:])
            sys.exit(f"EPS export failed for {fig['name']}")
        print(f"  exported {eps}")
    shutil.rmtree(WORK_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()
