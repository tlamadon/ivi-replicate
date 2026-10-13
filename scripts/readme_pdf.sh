#!/usr/bin/env bash
# Build README.pdf from README.md (JOLE requires the README as a PDF).
# Needs pandoc and xelatex on PATH, and the DejaVu fonts; on NixOS:
#   nix shell nixpkgs#pandoc -c scripts/readme_pdf.sh
set -euo pipefail
cd "$(dirname "$0")/.."
pandoc README.md -o README.pdf \
  --from markdown+task_lists-tex_math_dollars-raw_tex --columns=60 --lua-filter scripts/readme/pdf.lua \
  --pdf-engine=xelatex --toc --toc-depth=2 \
  -M title="Replication package for: Indirect Variational Inference: Applications to Earnings Dynamics" \
  -M author="Neele Balke, Stéphane Bonhomme and Thibaut Lamadon" \
  -M date="Journal of Labor Economics (MS 25521R1)" \
  -V geometry:margin=2cm -V fontsize=10pt -V colorlinks=true \
  -V mainfont="DejaVu Serif" -V sansfont="DejaVu Sans" -V monofont="DejaVu Sans Mono" \
  -V monofontoptions="Scale=0.8"
echo "wrote README.pdf"
