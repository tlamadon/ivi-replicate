#!/usr/bin/env bash
# Full replication: data construction -> estimation -> figures and tables.
#
#   ./run_all.sh                     # layers 1-3, using the published bootstrap
#   ./run_all.sh --with-bootstrap    # also rerun the 2 x 100 bootstrap (~210 GPU-hours)
#   ./run_all.sh --figures-only      # layer 3 only, from the published bundles (~2 min, no data, no GPU)
#
# Needs the three ICPSR 210782 files in data/raw/ (see data/raw/README.md)
# unless --figures-only. Every step is a `replicate.py` call; see README.md.
set -euo pipefail
cd "$(dirname "$0")"

RUN=(uv run --frozen python replicate.py)
BOOT=()
case "${1:-}" in
  --figures-only)
    "${RUN[@]}" run --layer 3 --force
    exit 0 ;;
  --with-bootstrap) BOOT=(--with-bootstrap) ;;
  "") ;;
  *) echo "usage: $0 [--with-bootstrap | --figures-only]" >&2; exit 2 ;;
esac

RAW="${BPP_EXT:-data/raw}"
for f in data.dta tax9192.dta natpr.dta; do
  if [ ! -f "$RAW/$f" ]; then
    echo "missing $RAW/$f: download it from ICPSR 210782 (see data/raw/README.md)" >&2
    exit 1
  fi
done

"${RUN[@]}" run --layer 1 --force                 # PSID earnings matrix + sample statistics
"${RUN[@]}" verify-data                           # sha256 of output/data/bpp_y_matrix.npy
"${RUN[@]}" run --layer 2 --force "${BOOT[@]}"    # estimations and simulations -> bundles
"${RUN[@]}" run --layer 3 --force                 # figures and tables
"${RUN[@]}" compare                               # regenerated vs published bundles
