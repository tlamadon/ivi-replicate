#!/usr/bin/env bash
# Build the Dataverse upload bundle dist/ivi-replicate-<version>.zip from the
# committed tree (git archive: untracked or ignored files, such as data/raw/*.dta
# or output/data/*.npy, can never end up in it), then verify that it holds no
# PSID data:
#   - no data-format file (.dta .npy .npz .parquet .feather .pkl .h5 .rds .sav
#     .csv .pth .pt ...) and nothing under data/ except data/raw/README.md;
#   - the only JSON results are the audited published outputs (aggregates).
# Writes dist/MANIFEST.txt (path, size, sha256 of every file in the bundle).
set -euo pipefail
cd "$(dirname "$0")/.."

[ -z "$(git status --porcelain)" ] || { echo "commit your changes first" >&2; exit 1; }
version=$(grep -m1 '^version = ' pyproject.toml | cut -d'"' -f2)
name="ivi-replicate-${version}"
mkdir -p dist
rm -rf "dist/${name}" "dist/${name}.zip"

git archive --format=tar --prefix="${name}/" HEAD -- . ':(exclude)CLAUDE.md' | tar -x -C dist

# ---- verification ---------------------------------------------------------
cd "dist/${name}"
fail=0
bad=$(find . -type f | grep -iE '\.(dta|npy|npz|parquet|feather|pkl|pickle|h5|hdf5|rds|rda|sav|sas7bdat|csv|pth|pt|xlsx?|dat)$' || true)
[ -z "$bad" ] || { echo "FORBIDDEN data-format files:"; echo "$bad"; fail=1; }

extra_data=$(find ./data -type f ! -path ./data/raw/README.md 2>/dev/null || true)
[ -z "$extra_data" ] || { echo "UNEXPECTED files under data/:"; echo "$extra_data"; fail=1; }

# Published outputs reviewed in the PSID audit: estimates, simulation results
# and aggregate sample statistics only.
expected_outputs="./output/bundles/2026-07-13-bpp-hetero-mean-indep-bundle.json
./output/bundles/ar1n_binding_mu1_bundle.json
./output/bundles/ar1n_mu1_profile_bundle.json
./output/bundles/ar1n_parameter_summary.json
./output/bundles/ar1n_quiver_bundle.json
./output/bundles/bundle_psid_heteroscale.json
./output/bundles/cells/_bpp_hetero_scale_ivi_bootstrap_k40/bootstrap_theta.json
./output/bundles/cells/_bpp_hetero_scale_ivi_bootstrap_k40_uncond/bootstrap_theta.json
./output/bundles/contour_bundle.json
./output/bundles/hetero_scale_parameter_summary.json
./output/bundles/hockeystick_parameter_summary.json
./output/bundles/hockeystick_parameter_summary_longt.json
./output/bundles/time-to-convergence-hockey-bundle.json
./output/data/bpp_selection_table.json"
actual_outputs=$(find ./output -type f | LC_ALL=C sort)
if [ "$actual_outputs" != "$(echo "$expected_outputs" | LC_ALL=C sort)" ]; then
  echo "output/ differs from the audited list:"
  diff <(echo "$expected_outputs" | LC_ALL=C sort) <(echo "$actual_outputs") || true
  fail=1
fi

big=$(find . -type f -size +2M)
[ -z "$big" ] || { echo "files over 2 MB (check them):"; echo "$big"; fail=1; }

[ "$fail" = 0 ] || { echo "bundle verification FAILED" >&2; exit 1; }

find . -type f | LC_ALL=C sort | while read -r f; do
  printf '%s\t%s\t%s\n' "${f#./}" "$(stat -c %s "$f")" "$(sha256sum "$f" | cut -d' ' -f1)"
done > ../MANIFEST.txt
cd ..
python3 -c "import shutil, sys; shutil.make_archive(sys.argv[1], 'zip', '.', sys.argv[1])" "${name}"
echo "verified: $(wc -l < MANIFEST.txt) files, no PSID data"
echo "wrote dist/${name}.zip ($(du -h "${name}.zip" | cut -f1)) and dist/MANIFEST.txt"
