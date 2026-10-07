# ml-earnings (replication branch) — notes for Claude

This branch is the frozen replication package: layer 1 builds the PSID data,
layer 2 regenerates the JSON bundles, layer 3 builds the paper's figures and
tables from them. Code lives in scripts/, mlye/, replicate.py; everything the
pipeline writes lives under output/{data,bundles,figures,manifests,logs}. The
published outputs are tracked at those same paths. See README.md.

- `replicate.py` is the single source of truth for what runs: every task,
  its layer, exact env settings, seeds, deps and resources, and the figure
  registry (`FIGURES`). Change settings there, never by editing script
  defaults. Regenerate `workflows/hut.replication_*.json` with `python replicate.py hut`
  after any change to layers 1-2 (tests fail otherwise).
- Entry points live in `scripts/`; figure/table generators are
  `scripts/fig_<name>.py` (stdlib + numpy, write an `\input`-able fragment),
  compiled by `scripts/fig_compile.py` with Tectonic (pinned in `mise.toml`).
- Regenerating overwrites the tracked published outputs in output/: check
  with `git diff output/` / `python replicate.py compare`; never commit
  regenerated bundles over the published ones unless that is the intent;
  `git checkout output/` restores them. Prefer `--workdir` for experiments.
- Every stochastic entry point must call `mlye.seeding.seed_everything`
  first (enforced by tests/test_replication.py).
- Keep `uv.lock`, `requirements.lock` and the `mlye-replication` stack in
  `scripthut.yaml` in sync (also tested). The stack must also exist in
  `main`'s scripthut.yaml: the server reads stacks from the source's branch.
  Workflows are found via the source's server-side `workflows_glob`
  (needs to match `workflows/hut.*.json`, e.g. `**/hut.*.json`).
- `python replicate.py run --smoke --layer 1,2 --workdir /tmp/x` exercises
  the compute DAG in minutes; `python replicate.py run --layer 3 --force`
  builds all figures from output/bundles in ~2 min;
  `uv run pytest` for the consistency checks.
- Do not commit the local `figures/` working folder (git-ignored).

Local GPU (NixOS): run inside `nix develop`, add a 64-bit zlib to
LD_LIBRARY_PATH, and set `TRITON_LIBCUDA_PATH=/run/opengl-driver/lib`.
GPU: RTX 5090, 32 GB (driver 595).
