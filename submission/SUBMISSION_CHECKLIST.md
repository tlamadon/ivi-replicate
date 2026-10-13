# JOLE final submission checklist — MS 25521R1

*Indirect Variational Inference: Applications to Earnings Dynamics*
(Balke, Bonhomme, Lamadon). Branch `jole-final`.

Legend: ✅ done · 🟡 prepared, needs a step from you · ⬜ not started ·
⏸ deferred · **[YOU]** = needs you (or a co-author), not Claude.

## Replication package

| Item | Status | Notes |
|---|---|---|
| PSID audit of repo and git history | ✅ | No PSID microdata in the tree, history, tags or release assets. Tracked outputs are estimates and aggregate statistics only. |
| History purge | ✅ not needed | Nothing PSID-derived was ever committed. |
| Inputs in git-ignored `data/raw/`, data formats blocked in `.gitignore` | ✅ | `data/raw/README.md` documents files and checksums. |
| Single entry point | ✅ | `run_all.sh` (layers 1 → 2 → 3, data check, compare). |
| Pinned environment | ✅ | `uv.lock`, `requirements.lock`, `mise.toml`. |
| Runtime and hardware documented | ✅ | From the 2026-10-05 replication run manifests. |
| README (SSDE template), Markdown + PDF | ✅ | `README.md`, `README.pdf` (`scripts/readme_pdf.sh`). |
| LICENSE | 🟡 **[YOU]** | MIT (code) + CC BY 4.0 (results, docs) is a default; confirm with co-authors. |
| Confirm ICPSR 210782 version | 🟡 **[YOU]** | README cites V2. Check that a fresh V2 download matches the checksums in `data/raw/README.md`. |
| PSID citation year | 🟡 **[YOU]** | Fill `[YEAR-OF-DOWNLOAD-PLACEHOLDER]` in README References. |
| Full run from a clean clone | ⬜ **[YOU]** | `./run_all.sh` on a machine with the data; check that `verify-data` passes and `compare` is clean. Optional, since the 2026-10-05 run already exists. |
| Dataverse bundle built and verified | ✅ | `scripts/make_dataverse_bundle.sh` → `dist/ivi-replicate-<version>.zip` + `dist/MANIFEST.txt`. Rebuild after the last commit. |
| Dataverse metadata | 🟡 **[YOU]** | `submission/DATAVERSE_METADATA.txt`; fill affiliations, ORCIDs, contact email. |
| **Dataverse dataset created and cited** | ⬜ **[YOU]** | Create in the JOLE collection (https://dataverse.harvard.edu/dataverse/JOLE, not "JOLEE"), upload, then put the DOI in README (`[DATAVERSE-DOI-PLACEHOLDER]`, rebuild PDF) and in the paper. |
| **ICPSR–PSID extract deposited and cited** | ✅ not needed | We use the BPP deposit unchanged (ICPSR 210782 V2, already in the ICPSR–PSID Data Extract Repository) and share no PSID files of our own. Cite `10.3886/E210782V2` + the PSID. Re-check if the acceptance letter or the "Instructions for Authors of Accepts" says otherwise. |
| Push `jole-final`, tag a release | ⬜ **[YOU]** | Claude does not push. Consider bumping `version` in `pyproject.toml` (and `uv.lock`) before tagging, so the bundle name matches. |

## Manuscript (Editorial Manager) — deferred

| Item | Status | Notes |
|---|---|---|
| Data citations in text and bibliography (Dataverse package; BPP/ICPSR 210782; PSID) | ⏸ | Needs the LaTeX repo and the "Instructions for Authors of Accepts" (JOLE files zip). |
| PSID funding acknowledgment | ⏸ **[YOU]** | PSID asks publications to acknowledge its funding sources; check the PSID citation FAQ for current wording. |
| **Editable .tex bundle** (flat folder, compiles from a clean copy) | ⏸ | Needs the LaTeX repo. |
| **Separate figure files** (.eps) | 🟡 | `fig-eps` (layer 3) already writes `output/figures/eps/<number>_<name>.eps` for every main-text item. Still to do: black-and-white readability check. |
| **Alt text** after each caption | ⏸ | Draft from plotting code and captions; follow JOLE alt-text length guidance. |
| **Web-only appendix PDF** | ⏸ | Needs the LaTeX repo. |
| Numbers in the text checked against the bundles | ⏸ | README leaves "all numbers in text" unchecked until this is done. |

## Publisher forms and author details

| Item | Status | Notes |
|---|---|---|
| **Publication agreement** signed by all three authors (or one on behalf of all) | ⬜ **[YOU]** | |
| **Open-access decision** | ⬜ **[YOU]** | |
| **Author emails and postal addresses** entered in Editorial Manager | ⬜ **[YOU]** | All three authors. |
| PubMed Central (PMCID) | ⬜ **[YOU]** | PSID requests that articles using its data get a PMCID. Check whether the journal deposits for you or whether you must submit to PMC. |
