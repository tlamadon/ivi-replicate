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
| LICENSE | ✅ | MIT License for the whole package (decided 2026-10-13). |
| ICPSR 210782 reference | ✅ | Checked 2026-10-13 against the DOI registry: V2 (issued 2025) is the latest version; DOI 10.3886/E210782V2 resolves to the ICPSR–PSID Data Extract Repository (landing page OK); citation year 2025. |
| ICPSR 210782 download matches the package | ✅ | Verified 2026-10-13 with `scripts/extract_icpsr_inputs.py`: the V2 download (`ICPSR_210782-V2.1.zip`) gives all three files with the expected SHA-256. |
| PSID citation year | ✅ | 2025. |
| Full run from a clean clone | ⬜ **[YOU]** | `./run_all.sh` on a machine with the data; check that `verify-data` passes and `compare` is clean. Optional, since the 2026-10-05 run already exists. |
| Dataverse bundle built and verified | ✅ | `scripts/make_dataverse_bundle.sh` → `dist/ivi-replicate-<version>.zip` + `dist/MANIFEST.txt`. Rebuild after the last commit. |
| Dataverse metadata | ✅ | Entered in Dataverse (draft saved and submitted). |
| **Dataverse dataset created and cited** | 🟡 | Dataset doi:10.7910/DVN/X7CAT0 (JOLE collection); zip uploaded and submitted for review 2026-10-13; DOI in README. Remaining: JOLE data editor's review/publication; cite the DOI in the paper (manuscript tasks); update "Related Publication" once the article DOI exists. |
| **ICPSR–PSID extract deposited and cited** | ✅ not needed | We use the BPP deposit unchanged (ICPSR 210782 V2, already in the ICPSR–PSID Data Extract Repository) and share no PSID files of our own. Cite `10.3886/E210782V2` + the PSID. Re-check if the acceptance letter or the "Instructions for Authors of Accepts" says otherwise. |
| Push `jole-final` | ✅ | Pushed 2026-10-13. Optional: tag a release. |

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
