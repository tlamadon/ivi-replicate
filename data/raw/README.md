# data/raw — input data (not included)

The analysis starts from three files in the replication package of Blundell,
Pistaferri and Preston (2008, AER), openICPSR project 210782, version V2:
<https://www.openicpsr.org/openicpsr/project/210782/view>.

They are PSID-derived and cannot be redistributed. Download them yourself
(free openICPSR account; accept the ICPSR and PSID terms of use) and place
them in this directory:

| File          | Content                                              |
|---------------|------------------------------------------------------|
| `data.dta`    | BPP's PSID household-year panel (from create1/create2) |
| `tax9192.dta` | TAXSIM federal income taxes for 1991–92              |
| `natpr.dta`   | National price index (`cpi`)                         |

Expected SHA-256 checksums (`sha256sum data/raw/*.dta`):

```text
c1d1949748e224f03b80803b01ea7775cb043cae1da5e89a8717452591ecfd41  data.dta
0af8a9b32b1b6d3c3d80e2f0830f662bf82c23f1d51a3c27d601f07efb17af81  tax9192.dta
7044d5511d7a3caa079688fe3513043907162373403f7dd53bfeaf1fa2a4cd9b  natpr.dta
```

`python3 scripts/extract_icpsr_inputs.py PATH/TO/the-downloaded.zip` extracts
the three files from the download into this directory and checks them.
To keep them elsewhere, set `BPP_EXT` to their directory. Everything in this
directory except this README is git-ignored.
