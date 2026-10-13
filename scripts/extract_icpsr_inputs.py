"""Extract the three ICPSR 210782 input files into data/raw/ and verify them.

Usage: python scripts/extract_icpsr_inputs.py PATH/TO/ICPSR_210782-V2.zip

Searches the downloaded package (including zip files nested inside it) for
data.dta, tax9192.dta and natpr.dta, writes them to data/raw/, and checks
their SHA-256 against the values the package was built with. Prints only
file names, sizes and checksums. Stdlib only.
"""
import hashlib
import io
import sys
import zipfile
from pathlib import Path

EXPECTED = {
    "data.dta": "c1d1949748e224f03b80803b01ea7775cb043cae1da5e89a8717452591ecfd41",
    "tax9192.dta": "0af8a9b32b1b6d3c3d80e2f0830f662bf82c23f1d51a3c27d601f07efb17af81",
    "natpr.dta": "7044d5511d7a3caa079688fe3513043907162373403f7dd53bfeaf1fa2a4cd9b",
}
OUT = Path(__file__).resolve().parent.parent / "data" / "raw"


def find(zf, found, where):
    """Collect the wanted members of zf (recursing into nested zips)."""
    for info in zf.infolist():
        base = info.filename.rsplit("/", 1)[-1]
        if base in EXPECTED and base not in found:
            found[base] = (f"{where}:{info.filename}", zf.read(info))
        elif base.lower().endswith(".zip"):
            find(zipfile.ZipFile(io.BytesIO(zf.read(info))), found, f"{where}:{info.filename}")


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    found = {}
    find(zipfile.ZipFile(sys.argv[1]), found, Path(sys.argv[1]).name)
    OUT.mkdir(parents=True, exist_ok=True)
    ok = True
    for name, sha in EXPECTED.items():
        if name not in found:
            print(f"MISSING   {name}")
            ok = False
            continue
        src, blob = found[name]
        got = hashlib.sha256(blob).hexdigest()
        (OUT / name).write_bytes(blob)
        status = "OK" if got == sha else "MISMATCH"
        ok &= got == sha
        print(f"{status:9} {name:12} {len(blob):>10} bytes  {got}  (from {src})")
    print("all three inputs verified; they are in data/raw/" if ok else
          "verification FAILED: tell the authors which file differs")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
