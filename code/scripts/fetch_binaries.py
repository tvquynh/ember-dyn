#!/usr/bin/env python3
"""Retrieve the original PE binaries for EMBER-Dyn samples, by SHA-256.

EMBER-Dyn does not redistribute raw PE binaries (see the manuscript,
"Obtaining raw binaries", and DATASHEET). Following the EMBER / EMBER2024
convention, the released manifest ships SHA-256 identifiers, and this helper
re-acquires the bytes on the reader's own machine from public malware-sharing
sources, keyed by hash:

  1. MalwareBazaar (abuse.ch) -- free Community API, requires a free Auth-Key.
     Indexes *confirmed malware only*: benign samples and older submissions are
     typically absent, so coverage is partial and skewed against the benign and
     long-tail cells.
  2. VirusTotal -- file download endpoint, *premium API only* (the free public
     API does not permit file download).

Each downloaded file is verified: its SHA-256 must equal the requested hash, or
it is discarded. Files already present in the output directory are skipped, so
the run is resumable.

WARNING -- this downloads LIVE malware. Run only inside an isolated,
network-restricted analysis VM. Do not execute the downloaded files on a host
you care about. The retrieved bytes are governed by the source platforms' terms
of use, not by the EMBER-Dyn licence.

Credentials are read from CLI flags or the environment (MB_API_KEY,
VT_API_KEY); nothing is hard-coded.

Examples
--------
    export MB_API_KEY=...            # free key from bazaar.abuse.ch
    export VT_API_KEY=...            # premium key (optional fallback)
    python fetch_binaries.py --manifest manifest.csv --out ./pe --limit 50

Dependencies: pandas, requests, tqdm; pyzipper (preferred, for AES ZIPs) with a
standard-library zipfile fallback. See requirements.txt.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import os
import sys
import time
import zipfile

import pandas as pd
import requests
from tqdm import tqdm

try:  # AES-encrypted ZIPs (MalwareBazaar) need pyzipper; ZipCrypto works with stdlib
    import pyzipper  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    pyzipper = None

MB_API = "https://mb-api.abuse.ch/api/v1/"
VT_API = "https://www.virustotal.com/api/v3/files/{sha256}/download"
ZIP_PASSWORD = b"infected"  # de-facto community password for malware archives


def sha256_of(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def extract_single(zip_bytes: bytes, password: bytes) -> bytes | None:
    """Return the single member of a password-protected ZIP, or None."""
    # Prefer pyzipper (handles AES + ZipCrypto); fall back to stdlib (ZipCrypto only).
    for opener in ((pyzipper.AESZipFile,) if pyzipper else ()) + (zipfile.ZipFile,):
        try:
            with opener(io.BytesIO(zip_bytes)) as zf:  # type: ignore[operator]
                names = [n for n in zf.namelist() if not n.endswith("/")]
                if not names:
                    return None
                return zf.read(names[0], pwd=password)
        except Exception:  # noqa: BLE001 - try the next opener (wrong cipher / bad zip)
            continue
    return None


def from_malwarebazaar(sha256: str, key: str, session: requests.Session) -> bytes | None:
    """Fetch a sample from MalwareBazaar as a password-protected ZIP."""
    resp = session.post(
        MB_API,
        data={"query": "get_file", "sha256_hash": sha256},
        headers={"Auth-Key": key},
        timeout=60,
    )
    ctype = resp.headers.get("Content-Type", "")
    if resp.status_code != 200 or "application/zip" not in ctype:
        return None  # error responses are JSON ({"query_status": "file_not_found"} etc.)
    return extract_single(resp.content, ZIP_PASSWORD)


def from_virustotal(sha256: str, key: str, session: requests.Session) -> bytes | None:
    """Fetch a sample from VirusTotal (premium download endpoint)."""
    resp = session.get(
        VT_API.format(sha256=sha256),
        headers={"x-apikey": key},
        timeout=120,
        allow_redirects=True,
    )
    if resp.status_code != 200 or not resp.content:
        return None
    return resp.content


def load_hashes(manifest: str, column: str, limit: int | None) -> list[str]:
    df = pd.read_csv(manifest, usecols=lambda c: c == column)
    hashes = (
        df[column].dropna().astype(str).str.strip().str.lower().unique().tolist()
    )
    if limit:
        hashes = hashes[:limit]
    return hashes


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, help="path to manifest.csv (or manifest_open.csv)")
    ap.add_argument("--out", default="./pe", help="output directory for retrieved binaries")
    ap.add_argument("--column", default="sha256", help="hash column in the manifest")
    ap.add_argument("--limit", type=int, default=None, help="process only the first N unique hashes")
    ap.add_argument("--mb-key", default=os.environ.get("MB_API_KEY"), help="MalwareBazaar Auth-Key (or env MB_API_KEY)")
    ap.add_argument("--vt-key", default=os.environ.get("VT_API_KEY"), help="VirusTotal premium key (or env VT_API_KEY)")
    ap.add_argument("--sleep", type=float, default=1.0, help="seconds to wait between requests (be polite)")
    args = ap.parse_args(argv)

    if not args.mb_key and not args.vt_key:
        ap.error("at least one of --mb-key / --vt-key (or MB_API_KEY / VT_API_KEY) is required")
    if pyzipper is None:
        print("[warn] pyzipper not installed; AES-encrypted ZIPs may fail "
              "(pip install pyzipper).", file=sys.stderr)

    os.makedirs(args.out, exist_ok=True)
    hashes = load_hashes(args.manifest, args.column, args.limit)
    if not hashes:
        print(f"[error] no hashes found in column '{args.column}' of {args.manifest}", file=sys.stderr)
        return 1
    session = requests.Session()

    got = skipped = failed = mismatch = 0
    for sha256 in tqdm(hashes, desc="fetch", unit="file"):
        dest = os.path.join(args.out, sha256)
        if os.path.exists(dest):
            skipped += 1
            continue

        data = None
        if args.mb_key:
            data = from_malwarebazaar(sha256, args.mb_key, session)
        if data is None and args.vt_key:
            data = from_virustotal(sha256, args.vt_key, session)

        if data is None:
            failed += 1
        elif sha256_of(data) != sha256:
            mismatch += 1  # integrity failure: do not write a wrong file
        else:
            with open(dest, "wb") as fh:
                fh.write(data)
            got += 1

        if args.sleep:
            time.sleep(args.sleep)

    total = len(hashes)
    print(
        f"\nretrieved {got} / {total}  "
        f"(skipped-existing {skipped}, not-found {failed}, hash-mismatch {mismatch}); "
        f"coverage {100.0 * (got + skipped) / total:.1f}% -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
