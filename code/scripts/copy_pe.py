"""Phase 2 — Copy 10,481 PE binaries from EMBER source to ember-dyn staging.

Source: ./data/pe/{train,test,challenge}/<sha256>
Dest:   ./output/staging/pe/{train,val,test,challenge}/<sha256>

Mapping per row of manifest.csv:
  ember_subset = where PE lives on disk  (train/test/challenge)
  dyn_split    = where to put it          (train/val/test/challenge)
"""
from __future__ import annotations

import argparse
import shutil
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

SRC_ROOT = Path("./data/pe")
DST_ROOT = Path("./output/staging/pe")

def copy_one(row: dict) -> dict:
    sha = row["sha256"]
    ember_subset = row["ember_subset"]
    dyn_split = row["dyn_split"]

    src = SRC_ROOT / ember_subset / sha
    dst = DST_ROOT / dyn_split / sha

    if not src.exists():
        return {"sha256": sha, "dyn_split": dyn_split, "status": "src_missing"}
    if dst.exists():
        size = dst.stat().st_size
        return {"sha256": sha, "dyn_split": dyn_split, "size": size, "status": "exists_skip"}

    try:
        shutil.copy2(src, dst)
    except Exception as e:
        return {"sha256": sha, "dyn_split": dyn_split, "status": f"copy_err:{e}"}

    # Verify
    size = dst.stat().st_size
    try:
        with open(dst, "rb") as f:
            magic = f.read(2)
        magic_ok = magic == b"\x4d\x5a"
    except Exception:
        magic_ok = False

    return {
        "sha256": sha,
        "dyn_split": dyn_split,
        "ember_subset": ember_subset,
        "size": size,
        "magic_ok": magic_ok,
        "status": "copied" if magic_ok else "copied_bad_magic",
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--audit-out", default="./output/staging/meta/pe_size_audit.csv")
    args = ap.parse_args()

    manifest = pd.read_csv(args.manifest, usecols=["sha256", "ember_subset", "dyn_split"])
    print(f"Manifest rows: {len(manifest):,}")
    print(f"By dyn_split:\n{manifest['dyn_split'].value_counts().to_string()}")

    # Create destination dirs
    for split in ["train", "val", "test", "challenge"]:
        (DST_ROOT / split).mkdir(parents=True, exist_ok=True)
    print(f"\nDestination root: {DST_ROOT}")

    rows = manifest.to_dict(orient="records")
    print(f"\nCopying {len(rows):,} files with {args.workers} workers .")
    with Pool(processes=args.workers) as pool:
        results = pool.map(copy_one, rows, chunksize=64)

    # Save audit
    audit = pd.DataFrame(results)
    Path(args.audit_out).parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(args.audit_out, index=False)
    print(f"\nAudit saved: {args.audit_out}")

    # Report
    print("\n=== Status counts ===")
    print(audit["status"].value_counts().to_string())
    print("\n=== Per dyn_split ===")
    print(audit.groupby("dyn_split").size().to_string())

    # Sanity
    n_copied = (audit["status"].isin(["copied", "exists_skip"])).sum()
    n_total = len(manifest)
    print(f"\nCopied/Existing: {n_copied:,} / {n_total:,}")
    if "magic_ok" in audit.columns:
        n_magic = audit["magic_ok"].sum() if audit["magic_ok"].dtype == bool else (audit["magic_ok"] == True).sum()
        print(f"Magic MZ OK: {n_magic:,}")
    if n_copied != n_total:
        print("WARN: not all files were copied successfully")

if __name__ == "__main__":
    main()
