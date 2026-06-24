"""Phase 4 — Filter all clean-week JSONL files into 4 dyn-split JSONL outputs.

Streams every .jsonl in ./data/ember2024/jsonl/, parses sha256 from each
line, and routes matched lines into one of train/val/test/challenge outputs.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

SRC = Path("./data/ember2024/jsonl")
DST = Path("./output/staging/jsonl")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    args = ap.parse_args()

    manifest = pd.read_csv(args.manifest, usecols=["sha256", "dyn_split"])
    sha_to_split = dict(zip(manifest["sha256"], manifest["dyn_split"]))
    print(f"Manifest sha256: {len(sha_to_split):,}")

    DST.mkdir(parents=True, exist_ok=True)
    out_files = {
        ds: open(DST / f"ember_dyn_{ds}.jsonl", "w", encoding="utf-8")
        for ds in ["train", "val", "test", "challenge"]
    }

    jsonl_files = sorted(SRC.glob("*.jsonl"))
    print(f"Source JSONL files: {len(jsonl_files)}")

    total_read = 0
    total_kept = 0
    kept_by_split = {ds: 0 for ds in out_files}
    t0 = time.time()

    for i, fp in enumerate(jsonl_files, 1):
        n_file = 0
        n_kept_file = 0
        with open(fp, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                n_file += 1
                try:
                    rec = json.loads(line)
                    sha = rec.get("sha256")
                    if sha and sha in sha_to_split:
                        ds = sha_to_split[sha]
                        out_files[ds].write(line)
                        kept_by_split[ds] += 1
                        n_kept_file += 1
                except json.JSONDecodeError:
                    continue
        total_read += n_file
        total_kept += n_kept_file
        if i % 25 == 0 or i == len(jsonl_files):
            elapsed = time.time() - t0
            print(f"  [{i}/{len(jsonl_files)}] {fp.name[:50]:<50s} "
                  f"read={n_file:,} kept={n_kept_file}  "
                  f"total_kept={total_kept:,}  elapsed={elapsed:.1f}s")

    for f in out_files.values():
        f.close()

    print(f"\nTotal read:   {total_read:,}")
    print(f"Total kept:   {total_kept:,}")
    print(f"Expected:     {len(sha_to_split):,}")
    print("By split:")
    for ds, n in kept_by_split.items():
        target = (manifest["dyn_split"] == ds).sum()
        ok = "OK" if n == target else f"MISMATCH (expected {target})"
        print(f"  {ds:<10s}: {n:,}  [{ok}]")

    if total_kept != len(sha_to_split):
        print(f"\nWARN: kept {total_kept} != manifest {len(sha_to_split)}")

if __name__ == "__main__":
    main()
