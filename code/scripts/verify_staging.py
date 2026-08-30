"""Phase 5: verify EMBER-Dyn staging folder integrity.

Checks:
  1. PE count per dyn_split matches manifest
  2. PE files all have MZ magic bytes (sample 200)
  3. PE sha256 set == manifest sha256 set (per split)
  4. Parquet rows match manifest count per split
  5. Parquet sha256 set == manifest sha256 set (per split)
  6. Parquet feature columns == 2568
  7. JSONL records count matches manifest per split
  8. JSONL sha256 set == manifest sha256 set (per split)
  9. Cross-modality: 50 random sha256 exist in all 3 (PE + parquet + JSONL)
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd

PE_ROOT = Path("./output/staging/pe")
PARQUET_ROOT = Path("./output/staging/parquet")
JSONL_ROOT = Path("./output/staging/jsonl")
SPLITS = ["train", "val", "test", "challenge"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    args = ap.parse_args()

    manifest = pd.read_csv(args.manifest, usecols=["sha256", "dyn_split"])
    print(f"Manifest rows: {len(manifest):,}")

    failures = []
    successes = []

    def check(name: str, ok: bool, detail: str = ""):
        prefix = "PASS" if ok else "FAIL"
        print(f"  [{prefix}] {name}  {detail}")
        (successes if ok else failures).append((name, detail))

    print("\n=== 1. PE count per split ===")
    pe_sha_by_split: dict[str, set[str]] = {}
    for split in SPLITS:
        d = PE_ROOT / split
        if not d.exists():
            check(f"pe_dir_exists [{split}]", False, str(d))
            pe_sha_by_split[split] = set()
            continue
        sha_set = set(f.name for f in d.iterdir() if f.is_file())
        pe_sha_by_split[split] = sha_set
        expected = (manifest["dyn_split"] == split).sum()
        check(f"pe_count [{split}]", len(sha_set) == expected,
              f"got {len(sha_set):,} expected {expected:,}")

    print("\n=== 2. PE magic MZ (sample 200 random) ===")
    all_sha_disk = []
    for split in SPLITS:
        all_sha_disk.extend([(split, sha) for sha in pe_sha_by_split[split]])
    random.seed(42)
    sample = random.sample(all_sha_disk, min(200, len(all_sha_disk)))
    bad_magic = 0
    for split, sha in sample:
        with open(PE_ROOT / split / sha, "rb") as f:
            if f.read(2) != b"\x4d\x5a":
                bad_magic += 1
    check("pe_magic_mz_sample200", bad_magic == 0,
          f"bad_magic={bad_magic} / 200")

    print("\n=== 3. PE sha256 set match manifest per split ===")
    for split in SPLITS:
        manif = set(manifest.loc[manifest["dyn_split"] == split, "sha256"])
        disk = pe_sha_by_split[split]
        only_manif = manif - disk
        only_disk = disk - manif
        ok = len(only_manif) == 0 and len(only_disk) == 0
        check(f"pe_set [{split}]", ok,
              f"only_in_manifest={len(only_manif)} only_on_disk={len(only_disk)}")

    print("\n=== 4-6. Parquet checks ===")
    parquet_sha_by_split: dict[str, set[str]] = {}
    for split in SPLITS:
        fp = PARQUET_ROOT / f"ember_dyn_{split}.parquet"
        if not fp.exists():
            check(f"parquet_exists [{split}]", False, str(fp))
            parquet_sha_by_split[split] = set()
            continue
        df = pd.read_parquet(fp)
        expected = (manifest["dyn_split"] == split).sum()
        check(f"parquet_rows [{split}]", len(df) == expected,
              f"got {len(df):,} expected {expected:,}")

        feat_cols = [c for c in df.columns if c.startswith("feature_")]
        check(f"parquet_feature_dim [{split}]", len(feat_cols) == 2568,
              f"got {len(feat_cols)}")

        manif_set = set(manifest.loc[manifest["dyn_split"] == split, "sha256"])
        parq_set = set(df["sha256"])
        ok = manif_set == parq_set
        check(f"parquet_sha_set [{split}]", ok,
              f"manif^parq diff = {len(manif_set ^ parq_set)}")
        parquet_sha_by_split[split] = parq_set

    print("\n=== 7-8. JSONL checks ===")
    jsonl_sha_by_split: dict[str, set[str]] = {}
    for split in SPLITS:
        fp = JSONL_ROOT / f"ember_dyn_{split}.jsonl"
        if not fp.exists():
            check(f"jsonl_exists [{split}]", False, str(fp))
            jsonl_sha_by_split[split] = set()
            continue
        sha_set = set()
        with open(fp, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    sha_set.add(rec["sha256"])
                except (json.JSONDecodeError, KeyError):
                    pass
        jsonl_sha_by_split[split] = sha_set
        expected = (manifest["dyn_split"] == split).sum()
        check(f"jsonl_count [{split}]", len(sha_set) == expected,
              f"got {len(sha_set):,} expected {expected:,}")
        manif_set = set(manifest.loc[manifest["dyn_split"] == split, "sha256"])
        ok = sha_set == manif_set
        check(f"jsonl_sha_set [{split}]", ok,
              f"diff = {len(manif_set ^ sha_set)}")

    print("\n=== 9. Cross-modality: 50 random sha256 in all 3 ===")
    random.seed(123)
    sample = random.sample(manifest.to_dict("records"), min(50, len(manifest)))
    missing = 0
    for row in sample:
        sha, split = row["sha256"], row["dyn_split"]
        in_pe = sha in pe_sha_by_split[split]
        in_pq = sha in parquet_sha_by_split[split]
        in_js = sha in jsonl_sha_by_split[split]
        if not (in_pe and in_pq and in_js):
            missing += 1
    check("cross_modality_50sample", missing == 0,
          f"missing in any modality = {missing} / 50")

    print(f"\n=== Summary ===")
    print(f"PASS: {len(successes)}")
    print(f"FAIL: {len(failures)}")
    if failures:
        print("\nFailed checks:")
        for name, detail in failures:
            print(f"  - {name}: {detail}")
        raise SystemExit(1)
    print("\nAll checks PASSED ✓")

if __name__ == "__main__":
    main()
