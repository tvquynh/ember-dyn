"""Comprehensive PE <-> parquet <-> JSONL hash verification.

Four phases:
  1. Build SHA-256 reference sets from manifest.csv, 4 parquet files,
     4 JSONL files, and 4 PE folders.
  2. Set-equality cross-checks (fast, IO-light).
  3. Compute actual SHA-256 hash of each PE file content using 8 parallel
     workers, compare to filename stem + parquet/JSONL records.
  4. Report PASS/FAIL with full mismatch listing.

Report-after-all-done mode (no early exit on first mismatch).
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(".")
SPLITS = ["train", "val", "test", "challenge"]
N_WORKERS = 8
EXPECTED_COUNT = {"train": 7000, "val": 997, "test": 1907, "challenge": 96}

# -------------------------------------------------------------------
# Worker function (must be picklable, top-level)
# -------------------------------------------------------------------
def hash_pe(arg: tuple[str, str]) -> tuple[str, str, str, int]:
    """Hash one PE file. Returns (split, expected_stem, actual_sha256, size_bytes)."""
    split, path_str = arg
    p = Path(path_str)
    h = hashlib.sha256()
    size = 0
    with open(p, "rb") as f:
        while True:
            chunk = f.read(1 << 20)  # 1 MB chunks
            if not chunk:
                break
            h.update(chunk)
            size += len(chunk)
    return (split, p.stem.lower(), h.hexdigest(), size)

def main():
    t_start = time.time()

    # ---------------------------------------------------------------
    # Phase 1: Build reference SHA-256 sets
    # ---------------------------------------------------------------
    print("=" * 78)
    print("EMBER-Dyn PE hash verification")
    print("=" * 78)
    print()
    print("Phase 1: Loading reference SHA-256 sets.")
    p1 = time.time()

    manifest = pd.read_csv(ROOT / "dataset" / "manifest.csv")
    manifest_sha = set(manifest["sha256"].str.lower())
    print(f"  manifest.csv         : {len(manifest_sha):,} unique sha256")

    parquet_sha = {}
    for s in SPLITS:
        df = pq.read_table(
            ROOT / "dataset" / "static" / f"{s}.parquet",
            columns=["sha256"],
        ).to_pandas()
        parquet_sha[s] = set(df["sha256"].str.lower())
        print(f"  parquet [{s:9s}]: {len(parquet_sha[s]):,} unique sha256")

    jsonl_sha = {}
    for s in SPLITS:
        shas = set()
        with open(
            ROOT / "dataset" / "static" / f"{s}.jsonl",
            "r", encoding="utf-8"
        ) as f:
            for line in f:
                shas.add(json.loads(line)["sha256"].lower())
        jsonl_sha[s] = shas
        print(f"  JSONL   [{s:9s}]: {len(jsonl_sha[s]):,} unique sha256")

    pe_paths = {}
    pe_sha = {}
    for s in SPLITS:
        pe_dir = ROOT / "pe_local" / s
        paths = sorted(
            p for p in pe_dir.iterdir()
            if p.is_file() and p.suffix.lower() != ".md"
        )
        pe_paths[s] = paths
        pe_sha[s] = {p.stem.lower() for p in paths}
        print(f"  PE      [{s:9s}]: {len(pe_sha[s]):,} files")

    print(f"  (phase 1: {time.time() - p1:.1f}s)")

    # ---------------------------------------------------------------
    # Phase 2: Set equality cross-checks
    # ---------------------------------------------------------------
    print()
    print("Phase 2: Set-equality cross-checks.")
    p2 = time.time()

    issues_phase2 = []

    # Count checks
    for s in SPLITS:
        n_pq, n_jl, n_pe = len(parquet_sha[s]), len(jsonl_sha[s]), len(pe_sha[s])
        exp = EXPECTED_COUNT[s]
        if not (n_pq == n_jl == n_pe == exp):
            issues_phase2.append(
                f"  Count mismatch [{s}]: pq={n_pq} jl={n_jl} pe={n_pe} (expected {exp})"
            )

    # Per-split 3-way equality
    for s in SPLITS:
        diff_pq_jl = parquet_sha[s] ^ jsonl_sha[s]
        diff_pq_pe = parquet_sha[s] ^ pe_sha[s]
        diff_jl_pe = jsonl_sha[s] ^ pe_sha[s]
        if diff_pq_jl:
            issues_phase2.append(f"  [{s}] parquet XOR JSONL: {len(diff_pq_jl)} mismatched sha256")
        if diff_pq_pe:
            issues_phase2.append(f"  [{s}] parquet XOR PE files: {len(diff_pq_pe)} mismatched sha256")
        if diff_jl_pe:
            issues_phase2.append(f"  [{s}] JSONL XOR PE files: {len(diff_jl_pe)} mismatched sha256")

    # Manifest = union of all PE
    all_pe = set().union(*pe_sha.values())
    diff_manifest_pe = manifest_sha ^ all_pe
    if diff_manifest_pe:
        issues_phase2.append(f"  manifest XOR all-PE: {len(diff_manifest_pe)} mismatched sha256")

    if issues_phase2:
        print("  FAIL — set-equality issues found:")
        for line in issues_phase2:
            print(line)
    else:
        print("  PASS — all set-equality checks (parquet = JSONL = PE files = manifest)")

    print(f"  (phase 2: {time.time() - p2:.1f}s)")

    # ---------------------------------------------------------------
    # Phase 3: Compute SHA-256 of each PE file content
    # ---------------------------------------------------------------
    print()
    print(f"Phase 3: Computing SHA-256 of {sum(len(pe_paths[s]) for s in SPLITS):,} "
          f"PE files with {N_WORKERS} workers.")
    p3 = time.time()

    args = [(s, str(p)) for s in SPLITS for p in pe_paths[s]]
    total_files = len(args)

    with mp.Pool(N_WORKERS) as pool:
        results = []
        last_report = time.time()
        for i, res in enumerate(pool.imap_unordered(hash_pe, args, chunksize=50), 1):
            results.append(res)
            if time.time() - last_report > 5.0:
                pct = 100 * i / total_files
                rate = i / (time.time() - p3)
                eta = (total_files - i) / rate
                print(f"  .{i:5d} / {total_files} ({pct:5.1f}%)  "
                      f"{rate:.0f} files/s  ETA {eta:.0f}s")
                last_report = time.time()

    elapsed_phase3 = time.time() - p3
    total_bytes = sum(r[3] for r in results)
    print(f"  (phase 3: {elapsed_phase3:.1f}s, "
          f"{total_bytes / 1024 / 1024 / 1024:.2f} GB, "
          f"{total_bytes / 1024 / 1024 / elapsed_phase3:.0f} MB/s)")

    # ---------------------------------------------------------------
    # Phase 4: Compare actual hash vs expected (filename + parquet + JSONL)
    # ---------------------------------------------------------------
    print()
    print("Phase 4: Cross-comparison of actual hashes.")
    p4 = time.time()

    name_mismatches = []   # filename != actual hash
    parquet_only = []      # expected sha in parquet but missing in actual
    jsonl_only = []        # expected sha in JSONL but missing in actual

    # Build actual hash set per split
    actual_hashes = defaultdict(set)
    for split, stem, actual, size in results:
        actual_hashes[split].add(actual)
        if stem != actual:
            name_mismatches.append((split, stem, actual, size))

    # Cross-check actual_hashes set vs parquet/JSONL set per split
    for s in SPLITS:
        # Things in parquet but not in actual (should be empty if all hashes match)
        diff_pq_not_actual = parquet_sha[s] - actual_hashes[s]
        diff_jl_not_actual = jsonl_sha[s] - actual_hashes[s]
        if diff_pq_not_actual:
            parquet_only.append((s, len(diff_pq_not_actual), list(diff_pq_not_actual)[:5]))
        if diff_jl_not_actual:
            jsonl_only.append((s, len(diff_jl_not_actual), list(diff_jl_not_actual)[:5]))

    print(f"  (phase 4: {time.time() - p4:.1f}s)")

    # ---------------------------------------------------------------
    # Final report
    # ---------------------------------------------------------------
    print()
    print("=" * 78)
    print("FINAL REPORT")
    print("=" * 78)
    print(f"Total PE files hashed:   {total_files:,}")
    print(f"Total bytes hashed:      {total_bytes / 1024 / 1024 / 1024:.2f} GB")
    print(f"Total time:              {time.time() - t_start:.1f}s")
    print()
    print("PHASE 2  Set-equality cross-checks:")
    print(f"  {'FAIL' if issues_phase2 else 'PASS'} ({len(issues_phase2)} issues)")
    for line in issues_phase2[:20]:
        print(line)
    print()
    print("PHASE 3  PE file content hashing:")
    print(f"  Hashed: {total_files:,} files")
    print(f"  Filename matches actual hash: "
          f"{total_files - len(name_mismatches):,} / {total_files:,}")
    if name_mismatches:
        print(f"  FAIL — {len(name_mismatches)} files where filename != actual hash:")
        for split, stem, actual, size in name_mismatches[:20]:
            print(f"    {split}/{stem}.bin  size={size}  actual={actual}")
        if len(name_mismatches) > 20:
            print(f"    . and {len(name_mismatches) - 20} more")
    else:
        print("  PASS — every PE filename matches its actual SHA-256")

    print()
    print("PHASE 4  Cross-modality consistency:")
    if not parquet_only and not jsonl_only:
        print("  PASS — actual hashes match parquet and JSONL sha256 records exactly")
    else:
        if parquet_only:
            print(f"  FAIL — parquet sha256 not found in actual hashes:")
            for s, n, sample in parquet_only:
                print(f"    [{s}]: {n} mismatches; sample: {sample}")
        if jsonl_only:
            print(f"  FAIL — JSONL sha256 not found in actual hashes:")
            for s, n, sample in jsonl_only:
                print(f"    [{s}]: {n} mismatches; sample: {sample}")

    print()
    print("=" * 78)
    all_pass = (
        not issues_phase2
        and not name_mismatches
        and not parquet_only
        and not jsonl_only
    )
    print(f"  OVERALL: {'PASS' if all_pass else 'FAIL'}")
    print("=" * 78)

if __name__ == "__main__":
    main()
