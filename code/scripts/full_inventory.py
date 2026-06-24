"""Comprehensive inventory + statistics for ember-dyn-staging/v1.

Covers:
  1. Directory tree + file sizes
  2. Manifest stats: per subset, family, dyn_split, ember_subset, file_type,
     time_bucket, label, confidence, detection_ratio histograms
  3. Capa/MITRE/MBC stats from parquet
  4. PE binary stats (size distribution)
  5. JSONL/parquet/PE inventory cross-modal
"""
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("./output/staging")

def section(title: str):
    print("=" * 78)
    print(f"  {title}")
    print("=" * 78)

def humanize(n: int) -> str:
    for u in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} PB"

def file_count_and_size(d: Path):
    files = [f for f in d.rglob("*") if f.is_file()]
    return len(files), sum(f.stat().st_size for f in files)

def main():
    section("1. DIRECTORY TREE + FILE SIZES")
    for sub in ["pe", "parquet", "jsonl", "meta", "logs", "scripts"]:
        d = ROOT / sub
        if not d.exists():
            print(f"{sub}/  (missing)")
            continue
        n_files, sz = file_count_and_size(d)
        print(f"{sub}/ ({n_files:,} files, {humanize(sz)})")
        if sub == "pe":
            for split in ["train", "val", "test", "challenge"]:
                sd = d / split
                n, s = file_count_and_size(sd)
                print(f"  {split}/  {n:,} files  {humanize(s)}")
        elif sub in {"parquet", "jsonl"}:
            for f in sorted(d.iterdir()):
                print(f"  {f.name}  {humanize(f.stat().st_size)}")
        else:
            for f in sorted(d.iterdir()):
                if f.is_file():
                    print(f"  {f.name}  {humanize(f.stat().st_size)}")

    # Root-level files
    root_files = [f for f in ROOT.iterdir() if f.is_file()]
    if root_files:
        print()
        for f in sorted(root_files):
            print(f"{f.name}  {humanize(f.stat().st_size)}")

    section("2. MANIFEST STATS")
    m = pd.read_csv(ROOT / "manifest.csv")
    print(f"Total rows: {len(m):,}  |  columns ({len(m.columns)}): {list(m.columns)}")

    print(f"\n--- by subset ---")
    print(m.groupby("subset").size().sort_values(ascending=False).to_string())

    print(f"\n--- by dyn_split ---")
    print(m["dyn_split"].value_counts().to_string())

    print(f"\n--- by ember_subset (where PE lives on source) ---")
    print(m["ember_subset"].value_counts().to_string())

    print(f"\n--- by label (binary) ---")
    print(m["label"].value_counts().to_string())

    print(f"\n--- by file_type ---")
    print(m["file_type"].value_counts().to_string())

    print(f"\n--- by time_bucket ---")
    print(m["time_bucket"].value_counts().to_string())

    print(f"\n--- top 25 family (by subset_class for A/B/C; family for E2/E3) ---")
    print(m["subset_class"].value_counts().head(25).to_string())

    print(f"\n--- subset × dyn_split crosstab ---")
    print(pd.crosstab(m["subset"], m["dyn_split"]).to_string())

    print(f"\n--- subset × time_bucket crosstab ---")
    print(pd.crosstab(m["subset"], m["time_bucket"]).to_string())

    print(f"\n--- file_type × dyn_split crosstab ---")
    print(pd.crosstab(m["file_type"], m["dyn_split"]).to_string())

    print(f"\n--- family_confidence distribution ---")
    fc = m["family_confidence"].dropna()
    print(f"  n={len(fc):,}  mean={fc.mean():.3f}  median={fc.median():.3f}  min={fc.min():.3f}  max={fc.max():.3f}")
    for lo, hi in [(0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]:
        n = ((fc >= lo) & (fc < hi)).sum()
        print(f"  [{lo:.1f}, {hi:.2f}): {n:,}")

    print(f"\n--- detection_ratio distribution ---")
    dr = m["detection_ratio"].astype(float).dropna()
    print(f"  n={len(dr):,}  mean={dr.mean():.3f}  median={dr.median():.3f}")
    for lo, hi in [(-0.001, 0), (0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.001)]:
        n = ((dr >= lo) & (dr < hi)).sum() if lo > 0 else (dr == 0).sum() if hi == 0 else ((dr >= lo) & (dr < hi)).sum()
        print(f"  [{lo:.3f}, {hi:.3f}): {n:,}")

    print(f"\n--- first_submission_date range ---")
    dates = pd.to_datetime(m["first_submission_date"], unit="s", errors="coerce")
    print(f"  range: {dates.min()}  →  {dates.max()}")
    print(f"  by year:")
    print(dates.dt.year.value_counts().sort_index().to_string())
    print(f"  by month (top 16):")
    print(dates.dt.to_period("M").value_counts().sort_index().to_string())

    section("3. CAPA / MITRE / MBC ANNOTATION STATS (from parquet)")
    parq_root = ROOT / "parquet"
    annot_dfs = []
    for split in ["train", "val", "test", "challenge"]:
        fp = parq_root / f"ember_dyn_{split}.parquet"
        if fp.exists():
            annot_dfs.append(pd.read_parquet(fp, columns=[
                "sha256", "behavior", "file_property", "packer", "exploit", "group",
                "n_caps", "n_ttps", "n_mbc"
            ]))
    annot = pd.concat(annot_dfs, ignore_index=True)
    print(f"Total rows: {len(annot):,}")

    for col in ["behavior", "file_property", "packer", "exploit", "group"]:
        annot[f"{col}_len"] = annot[col].apply(lambda x: len(x) if hasattr(x, '__len__') else 0)
        nonempty = int((annot[f"{col}_len"] > 0).sum())
        print(f"\n  {col}: nonempty={nonempty:,} ({100*nonempty/len(annot):.1f}%)")
        all_vals = []
        for lst in annot[col]:
            if hasattr(lst, '__len__'):
                for v in lst:
                    all_vals.append(str(v).lower())
        if all_vals:
            c = Counter(all_vals)
            for v, n in c.most_common(10):
                print(f"      {v}: {n:,}")

    print(f"\n  n_caps:  mean={annot['n_caps'].mean():.1f}  median={annot['n_caps'].median():.0f}  max={annot['n_caps'].max()}  nonzero={(annot['n_caps']>0).sum():,}")
    print(f"  n_ttps:  mean={annot['n_ttps'].mean():.1f}  median={annot['n_ttps'].median():.0f}  max={annot['n_ttps'].max()}  nonzero={(annot['n_ttps']>0).sum():,}")
    print(f"  n_mbc:   mean={annot['n_mbc'].mean():.1f}  median={annot['n_mbc'].median():.0f}  max={annot['n_mbc'].max()}  nonzero={(annot['n_mbc']>0).sum():,}")

    section("4. PE BINARY SIZE DISTRIBUTION")
    audit = pd.read_csv(ROOT / "meta" / "pe_size_audit.csv")
    print(f"Total PE files: {len(audit):,}")
    sizes = audit["size"].dropna()
    print(f"  min={humanize(int(sizes.min()))}  max={humanize(int(sizes.max()))}")
    print(f"  mean={humanize(int(sizes.mean()))}  median={humanize(int(sizes.median()))}")
    print(f"  total={humanize(int(sizes.sum()))}")
    print(f"\n  size buckets:")
    for lo, hi, lbl in [(0, 100e3, "<100 KB"), (100e3, 1e6, "100KB-1MB"),
                        (1e6, 5e6, "1-5 MB"), (5e6, 10e6, "5-10 MB"),
                        (10e6, 50e6, "10-50 MB"), (50e6, np.inf, ">50 MB")]:
        n = ((sizes >= lo) & (sizes < hi)).sum()
        print(f"    {lbl}: {n:,}")
    print(f"\n  by dyn_split:")
    print(audit.groupby("dyn_split")["size"].agg(["count", "sum", "mean", "median"]).to_string())

    section("5. JSONL STATS")
    for split in ["train", "val", "test", "challenge"]:
        fp = ROOT / "jsonl" / f"ember_dyn_{split}.jsonl"
        sz = fp.stat().st_size if fp.exists() else 0
        n = sum(1 for _ in open(fp, "r", encoding="utf-8", errors="replace")) if fp.exists() else 0
        avg = sz / n if n else 0
        print(f"  {split}: {n:,} records  {humanize(sz)} total  ~{humanize(int(avg))}/record")

    section("6. STORAGE SUMMARY")
    print(f"{'Component':<25s}{'Files':>10s}{'Size':>15s}")
    print("-" * 50)
    total_files, total_size = 0, 0
    for sub, label in [("pe", "PE binaries"), ("parquet", "Parquet features"),
                       ("jsonl", "JSONL raw"), ("meta", "Metadata"),
                       ("logs", "Logs"), ("scripts", "Scripts")]:
        d = ROOT / sub
        if d.exists():
            n, s = file_count_and_size(d)
            print(f"{label:<25s}{n:>10,d}{humanize(s):>15s}")
            total_files += n
            total_size += s
    root_files = [f for f in ROOT.iterdir() if f.is_file()]
    root_size = sum(f.stat().st_size for f in root_files)
    print(f"{'Root files':<25s}{len(root_files):>10,d}{humanize(root_size):>15s}")
    total_files += len(root_files)
    total_size += root_size
    print("-" * 50)
    print(f"{'TOTAL':<25s}{total_files:>10,d}{humanize(total_size):>15s}")

if __name__ == "__main__":
    main()
