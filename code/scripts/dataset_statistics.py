"""Comprehensive descriptive statistics for the consolidated EMBER-Dyn archive."""

from __future__ import annotations
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(".")
SPLITS = ["train", "val", "test", "challenge"]

def load_all() -> pd.DataFrame:
    """Load full 10,000-row table with all metadata columns."""
    parts = []
    for split in SPLITS:
        df = pq.read_table(
            ROOT / "dataset" / "static" / f"{split}.parquet",
            columns=[
                "sha256", "file_type", "label", "week_id",
                "first_submission_date", "detection_ratio",
                "family", "family_confidence",
                "behavior", "file_property", "packer", "exploit", "group",
                "caps_list", "ttps_list", "mbc_list",
                "subset", "subset_class", "dyn_split", "time_bucket",
            ],
        ).to_pandas()
        parts.append(df)
    return pd.concat(parts, ignore_index=True)

def fmt_pct(num: int, total: int) -> str:
    return f"{num:,} ({100.0 * num / total:5.2f}%)"

def list_len(s):
    """Length of list/array element, robust to None."""
    return s.apply(lambda v: 0 if v is None else len(v))

def print_block(title: str, char: str = "="):
    print()
    print(char * 78)
    print(f"  {title}")
    print(char * 78)

def main():
    df = load_all()
    N = len(df)

    print_block(f"EMBER-Dyn descriptive statistics  -  N = {N:,} samples", "=")

    # ----- 1. Overall split structure -----
    print_block("1. Split structure", "-")
    print(f"{'Split':12s} {'Count':>10s} {'%':>8s}")
    print("-" * 32)
    for s in SPLITS:
        n = (df["dyn_split"] == s).sum()
        print(f"  {s:10s} {n:>10,} {100*n/N:>7.2f}%")
    print(f"  {'TOTAL':10s} {N:>10,} {100.0:>7.2f}%")

    # ----- 2. Label distribution -----
    print_block("2. Label distribution", "-")
    benign = (df["label"] == 0).sum()
    malicious = (df["label"] == 1).sum()
    print(f"  benign      (label=0):  {fmt_pct(benign, N)}")
    print(f"  malicious   (label=1):  {fmt_pct(malicious, N)}")

    # ----- 3. File type distribution -----
    print_block("3. File type distribution", "-")
    ft = df["file_type"].value_counts()
    for k, v in ft.items():
        print(f"  {str(k):20s}: {fmt_pct(v, N)}")

    # ----- 4. Subset breakdown -----
    print_block("4. Subset composition", "-")
    subset_order = ["A", "B", "C", "D", "E1", "E2", "E3", "F", "Challenge"]
    subset_purpose = {
        "A": "Known family (multi-class 5-way)",
        "B": "Open-set pool (out-of-distribution)",
        "C": "Few-shot pool (meta-learning)",
        "D": "Benign (binary detection negatives)",
        "E1": "Exploit-bearing rare gems",
        "E2": "APT group attribution",
        "E3": "Packer non-UPX",
        "F": "Broad-coverage family pool",
        "Challenge": "Time-distinct hold-out",
    }
    print(f"{'Subset':12s} {'Count':>8s} {'%':>7s}  Purpose")
    print("-" * 78)
    for sub in subset_order:
        n = (df["subset"] == sub).sum()
        print(f"  {sub:10s} {n:>8,} {100*n/N:>6.2f}%  {subset_purpose[sub]}")

    # ----- 5. Family distribution -----
    print_block("5. Family distribution (AVClass)", "-")
    families = df["family"].dropna()
    families = families[families != ""]
    n_unique = families.nunique()
    print(f"  Unique families:      {n_unique}")
    print(f"  Samples with family:  {fmt_pct(len(families), N)}")
    print(f"  Samples without family: {fmt_pct(N - len(families), N)}")
    print()
    print("  Top 20 families by sample count:")
    top20 = families.value_counts().head(20)
    for fam, n in top20.items():
        print(f"    {str(fam):24s}: {n:>5,}")

    # ----- 6. Time distribution -----
    print_block("6. Time bucket distribution", "-")
    tb = df["time_bucket"].value_counts()
    for k, v in tb.items():
        print(f"  {str(k):10s} (relative to 2024-09 cut): {fmt_pct(v, N)}")

    print()
    print("  Monthly first_submission_date breakdown:")
    df["fsd"] = pd.to_datetime(df["first_submission_date"], unit="s", errors="coerce")
    df["fsd_month"] = df["fsd"].dt.to_period("M").astype(str)
    monthly = df["fsd_month"].value_counts().sort_index()
    for m, n in monthly.items():
        bar = "#" * int(40 * n / monthly.max())
        print(f"    {m}  {n:>5,}  {bar}")

    # ----- 7. Detection ratio -----
    print_block("7. Detection ratio distribution", "-")
    mal = df[df["label"] == 1]
    ben = df[df["label"] == 0]
    print(f"  Benign  (n={len(ben):,}): mean={ben['detection_ratio'].mean():.3f}, max={ben['detection_ratio'].max():.3f}")
    print(f"  Malicious (n={len(mal):,}):")
    print(f"    mean={mal['detection_ratio'].mean():.3f}, median={mal['detection_ratio'].median():.3f}")
    print()
    bands = [(0.0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.001)]
    print("  Malicious sample bands:")
    for lo, hi in bands:
        n = ((mal["detection_ratio"] >= lo) & (mal["detection_ratio"] < hi)).sum()
        label = f"[{lo:.1f}, {hi:.1f})" if hi <= 1.0 else f"[{lo:.1f}, 1.0]"
        print(f"    {label:14s}: {fmt_pct(n, len(mal))}")

    # ----- 8. AVClass family confidence -----
    print_block("8. AVClass family_confidence (malicious only)", "-")
    fc = mal["family_confidence"].dropna()
    print(f"  Samples with confidence:  {len(fc):,}")
    print(f"  mean: {fc.mean():.3f}   median: {fc.median():.3f}   min: {fc.min():.3f}   max: {fc.max():.3f}")
    bins = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0001]
    labels = ["[0.0, 0.3)", "[0.3, 0.5)", "[0.5, 0.7)", "[0.7, 0.9)", "[0.9, 1.0]"]
    counts = pd.cut(fc, bins=bins, include_lowest=True, labels=labels).value_counts().sort_index()
    print("  Confidence bands:")
    for k, n in counts.items():
        print(f"    {str(k):14s}: {fmt_pct(n, len(fc))}")

    # ----- 9. Annotation coverage -----
    print_block("9. Annotation coverage (population-wide)", "-")

    def coverage(col: str, label: str):
        n = (list_len(df[col]) > 0).sum()
        print(f"  {label:36s}: {fmt_pct(n, N)}")
        return n

    coverage("behavior",      "behavior  tag (capa behaviour)")
    coverage("file_property", "file_property tag")
    coverage("packer",        "packer tag")
    coverage("exploit",       "exploit tag")
    coverage("group",         "group (APT) tag")
    print()
    coverage("caps_list",     "caps_list (full capa identifiers)")
    coverage("ttps_list",     "ttps_list (full MITRE ATT&CK techniques)")
    coverage("mbc_list",      "mbc_list (full MBC behaviour identifiers)")

    # ----- 10. Capa / TTP / MBC statistics per sample -----
    print_block("10. Capa / TTP / MBC counts per sample", "-")
    for col in ["caps_list", "ttps_list", "mbc_list"]:
        lens = list_len(df[col])
        nz = lens[lens > 0]
        print(f"  {col:12s}: mean={lens.mean():6.2f}   median={int(lens.median())}   max={lens.max():>4d}   non-zero mean={nz.mean():6.2f}")

    # ----- 11. Top capa capabilities -----
    print_block("11. Top capa capabilities (across all samples)", "-")
    capa_counter = Counter()
    for lst in df["caps_list"]:
        if lst is None:
            continue
        capa_counter.update(lst)
    print(f"  Unique capa capabilities: {len(capa_counter):,}")
    print(f"  Total occurrences:        {sum(capa_counter.values()):,}")
    print()
    print("  Top 20 capa capabilities:")
    for cap, n in capa_counter.most_common(20):
        print(f"    {cap[:60]:60s}  {n:>6,}")

    # ----- 12. Top MITRE ATT&CK techniques -----
    print_block("12. Top MITRE ATT&CK techniques", "-")
    ttp_counter = Counter()
    for lst in df["ttps_list"]:
        if lst is None:
            continue
        ttp_counter.update(lst)
    print(f"  Unique techniques:  {len(ttp_counter):,}")
    print(f"  Total occurrences:  {sum(ttp_counter.values()):,}")
    print()
    print("  Top 20 techniques:")
    for ttp, n in ttp_counter.most_common(20):
        print(f"    {ttp[:60]:60s}  {n:>6,}")

    # ----- 13. Top MBC behaviours -----
    print_block("13. Top MBC behaviours", "-")
    mbc_counter = Counter()
    for lst in df["mbc_list"]:
        if lst is None:
            continue
        mbc_counter.update(lst)
    print(f"  Unique behaviours:  {len(mbc_counter):,}")
    print(f"  Total occurrences:  {sum(mbc_counter.values()):,}")
    print()
    print("  Top 20 behaviours:")
    for b, n in mbc_counter.most_common(20):
        print(f"    {b[:60]:60s}  {n:>6,}")

    # ----- 14. APT group distribution (full) -----
    print_block("14. APT group distribution (full list)", "-")
    grp_counter = Counter()
    for lst in df["group"]:
        if lst is None:
            continue
        grp_counter.update(lst)
    print(f"  Unique APT groups:  {len(grp_counter):,}")
    print(f"  Total tagged:       {sum(grp_counter.values()):,}")
    print()
    for g, n in grp_counter.most_common():
        print(f"    {g:24s}  {n:>5,}")

    # ----- 15. Exploit distribution (full) -----
    print_block("15. Exploit distribution (full list of CVEs / IDs)", "-")
    exp_counter = Counter()
    for lst in df["exploit"]:
        if lst is None:
            continue
        exp_counter.update(lst)
    print(f"  Unique exploits:    {len(exp_counter):,}")
    print(f"  Total tagged:       {sum(exp_counter.values()):,}")
    print()
    for e, n in exp_counter.most_common():
        print(f"    {e:28s}  {n:>5,}")

    # ----- 16. Packer distribution (full) -----
    print_block("16. Packer distribution (full list)", "-")
    pkg_counter = Counter()
    for lst in df["packer"]:
        if lst is None:
            continue
        pkg_counter.update(lst)
    print(f"  Unique packers:     {len(pkg_counter):,}")
    print(f"  Total tagged:       {sum(pkg_counter.values()):,}")
    print()
    for p, n in pkg_counter.most_common():
        print(f"    {p:24s}  {n:>5,}")

    # ----- 17. Behavior tag distribution (top) -----
    print_block("17. Behavior tag distribution (top 20)", "-")
    beh_counter = Counter()
    for lst in df["behavior"]:
        if lst is None:
            continue
        beh_counter.update(lst)
    print(f"  Unique behaviour tags: {len(beh_counter):,}")
    print(f"  Total tagged:           {sum(beh_counter.values()):,}")
    print()
    for b, n in beh_counter.most_common(20):
        print(f"    {b:24s}  {n:>5,}")

    # ----- 18. Per-split breakdown (label x subset) -----
    print_block("18. Per-split cross-tabs", "-")
    print()
    print("  Split x Subset:")
    pivot = pd.crosstab(df["dyn_split"], df["subset"], margins=True)
    pivot = pivot.reindex(index=SPLITS + ["All"], columns=subset_order + ["All"], fill_value=0)
    print(pivot.to_string())
    print()
    print("  Split x file_type:")
    pivot = pd.crosstab(df["dyn_split"], df["file_type"], margins=True)
    pivot = pivot.reindex(index=SPLITS + ["All"], fill_value=0)
    print(pivot.to_string())
    print()
    print("  Split x label:")
    pivot = pd.crosstab(df["dyn_split"], df["label"], margins=True)
    pivot = pivot.reindex(index=SPLITS + ["All"], fill_value=0)
    pivot.columns = ["benign", "malicious", "All"]
    print(pivot.to_string())

    # ----- 19. PE binary on-disk statistics -----
    print_block("19. PE binary on-disk statistics", "-")
    total_bytes = 0
    sizes = []
    for split in SPLITS:
        pe_dir = ROOT / "pe_local" / split
        for p in pe_dir.iterdir():
            if p.is_file() and p.suffix.lower() != ".md":
                sz = p.stat().st_size
                sizes.append(sz)
                total_bytes += sz
    sizes = np.array(sizes)
    print(f"  Total PE files:        {len(sizes):,}")
    print(f"  Total disk usage:      {total_bytes / 1024 / 1024 / 1024:.2f} GB")
    print(f"  Mean file size:        {sizes.mean() / 1024:.1f} KB")
    print(f"  Median file size:      {np.median(sizes) / 1024:.1f} KB")
    print(f"  Min file size:         {sizes.min() / 1024:.1f} KB")
    print(f"  Max file size:         {sizes.max() / 1024 / 1024:.2f} MB")
    print(f"  Std file size:         {sizes.std() / 1024:.1f} KB")

    # ----- 20. Disk footprint of the consolidated archive -----
    print_block("20. Consolidated archive disk footprint", "-")
    for name, sub in [
        ("paper/",     "paper"),
        ("code/",      "code"),
        ("dataset/",   "dataset"),
        ("pe_local/",  "pe_local"),
    ]:
        sub_path = ROOT / sub
        total = sum(p.stat().st_size for p in sub_path.rglob("*") if p.is_file())
        print(f"  {name:14s} {total / 1024 / 1024 / 1024:>8.2f} GB  ({total / 1024 / 1024:,.0f} MB)")

    print()
    print("=" * 78)
    print(f"  End of statistics. Run finished at {datetime.now().isoformat(timespec='seconds')}")
    print("=" * 78)

if __name__ == "__main__":
    main()
