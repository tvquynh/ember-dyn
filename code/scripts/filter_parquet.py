"""Phase 3 — Filter clean-week parquet (3 ember splits) → 4 dyn-split parquets.

For each ember_subset (train/test/challenge), load the parquet and gather
matching sha256s, routing each to the proper dyn_split (train/val/test/challenge).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

CW = Path("./data/ember2024/parquet")
DST = Path("./output/staging/parquet")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    args = ap.parse_args()

    manifest = pd.read_csv(args.manifest, usecols=["sha256", "ember_subset", "dyn_split"])
    print(f"Manifest rows: {len(manifest):,}")

    # Group manifest sha256 by (ember_subset, dyn_split)
    routing = manifest.groupby(["ember_subset", "dyn_split"])["sha256"].apply(set).to_dict()

    DST.mkdir(parents=True, exist_ok=True)

    # For each dyn_split, accumulate matching rows across ember_subsets
    accumulators: dict[str, list[pd.DataFrame]] = {ds: [] for ds in ["train", "val", "test", "challenge"]}

    for ember_subset in ["train", "test", "challenge"]:
        print(f"\n[load] ember_{ember_subset}.parquet .")
        df = pd.read_parquet(CW / f"ember2024_{ember_subset}.parquet")
        print(f"  rows: {len(df):,} cols: {len(df.columns)}")

        for dyn_split in ["train", "val", "test", "challenge"]:
            wanted = routing.get((ember_subset, dyn_split), set())
            if not wanted:
                continue
            sub = df[df["sha256"].isin(wanted)]
            print(f"  → ({ember_subset} → {dyn_split}): {len(sub):,} rows")
            accumulators[dyn_split].append(sub)

    # Save per dyn_split
    print("\n[save] per dyn_split .")
    for dyn_split, parts in accumulators.items():
        if not parts:
            print(f"  {dyn_split}: empty, skip")
            continue
        merged = pd.concat(parts, ignore_index=True)
        before = len(merged)
        merged = merged.drop_duplicates(subset="sha256", keep="first")
        after = len(merged)
        if before != after:
            print(f"  {dyn_split}: dropped {before - after} duplicate sha256")
        out_path = DST / f"ember_dyn_{dyn_split}.parquet"
        merged.to_parquet(out_path, compression="snappy", index=False)
        size_mb = out_path.stat().st_size / 1e6
        print(f"  {dyn_split}: {len(merged):,} rows → {out_path.name} ({size_mb:.1f} MB)")

    print("\nDone.")

if __name__ == "__main__":
    main()
