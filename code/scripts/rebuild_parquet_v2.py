"""Rebuild EMBER-Dyn parquet files with extended schema v2.

Adds 4 columns (sha1, caps_list, ttps_list, mbc_list) from raw JSONL.
Removes 3 redundant count columns (n_caps, n_ttps, n_mbc).
Appends 4 EMBER-Dyn-specific columns from manifest (subset, subset_class,
dyn_split, time_bucket).

Also generates 4 annotation parquets (long-form, sha256 -> capability pairs)
for direct use with sklearn.preprocessing.MultiLabelBinarizer.

Input  : dataset/static/<split>.parquet  (current 2,587 cols)
         dataset/static/<split>.jsonl
         dataset/manifest.csv  (must contain subset, subset_class, dyn_split, time_bucket)

Output : dataset/static/<split>.parquet  (rebuilt, 2,592 cols)
         dataset/annotations/capa_caps.parquet
         dataset/annotations/mitre_ttps.parquet
         dataset/annotations/mbc_behaviors.parquet
         dataset/annotations/group_apt.parquet
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(".")
STATIC = ROOT / "dataset" / "static"
ANNOT = ROOT / "dataset" / "annotations"
MANIFEST = ROOT / "dataset" / "manifest.csv"

SPLITS = ["train", "val", "test", "challenge"]
DROP_COUNTS = ["n_caps", "n_ttps", "n_mbc"]
DROP_REDUNDANT_MANIFEST = ["ember_subset", "det_ratio"]  # duplicates in manifest

def parse_jsonl_sidecar(jsonl_path: Path) -> pd.DataFrame:
    """Parse JSONL and extract per-sample sha1 + capability lists.

    Returns DataFrame indexed by sha256 with columns:
        sha1, caps_list, ttps_list, mbc_list
    """
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            caps = r.get("caps", []) or []
            ttps = r.get("ttps", []) or []
            mbc = r.get("mbc", []) or []
            # Normalize capability list elements to strings.
            # Many entries are objects like {"name": ., "namespace": .} for caps,
            # {"tactic": ., "technique": .} for ttps, {"objective": ., "behavior": .} for mbc.
            caps_str = [
                c if isinstance(c, str)
                else (c.get("Capability") or c.get("capability") or "" if isinstance(c, dict) else str(c))
                for c in caps
            ]
            ttps_str = [
                t if isinstance(t, str)
                else (t.get("Technique") or t.get("technique") or "" if isinstance(t, dict) else str(t))
                for t in ttps
            ]
            mbc_str = [
                m if isinstance(m, str)
                else (m.get("Behavior") or m.get("behavior") or "" if isinstance(m, dict) else str(m))
                for m in mbc
            ]
            # Filter empty strings
            caps_str = [c for c in caps_str if c]
            ttps_str = [t for t in ttps_str if t]
            mbc_str = [m for m in mbc_str if m]
            rows.append({
                "sha256": r["sha256"],
                "sha1": r.get("sha1", ""),
                "caps_list": caps_str,
                "ttps_list": ttps_str,
                "mbc_list": mbc_str,
            })
    df = pd.DataFrame(rows)
    # Dedupe by sha256 (JSONL may contain duplicates from source weekly files)
    df = df.drop_duplicates(subset="sha256", keep="first")
    return df.set_index("sha256")

def rebuild_split_parquet(split: str, manifest_df: pd.DataFrame) -> int:
    """Rebuild one split parquet with v2 schema. Returns row count."""
    parquet_path = STATIC / f"{split}.parquet"
    jsonl_path = STATIC / f"{split}.jsonl"

    print(f"  [{split}] reading parquet ({parquet_path.stat().st_size/1024/1024:.1f} MB).")
    df = pq.read_table(parquet_path).to_pandas()
    print(f"  [{split}] parquet rows: {len(df):,}, cols: {len(df.columns)}")

    print(f"  [{split}] parsing JSONL sidecar.")
    sidecar = parse_jsonl_sidecar(jsonl_path)
    print(f"  [{split}] sidecar rows: {len(sidecar):,}")

    # Drop redundant count columns
    df = df.drop(columns=[c for c in DROP_COUNTS if c in df.columns], errors="ignore")

    # Join sha1 + 3 capability list columns from JSONL sidecar
    df = df.merge(sidecar, left_on="sha256", right_index=True, how="left")
    # Position sha1 right after md5; caps_list/ttps_list/mbc_list right after group
    # Reorder: keep all metadata first in canonical order, then features
    canonical_meta = [
        "sha256", "md5", "sha1", "tlsh",
        "file_type", "label", "week_id",
        "first_submission_date", "last_analysis_date", "detection_ratio",
        "family", "family_confidence",
        "behavior", "file_property", "packer", "exploit", "group",
        "caps_list", "ttps_list", "mbc_list",
    ]
    feat_cols = [c for c in df.columns if c.startswith("feature_")]

    # Append 4 EMBER-Dyn-specific columns from manifest
    df = df.merge(
        manifest_df[["sha256", "subset", "subset_class", "dyn_split", "time_bucket"]],
        on="sha256", how="left",
    )
    ed_cols = ["subset", "subset_class", "dyn_split", "time_bucket"]

    # Final ordering: 20 EMBER2024 meta + 4 EMBER-Dyn meta + 2,568 features
    df = df[canonical_meta + ed_cols + feat_cols]
    print(f"  [{split}] rebuilt parquet cols: {len(df.columns)} (expect 2,592)")

    # Write back
    out_path = STATIC / f"{split}.parquet"
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, out_path, compression="snappy")
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"  [{split}] wrote {out_path.name} ({size_mb:.1f} MB)")
    return len(df)

def build_annotation_parquets():
    """Build 4 long-form annotation parquets from all rebuilt static parquets."""
    print()
    print("Building annotation parquets (long-form for MultiLabelBinarizer).")

    all_caps, all_ttps, all_mbc, all_group = [], [], [], []
    for split in SPLITS:
        df = pq.read_table(
            STATIC / f"{split}.parquet",
            columns=["sha256", "caps_list", "ttps_list", "mbc_list", "group"],
        ).to_pandas()

        def _iter(v):
            """Yield non-empty strings from a list/array/None."""
            if v is None:
                return
            for x in v:
                if x:
                    yield x
        for _, row in df.iterrows():
            sha = row["sha256"]
            for cap in _iter(row["caps_list"]):
                all_caps.append({"sha256": sha, "capability": cap, "namespace": ""})
            for ttp in _iter(row["ttps_list"]):
                all_ttps.append({"sha256": sha, "tactic": "", "technique": ttp})
            for mb in _iter(row["mbc_list"]):
                all_mbc.append({"sha256": sha, "objective": "", "behavior": mb})
            for grp in _iter(row["group"]):
                all_group.append({"sha256": sha, "group": grp})

    for rows, name in [
        (all_caps, "capa_caps"),
        (all_ttps, "mitre_ttps"),
        (all_mbc, "mbc_behaviors"),
        (all_group, "group_apt"),
    ]:
        out = ANNOT / f"{name}.parquet"
        if rows:
            pd.DataFrame(rows).to_parquet(out, compression="snappy", index=False)
            print(f"  {name}.parquet: {len(rows):,} rows ({out.stat().st_size/1024:.1f} KB)")
        else:
            # Empty placeholder
            pd.DataFrame(columns=["sha256"]).to_parquet(out, compression="snappy", index=False)
            print(f"  {name}.parquet: 0 rows (placeholder)")

def main():
    print("=" * 60)
    print("EMBER-Dyn parquet rebuild v2 (schema extension)")
    print("=" * 60)
    print(f"ROOT: {ROOT}")
    print()

    # Load manifest, drop redundant columns
    print("Loading manifest.")
    manifest = pd.read_csv(MANIFEST)
    manifest = manifest.drop(
        columns=[c for c in DROP_REDUNDANT_MANIFEST if c in manifest.columns],
        errors="ignore",
    )
    print(f"  Manifest: {len(manifest):,} rows, {len(manifest.columns)} cols")

    # Rebuild each split parquet
    print()
    print("Rebuilding split parquets.")
    total = 0
    for split in SPLITS:
        n = rebuild_split_parquet(split, manifest)
        total += n
    print(f"\n  Total rows across splits: {total:,} (expect 10,000)")

    # Build annotation parquets
    build_annotation_parquets()

    print()
    print("=" * 60)
    print("Rebuild complete.")
    print("=" * 60)

if __name__ == "__main__":
    main()
