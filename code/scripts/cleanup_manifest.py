"""Cleanup manifest.csv — drop 5 legacy columns to match paper claim (24 cols).

Legacy columns dropped:
  - det_ratio     (replaced by detection_ratio)
  - ember_subset  (replaced by subset)
  - n_caps        (full caps_list now in parquet)
  - n_ttps        (full ttps_list now in parquet)
  - n_mbc         (full mbc_list now in parquet)
"""

import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(".")
mf = ROOT / "dataset" / "manifest.csv"

print(f"Loading {mf}")
df = pd.read_csv(mf)
print(f"  Before: {len(df):,} rows x {len(df.columns)} cols")

legacy_cols = ["det_ratio", "ember_subset", "n_caps", "n_ttps", "n_mbc"]
to_drop = [c for c in legacy_cols if c in df.columns]
df = df.drop(columns=to_drop)
print(f"  Dropped: {to_drop}")

# Pull sha1 + caps_list + ttps_list + mbc_list from parquets to match paper Table 3.
# For list[str] cols, JSON-serialize for CSV portability.
print(f"  Pulling sha1 + caps_list + ttps_list + mbc_list from parquets.")
extra_cols = ["sha1", "caps_list", "ttps_list", "mbc_list"]
parts = []
for split in ["train", "val", "test", "challenge"]:
    p = ROOT / "dataset" / "static" / f"{split}.parquet"
    table = pq.read_table(p, columns=["sha256"] + extra_cols).to_pandas()
    parts.append(table)
extra_df = pd.concat(parts, ignore_index=True)

# JSON-serialise list-typed cols for CSV portability
for c in ["caps_list", "ttps_list", "mbc_list"]:
    extra_df[c] = extra_df[c].apply(
        lambda v: json.dumps(list(v)) if v is not None else "[]"
    )

# Merge by sha256 (left join — manifest is authoritative)
df = df.merge(extra_df, on="sha256", how="left")

# Reorder to canonical 24-col schema (paper Table 3)
canonical = [
    "sha256", "md5", "sha1", "tlsh",
    "file_type", "label", "week_id",
    "first_submission_date", "last_analysis_date",
    "detection_ratio", "family", "family_confidence",
    "behavior", "file_property", "packer", "exploit", "group",
    "caps_list", "ttps_list", "mbc_list",
    "subset", "subset_class", "dyn_split", "time_bucket",
]
df = df[canonical]
print(f"  After:  {len(df):,} rows x {len(df.columns)} cols (canonical 24)")

df.to_csv(mf, index=False)
print(f"  Wrote {mf} ({mf.stat().st_size / 1024 / 1024:.2f} MB)")
print()
print(f"Final schema (24 cols):")
for c in df.columns:
    sample = str(df[c].iloc[0])[:60]
    print(f"  {c:25s} : {sample}")
