"""Generate splits.json — per-task SHA-256 lists for all 15 benchmark tasks.

Includes:
  - default            : standard train/val/test/challenge split (T1, T2, T5-T7)
  - T4_open_set        : Subset A (known) vs Subset B (unknown)
  - T6_few_shot        : Subset C 8 families × 50 samples pool
  - T7_packer          : non-UPX packer positives (packer field; UPX-only excluded) vs no-packer negatives
  - T8_exploit         : exploit-bearing positives (exploit field non-empty) vs rest negatives
  - T9_apt             : APT group attribution (Subset E2)
  - T10-T12_multilabel : per-sample multi-label (uses default split)
  - T13_per_arch       : per-file_type splits (Win32 / Win64 / .NET)
  - T15_challenge      : 96 time-distinct challenge samples
"""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(".")
_m = ROOT / "dataset" / "manifest.csv"
if not _m.exists():
    _m = ROOT / "dataset" / "manifest_open.csv"   # open-tier fallback (all needed columns are in the 26)
df = pd.read_csv(_m)
print(f"Loaded manifest: {len(df):,} rows x {len(df.columns)} cols")

def sha_list(filter_mask) -> list[str]:
    return df.loc[filter_mask, "sha256"].str.lower().tolist()

def parse_list(cell) -> list:
    """Parse a JSON-array-string capa field cell (e.g. '[\"themida\"]') to a list."""
    if isinstance(cell, list):
        return cell
    s = str(cell).strip()
    if s in ("", "[]", "None", "nan", "NaN"):
        return []
    try:
        v = json.loads(s)
    except Exception:
        return []
    return v if isinstance(v, list) else [v]

splits = {}

# Default split (T1, T2, T5, T6, T7, T10-T14, T15)
splits["default"] = {
    split: sha_list(df["dyn_split"] == split)
    for split in ["train", "val", "test", "challenge"]
}

# T4 open-set: Subset A train as known, Subset B as unknown
splits["T4_open_set"] = {
    "known_train": sha_list(
        (df["subset"] == "A") & (df["dyn_split"] == "train")
    ),
    "known_val": sha_list(
        (df["subset"] == "A") & (df["dyn_split"] == "val")
    ),
    "known_test": sha_list(
        (df["subset"] == "A") & (df["dyn_split"] == "test")
    ),
    "unknown_pool": sha_list(df["subset"] == "B"),
}

# T6 few-shot: Subset C 8 families
splits["T6_few_shot"] = {
    "support_query_pool": sha_list(df["subset"] == "C"),
}

# T7 packer detection: positives = packer field carries a non-UPX entry;
# negatives = packer field empty; UPX-only samples are excluded (not a clean
# negative). Subset E3 is a curated slice of the positives, not the task label
# (matches task_definitions.json T7 and run_baselines_final.py).
_packer = df["packer"].map(parse_list)
splits["T7_packer"] = {
    "positives": sha_list(_packer.map(lambda xs: any(str(x).strip().lower() != "upx" for x in xs))),
    "negatives": sha_list(_packer.map(lambda xs: len(xs) == 0)),
}

# T8 exploit detection: positives = exploit field non-empty; negatives = empty.
# Subset E1 is a curated slice of the positives, not the task label
# (matches task_definitions.json T8 and run_baselines_final.py).
_exploit = df["exploit"].map(parse_list)
splits["T8_exploit"] = {
    "positives": sha_list(_exploit.map(lambda xs: len(xs) > 0)),
    "negatives": sha_list(_exploit.map(lambda xs: len(xs) == 0)),
}

# T9 APT attribution (E2 multi-class)
splits["T9_apt"] = {
    "samples": sha_list(df["subset"] == "E2"),
}

# T13 per-architecture (Win32 / Win64 / .NET)
splits["T13_per_arch"] = {
    "Win32": sha_list(df["file_type"] == "Win32"),
    "Win64": sha_list(df["file_type"] == "Win64"),
    "Dot_Net": sha_list(df["file_type"] == "Dot_Net"),
}

# T15 challenge (96 time-distinct hold-out)
splits["T15_challenge"] = {
    "samples": sha_list(df["dyn_split"] == "challenge"),
}

# Verify counts
for task_name, task_splits in splits.items():
    if isinstance(task_splits, dict):
        counts = {k: len(v) for k, v in task_splits.items()}
        print(f"  {task_name}: {counts}")

out = ROOT / "dataset" / "splits.json"
with open(out, "w", encoding="utf-8", newline="\n") as f:
    json.dump(splits, f, indent=2)
    f.write("\n")
print(f"\nWrote {out} ({out.stat().st_size / 1024:.1f} KB)")
