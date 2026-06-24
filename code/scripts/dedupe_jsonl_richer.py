"""Dedupe JSONL by SHA-256 keeping the RICHER record per group.

Richness score per record = len(caps) + len(ttps) + len(mbc).
For tie-break, fall back to lexicographically earlier line.

Used to fix the train split which has 7,293 raw lines for 7,000 unique
sha256 keys (one duplicate carried over from EMBER2024 weekly source files).

After dedupe, the corresponding parquet must be rebuilt so its caps_list,
ttps_list, and mbc_list columns reflect the winning record.
"""

from __future__ import annotations
import json
import shutil
from collections import defaultdict
from pathlib import Path

ROOT = Path(".")
SPLITS = ["train", "val", "test", "challenge"]

def _richness(rec: dict) -> int:
    """Score a record by total number of capa + TTP + MBC entries."""
    return (
        len(rec.get("caps", []) or [])
        + len(rec.get("ttps", []) or [])
        + len(rec.get("mbc", []) or [])
    )

def dedupe_jsonl(jsonl_path: Path) -> tuple[int, int, int]:
    """Dedupe a JSONL file in-place. Returns (orig, unique, duplicates_dropped)."""
    by_sha: dict[str, dict] = {}
    by_sha_richness: dict[str, int] = {}
    n_orig = 0
    n_dup_replaced = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            n_orig += 1
            rec = json.loads(line)
            sha = rec["sha256"].lower()
            r = _richness(rec)
            if sha in by_sha:
                # Duplicate: keep the richer record
                if r > by_sha_richness[sha]:
                    by_sha[sha] = rec
                    by_sha_richness[sha] = r
                    n_dup_replaced += 1
            else:
                by_sha[sha] = rec
                by_sha_richness[sha] = r

    n_unique = len(by_sha)
    n_dropped = n_orig - n_unique

    # Backup original then write deduped
    backup = jsonl_path.with_suffix(".jsonl.predupe.bak")
    if not backup.exists():
        shutil.copy2(jsonl_path, backup)

    # Write in sorted SHA-256 order for reproducibility
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for sha in sorted(by_sha):
            f.write(json.dumps(by_sha[sha], ensure_ascii=False))
            f.write("\n")

    return n_orig, n_unique, n_dropped

def main():
    print("=" * 70)
    print("Smart dedupe of EMBER-Dyn JSONL (richer-keep)")
    print("=" * 70)
    for split in SPLITS:
        p = ROOT / "dataset" / "static" / f"{split}.jsonl"
        n_orig, n_unique, n_dropped = dedupe_jsonl(p)
        marker = "DEDUPED" if n_dropped > 0 else "no-op"
        print(f"  [{split}] {n_orig:,} -> {n_unique:,} ({n_dropped} dropped) [{marker}]")
    print("=" * 70)
    print("Note: backups saved as *.jsonl.predupe.bak in same folder.")
    print("Next step: re-run rebuild_parquet_v2.py to sync caps_list/ttps_list/mbc_list")

if __name__ == "__main__":
    main()
