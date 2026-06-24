"""Enrich the 483 F samples to full coverage.

The F broad-coverage pool provides additional family coverage using static feature
vectors and CAPE detonation, but four EMBER2024-inherited fields were never
back-joined into the static parquets (sha1, caps_list, ttps_list, mbc_list)
and the manifest was never synced to the metadata the parquets already held
(md5, file_type, detection_ratio, family, capa tag lists, ...).

This script sources the four missing fields from the EMBER2024 clean-week
JSONL (authoritative upstream), writes them into the static parquets, syncs
the manifest from the parquets, and regenerates the four long-form annotation
parquets so that all 10,000 samples have uniform tri-modal + annotation
coverage.

Run on the host that holds jsonl_clean-week (Windows dev copy).
"""
from __future__ import annotations

import json
import os
import re
import glob
import shutil
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Set EMBER_DYN_DATASET to your dataset root (holding manifest.csv, static/,
# annotations/). Set EMBER_DYN_JSONL_DIR to the upstream EMBER2024 clean-week
# JSONL directory used to source the four inherited fields.
DATASET = Path(os.environ.get("EMBER_DYN_DATASET", "./dataset"))
STATIC = DATASET / "static"
ANNOT = DATASET / "annotations"
MANIFEST = DATASET / "manifest.csv"
BACKUP = DATASET / "_pre_enrich_backup"
JSONL_GLOB = os.path.join(
    os.environ.get("EMBER_DYN_JSONL_DIR", "./jsonl_clean-week"), "**", "*.jsonl")
SPLITS = ["train", "val", "test", "challenge"]

# EMBER2024-inherited columns synced parquet -> manifest for F rows.
# (scalars + list columns; list columns serialised to JSON strings in the CSV)
SCALAR_COLS = ["md5", "sha1", "tlsh", "file_type", "week_id",
               "last_analysis_date", "detection_ratio",
               "family", "family_confidence"]
LIST_COLS = ["behavior", "file_property", "packer", "exploit", "group",
             "caps_list", "ttps_list", "mbc_list"]

DRY_RUN = "--write" not in sys.argv


# ---- normalisation identical to rebuild_parquet_v2.parse_jsonl_sidecar ----
def norm(lst, *keys):
    out = []
    for c in (lst or []):
        if isinstance(c, str):
            s = c
        elif isinstance(c, dict):
            s = ""
            for k in keys:
                s = c.get(k) or c.get(k.lower()) or ""
                if s:
                    break
        else:
            s = str(c)
        if s:
            out.append(s)
    return out


def as_pylist(v):
    """Return a clean python list from a parquet list cell (ndarray/list/None)."""
    if v is None:
        return []
    return [str(x) for x in list(v)]


def build_lut(coverage):
    """Scan clean-week JSONL for the F sha256 -> missing fields."""
    rgx = re.compile(r'"sha256"\s*:\s*"([0-9a-fA-F]{64})"')
    lut = {}
    files = sorted(glob.glob(JSONL_GLOB, recursive=True))
    print(f"  scanning {len(files)} JSONL files for {len(coverage)} F hashes")
    for p in files:
        if len(lut) >= len(coverage):
            break
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                m = rgx.search(line[:400])
                if not m:
                    continue
                sha = m.group(1)
                if sha in coverage and sha not in lut:
                    r = json.loads(line)
                    lut[sha] = {
                        "sha1": r.get("sha1", "") or "",
                        "caps_list": norm(r.get("caps"), "Capability"),
                        "ttps_list": norm(r.get("ttps"), "Technique"),
                        "mbc_list": norm(r.get("mbc"), "Behavior"),
                    }
    return lut


def main():
    print("=" * 64)
    print("EMBER-Dyn F enrichment", "(DRY-RUN)" if DRY_RUN else "(WRITE)")
    print("=" * 64)

    manifest = pd.read_csv(MANIFEST)
    coverage = set(manifest.loc[manifest.subset == "F", "sha256"])
    print(f"F samples: {len(coverage)}")

    print("\n[1] Building lookup from clean-week JSONL")
    lut = build_lut(coverage)
    print(f"  found {len(lut)}/{len(coverage)} in JSONL")
    missing = coverage - set(lut)
    if missing:
        print(f"  WARNING: {len(missing)} F not found in JSONL (left empty)")
    # coverage of the sourced fields
    n_caps = sum(1 for v in lut.values() if v["caps_list"])
    n_sha1 = sum(1 for v in lut.values() if v["sha1"])
    print(f"  with non-empty caps_list: {n_caps}/{len(lut)}; with sha1: {n_sha1}/{len(lut)}")
    # sample
    for sha in list(lut)[:2]:
        v = lut[sha]
        print(f"   {sha[:16]} sha1={v['sha1'][:12]} caps={v['caps_list'][:2]} mbc={v['mbc_list'][:2]}")

    if DRY_RUN:
        print("\n[DRY-RUN] no files written. Re-run with --write to apply.")
        return

    # ---- backup ----
    print("\n[2] Backing up originals ->", BACKUP)
    BACKUP.mkdir(exist_ok=True)
    shutil.copy2(MANIFEST, BACKUP / "manifest.csv")
    for sp in SPLITS:
        shutil.copy2(STATIC / f"{sp}.parquet", BACKUP / f"{sp}.parquet")
    for name in ["capa_caps", "mitre_ttps", "mbc_behaviors", "group_apt"]:
        src = ANNOT / f"{name}.parquet"
        if src.exists():
            shutil.copy2(src, BACKUP / f"{name}.parquet")

    # ---- patch static parquets ----
    # Rebuild the four affected columns as whole python lists (uniform
    # list-of-lists) so pyarrow infers a clean list<string> type; assigning a
    # list into a single .at cell is fragile, so we never do that.
    print("\n[3] Patching static parquets (sha1 + caps/ttps/mbc lists for F)")
    for sp in SPLITS:
        path = STATIC / f"{sp}.parquet"
        df = pq.read_table(path).to_pandas()
        shas = df["sha256"].tolist()
        n_tu = sum(1 for s in shas if s in coverage)
        if n_tu == 0:
            print(f"  [{sp}] no F rows, unchanged")
            continue
        sha1_col = df["sha1"].tolist()
        caps_col = df["caps_list"].tolist()
        ttps_col = df["ttps_list"].tolist()
        mbc_col = df["mbc_list"].tolist()
        patched = 0
        for i, sha in enumerate(shas):
            if sha in lut:
                sha1_col[i] = lut[sha]["sha1"]
                caps_col[i] = lut[sha]["caps_list"]
                ttps_col[i] = lut[sha]["ttps_list"]
                mbc_col[i] = lut[sha]["mbc_list"]
                patched += 1
            else:
                # normalise existing cells to plain lists for uniform typing
                caps_col[i] = as_pylist(caps_col[i])
                ttps_col[i] = as_pylist(ttps_col[i])
                mbc_col[i] = as_pylist(mbc_col[i])
        df["sha1"] = sha1_col
        df["caps_list"] = caps_col
        df["ttps_list"] = ttps_col
        df["mbc_list"] = mbc_col
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False),
                       path, compression="snappy")
        print(f"  [{sp}] patched {patched}/{n_tu} F rows -> {path.name}")

    # ---- sync manifest from patched parquets ----
    print("\n[4] Syncing manifest F rows from patched static parquets")
    parts = []
    for sp in SPLITS:
        cols = ["sha256"] + SCALAR_COLS + LIST_COLS
        t = pq.read_table(STATIC / f"{sp}.parquet", columns=cols).to_pandas()
        parts.append(t[t["sha256"].isin(coverage)])
    src = pd.concat(parts).set_index("sha256")
    man = manifest.set_index("sha256")
    for sha in coverage:
        if sha not in src.index:
            continue
        for c in SCALAR_COLS:
            val = src.at[sha, c]
            man.at[sha, c] = val
        for c in LIST_COLS:
            man.at[sha, c] = json.dumps(as_pylist(src.at[sha, c]))
    man = man.reset_index()
    # restore original column order
    man = man[manifest.columns.tolist()]
    # integer columns become float64 once F NaNs are read; now that every
    # row is populated, cast back so the CSV keeps "5" rather than "5.0"
    for c in ["week_id", "first_submission_date", "last_analysis_date"]:
        if c in man.columns and man[c].notna().all():
            man[c] = man[c].astype("int64")
    man.to_csv(MANIFEST, index=False)
    print(f"  manifest written ({len(man)} rows, {len(man.columns)} cols)")

    # ---- regenerate annotation parquets (build_annotation_parquets logic) ----
    print("\n[5] Regenerating annotation parquets from patched static parquets")
    all_caps, all_ttps, all_mbc, all_group = [], [], [], []

    def _iter(v):
        if v is None:
            return
        for x in v:
            if x:
                yield x

    for sp in SPLITS:
        df = pq.read_table(STATIC / f"{sp}.parquet",
                           columns=["sha256", "caps_list", "ttps_list",
                                    "mbc_list", "group"]).to_pandas()
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
    for rows, name in [(all_caps, "capa_caps"), (all_ttps, "mitre_ttps"),
                       (all_mbc, "mbc_behaviors"), (all_group, "group_apt")]:
        out = ANNOT / f"{name}.parquet"
        pd.DataFrame(rows).to_parquet(out, compression="snappy", index=False)
        tu_rows = sum(1 for r in rows if r["sha256"] in coverage)
        print(f"  {name}.parquet: {len(rows):,} rows (F contributes {tu_rows})")

    print("\n[done] enrichment complete.")


if __name__ == "__main__":
    main()
