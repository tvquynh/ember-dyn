"""Phase 0 — Purposively sample the EMBER-Dyn subset from the EMBER2024 PE corpus.

Strategy:
  Subset A  = 5 known families x 1,200 = 6,000  (conf>=0.7, det>=0.5)
  Subset B  = 10 open-set families x 150 = 1,500 (conf>=0.5, det>=0.5)
  Subset C  = 8 few-shot families x 50  = 400    (conf>=0.5, det>=0.5)
  Subset D  = 2,000 benign                       (label==0, det==0)
  Subset E1 = ALL exploit-bearing                (exploit list non-empty)
  Subset E2 = 200 APT-group attributed           (group list non-empty)
  Subset E3 = 200 packer non-UPX                 (packer non-empty AND not UPX)
  Challenge = 100 (kept as-is from EMBER2024 challenge split)
  TOTAL ~= 10,481

Stratification:
  time_bucket : pre-2024-09 vs post-2024-09 -> 50/50 within each subset
  file_type   : Win32/Win64/Dot_Net match population
  train/val/test = 70/10/20 stratified per (subset x family x time_bucket x file_type)

Output: manifest.csv at --output path

NOTE: this script documents the purposive selection criteria only.
The released EMBER-Dyn comprises 10,000 samples including the broad-coverage
family pool (subset F) and a near-duplicate-disjoint (TLSH) re-split; the released
manifest.csv is the canonical record of the final sample list and the assigned
train/val/test/challenge split. Exact regeneration requires the upstream
EMBER2024 clean-week source (samples are identified by SHA-256).
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

PE_ROOT = Path("./data/pe")
CW = Path("./data/ember2024/parquet")

# Family configs
A_FAMILIES = ["berbew", "cosmu", "expiro", "xmrig", "virlock"]
B_FAMILIES = ["upatre", "glupteba", "stealc", "lummastealer", "agenttesla",
              "floxif", "zeus", "xworm", "redline", "amadey"]
# (grandoreiro pool 30 on disk subset → replaced with zeus, banker analog)
C_FAMILIES = ["blackmoon", "remcos", "banload", "virut", "cobaltstrike",
              "ekstak", "sality", "flystudio"]
# (debrisa pool 0 → replaced with remcos, RAT analog)

# Per-subset filters
A_CONF_MIN = 0.7
A_VIRLOCK_FALLBACK = 0.65
B_CONF_MIN = 0.5
C_CONF_MIN = 0.5
DET_RATIO_MIN = 0.5

# Stratification proportions
FILETYPE_PCT_MAL = {"Win32": 0.75, "Win64": 0.15, "Dot_Net": 0.10}
FILETYPE_PCT_BEN = {"Win32": 0.60, "Win64": 0.25, "Dot_Net": 0.15}
TIME_CUTOFF = pd.Timestamp("2024-09-01")
SPLIT_RATIOS = (0.70, 0.10, 0.20)  # train, val, test

# APT group desired counts
APT_TARGET = {"gamaredon": 100, "turla": 30, "equationgroup": 20,
              "molerats": 19, "knotweed": 5}  # rest 26 from other groups

# Packer non-UPX desired counts
PACKER_TARGET = {"themida": 70, "nsis": 50, "vmprotect": 50,
                 "enigmaprotector": 30}  # other packers fill rest

# ---------------------------------------------------------------------------
def load_full() -> pd.DataFrame:
    """Load 3 splits clean-week parquet + add ember_subset column."""
    cols = ["sha256", "md5", "tlsh", "file_type", "label", "week_id",
            "first_submission_date", "last_analysis_date", "detection_ratio",
            "family", "family_confidence",
            "behavior", "file_property", "packer", "exploit", "group",
            "n_caps", "n_ttps", "n_mbc"]
    parts = []
    for split in ["train", "test", "challenge"]:
        df = pd.read_parquet(CW / f"ember2024_{split}.parquet", columns=cols)
        df["ember_subset"] = split
        parts.append(df)
    full = pd.concat(parts, ignore_index=True)
    return full

def restrict_to_disk_pe(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows whose sha256 exists as an available PE file."""
    sha_on_disk = set()
    for split in ["train", "test", "challenge"]:
        d = PE_ROOT / split
        sha_on_disk.update(os.listdir(d))
    df = df[df["sha256"].isin(sha_on_disk)].copy()
    return df

def add_columns(df: pd.DataFrame) -> pd.DataFrame:
    df["date"] = pd.to_datetime(df["first_submission_date"], unit="s", errors="coerce")
    df["time_bucket"] = np.where(df["date"] < TIME_CUTOFF, "pre", "post")
    # detection_ratio in clean-week parquet is already float64 ratio
    df["det_ratio"] = df["detection_ratio"].astype(float)
    df["family_norm"] = df["family"].fillna("").str.lower()
    # Force list-type for capa columns (some rows may have None/ndarray)
    for col in ["behavior", "packer", "exploit", "group"]:
        df[col + "_list"] = df[col].apply(
            lambda x: [str(v).lower() for v in x] if hasattr(x, "__iter__") and not isinstance(x, str) else []
        )
    return df

# ---------------------------------------------------------------------------
def stratified_pick(pool: pd.DataFrame, n: int, ft_target: dict, rng: np.random.Generator):
    """Sample n rows from pool, stratified by (time_bucket × file_type).
    Target proportions: time=50/50, file_type=ft_target. Fallback to other cells if undersampled.
    """
    if len(pool) <= n:
        return pool.copy()
    cells = []
    for tb in ["pre", "post"]:
        for ft, pct in ft_target.items():
            target = int(round(n * 0.5 * pct))
            sub = pool[(pool["time_bucket"] == tb) & (pool["file_type"] == ft)]
            if len(sub) >= target:
                picked = sub.sample(n=target, random_state=int(rng.integers(0, 2**31)))
            else:
                picked = sub.copy()  # take all, will compensate later
            cells.append(picked)
    out = pd.concat(cells, ignore_index=True)
    # Compensate shortfall from any leftover pool
    if len(out) < n:
        used_sha = set(out["sha256"])
        leftover = pool[~pool["sha256"].isin(used_sha)]
        extra = leftover.sample(n=min(n - len(out), len(leftover)),
                                 random_state=int(rng.integers(0, 2**31)))
        out = pd.concat([out, extra], ignore_index=True)
    return out.head(n)

def assign_split(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Stratified train/val/test split per (subset × subset_class × time_bucket × file_type).
    Challenge subset always stays in 'challenge' split.
    """
    df = df.copy()
    df["dyn_split"] = ""

    # Challenge → all 'challenge'
    challenge_mask = df["subset"] == "Challenge"
    df.loc[challenge_mask, "dyn_split"] = "challenge"

    # Others: stratified 70/10/20
    rest = df[~challenge_mask].copy()
    strat_cols = ["subset", "subset_class", "time_bucket", "file_type"]

    train_ratio, val_ratio, _ = SPLIT_RATIOS

    split_assignments = []
    for cell_key, group in rest.groupby(strat_cols, dropna=False):
        n = len(group)
        idx = rng.permutation(n)
        n_train = int(np.round(n * train_ratio))
        n_val = int(np.round(n * val_ratio))
        # rest goes to test (dtype=object to allow longer strings like "train")
        splits = np.array(["test"] * n, dtype=object)
        splits[idx[:n_train]] = "train"
        splits[idx[n_train:n_train + n_val]] = "val"
        group = group.copy()
        group["dyn_split"] = splits
        split_assignments.append(group)

    rest_assigned = pd.concat(split_assignments, ignore_index=True)
    final = pd.concat([df[challenge_mask], rest_assigned], ignore_index=True)
    return final

# ---------------------------------------------------------------------------
def pick_subset_a(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for fam in A_FAMILIES:
        conf = A_VIRLOCK_FALLBACK if fam == "virlock" else A_CONF_MIN
        pool = df[(df["label"] == 1) & (df["family_norm"] == fam) &
                  (df["family_confidence"].fillna(0) >= conf) &
                  (df["det_ratio"].fillna(0) >= DET_RATIO_MIN)]
        picked = stratified_pick(pool, 1200, FILETYPE_PCT_MAL, rng)
        picked = picked.assign(subset="A", subset_class=fam)
        rows.append(picked)
        print(f"  A {fam}: pool {len(pool):,} -> picked {len(picked):,}")
    return pd.concat(rows, ignore_index=True)

def pick_subset_b(df: pd.DataFrame, used_sha: set, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for fam in B_FAMILIES:
        pool = df[(df["label"] == 1) & (df["family_norm"] == fam) &
                  (df["family_confidence"].fillna(0) >= B_CONF_MIN) &
                  (df["det_ratio"].fillna(0) >= DET_RATIO_MIN) &
                  (~df["sha256"].isin(used_sha))]
        picked = stratified_pick(pool, 150, FILETYPE_PCT_MAL, rng)
        picked = picked.assign(subset="B", subset_class=fam)
        rows.append(picked)
        print(f"  B {fam}: pool {len(pool):,} -> picked {len(picked):,}")
    return pd.concat(rows, ignore_index=True)

def pick_subset_c(df: pd.DataFrame, used_sha: set, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for fam in C_FAMILIES:
        pool = df[(df["label"] == 1) & (df["family_norm"] == fam) &
                  (df["family_confidence"].fillna(0) >= C_CONF_MIN) &
                  (df["det_ratio"].fillna(0) >= DET_RATIO_MIN) &
                  (~df["sha256"].isin(used_sha))]
        picked = stratified_pick(pool, 50, FILETYPE_PCT_MAL, rng)
        picked = picked.assign(subset="C", subset_class=fam)
        rows.append(picked)
        print(f"  C {fam}: pool {len(pool):,} -> picked {len(picked):,}")
    return pd.concat(rows, ignore_index=True)

def pick_subset_d(df: pd.DataFrame, used_sha: set, rng: np.random.Generator) -> pd.DataFrame:
    pool = df[(df["label"] == 0) &
              (df["det_ratio"].fillna(-1) == 0) &
              (~df["sha256"].isin(used_sha))]
    picked = stratified_pick(pool, 2000, FILETYPE_PCT_BEN, rng)
    picked = picked.assign(subset="D", subset_class="benign")
    print(f"  D benign: pool {len(pool):,} -> picked {len(picked):,}")
    return picked

def pick_subset_e1_exploit(df: pd.DataFrame, used_sha: set) -> pd.DataFrame:
    mask = df["exploit_list"].apply(lambda x: len(x) > 0) & ~df["sha256"].isin(used_sha)
    picked = df[mask].copy()
    picked = picked.assign(subset="E1", subset_class="exploit")
    print(f"  E1 exploit: pool {int(mask.sum()):,} -> picked {len(picked):,}")
    return picked

def pick_subset_e2_apt(df: pd.DataFrame, used_sha: set, rng: np.random.Generator) -> pd.DataFrame:
    # Each sample's group_list could have multiple; take primary (first)
    rows = []
    used = set(used_sha)
    for group, want in APT_TARGET.items():
        pool = df[df["group_list"].apply(lambda x: group in x) & ~df["sha256"].isin(used)]
        n_take = min(want, len(pool))
        if n_take > 0:
            picked = pool.sample(n=n_take, random_state=int(rng.integers(0, 2**31)))
            picked = picked.assign(subset="E2", subset_class=f"apt_{group}")
            rows.append(picked)
            used.update(picked["sha256"])
            print(f"  E2 {group}: pool {len(pool):,} -> picked {len(picked):,}")
    # Other groups
    target_total = sum(APT_TARGET.values()) + 26  # ~200
    so_far = sum(len(r) for r in rows)
    remain = target_total - so_far
    if remain > 0:
        other_groups = set()
        for x in df["group_list"]:
            other_groups.update(x)
        other_groups -= set(APT_TARGET.keys())
        pool = df[df["group_list"].apply(lambda x: any(g in other_groups for g in x)) & ~df["sha256"].isin(used)]
        n_take = min(remain, len(pool))
        if n_take > 0:
            picked = pool.sample(n=n_take, random_state=int(rng.integers(0, 2**31)))
            picked = picked.assign(subset="E2", subset_class="apt_other")
            rows.append(picked)
            print(f"  E2 other-groups: pool {len(pool):,} -> picked {len(picked):,}")
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=df.columns.tolist() + ["subset", "subset_class"])

def pick_subset_e3_packer(df: pd.DataFrame, used_sha: set, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    used = set(used_sha)
    for packer, want in PACKER_TARGET.items():
        pool = df[df["packer_list"].apply(lambda x: packer in x and "upx" not in x) &
                  ~df["sha256"].isin(used)]
        n_take = min(want, len(pool))
        if n_take > 0:
            picked = pool.sample(n=n_take, random_state=int(rng.integers(0, 2**31)))
            picked = picked.assign(subset="E3", subset_class=f"packer_{packer}")
            rows.append(picked)
            used.update(picked["sha256"])
            print(f"  E3 {packer}: pool {len(pool):,} -> picked {len(picked):,}")
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=df.columns.tolist() + ["subset", "subset_class"])

def pick_challenge(df: pd.DataFrame) -> pd.DataFrame:
    picked = df[df["ember_subset"] == "challenge"].copy()
    picked = picked.assign(subset="Challenge", subset_class="challenge")
    print(f"  Challenge: kept all {len(picked):,}")
    return picked

# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="./output/staging/manifest.csv")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    print("[1/4] Loading clean-week parquet (3 splits) .")
    df = load_full()
    print(f"  total rows: {len(df):,}")

    print("[2/4] Restricting to samples with an available PE file.")
    df = restrict_to_disk_pe(df)
    print(f"  on-disk rows: {len(df):,}")

    print("[3/4] Adding derived columns .")
    df = add_columns(df)

    print("[4/4] Sampling subsets .")
    selected = []

    # Challenge first (fixed)
    sel = pick_challenge(df)
    selected.append(sel)
    used = set(sel["sha256"])

    sel = pick_subset_a(df, rng)
    selected.append(sel)
    used.update(sel["sha256"])

    sel = pick_subset_b(df, used, rng)
    selected.append(sel)
    used.update(sel["sha256"])

    sel = pick_subset_c(df, used, rng)
    selected.append(sel)
    used.update(sel["sha256"])

    sel = pick_subset_d(df, used, rng)
    selected.append(sel)
    used.update(sel["sha256"])

    sel = pick_subset_e1_exploit(df, used)
    selected.append(sel)
    used.update(sel["sha256"])

    sel = pick_subset_e2_apt(df, used, rng)
    selected.append(sel)
    used.update(sel["sha256"])

    sel = pick_subset_e3_packer(df, used, rng)
    selected.append(sel)

    manifest = pd.concat(selected, ignore_index=True)
    print(f"\nTotal selected: {len(manifest):,}")

    # Assign dyn_split
    manifest = assign_split(manifest, rng)

    # Drop helper list columns from output (keep original capa cols)
    drop_cols = ["behavior_list", "packer_list", "exploit_list", "group_list",
                 "family_norm", "date"]
    manifest = manifest.drop(columns=[c for c in drop_cols if c in manifest.columns])

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out_path, index=False)
    print(f"\nSaved manifest: {out_path}")

    # Summary
    summary = {
        "total": len(manifest),
        "by_subset": dict(manifest["subset"].value_counts()),
        "by_dyn_split": dict(manifest["dyn_split"].value_counts()),
        "by_ember_subset": dict(manifest["ember_subset"].value_counts()),
        "by_file_type": dict(manifest["file_type"].value_counts()),
        "by_label": dict(manifest["label"].value_counts()),
        "by_time_bucket": dict(manifest["time_bucket"].value_counts()),
    }
    summary_path = out_path.parent / "meta" / "manifest_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Saved summary: {summary_path}")
    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2, default=str))

if __name__ == "__main__":
    main()
