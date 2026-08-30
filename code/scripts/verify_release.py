#!/usr/bin/env python3
"""verify_release.py - self-contained integrity check for the *published*
EMBER-Dyn open release (no build-staging layout required).

Usage:
    python verify_release.py /path/to/dataset

Re-runs checksum, manifest-count, parquet-schema, annotation-count,
split-coverage, and PII-residual checks directly on the released files. Exits
non-zero on any failure. Every released file (including the Full reports under
dynamic/full) is checksum-verified against checksums.sha256. The Small dynamic
reports are scanned for analyzer-account / guest-SID residual
(separator-agnostic); pass --deep to also content-scan the multi-GB Full
reports.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

EXPECT = {
    "rows": 10000,
    "label": {"1": 8126, "0": 1874},
    "splits": {"train": 7000, "val": 997, "test": 1907, "challenge": 96},
    "feature_dim": 2568,
    "static_cols": 2592,
    "open_manifest_cols": 26,
    "annotation_rows": {"capa_caps": 120545, "mitre_ttps": 58902, "mbc_behaviors": 106201, "group_apt": 217},
    "annotation_secondary": {"capa_caps": "namespace", "mitre_ttps": "tactic", "mbc_behaviors": "objective", "group_apt": "group"},
    "tasks": 15,
}
RECHECK_COLS = {"recheck_label", "recheck_detections", "recheck_date"}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(root_str: str, deep: bool = False) -> int:
    root = Path(root_str)
    checks: list[tuple[str, bool, str]] = []

    def chk(name, ok, detail=""):
        checks.append((name, bool(ok), detail))

    # 1. checksums
    cs = root / "checksums.sha256"
    if cs.exists():
        bad = []
        for line in cs.read_text(encoding="utf-8").splitlines():
            if " *" not in line:
                continue
            digest, rel = line.split(" *", 1)
            fp = root / rel.lstrip("./")
            if not fp.exists() or sha256_file(fp) != digest.strip():
                bad.append(rel)
        chk("checksums.sha256 verifies", not bad, f"{len(bad)} mismatched/missing")
    else:
        chk("checksums.sha256 present", False, "missing")

    # 2. open manifest
    mo = pd.read_csv(root / "manifest_open.csv", dtype=str, keep_default_na=False)
    chk("manifest rows == 10000", len(mo) == EXPECT["rows"], str(len(mo)))
    chk("manifest_open columns == 26", mo.shape[1] == EXPECT["open_manifest_cols"], str(mo.shape[1]))
    chk("no recheck cols in open manifest", not (RECHECK_COLS & set(mo.columns)), str(RECHECK_COLS & set(mo.columns)))
    chk("label balance 8126/1874", mo["label"].value_counts().to_dict() == EXPECT["label"], mo["label"].value_counts().to_dict())
    chk("dyn_split sizes", mo["dyn_split"].value_counts().to_dict() == EXPECT["splits"], mo["dyn_split"].value_counts().to_dict())

    # 3. static parquets
    man_sha = set(mo["sha256"].str.lower())
    for sp, n in EXPECT["splits"].items():
        t = pq.read_table(root / "static" / f"{sp}.parquet")
        feat = [c for c in t.column_names if c.startswith("feature_")]
        psha = {s.lower() for s in t.column("sha256").to_pylist()}
        man_sp = set(mo.loc[mo["dyn_split"] == sp, "sha256"].str.lower())
        chk(f"static/{sp}.parquet cols == 2592", t.num_columns == EXPECT["static_cols"], str(t.num_columns))
        chk(f"static/{sp}.parquet feature dim == 2568", len(feat) == EXPECT["feature_dim"], str(len(feat)))
        chk(f"static/{sp}.parquet sha set == manifest", psha == man_sp, f"{len(psha)} vs {len(man_sp)}")

    # 4. annotation parquets
    for fn, rows in EXPECT["annotation_rows"].items():
        t = pq.read_table(root / "annotations" / f"{fn}.parquet")
        sec = EXPECT["annotation_secondary"][fn]
        chk(f"annotations/{fn} rows == {rows}", t.num_rows == rows, str(t.num_rows))
        nonempty = sum(1 for v in t.column(sec).to_pylist() if v) if sec in t.column_names else 0
        chk(f"annotations/{fn} '{sec}' populated", nonempty == t.num_rows, f"{nonempty}/{t.num_rows}")

    # 5. splits.json
    sj = json.loads((root / "splits.json").read_text())
    td = json.loads((root / "task_definitions.json").read_text())
    ref = {t.get("split_name") for t in td.values() if t.get("split_name")}
    chk("splits.json covers all task split_names", ref <= set(sj.keys()), str(sorted(ref - set(sj.keys()))))
    if "default" in sj:
        for sp in EXPECT["splits"]:
            a = set(x.lower() for x in sj["default"][sp])
            b = set(mo.loc[mo["dyn_split"] == sp, "sha256"].str.lower())
            chk(f"splits.json default.{sp} == manifest", a == b, f"{len(a)} vs {len(b)}")
    chk("task_definitions has 15 tasks", len(td) == EXPECT["tasks"], str(len(td)))

    # 6. static parquet Arrow schema identical across splits
    schemas = {sp: pq.read_schema(root / "static" / f"{sp}.parquet") for sp in EXPECT["splits"]}
    base = schemas["train"]
    for sp in ("val", "test", "challenge"):
        chk(f"static/{sp}.parquet schema == train", schemas[sp].equals(base), "Arrow schema mismatch")

    # 7. near-duplicate-disjoint leakage: every eval sample's nearest-train TLSH >= 30
    import csv as _csv
    tl = root / "tlsh_leakage.csv"
    if tl.exists():
        dists = []
        with open(tl, newline="") as f:
            for row in _csv.DictReader(f):
                col = next((k for k in row if "dist" in k.lower()), None)
                if col and str(row[col]).strip():
                    dists.append(int(float(row[col])))
        chk("tlsh_leakage all >= 30 (no cross-side near-dup)",
            bool(dists) and min(dists) >= 30, f"min={min(dists) if dists else 'n/a'} n={len(dists)}")
    else:
        chk("tlsh_leakage.csv present", False, "missing")

    # 8. PII-residual scan on released dynamic reports (separator-agnostic):
    #    the analyzer account 'cape' in any path form (incl. reduced-schema
    #    rewrites : | _) and the constant guest SID must not survive.
    import re as _re
    import tarfile as _tar
    try:
        import zstandard as _zstd
    except Exception:
        _zstd = None
    RES_CAPE = _re.compile(rb'(?i)(?:[\x5c/:|_]cape(?![a-z0-9])|\bcape(?=[\x5c/:|]))')
    RES_SID = _re.compile(rb'S-1-5-21-\d+-\d+-\d+')

    def _scan_tier(dirp, ds_map=None):
        creps = sreps = members = ds_mismatch = 0
        for tarp in sorted(dirp.glob("*.tar.zst")):
            with open(tarp, "rb") as f, _zstd.ZstdDecompressor().stream_reader(f) as rd:
                with _tar.open(fileobj=rd, mode="r|") as t:
                    for m in t:
                        if not m.isfile():
                            continue
                        members += 1
                        raw = t.extractfile(m).read()
                        if RES_CAPE.search(raw):
                            creps += 1
                        if RES_SID.search(raw):
                            sreps += 1
                        if ds_map is not None:
                            sha = m.name.rsplit("/", 1)[-1][:-5].lower()
                            beh = json.loads(raw).get("behavior", {})
                            summ = beh.get("summary", {})
                            has = (isinstance(summ, dict) and any(summ.values())) or bool(beh.get("apistats"))
                            if (ds_map.get(sha) == "behavior") != bool(has):
                                ds_mismatch += 1
        return creps, sreps, members, ds_mismatch

    small_dir = root / "dynamic" / "small"
    if _zstd and small_dir.is_dir():
        ds_map = dict(zip(mo["sha256"].str.lower(), mo["dynamic_status"]))
        c, s, n, dsm = _scan_tier(small_dir, ds_map)
        chk("open Small: members == 10000", n == EXPECT["rows"], str(n))
        chk("open Small: no analyzer-account residual", c == 0, f"{c} reports")
        chk("open Small: no guest-SID residual", s == 0, f"{s} reports")
        chk("open Small: dynamic_status == content (behavior iff non-empty summary/apistats)", dsm == 0, f"{dsm} mismatches")
    elif not _zstd:
        chk("zstandard available for PII scan", False, "pip install zstandard")

    # 9. Full reports are covered by checksums.sha256 (section 1) and content-scanned with --deep.
    full_dir = root / "dynamic" / "full"
    if deep and _zstd and full_dir.is_dir():
        c, s, n, _ = _scan_tier(full_dir)
        chk("Full reports: members == 10000", n == EXPECT["rows"], str(n))
        chk("Full reports: no analyzer-account residual", c == 0, f"{c} reports")
        chk("Full reports: no guest-SID residual", s == 0, f"{s} reports")
    elif deep and not _zstd:
        chk("zstandard available for Full PII scan", False, "pip install zstandard")

    # report
    npass = sum(1 for _, ok, _ in checks if ok)
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if not ok else ""))
    print(f"\n{npass}/{len(checks)} checks passed")
    return 0 if npass == len(checks) else 1


if __name__ == "__main__":
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    deep = "--deep" in sys.argv[1:]
    if len(pos) != 1:
        print("usage: python verify_release.py /path/to/dataset [--deep]", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(pos[0], deep))
