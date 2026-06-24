"""Run 10 verification invariants on the consolidated EMBER-Dyn archive."""

from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(".")
SPLITS = ["train", "val", "test", "challenge"]
EXPECTED_COUNTS = {"train": 7000, "val": 997, "test": 1907, "challenge": 96}

def main():
    results = []

    # I1: paper PDF exists and is ~700 KB
    pdf = ROOT / "paper" / "manuscript" / "main.pdf"
    size_kb = pdf.stat().st_size / 1024 if pdf.exists() else 0
    results.append(("I1 Manuscript PDF", pdf.exists() and 500 < size_kb < 900, f"{size_kb:.0f} KB"))

    # I2: 6 figures
    figs = list((ROOT / "paper" / "manuscript" / "figures").glob("fig*.pdf"))
    results.append(("I2 Figures", len(figs) == 6, f"{len(figs)} files"))

    # I3: 7 scripts
    scripts = list((ROOT / "code" / "scripts").glob("*.py"))
    results.append(("I3 Scripts", len(scripts) >= 7, f"{len(scripts)} files"))

    # I4: manifest 10,000 rows
    manifest = pd.read_csv(ROOT / "dataset" / "manifest.csv")
    results.append(("I4 Manifest rows", len(manifest) == 10000, f"{len(manifest):,} rows"))

    # I5: 4 static parquets, each correct row + col count
    parquet_ok = True
    parquet_msg = []
    for split in SPLITS:
        fp = pq.ParquetFile(ROOT / "dataset" / "static" / f"{split}.parquet")
        n = fp.metadata.num_rows
        c = len(fp.schema_arrow)
        expected_n = EXPECTED_COUNTS[split]
        ok = n == expected_n and c == 2592
        parquet_ok = parquet_ok and ok
        parquet_msg.append(f"{split}={n}r/{c}c")
    results.append(("I5 Parquets v2 schema (2,592 cols)", parquet_ok, " ".join(parquet_msg)))

    # I6: 4 JSONL files
    jsonls = list((ROOT / "dataset" / "static").glob("*.jsonl")) + list((ROOT / "dataset" / "static").glob("*.jsonl.zst"))
    results.append(("I6 JSONL files", len(jsonls) == 4, f"{len(jsonls)} files"))

    # I7: 4 annotation parquets
    annots = list((ROOT / "dataset" / "annotations").glob("*.parquet"))
    results.append(("I7 Annotation parquets", len(annots) == 4, f"{len(annots)} files"))

    # I8: PE counts per split
    pe_ok = True
    pe_msg = []
    for split in SPLITS:
        pe_dir = ROOT / "pe_local" / split
        n = len(list(pe_dir.glob("*"))) if pe_dir.exists() else 0
        expected = EXPECTED_COUNTS[split]
        ok = n == expected
        pe_ok = pe_ok and ok
        pe_msg.append(f"{split}={n}")
    results.append(("I8 PE binary counts", pe_ok, " ".join(pe_msg)))

    # I9: Cross-modal SHA-256 set equality on 50-sample spot-check
    test_pq = pq.read_table(
        ROOT / "dataset" / "static" / "test.parquet",
        columns=["sha256"],
    ).to_pandas()
    sample_50 = set(test_pq["sha256"].sample(n=50, random_state=42).tolist())
    pe_test_files = {p.stem for p in (ROOT / "pe_local" / "test").iterdir()}
    overlap = sample_50 & pe_test_files
    results.append(("I9 Cross-modal SHA-256 spot-check (50)", len(overlap) == 50, f"{len(overlap)}/50 match"))

    # I10: top-level docs exist
    docs = [
        ROOT / "README.md",
        ROOT / "LICENSE-CC-BY-4.0.txt",
        ROOT / "LICENSE-MIT.txt",
        ROOT / "CITATION.cff",
        ROOT / "pe_local" / "README.md",
        ROOT / "code" / "README.md",
        ROOT / "code" / "RUNBOOK.md",
    ]
    n_docs = sum(1 for d in docs if d.exists())
    results.append(("I10 Top-level docs", n_docs == 7, f"{n_docs}/7 docs"))

    # Report
    print("=" * 70)
    print("EMBER-Dyn consolidation verification")
    print("=" * 70)
    n_pass = 0
    for name, ok, msg in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name:42s} | {msg}")
        if ok:
            n_pass += 1
    print("=" * 70)
    print(f"  {n_pass}/{len(results)} invariants passed")
    print("=" * 70)

if __name__ == "__main__":
    main()
