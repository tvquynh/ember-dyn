# EMBER-Dyn RUNBOOK

Step-by-step guide to rebuild the EMBER-Dyn curated subset from the upstream
EMBER2024 release.

## Prerequisites

- Access to the EMBER2024 clean-week parquet (train, test, challenge) — published
  at <https://github.com/FutureComputing4AI/EMBER2024>.
- Access to the EMBER2024 raw JSONL (per-week files).
- Python ≥ 3.10 with `pandas`, `pyarrow`, `numpy`, `scikit-learn`, `lightgbm`,
  `xgboost`, `torch`, `zstandard` (see `requirements.txt`).
- For dynamic detonation: a Linux host with KVM/QEMU, libvirt, MongoDB,
  PostgreSQL, INetSim, and CAPEv2 (follow the upstream CAPEv2 project
  documentation at <https://github.com/kevoreilly/CAPEv2>).
- For raw bytes: VirusTotal Premium access, institutional malware-sharing
  arrangements (MalwareBazaar, MalShare), or direct contact with the EMBER2024
  team — needed only by the dynamic-detonation step, not for sample selection.

## Pipeline steps

### 1. Purposive sample selection

```bash
# Inputs are read from ./data/ember2024/parquet and ./data/pe (see the script header).
python scripts/sample_ember_dyn.py --output manifest_initial.csv --seed 42
```

This performs the **initial** purposive selection: each subset is defined by an
explicit purpose plus an eligibility criterion on EMBER2024 metadata
(family, `family_confidence`, `detection_ratio`, tag presence), prioritising
high-`family_confidence` samples, with proportional stratification across
family, time bucket, and file type.

The released EMBER-Dyn is then finalised by the broad-coverage pool pass
(`scripts/enrich_coverage_pool.py`, which curates the 483 subset-`F` samples) and a
near-duplicate-disjoint (TLSH) re-split, together yielding the final
**10,000-sample** `manifest.csv` (columns `subset`, `subset_class`,
`dyn_split`, `time_bucket`). The released `manifest.csv` is the **canonical**
record of the final sample list and split; because the upstream EMBER2024
clean-week source is not redistributed here, exact regeneration of the list
requires that source (samples are identified by SHA-256 for retrieval). The
split (`dyn_split`) is recorded in the manifest; `splits.json` is regenerated
from it via `scripts/generate_splits_json.py` and the benchmark task
definitions via `scripts/generate_task_definitions.py`. The near-duplicate
disjointness (no evaluation sample within TLSH 30 of any train/validation
sample) is recorded per-sample in `tlsh_leakage.csv`; `scripts/verify_release.py`
re-validates that this shipped leakage record contains no evaluation sample
below TLSH 30. (It re-checks the precomputed distances rather than recomputing
the all-pairs TLSH re-split, which requires the un-redistributed upstream
EMBER2024 source; the released `dyn_split` is the canonical record.)

### 2. Filter parquet to selected samples

```bash
python scripts/filter_parquet.py \
       --manifest manifest.csv \
       --ember2024-parquet /path/to/ember2024_train.parquet \
       --output-dir dataset/static/
```

Produces four split-specific parquet files
(`{train,val,test,challenge}.parquet`).

### 3. Filter raw JSONL

```bash
python scripts/filter_jsonl.py \
       --manifest manifest.csv \
       --jsonl-dir /path/to/ember2024_jsonl/ \
       --output-dir dataset/static/
```

Produces four split-specific JSONL files
(`{train,val,test,challenge}.jsonl`) followed by zstd compression to
`*.jsonl.zst` via `scripts/compress_jsonl.py`.

### 4. Extend parquet schema (v2)

```bash
python scripts/rebuild_parquet_v2.py
```

Adds four columns to each static parquet (`sha1`, `caps_list`, `ttps_list`,
`mbc_list`) by parsing the JSONL sidecars. Generates four long-form annotation
parquets (`dataset/annotations/{capa_caps,mitre_ttps,mbc_behaviors,group_apt}.parquet`).

### 5. (Optional) Obtain raw PE binaries

If you want to run dynamic detonation, you must obtain the 10,000 PE binaries
matching the SHA-256 identifiers in the manifest. EMBER-Dyn does not
redistribute binaries; obtain them via:

- VirusTotal Premium API
- Institutional malware-sharing arrangements (MalwareBazaar, MalShare, etc.)
- Direct contact with the EMBER2024 team

Place them under `pe_local/<split>/<SHA-256>.bin`.

### 6. (Optional) Run dynamic detonation

See the upstream CAPEv2 project documentation
(<https://github.com/kevoreilly/CAPEv2>) for the full sandbox setup
walkthrough. Once detonation has produced per-task `report.json`
files under the CAPE storage directory, the dynamic Small variant is
extracted by selecting 13 `behavior.summary` keys plus 23
`static.pe` keys per sample (the Avast-CTU 2022 schema), and the
Full variant copies the entire `report.json`. Final tarballs
(`{train,val,test,challenge}.tar.zst`) are produced by per-split
tar+zstd at compression level 19.

### 7. PII redaction on dynamic tarballs

Apply the best-effort PII filter to every Small and Full tarball
before release. The filter is deterministic — re-running it on the
same raw reports produces an identical redacted bundle.

```bash
python scripts/redact_pii.py raw/small/train.tar.zst dataset/dynamic/small/train.tar.zst --workers 16
python scripts/redact_pii.py raw/full/train.tar.zst  dataset/dynamic/full/train.tar.zst  --workers 16
# … repeat for val / test / challenge
```

The filter substitutes `C:\Users\<name>` → `C:\Users\<user>`,
removes email addresses, scrubs `/user:`, `runas /user:`, and
`password=` patterns. Observed network destinations are preserved.
On a 16-core machine the Full train tarball (155 GB decompressed)
takes ~40 minutes; the Small tarballs take seconds. See the paper's
Ethics statement for the full policy.

### 8. Verify

```bash
python scripts/verify_staging.py
```

`verify_staging.py` runs 9 checks (against the build-staging layout, which
includes the raw PE files):

1. PE file count per split matches the manifest (7,000 / 997 / 1,907 / 96)
2. PE files all carry MZ magic bytes (200-sample spot-check)
3. PE SHA-256 set == manifest SHA-256 set (per split)
4. Parquet row counts match the manifest (per split)
5. Parquet SHA-256 set == manifest SHA-256 set (per split)
6. Parquet feature-column count == 2,568
7. JSONL record counts match the manifest (per split)
8. JSONL SHA-256 set == manifest SHA-256 set (per split)
9. Cross-modality: 50 random SHA-256 exist in PE + parquet + JSONL

(The annotation row-count, manifest/parquet column-count, and feature-NaN
checks run in `verify_release.py` on the published bundle, not here.)

`verify_staging.py` runs against the build-staging layout. For a downloader who
only has the published bundle, `scripts/verify_release.py` re-runs the
checksum, manifest-count, parquet-schema, annotation-count, and split-coverage
checks directly on the released open-tier files:

```bash
python scripts/verify_release.py /path/to/dataset
```

All checks must pass on the v1 release.

### 9. Reproduce baseline tables

Reproduce the Technical Validation result tables (tasks T1–T15, the
dynamic-only TD1 baseline, and the diversity metrics) by running the
authoritative baseline runner. Point `EMBER_DYN_BASELINES` at the
dataset root holding `manifest.csv`, `splits.json`, and the static
parquets:

```bash
export EMBER_DYN_BASELINES=../dataset
python ../baselines/run_baselines_final.py
```

The script writes `results_final.json`, containing the T1–T15 +
TD1 + diversity numbers reported in the manuscript's Technical
Validation section.

## Estimated total runtime

- Steps 1–4 (sample selection + filtering + schema extension): ~10 minutes
- Step 5 (PE acquisition): hours to days depending on access method
- Step 6 (dynamic detonation): on the order of days for the full 10,000-sample
  pass on a 12-VM CAPEv2 deployment; throughput depends on host hardware
- Step 7 (PII redaction): a few minutes per tarball
- Step 8 (verification): ~1 minute
- Step 9 (reproduce baseline tables): about 4.4 hours (~15,850 s) for the full
  five-seed LightGBM suite on a 60-core CPU server; a single-seed smoke run
  completes in tens of minutes
