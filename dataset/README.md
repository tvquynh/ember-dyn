# EMBER-Dyn

A coordinated tri-modal, 10,000-sample malware benchmark drawn from the
EMBER2024 portable-executable (PE) corpus, designed for static, dynamic, and
capability-grounded detection research on a single workstation or a free cloud
notebook tier.

Each sample is materialised in coordinated modalities and linked by SHA-256:
a 2,568-dimensional EMBER static feature vector, the raw hierarchical EMBER
JSON record, a CAPEv2 sandbox behavioural report (a compact *Small* variant
and a complete *Full* variant), and four capability-annotation tables
(capa capabilities, MITRE ATT&CK techniques, MBC behaviours, APT group tags).

- **Samples:** 10,000 (8,126 malicious across 245 AVClass families; 1,874 benign)
- **Collection window:** 2023–2024 (inherited from EMBER2024)
- **Splits:** train 7,000 / val 997 / test 1,907 / challenge 96 (eval-side near-duplicate-disjoint: test/challenge ≥ TLSH 30 from train+val)
- **Pre-defined tasks:** 15 (`task_definitions.json`)
- **License (open tier):** CC-BY-4.0 for EMBER-Dyn original artefacts (CAPE dynamic reports, curation columns, manifests); EMBER2024-derived artefacts (static feature parquets, raw EMBER JSON, capa/MITRE/MBC annotations) inherit the upstream **Apache-2.0** licence of EMBER2024.

## Citation

If you use EMBER-Dyn, please cite the accompanying data descriptor and the
upstream EMBER2024 corpus. (The dataset DOI is assigned on archival deposit and
will appear here.)

```
@article{huynh2026emberdyn,
  title   = {EMBER-Dyn: A Coordinated Tri-Modal Malware Benchmark for Static,
             Dynamic, and Capability-Grounded Detection Research},
  author  = {Huynh, Trong-Thua and Huynh, De-Thu and Nguyen, Minh Thang and Trinh, Van-Quynh},
  journal = {IEEE Access},
  year    = {2026},
  note    = {DOI to be assigned on deposit}
}
```

## File layout

```
manifest_open.csv          # 26-column registry of all 10,000 samples (open tier)
manifest.csv               # 29 columns = manifest_open + 3 re-check columns (controlled tier)
manifest_recheck.csv       # sha256 + the 3 Q1-2026 re-check columns (controlled tier)
splits.json                # split membership (train/val/test/challenge)
task_definitions.json      # the 15 pre-defined benchmark tasks (T1..T15)
tlsh_leakage.csv           # each eval sample's nearest-train TLSH distance (open tier)
failed_detonation_samples.json   # 1,001 candidate hashes excluded (controlled tier)
checksums.sha256           # SHA-256 of every open-tier file (23 entries)
controlled_checksums.sha256  # SHA-256 of every controlled-tier file (7 entries)
CITATION.cff               # citation metadata
LICENSE-CC-BY-4.0.txt      # open-tier licence

static/
  {train,val,test,challenge}.parquet   # 2,568 EMBER features + metadata per split
  {train,val,test,challenge}.jsonl.zst # raw hierarchical EMBER JSON records

dynamic/
  small/{train,val,test,challenge}.tar.zst  # Reduced CAPE reports (open tier)
  full/{train,val,test,challenge}.tar.zst   # full CAPE report.json (controlled tier)

annotations/
  capa_caps.parquet      # sha256 -> capa capability
  mitre_ttps.parquet     # sha256 -> MITRE ATT&CK technique
  mbc_behaviors.parquet  # sha256 -> MBC behaviour
  group_apt.parquet      # sha256 -> APT group tag
```

Each `dynamic/**/*.tar.zst` member is named `<sha256>.json`. The Small report
carries 13 behavioural-summary fields, a per-sample `apistats` API-call
histogram, and a 23-field static PE block; the Full report carries the complete
CAPE sections.

## Subsets (`subset` column)

| Subset | n | Meaning |
|---|---:|---|
| A | 5,359 | five known families (high-confidence) |
| B | 1,359 | open-set / unknown-family pool |
| C | 381 | few-shot rare families |
| D | 1,845 | benign |
| E1 | 109 | exploit-bearing (rare gem) |
| E2 | 177 | APT group-tagged (rare gem) |
| E3 | 191 | non-UPX packer (rare gem) |
| F | 483 | broad-coverage family pool |
| Challenge | 96 | initially-evasive hold-out |

## Quick start

```python
import pandas as pd
m = pd.read_csv("manifest_open.csv")   # open tier; use manifest.csv under controlled access
print(len(m), "samples;", m.label.value_counts().to_dict())

train = pd.read_parquet("static/train.parquet")
feat = [c for c in train.columns if c.startswith("feature_")]
print(len(feat), "static features")   # 2568
```

Read a Small behavioural report:

```python
import tarfile, zstandard as zstd, io, json
with open("dynamic/small/test.tar.zst", "rb") as fh:
    buf = io.BytesIO(zstd.ZstdDecompressor().stream_reader(fh).read())
buf.seek(0)
with tarfile.open(fileobj=buf, mode="r:") as tar:
    member = next(m for m in tar if m.name.endswith(".json"))
    report = json.loads(tar.extractfile(member).read())
print(report["behavior"]["summary"].keys(), len(report["behavior"]["apistats"]))
```

## Dynamic-modality coverage

Behavioural reports with captured execution are available for 9,086 of the
10,000 samples (90.9%); the remaining 914 produced a structurally valid but
behaviourally empty report under the single sandbox configuration and are
flagged by the `dynamic_status` column (`behavior` / `no_behavior`). The Small
report schema is identical across all 10,000 samples.

## Splits and leakage

Splits are near-duplicate-disjoint: the minimum cross-side TLSH distance between
any train/val sample and any test/challenge sample is 30, so there are no
cross-split near-duplicate edges and train-to-test memorisation is removed as a
confound. Three identical-TLSH groups are retained, all on the train+val side
(two wholly within train, one spanning train and val); none cross into test or
challenge. The 96-sample challenge split contains a single intra-split
near-duplicate pair at TLSH distance 15, which does not affect the challenge task
(scored against test-split benign negatives). `tlsh_leakage.csv` ships each
evaluation sample's nearest-train TLSH distance (minimum 30).

## Access tiers

- **Open tier (CC-BY-4.0):** `manifest_open.csv`, `splits.json`,
  `task_definitions.json`, `tlsh_leakage.csv`, all static parquets and raw JSONL,
  all annotation parquets, and the Small dynamic reports.
- **Controlled-access tier (request + data-use agreement):** `manifest.csv` and
  `manifest_recheck.csv` (both carry the per-sample Q1-2026 re-check fields), the
  Full CAPE reports, and `failed_detonation_samples.json`.

## Provenance and ethics

EMBER-Dyn redistributes static feature vectors, raw EMBER JSON, sandbox
behavioural reports, and capability annotations derived from PE binaries indexed
by EMBER2024. It does **not** redistribute live malware binaries. Researchers
reproducing the dataset must independently re-acquire the binaries via the same
channels (e.g. VirusTotal or institutional access) using the SHA-256 identifiers.
Use is intended for defensive security research and education.
