# EMBER-Dyn

A coordinated static-dynamic malware benchmark with capability-grounded
annotations.

**Zenodo archival DOI:** [10.5281/zenodo.22178427](https://doi.org/10.5281/zenodo.22178427).
[![EMBER-Dyn-original: CC-BY-4.0](https://img.shields.io/badge/EMBER--Dyn--original-CC--BY--4.0-green)](https://creativecommons.org/licenses/by/4.0/)
[![EMBER2024-derived: Apache-2.0](https://img.shields.io/badge/EMBER2024--derived-Apache--2.0-blue)](https://www.apache.org/licenses/LICENSE-2.0)
[![Code: MIT](https://img.shields.io/badge/code-MIT-orange)](LICENSE-MIT.txt)

License scope is artifact-specific: EMBER-Dyn-original artifacts use
CC-BY-4.0, EMBER2024-derived artifacts retain Apache-2.0, and code uses MIT.

**Version**: v1.0.0 · **Maintainer**: Van-Quynh Trinh (`quynhtv@ptithcm.edu.vn`) ·
**Corresponding author**: Trong-Thua Huynh (`thuaht@ptit.edu.vn`)

---

## 1. What is EMBER-Dyn?

EMBER-Dyn is a **10,000-sample** purposively curated subset of the EMBER2024
benchmark corpus (Joyce *et al.*, KDD 2025). It coordinates two measured views
and a semantic annotation layer:

| Release component | Files | Description |
|---|---|---|
| **Static features** | `dataset/static/*.parquet` + `*.jsonl.zst` | 2,568-dim EMBER feature vector + full raw EMBER JSON record |
| **Dynamic Small (Avast-CTU 2022 schema)** | `dataset/dynamic/small/*.tar.zst` (~41 MB) | 13 `behavior.summary` keys + a per-sample `apistats` API-call histogram + 23 `static.pe` keys |
| **Dynamic Full (CAPE report.json)** | `dataset/dynamic/full/*.tar.zst` (~3.0 GB) | Dynamic CAPE keys (behavior, signatures, network, ttps, …); binary-payload sections (dropped, procdump, procmemory, CAPE.payloads) removed for safety |
| **Annotations** | `dataset/annotations/*.parquet` | static-derived capa capabilities, MITRE ATT&CK techniques, MBC behaviours, and weak group tags |

Non-empty behavioral summary or API-count observations are available for 9,086 of the 10,000
samples (90.86%); the remaining 914 have a structurally valid report with an
empty summary and API map and are flagged by `dynamic_status`. This status does
not prove successful execution or anti-analysis behavior. The
dataset additionally ships a `failed_detonation_samples.json` artefact
recording 1,001 SHA-256 hashes of *candidate* samples that did not
produce a valid behavioural report under our single CAPEv2
configuration and were excluded from the 10,000 (zero overlap with the
manifest). We deliberately **do not** label these as confirmed anti-VM
evaders. Non-execution under one configuration can equally reflect a
missing dependency or environment mismatch.

---

## 2. Quick numbers

| Item | Value |
|---|---|
| Total samples | **10,000** (8,126 malicious + 1,874 benign) |
| Splits | train 7,000 · val 997 · test 1,907 · challenge 96 (eval-side near-duplicate-disjoint: test/challenge ≥ TLSH 30 from train+val) |
| Families | 245 AVClass families (5 primary in Subset A) |
| Static feature dim | 2,568 (EMBER feature version 3, introduced by EMBER2024) |
| Manifest columns | 29 full (`manifest.csv`, canonical) · 26 without re-check (`manifest_open.csv`) · 3 aggregate re-check columns |
| Static parquet coverage | 10,000 / 10,000 (100%) |
| Dynamic observed-content coverage | 9,086 / 10,000 (90.86%); flagged by `dynamic_status` |
| Annotation parquets | 4 (capa, MITRE, MBC, APT) |
| Compressed size | final compressed core archive ~95 MB · Full dynamic reports ~3.0 GB |
| Benchmark tasks predefined | 15 in `task_definitions.json` |

---

## 3. Quickstart

### Install dependencies

```bash
pip install pandas pyarrow lightgbm scikit-learn zstandard
```

### Load static features (3 lines)

```python
import pandas as pd
df = pd.read_parquet("dataset/static/train.parquet")
X = df.filter(regex=r"^feature_").values    # (7000, 2568) float32
y = df["label"].values                       # 0 = benign, 1 = malicious
```

### Train a LightGBM binary detector

```python
import lightgbm as lgb
clf = lgb.LGBMClassifier(
    n_estimators=500, num_leaves=64,
    min_data_in_leaf=100, learning_rate=0.05,
    n_jobs=-1, random_state=42,
)
clf.fit(X, y)
```

### Inspect capabilities

```python
df["caps_list"].head()        # capa capability identifiers per sample
df["ttps_list"].head()        # MITRE ATT&CK technique identifiers
df["mbc_list"].head()         # MBC behaviour identifiers
df["subset"].value_counts()   # A / B / C / D / E1 / E2 / E3 / F / Challenge
df["time_bucket"].value_counts()   # pre / post 2024-09
# dynamic_status is a manifest field (not a static-parquet column).
# 'behavior' = non-empty behavior.summary OR non-empty apistats (the manifest rule).
# Most reports have both; ~57 have a summary but empty apistats and 66 have
# apistats but an empty summary. Check len(report["behavior"]["apistats"])>0
# when building API-histogram features.
pd.read_csv("dataset/manifest_open.csv")["dynamic_status"].value_counts()  # behavior / no_behavior
```

### Decode Small dynamic variant

```bash
mkdir -p out_small
tar -I zstd -xf dataset/dynamic/small/train.tar.zst -C out_small/
# Each <sha256>.json: behavior.summary (13 keys) + behavior.apistats + static.pe (23 keys)
```

### Decode Full dynamic variant

```bash
mkdir -p out_full
tar -I zstd -xf dataset/dynamic/full/train.tar.zst -C out_full/
# Each <sha256>.json is a CAPE report.json (binary-payload sections removed)
```

### Multi-label training with annotation parquets

```python
from sklearn.preprocessing import MultiLabelBinarizer
caps = pd.read_parquet("dataset/annotations/capa_caps.parquet")
# Annotation parquets keep occurrence multiplicity. The capa `capability` column
# is already distinct per sha (so the drop_duplicates below is a no-op here), but
# mitre `technique` (~37% duplicate rows), mbc `behavior` (~12%), and the coarse
# grouping columns (namespace/tactic/objective, 19-68%) DO repeat -- de-duplicate
# on whichever column you count/group by before any value_counts()/frequency
# analysis. The manifest *_list columns are already distinct.
caps = caps.drop_duplicates(["sha256", "capability"])
y_multi = caps.groupby("sha256")["capability"].agg(list)
mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(y_multi)
```

See `notebooks/` for end-to-end Colab quickstarts (LightGBM, XGBoost,
MLP-PyTorch, Random Forest, TF-IDF+LightGBM dynamic).

---

## 4. Repository layout

```
ember-dyn/
├── README.md                          this file
├── DATASHEET.md                       Gebru-style datasheet
├── LICENSE-CC-BY-4.0.txt              dataset licence
├── LICENSE-MIT.txt                    code licence
├── CITATION.cff                       citation metadata
│
├── baselines/                         frozen results, runner, and vocabularies
│
├── code/
│   ├── README.md
│   ├── RUNBOOK.md                     end-to-end rebuild instructions
│   ├── requirements.txt
│   └── scripts/                       build-pipeline + verification scripts
│
└── notebooks/                         6 Colab notebooks (overview + 5 quickstarts)
```

The canonical EMBER-Dyn v1.0.0 dataset release is archived on Zenodo at DOI
`10.5281/zenodo.22178427`. It contains the final compressed core archive
(approximately 95 MB) and Full reports (approximately 3.0 GB). The core uses
plain Parquet/JSONL and loads with pandas/pyarrow.

---

## 5. Benchmark baselines

We report LightGBM baselines on the full 10,000-sample release,
mirroring the EMBER2024 reference hyper-parameter specification (500
boosting rounds with early stopping, 64 leaves, 100 minimum data
points per leaf, learning rate 0.05). Single-label detection tasks
(T1–T3, T7, T8) are five-seed means over seeds `42`, `123`, `456`, `789`, and
`1011`; the remaining tasks use the primary seed under their
task-specific protocol. Task IDs follow `task_definitions.json`.

### 5.1 Saturated standard tasks as a reference floor

| ID | Task | Metric | Value |
|---|---|---|---|
| T1 | Binary detection (static) | ROC AUC | **0.9985** |
| T2 | Family 5-class on Subset A | macro F1 | **1.000** |
| T7 | Non-UPX packer detection | ROC AUC / AP | 0.9981 / 0.9728 |
| T8 | Exploit detection | ROC AUC / AP | 0.9906 / 0.8641 |
| T4 | Open-set Subset A vs B (MSP / free-energy) | AUROC | 0.9992 / 0.9937 |
| T13 | Per-architecture (Win32 / Win64 / .NET) | ROC AUC | 0.998 / 0.999 / 1.000 |
| T10/T11/T12 | capa / MITRE / MBC multi-label, head third | macro F1 | 0.807 / 0.820 / 0.847 |

These ceilings are consistent with EMBER2024's reported test-set
performance (ROC AUC ~0.997) and confirm that EMBER-Dyn behaves as
expected on tasks where the EMBER feature set is known to be
discriminative.

### 5.2 Diagnostic tasks and protocol boundaries

| ID | Task | Headline metric | Boundary | Research direction |
|---|---|---|---|---|
| **T15** | Challenge malware vs test benign | Static AP 0.5218; frozen fusion AP 0.4576 | Class/source-split confounding | Stress comparison |
| **T9** | Exploratory weak-label group tag (six classes) | **macro F1 0.120** | One class has zero test support | Few-example group-tag prediction |
| T3 | Temporal separability (pre vs post 2024-09) | ROC AUC 0.855 | Not future-drift performance | Temporal distribution signal |
| **T6** | Few-shot family (Subset C, 5-way 5-/10-shot) | **Accuracy 0.394 / 0.411** | ~0.6 | Meta-learning, prototypical networks |
| T10–T12 | Long-tail multi-label, tail third | macro F1 0.38–0.45 | ~0.55 | Long-tail multi-label learning |
| T14 | Capability-count regression | R² 0.844 | 0.16 | Structured capability inference |
| **T5** | Validation-selected static-dynamic fusion | frozen fusion 0.9980 vs static 0.9985 ROC AUC | Difference CI includes zero | Missingness-aware cross-view models |

T5 fits 19,152 TF-IDF and 322 API-count features on training data only, for
19,474 learned columns total. This is a baseline representation, not a fixed
dataset dimension. Static
and dynamic models use validation early stopping; validation selects the frozen
`0.7 * static + 0.3 * dynamic` rule. Test ROC AUC/AP is 0.9984996969/
0.9996060029 for static, 0.9868129925/0.9956847269 for dynamic, and
0.9979676029/0.9994510333 for fusion. The fusion-minus-static confidence
interval includes zero, so the tested rule does not improve static-only.

### 5.3 Reproduce all numbers

```bash
cd baselines
export EMBER_DYN_BASELINES=/path/to/dataset   # the dataset root
python run_baselines_final.py                 # produces results_final.json (T1–T15 + TD1 + diversity)
```

---

## 6. 15 pre-defined tasks

`dataset/task_definitions.json` enumerates 15 tasks with their split
definitions and reference metrics:

| ID | Task | Reference metric |
|---|---|---|
| T1  | Binary detection                          | AUC, F1 |
| T2  | Family multi-class (5-way)                | Macro F1 |
| T3  | Temporal separability (pre vs. post 2024-09) | ROC AUC, F1 |
| T4  | Open-set detection (A known vs. B novel)  | AUROC, FPR@95%TPR |
| T5  | Validation-selected static-dynamic fusion | ROC AUC, AP, paired differences |
| T6  | Few-shot family (N=5, K in {5,10})        | Episodic accuracy |
| T7  | Non-UPX packer detection                  | AUC, F1 |
| T8  | Exploit / CVE-bearing                     | AUC, F1 |
| T9  | Exploratory weak-label group tag (6-way)  | Fixed-six-class macro F1 |
| T10 | capa capability multi-label               | Macro F1 |
| T11 | MITRE ATT&CK TTP multi-label              | Macro F1 |
| T12 | MBC behaviour multi-label                 | Macro F1 |
| T13 | Per-architecture detector (Win32/64/.NET) | AUC per `file_type` |
| T14 | Capability count regression               | MAE, R^2 |
| T15 | Challenge malware vs test benign          | AP / ROC AUC |

`run_baselines_final.py` (§ 5) reports reference baselines for all
fifteen tasks plus the dynamic-only TD1 intermediate and a model-class
diversity panel.

---

## 7. Reproducibility

The purposive selection stage from the EMBER2024 clean-week parquet is
deterministic: running `code/scripts/sample_ember_dyn.py` against the
same EMBER2024 release reproduces the criterion-based selection,
including the broad-coverage family pool (subset `F`). The released
10,000-sample set is finalised by a near-duplicate-disjoint (TLSH)
re-split; `dataset/manifest.csv` is
the canonical record of the final sample list and its
train/val/test/challenge assignment (samples are identified by SHA-256).
Selection is **purposive and
criterion-based**: each subset is defined by an explicit purpose plus
an explicit eligibility criterion on EMBER2024 metadata, with
proportional stratification across families and time buckets. Where
the eligible pool exceeds the target count, samples are drawn by
descending AVClass `family_confidence`. See the accompanying manuscript for the
full methodology and `code/RUNBOOK.md` for the step-by-step rebuild.

Reproducing the dynamic variants additionally requires raw PE binaries
plus a CAPEv2 + KVM/libvirt sandbox deployment. We do not redistribute
the binaries; re-acquire them by hash from public malware-sharing
sources using the SHA-256 identifiers in the manifest. The repository
ships a helper for this, `code/scripts/fetch_binaries.py`, which fetches
samples from MalwareBazaar (free API) with a VirusTotal (premium)
fallback. Coverage is partial: VirusTotal file download is premium-only
and MalwareBazaar indexes confirmed malware only (no benign or older
samples), so retrieval is skewed against the benign and long-tail cells.
The sandbox configuration is documented in the manuscript's Methods
section.

---

## 8. Known limitations

- **Dynamic observed-content coverage 90.86%**. For 9,086 of 10,000
  samples, the report contains a non-empty behavioral summary or API-count
  map. The other 914 have no such observed content under the single sandbox
  configuration and are flagged by `dynamic_status`. This flag does not
  establish whether a sample executed. Static features and annotations are
  provided for every sample.
- **Short time window**. EMBER2024 covers 15 months (2023-09 to
  2024-12), so EMBER-Dyn supports short-horizon temporal evaluation
  but not multi-year drift analysis.
- **Single sandbox configuration**. One Windows 10 22H2 guest
  template, INetSim simulated services, five-minute analysis budget.
  Behaviours requiring different OS versions, longer timeouts, or
  user interaction may be under-represented.
- **No raw PE redistribution**. By design, following the upstream
  EMBER2024 policy.

---

## 9. PII redaction policy

Every dynamic Small and Full report in v1.0 has been processed by a
deterministic, best-effort PII filter prior to release. The filter
applies, at the JSON-value level (never on raw bytes, so string escaping is
preserved), the following substitutions to every `.json` member of every
dynamic tarball:

| Source pattern | Replacement |
|---|---|
| account name in **any** `Users\<name>` path form (drive-letter, `%SystemDrive%`, `\Device\HarddiskVolume*`), except `Public`/`Default`/`All Users` | `Users\<user>` |
| `<host>\cape` and other analysis-account references | `<user>` |
| analyzer install paths (`\CAPE\…`, `C:\CAPE`, `/opt/CAPEv2`) | `<analyzer-path>` / `<sandbox-storage>` |
| `capemon` analyzer module | `<analyzer-module>` |
| `info.machine` record (name / label / IP / timestamps) | `<sandbox-host>` / `<sandbox-ip>` / `<ts>` |
| RFC1918 sandbox addresses | `<sandbox-ip>` |
| email addresses | `<email>` |
| `/user:<name>` and `password=<value>` | `<user>` / `<redacted>` |

The exact released filter is published as `code/scripts/redact_pii.py`, so running it on the upstream raw CAPE reports
reproduces the released redaction. To re-run the filter on a
fresh CAPE report set:

```bash
python code/scripts/redact_pii.py path/to/raw.tar.zst path/to/redacted.tar.zst
```

**Limitations of the filter.** This is a best-effort, regex-based
filter. Residual username fragments may persist in non-pattern
positions (for example, embedded inside other strings or encoded
artefacts). We recommend treating all string values in dynamic
reports as untrusted. Observed network destinations (IPs, domains,
TLS metadata) in Full reports are intentionally preserved because
they are essential for network-traffic-based detection research;
researchers should treat the Full variant as containing live threat
intelligence and follow defender-grade data-handling practices.

---

## 10. Ethics

EMBER-Dyn redistributes static feature vectors, raw EMBER JSON
records, and sandbox behavioural reports derived from PE binaries
indexed by EMBER2024. EMBER-Dyn does **not** redistribute live malware
binaries, dropped files, memory dumps, or packet captures. Project
documentation specifies three isolation controls: a libvirt bridge with
`forward='none'`, an `iptables FORWARD` drop rule, and INetSim simulated
services. The release does not contain operational network logs that
independently verify isolation or egress for every run. The release does not include
personally identifiable information or telemetry from third-party
endpoints. The distributed feature vectors and behavioural summaries
do not contain executable payloads.

Intended uses are research, education, and defensive security
evaluation in controlled environments.

---

## 11. Licence

- **Dataset**: EMBER-Dyn original artefacts (CAPE dynamic reports, curation
  columns, manifests) are **CC-BY-4.0**; EMBER2024-derived artefacts (static
  feature parquets, raw EMBER JSON, capa/MITRE/MBC annotation parquets) inherit
  the upstream **Apache-2.0** licence of EMBER2024 and are redistributed with
  attribution. See the Zenodo deposit licence files (`LICENSE-CC-BY-4.0.txt`, `LICENSE-Apache-2.0.txt`) for the per-artefact licence/provenance.
- **Code** (everything under `code/`): MIT
- **PE binaries**: not redistributed; see Section 7

---

## 12. Citation

```bibtex
@dataset{trinh2026emberdyn,
  author    = {Trinh, Van-Quynh and Huynh, Trong-Thua and
               Huynh, De-Thu and Nguyen, Minh Thang},
  title     = {EMBER-Dyn: A Coordinated Static-Dynamic Benchmark with
               Capability-Grounded Annotations for Malware Analysis},
  year      = {2026},
  publisher = {Zenodo},
  version   = {1.0.0},
  doi       = {10.5281/zenodo.22178427},
  url       = {https://doi.org/10.5281/zenodo.22178427}
}
```

Please also cite the upstream EMBER2024 paper:

```bibtex
@inproceedings{joyce2025ember2024,
  title     = {{EMBER2024 - A Benchmark Dataset for Holistic Evaluation
              of Malware Classifiers}},
  author    = {Joyce, Robert J. and Miller, Gideon and Roth, Phil and
              Zak, Richard and Zaresky-Williams, Elliott and Anderson,
              Hyrum and Raff, Edward and Holt, James},
  booktitle = {Proceedings of the 31st ACM SIGKDD Conference on
              Knowledge Discovery and Data Mining (KDD '25)},
  year      = {2025},
  doi       = {10.1145/3711896.3737431}
}
```

---

## 13. Contact

Corresponding author: Trong-Thua Huynh, `thuaht@ptit.edu.vn`, ORCID 0000-0003-3934-1067
Repository maintainer: Van-Quynh Trinh, `quynhtv@ptithcm.edu.vn`, ORCID 0009-0006-0514-6123

Issues and pull requests welcome at
<https://github.com/tvquynh/ember-dyn/issues>.
