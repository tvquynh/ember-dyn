# EMBER-Dyn

A coordinated tri-modal malware benchmark for static, dynamic, and
capability-grounded detection research.

![DOI](https://img.shields.io/badge/DOI-reserved%20(minted%20on%20release)-lightgrey)
[![Licence: CC-BY-4.0](https://img.shields.io/badge/dataset%20licence-CC--BY--4.0-green)](https://creativecommons.org/licenses/by/4.0/)
[![Licence: MIT](https://img.shields.io/badge/code%20licence-MIT-orange)](LICENSE-MIT.txt)

> **The dataset is released openly under CC-BY-4.0 on Zenodo** (DOI below). This
> repository is the **code-and-documentation mirror** (code, notebooks, reference
> baselines, datasheet). Download the Zenodo deposit (or use a provided copy),
> then unpack it; all data paths in this README are relative to that dataset root.

**Version**: v1.0.0 · **Maintainer**: Van-Quynh Trinh (`quynhtv@ptithcm.edu.vn`) ·
**Corresponding author**: Van-Quynh Trinh (`quynhtv@ptithcm.edu.vn`)

---

## 1. What is EMBER-Dyn?

EMBER-Dyn is a **10,000-sample** curated subset of the EMBER2024
benchmark corpus (Joyce *et al.*, KDD 2025) extended with three new
modalities that the upstream release does not provide:

| Modality | Files | Description |
|---|---|---|
| **Static features** | `dataset/static/*.parquet` + `*.jsonl.zst` | 2,568-dim EMBER feature vector + full raw EMBER JSON record |
| **Dynamic Small (Avast-CTU 2022 schema)** | `dataset/dynamic/small/*.tar.zst` (~41 MB) | 13 `behavior.summary` keys + a per-sample `apistats` API-call histogram + 23 `static.pe` keys |
| **Dynamic Full (CAPE report.json)** | `dynamic/full/*.tar.zst` (~3.0 GB, CC-BY-4.0, on Zenodo) | Dynamic CAPE keys (behavior, signatures, network, ttps, …); binary-payload sections (dropped, procdump, procmemory, CAPE.payloads) removed for safety |
| **Annotations** | `dataset/annotations/*.parquet` | capa capabilities, MITRE ATT&CK techniques, MBC behaviours, APT group |

Behavioural reports with captured execution are available for 9,086 of
the 10,000 samples (90.9%); the remaining 914 produced a structurally
valid but behaviourally empty report under the single sandbox
configuration and are flagged by the `dynamic_status` column. The
dataset additionally ships a `failed_detonation_samples.json` artefact
recording 1,001 SHA-256 hashes of *candidate* samples that did not
produce a valid behavioural report under our single CAPEv2
configuration and were excluded from the 10,000 (zero overlap with the
manifest). We deliberately **do not** label these as confirmed anti-VM
evaders — non-execution under one configuration can equally reflect a
missing dependency or environment mismatch.

---

## 2. Quick numbers

| Item | Value |
|---|---|
| Total samples | **10,000** (8,126 malicious + 1,874 benign) |
| Splits | train 7,000 · val 997 · test 1,907 · challenge 96 (eval-side near-duplicate-disjoint: test/challenge ≥ TLSH 30 from train+val) |
| Families | 245 AVClass families (5 primary in Subset A) |
| Static feature dim | 2,568 (EMBER feature version 3, introduced by EMBER2024) |
| Manifest columns | 29 full (= 26 + 3 aggregate re-check columns); a 26-column variant without re-check is also provided |
| Static parquet coverage | 10,000 / 10,000 (100%) |
| Dynamic coverage | 9,086 / 10,000 (90.9%) behaviour; flagged by `dynamic_status` |
| Annotation parquets | 4 (capa, MITRE, MBC, APT) |
| Compressed size | compact core ~116 MB · Full dynamic reports ~3.0 GB (all CC-BY on Zenodo) |
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
# dynamic_status is a manifest field (not a static-parquet column):
pd.read_csv("dataset/manifest_open.csv")["dynamic_status"].value_counts()  # behavior / no_behavior
```

### Decode Small dynamic variant

```bash
mkdir -p out_small
tar -I zstd -xf dataset/dynamic/small/train.tar.zst -C out_small/
# Each <sha256>.json: behavior.summary (13 keys) + behavior.apistats + static.pe (23 keys)
```

### Decode Full dynamic variant

The Full CAPE reports (~3.0 GB) are released openly on Zenodo under CC-BY-4.0.
After downloading the deposit:

```bash
mkdir -p out_full
tar -I zstd -xf dynamic/full/train.tar.zst -C out_full/
# Each <sha256>.json is a CAPE report.json (binary-payload sections removed)
```

### Multi-label training with annotation parquets

```python
from sklearn.preprocessing import MultiLabelBinarizer
caps = pd.read_parquet("dataset/annotations/capa_caps.parquet")
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
├── baselines/
│   ├── run_baselines_final.py         reference LightGBM baseline suite (T1–T15 + TD1)
│   └── results_final.json             the reported baseline numbers
│
├── code/
│   ├── README.md
│   ├── RUNBOOK.md                     end-to-end rebuild instructions
│   ├── requirements.txt
│   └── scripts/                       build-pipeline + verification scripts
│
└── notebooks/                         6 notebooks (overview + 5 quickstarts)
```

The dataset itself (all artefacts — static parquets and raw JSONL, the Small
**and** Full CAPE reports, annotation parquets, manifests, splits, task and
leakage files) is released openly under **CC-BY-4.0** and archived on **Zenodo**
with a persistent DOI (above). It is plain Parquet/JSON --- load with
pandas/pyarrow, no special tooling. This repository carries the code,
documentation, notebooks, and reference baselines that operate on it.

---

## 5. Benchmark baselines

We report LightGBM baselines on the full 10,000-sample release,
mirroring the EMBER2024 reference hyper-parameter specification (500
boosting rounds with early stopping, 64 leaves, 100 minimum data
points per leaf, learning rate 0.05). Single-label detection tasks
(T1–T3, T7, T8) are five-seed means over seeds {42, 123, 456, 789,
1011}; the remaining tasks use the primary seed under their
task-specific protocol. Task IDs follow `task_definitions.json`.

### 5.1 Saturated standard tasks — reference floor

| ID | Task | Metric | Value |
|---|---|---|---|
| T1 | Binary detection (static) | ROC AUC | **0.9985** |
| T2 | Family 5-class on Subset A | macro F1 | **1.000** |
| T7 | Non-UPX packer detection | ROC AUC / PR AUC | 0.9981 / 0.9728 |
| T8 | Exploit detection | ROC AUC / PR AUC | 0.9906 / 0.8641 |
| T4 | Open-set Subset A vs B (MSP / free-energy) | AUROC | 0.9992 / 0.9937 |
| T13 | Per-architecture (Win32 / Win64 / .NET) | ROC AUC | 0.998 / 0.999 / 1.000 |
| T10/T11/T12 | capa / MITRE / MBC multi-label, head third | macro F1 | 0.807 / 0.820 / 0.847 |

These ceilings are consistent with EMBER2024's reported test-set
performance (ROC AUC ~0.997) and confirm that EMBER-Dyn behaves as
expected on tasks where the EMBER feature set is known to be
discriminative.

### 5.2 🎯 Open research challenges — measurable headroom for the community

| ID | Task | Headline metric | Headroom | Research direction |
|---|---|---|---|---|
| **T15** | Challenge OOD (challenge + test benign) | **PR AUC 0.515**, ROC AUC 0.858, F1 0.347 | **0.48** | Challenge-set OOD / evasive detection |
| **T9** | APT group-tag prediction (Subset E2, 6-way) | **macro F1 0.144** | 0.86 | Few-example, weak-label group-tag prediction |
| T3 | Temporal distribution shift (pre vs. post 2024-09) | ROC AUC 0.855 | 0.15 | Robust representations under temporal shift |
| **T6** | Few-shot family (Subset C, 5-way 5-/10-shot) | **Accuracy 0.394 / 0.411** | ~0.6 | Meta-learning, prototypical networks |
| T10–T12 | Long-tail multi-label, tail third | macro F1 0.38–0.45 | ~0.55 | Long-tail multi-label learning |
| T14 | Capability-count regression | R² 0.844 | 0.16 | Structured capability inference |
| **T5** | Multi-modal fusion (static + dynamic) | best fusion 0.9987 vs static 0.9985 | CI [−0.0002,+0.0007] | Cross-modal gating, hierarchical MIL |

The dynamic-only baseline (TD1: TF-IDF over the behavioural-summary
fields concatenated with the `apistats` histogram) reaches ROC AUC
0.9877, yet naive late fusion of the static and dynamic probabilities
is **statistically indistinguishable** from the static-only ceiling
(paired bootstrap 95% CI straddling zero). The dynamic modality carries
independent signal; converting it into a gain over the static ceiling
with a dedicated multimodal architecture (gating, hierarchical MIL,
contrastive alignment) is the open problem.

### 5.3 Reproduce all numbers

```bash
cd baselines
export EMBER_DYN_BASELINES=../dataset          # the open-tier root in this repo
python run_baselines_final.py                  # produces results_final.json (T1–T15 + TD1 + diversity)
```

The suite reads `manifest_open.csv` (the three aggregate re-check columns in
the 29-column `manifest.csv` are not required for the fifteen benchmark tasks).

---

## 6. 15 pre-defined tasks

`dataset/task_definitions.json` enumerates 15 tasks with their split
definitions and reference metrics:

| ID | Task | Reference metric |
|---|---|---|
| T1  | Binary detection                          | AUC, F1 |
| T2  | Family multi-class (5-way)                | Macro F1 |
| T3  | Temporal distribution shift (pre/post 2024-09) | ROC AUC, F1 |
| T4  | Open-set detection (A known vs. B novel)  | AUROC, FPR@95%TPR |
| T5  | Hybrid static + dynamic fusion            | AUC vs. single-modality baseline |
| T6  | Few-shot family (N=5, K in {5,10})        | Episodic accuracy |
| T7  | Non-UPX packer detection                  | AUC, F1 |
| T8  | Exploit / CVE-bearing                     | AUC, F1 |
| T9  | APT group-tag prediction (6-way)          | Macro F1 |
| T10 | capa capability multi-label               | Macro F1 |
| T11 | MITRE ATT&CK TTP multi-label              | Macro F1 |
| T12 | MBC behaviour multi-label                 | Macro F1 |
| T13 | Per-architecture detector (Win32/64/.NET) | AUC per `file_type` |
| T14 | Capability count regression               | MAE, R^2 |
| T15 | Challenge OOD (96 time-distinct samples)  | PR AUC / ROC AUC |

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
re-split; `dataset/manifest_open.csv` is
the canonical record of the final sample list and its
train/val/test/challenge assignment (samples are identified by SHA-256;
the 29-column `manifest.csv` adds the three aggregate re-check columns).
Selection is **purposive and
criterion-based**: each subset is defined by an explicit purpose plus
an explicit eligibility criterion on EMBER2024 metadata, with
proportional stratification across families and time buckets. Where
the eligible pool exceeds the target count, samples are drawn by
descending AVClass `family_confidence`. See
`paper/manuscript/sections/03_methods.tex` for the full methodology and
`code/RUNBOOK.md` for the step-by-step rebuild.

Reproducing the dynamic variants additionally requires raw PE binaries
plus a CAPEv2 + KVM/libvirt sandbox deployment. We do not redistribute
the binaries; obtain them through VirusTotal Premium access or
institutional malware-sharing arrangements using the SHA-256
identifiers in the manifest. The sandbox configuration is documented
in the manuscript's Methods section.

---

## 8. Known limitations

- **Dynamic coverage 90.9%**. 9,086 of 10,000 samples produced a
  behavioural report with captured execution; the other 914 produced a
  structurally valid but behaviourally empty report under the single
  sandbox configuration and are flagged by the `dynamic_status` column.
  Static features and all annotations are provided for every sample, so
  downstream users can filter on dynamic availability without losing
  coverage.
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

The filter is published as `code/scripts/redact_pii.py` — the exact version
that produced the release — so running it on the upstream raw CAPE reports
reproduces the released redaction. To re-run the filter on a
fresh CAPE report set:

```bash
python code/scripts/redact_pii.py path/to/raw.tar.zst path/to/redacted.tar.zst
```

**Limitations of the filter.** This is a best-effort, regex-based
filter. Residual username fragments may persist in non-pattern
positions (for example, embedded inside other strings or encoded
artefacts) — we recommend treating all string values in dynamic
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
binaries, dropped files, memory dumps, or packet captures. Dynamic
detonation was performed on triple-isolated infrastructure
(`forward='none'` libvirt bridge, `iptables FORWARD` drop, INetSim
simulated services) with no public-internet egress observed during
detonation. The release does not include human-subject data,
personally identifiable information, or telemetry from third-party
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
  attribution. See the licence/provenance matrix in the paper (Table 2).
- **Code** (everything under `code/`): MIT
- **Manuscript** (everything under `paper/`): CC-BY-4.0 (post-publication)
- **PE binaries**: not redistributed — see § 7

---

## 12. Citation

```bibtex
@article{huynh2026emberdyn,
  title   = {EMBER-Dyn: A Coordinated Tri-Modal Malware Benchmark for Static,
             Dynamic, and Capability-Grounded Detection Research},
  author  = {Huynh, Trong-Thua and Huynh, De-Thu and Nguyen, Minh Thang
             and Trinh, Van-Quynh},
  journal = {IEEE Access},
  year    = {2026},
  doi     = {10.5281/zenodo.PLACEHOLDER}
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

Corresponding author: Van-Quynh Trinh — `quynhtv@ptithcm.edu.vn` — ORCID 0009-0006-0514-6123
Repository maintainer: Van-Quynh Trinh — `quynhtv@ptithcm.edu.vn` — ORCID 0009-0006-0514-6123

Issues and pull requests welcome at
<https://github.com/tvquynh/ember-dyn/issues>.
