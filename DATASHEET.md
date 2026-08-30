# Datasheet for EMBER-Dyn (Gebru-style)

Following the template proposed by Gebru et al., "Datasheets for Datasets" (CACM 2021).

---

## Motivation

**For what purpose was the dataset created?**
EMBER-Dyn was created to provide the malware-detection research community with a
compact coordinated benchmark combining static portable-executable features, raw EMBER
JSON records, and CAPEv2 sandbox dynamic behavioural reports for the same 10,000 PE
samples, joined with capa/MITRE ATT&CK/MBC annotations and structured around 15
pre-defined benchmark tasks.

**Who created the dataset?**
The four authors of this paper at the Posts and Telecommunications Institute of
Technology (PTIT), the Saigon International University (SIU), and the Ho Chi Minh
City University of Economics and Finance (UEF), Vietnam.

**Who funded the dataset creation?**
The accompanying paper does not declare a separate research grant or grant
identifier. It acknowledges institutional support with the statement: "This
research was supported by the Posts and Telecommunications Institute of
Technology (PTIT), Vietnam."

---

## Composition

**What do the instances represent?**
Each instance is one PE file from the EMBER2024 corpus, sampled between 2023-09-24
and 2024-12-14. Each instance carries: a 2568-dimensional static feature vector,
the raw EMBER JSON record, Small and Full CAPE sandbox reports, and a 29-column
manifest row (20 EMBER2024-inherited columns, 6 EMBER-Dyn curation columns, and
3 Q1-2026 VirusTotal re-check columns).

**How many instances total?**
10,000 instances, divided into four splits: train (7,000), validation (997),
test (1,907), challenge (96).

**Does the dataset contain all possible instances or a sample?**
Sample. The EMBER2024 source corpus has 3,238,315 instances; EMBER-Dyn contains a
deterministic, criterion-based purposive sample of 10,000 coordinated records. It is
not a probability sample or a statistically representative miniature of EMBER2024.
The exact intermediate eligible and submitted-candidate denominators cannot be
reconstructed from the release artifacts alone.

**What data does each instance consist of?**
Static feature vector (2,568 numerical features), raw EMBER JSON (hierarchical
record with histogram, byteentropy, strings, headers, sections, imports, exports,
data directories, rich header, authenticode, capa matches, ATT&CK matches, MBC
matches), CAPE Reduced JSON (`behavior.summary` 13 string lists, a per-sample
`apistats` API-call histogram, and `static.pe` 23 structural keys), and 29 manifest
columns (including `recheck_label`, `recheck_detections`, `recheck_date` from the
Q1-2026 re-check; the re-check fields are released as aggregate detection
counts only, not per-engine VirusTotal results).

**Is there a label or target?**
Yes. Each sample carries:
- Binary `label` (0=benign, 1=malicious)
- AVClass `family` string (or empty)
- `family_confidence` float
- `behavior`, `file_property`, `packer`, `exploit`, `group` list-valued capa tags
- `caps_list`, `ttps_list`, `mbc_list` capa/MITRE ATT&CK/MBC identifier lists
- `subset`, `subset_class`, `note` curation fields
- `dyn_split` train/val/test/challenge assignment
- `time_bucket` pre/post 2024-09
- `dynamic_status` behavior / no_behavior (report-content observation flag)

**Is any information missing?**
Several capa-derived fields are sparse by design (1.1% exploit, 2.2% group tags). The
`family` field is empty for benign samples (label=0) and for some low-confidence
malicious samples. Reports with non-empty behavioral summary or API-count content are
available for 9,086 of the 10,000 samples (90.86%); the remaining 914 have no such
observed content under the single sandbox configuration and are flagged by the
`dynamic_status` column. This flag does not establish whether a sample executed.
Static features and annotations remain available for every release sample. These
missing values are documented in the schema.

Observed-content coverage by subset (share of each subset with
`dynamic_status == behavior`):

| Subset | Behaviour / total | Coverage |
|---|---|---|
| A (known families)     | 4,921 / 5,359 | 91.8% |
| B (open-set pool)      | 1,290 / 1,359 | 94.9% |
| C (few-shot rare)      |   335 / 381   | 87.9% |
| D (benign)             | 1,674 / 1,845 | 90.7% |
| E1 (exploit-bearing)   |    99 / 109   | 90.8% |
| E2 (APT group-tagged)  |    70 / 177   | 39.5% |
| E3 (non-UPX packer)    |   175 / 191   | 91.6% |
| F (broad-coverage)     |   438 / 483   | 90.7% |
| Challenge              |    84 / 96    | 87.5% |
| **All**                | **9,086 / 10,000** | **90.86%** |

E2 group-tagged samples show lower observed-content coverage (39.55%). The available
records do not establish a cause. Challenge/T15 positives have 87.50% coverage and
T15 negatives have 91.33% coverage. Coverage is 98.52% for .NET, 94.67% for Win32,
and 77.53% for Win64. Associations between availability and subset and between
availability and file type have Cramer's V values of 0.2449 and 0.2603,
respectively.

**Are relationships between instances made explicit?**
No explicit relationships. Each instance is treated as independent.

**Are there recommended data splits?**
Yes. The recommended splits are train (7,000), validation (997), test (1,907),
and challenge (96). Researchers
should follow the split discipline described in the accompanying data descriptor
and `task_definitions.json` (evaluate only on the near-duplicate-disjoint
test/challenge splits).

**Are there errors, sources of noise, or redundancies?**
- AVClass family labels carry inherent multi-engine disagreement noise; the
  `family_confidence` field documents this per sample.
- The `first_submission_date` and `last_analysis_date` columns are stored as
  Unix epoch seconds (numeric), not ISO strings. Convert with
  `pd.to_datetime(col, unit='s')`.
- The `dynamic_status == 'behavior'` flag marks a report with a non-empty
  `behavior.summary` **or** a non-empty `apistats` histogram (the manifest rule);
  most have both, ~57 have a summary but empty `apistats`, and 66 have `apistats`
  but an empty summary. Check `len(apistats) > 0` when building API-histogram
  features to avoid silently collecting zero-vectors.
- The reference T5 baseline learns 19,152 TF-IDF features and 322 API-count
  features from training data only, for 19,474 columns total. The fitted
  vocabulary is released under `baselines/artifacts/`; this learned size is not
  an immutable dataset dimension.
- The annotation parquets retain occurrence multiplicity. The fine label
  columns are largely distinct (`capa.capability` 0% duplicate; `mbc.behavior`
  ≈12%; `mitre.technique` ≈37%), while the coarse grouping columns
  (`namespace`/`tactic`/`objective`) repeat heavily (≈19–68%). De-duplicate on
  whichever column you count or group by before presence/count/frequency
  analysis. The manifest `caps_list`/`ttps_list`/`mbc_list` columns are already
  distinct.
- Detection-ratio is a snapshot from `last_analysis_date`. The Q1-2026 re-check
  provides a second, dated aggregate detection count per sample over a median
  17-month gap (range 9-27 months). Under a three-state endpoint comparison,
  9,967/10,000 (99.67%) retain their original benign or malicious endpoint
  without entering grayware, 29/10,000 (0.29%) enter grayware, and 4/10,000
  (0.04%) cross directly between benign and malicious endpoints. Under the
  separate binary rule of at least five detections as malicious, 9,984/10,000
  (99.84%) remain on the same side and 16 change sides. These counts do not
  establish ground-truth label correctness or absence of drift.
- One duplicate SHA-256 in the EMBER2024 test parquet was deduplicated during
  EMBER-Dyn assembly.
- The 483 samples in subset `F` are a broad-coverage criterion-based draw from
  the EMBER2024 high-confidence pool; they carry full static, dynamic, annotation,
  and re-check coverage and are annotated by the `note` column.

**Is the dataset self-contained?**
Yes for the released static and dynamic representations and the
capability-grounded annotation layer. The original PE binaries are NOT
redistributed (neither is the
upstream EMBER2024, which publishes only features, metadata, and hashes);
researchers needing raw bytes re-acquire them by hash from public
malware-sharing sources using the SHA-256 identifiers in the manifest, e.g. via
the `code/scripts/fetch_binaries.py` helper (MalwareBazaar free API, with a
VirusTotal premium fallback). Coverage is partial and skewed against the benign
and long-tail cells.

**Does the dataset contain confidential data?**
No. All samples are public malware/benign hashes already published in EMBER2024.

**Could the dataset cause discomfort or risk?**
The CAPE Reduced reports contain references to file paths, registry keys, and
mutex names from real malware behaviour. These strings can include offensive
language fragments embedded in malware (e.g., crude C2 domain names). Researchers
should treat the dataset as containing potentially offensive textual artefacts.

---

## Collection Process

**How was the data acquired?**
- Static features and metadata: inherited from EMBER2024 clean-week parquet
  (Joyce et al. 2025).
- Raw EMBER JSON: inherited from EMBER2024 clean-week JSONL.
- CAPE Reduced JSON: derived from locally available CAPE reports. Project
  documentation describes isolated infrastructure; the release does not include
  operational network logs that independently verify isolation for every run.
- capa/MITRE/MBC annotations: inherited from EMBER2024 clean-week parquet.

**What mechanisms were used to collect the data?**
EMBER2024 itself collected samples via VirusTotal and partner antivirus
backends. EMBER-Dyn sampling is implemented in a deterministic Python script
(`scripts/sample_ember_dyn.py`). A structured spot check of 12 locally available
Full reports records CAPEv2 2.5. Project documentation describes Windows 10 22H2
Enterprise Evaluation guests under KVM/QEMU/libvirt, 12 virtual machines, a
300-second budget, and INetSim; these infrastructure details are not independently
recoverable from the reduced release records.

**Who was involved in the collection process?**
The four authors of this paper.

**Over what timeframe was the data collected?**
The underlying EMBER2024 source samples were collected between 2023-09-24 and
2024-12-14. EMBER-Dyn sample curation was finalized ahead of the Q1-2026
re-check. Timestamps in 12 inspected Full reports span 2026-05-23 to 2026-05-28;
the available report sample does not establish the complete dynamic-analysis date
range.

---

## Preprocessing / Cleaning / Labelling

**Was any preprocessing/cleaning/labelling done?**
- Construction records identify 136,611 SHA-256 hashes available on disk, but the
  corresponding candidate frame is not present in the release. This number documents
  a construction-stage record rather than a reproducible denominator for the final
  selection flow.
- Detection_ratio was already pre-parsed to float in clean-week parquet.
- One duplicate SHA-256 in the test parquet was deduplicated.
- CAPE reports were extracted in the Avast-CTU Reduced schema (`behavior.summary` +
  `static.pe`) using a post-processing hook.

**Is the raw data also saved?**
The full EMBER2024 source remains publicly available via Joyce et al.'s release.
Raw CAPE reports (including dropped binaries, memory dumps, pcaps) are not retained
in the release due to storage constraints (see Limitations).

**Is the preprocessing software publicly available?**
Yes. The MIT-licensed software and reproducibility documentation are maintained
at `https://github.com/tvquynh/ember-dyn`.

---

## Uses

**Has the dataset been used for any tasks already?**
Baseline binary detection, family classification, temporal separability, open-set, and capa
multi-label tasks (reference baselines in the code repository).

**What other tasks could the dataset be used for?**
The 15 pre-defined tasks cover binary detection, multi-class family, drift,
open-set detection, few-shot meta-learning, packer fingerprinting, exploit
detection, exploratory weak-label group-tag prediction, capa/MITRE/MBC multi-label,
per-architecture detection, capability count regression, and time-distinct
benchmarking. The challenge-malware versus test-benign task is a
source-split-confounded stress comparison, not a canonical out-of-distribution test.
Further uses include adversarial robustness studies, transfer learning to other
PE corpora, hierarchical multi-instance learning research, and threat-intelligence
classifier studies.

**Are there tasks for which the dataset should not be used?**
- Not suitable as a stand-alone product-grade malware detection corpus due to its
  compact 10,000-sample size.
- Not suitable for long-horizon (multi-year) drift study; the underlying corpus
  covers only 15 months.
- Not suitable as a substitute for live antivirus systems.

---

## Distribution

**Will the dataset be distributed?**
The release is distributed as coordinated archival components containing both manifests (26-column and 29-column with the
aggregate re-check fields), `manifest_recheck.csv`, splits, task definitions,
static parquets, raw EMBER JSONL, annotation parquets, the Small **and** Full
CAPE report variants, and the `failed_detonation_samples.json` artefact.
EMBER-Dyn-original artefacts are licensed CC-BY-4.0; EMBER2024-derived artefacts
(static parquets, raw JSONL, capa/MITRE/MBC annotations) retain the upstream
Apache-2.0 licence with attribution. See the deposit `LICENSE-CC-BY-4.0.txt` and
`LICENSE-Apache-2.0.txt` (NOTICE) for the per-artefact licence/provenance.

**How will it be distributed?**
The canonical archival release is distributed through Zenodo at
`https://doi.org/10.5281/zenodo.22178427`. Code and documentation are maintained
at `https://github.com/tvquynh/ember-dyn`.

**When will it be distributed?**
The v1.0.0 release accompanies the associated manuscript submission.

**Is the dataset subject to any IP-based restrictions?**
Both licences require attribution: CC-BY-4.0 to the EMBER-Dyn authors for the
original artefacts, and the upstream Apache-2.0 (with attribution to the
EMBER2024 corpus and retention of its NOTICE) for the EMBER2024-derived
artefacts. The re-check fields are released as aggregate detection counts only
(not per-engine VirusTotal results). Raw PE binaries are not distributed.

---

## Maintenance

**Who will support, host, and maintain the dataset?**
The PTIT research team at the contact `quynhtv@ptit.edu.vn` and
`thuaht@ptit.edu.vn` maintains the release through the Zenodo archival record at
DOI `10.5281/zenodo.22178427` and the EMBER-Dyn GitHub repository.

**Will the dataset be updated?**
The initial release is v1.0.0. We will issue patch releases for verified errors;
substantive updates (e.g., extending to longer time windows or alternative sandbox
configurations) will be versioned as v2.x.

**If the dataset relates to people, will their data be retained or removed?**
The dataset contains no personal information.

**Will older versions be supported?**
Version history is maintained through the Zenodo archival record.
