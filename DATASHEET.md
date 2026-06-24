# Datasheet for EMBER-Dyn (Gebru-style)

Following the template proposed by Gebru et al., "Datasheets for Datasets" (CACM 2021).

---

## Motivation

**For what purpose was the dataset created?**
EMBER-Dyn was created to provide the malware-detection research community with the
first compact public corpus, to our knowledge, combining static portable-executable features, raw EMBER
JSON records, and CAPEv2 sandbox dynamic behavioural reports for the same 10,000 PE
samples, joined with capa/MITRE ATT&CK/MBC annotations and structured around 15
pre-defined benchmark tasks.

**Who created the dataset?**
The four authors of this paper at the Posts and Telecommunications Institute of
Technology (PTIT), the Saigon International University (SIU), and the Ho Chi Minh
City University of Economics and Finance (UEF), Vietnam.

**Who funded the dataset creation?**
PTIT institutional resources. No external funding.

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
Sample. The EMBER2024 source corpus has 3,238,315 instances; EMBER-Dyn samples 0.3%
via deterministic, criterion-based purposive selection (no random component) to maximise multi-task utility at
minimum download cost.

**What data does each instance consist of?**
Static feature vector (2,568 numerical features), raw EMBER JSON (hierarchical
record with histogram, byteentropy, strings, headers, sections, imports, exports,
data directories, rich header, authenticode, capa matches, ATT&CK matches, MBC
matches), CAPE Reduced JSON (`behavior.summary` 13 string lists, a per-sample
`apistats` API-call histogram, and `static.pe` 23 structural keys), and 29 manifest
columns (including `recheck_label`, `recheck_detections`, `recheck_date` from the
Q1-2026 re-check; the per-sample re-check fields and the failed-detonation list are
provided under controlled access, not in the open CC-BY layer).

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
- `dynamic_status` behavior / no_behavior (captured-execution flag)

**Is any information missing?**
Several capa-derived fields are sparse by design (1.1% exploit, 2.2% APT group). The
`family` field is empty for benign samples (label=0) and for some low-confidence
malicious samples. Behavioural reports with captured execution are available for
9,086 of the 10,000 samples (90.9%); the remaining 914 produced a valid but
behaviourally empty report under the single sandbox configuration and are flagged by
the `dynamic_status` column (static features and all annotations are present for every
sample). These missing values are documented in the schema.

Behavioural-capture coverage by subset (share of each subset with
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
| **All**                | **9,086 / 10,000** | **90.9%** |

E2 (APT-attributed) shows markedly lower capture (39.5%): these older, often
targeted samples are the most prone to anti-analysis bail-out or unmet
environmental preconditions under the single Windows 10 22H2 / INetSim / 5-minute
configuration.

**Are relationships between instances made explicit?**
No explicit relationships. Each instance is treated as independent.

**Are there recommended data splits?**
Yes — train (7,000), val (997), test (1,907), challenge (96). Researchers
should follow the discipline described in the Usage Notes section of the paper.

**Are there errors, sources of noise, or redundancies?**
- AVClass family labels carry inherent multi-engine disagreement noise; the
  `family_confidence` field documents this per sample.
- The `first_submission_date` and `last_analysis_date` columns are stored as
  Unix epoch seconds (numeric), not ISO strings — convert with
  `pd.to_datetime(col, unit='s')`.
- Detection-ratio is a snapshot from `last_analysis_date`. The Q1-2026 re-check
  measures how much labels drift: 99.7% of labels are stable over a median 17-month gap (range 9-27 months).
  This 99.7% is the pooled rate, (8,112+1,855)/10,000 = 99.67% (8,112/8,126 = 99.8% malicious remain >=5
  detections; 1,855/1,874 = 99.0% benign remain 0); the pooled figure is higher than the benign per-class
  rate because malicious samples dominate the corpus. The inherited labels are thus temporally robust for this subset.
- One duplicate SHA-256 in the EMBER2024 test parquet was deduplicated during
  EMBER-Dyn assembly.
- The 483 samples in subset `F` are a broad-coverage criterion-based draw from
  the EMBER2024 high-confidence pool; they carry full static, dynamic, annotation,
  and re-check coverage and are annotated by the `note` column.

**Is the dataset self-contained?**
Yes for the three primary modalities (parquet, JSONL, CAPE Reduced) and all
annotations. The original PE binaries are NOT redistributed; researchers needing
raw bytes must obtain them through the EMBER2024 release channel.

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
- CAPE Reduced JSON: produced by detonating each sample in a CAPEv2 sandbox on
  isolated infrastructure (Methods section).
- capa/MITRE/MBC annotations: inherited from EMBER2024 clean-week parquet.

**What mechanisms were used to collect the data?**
EMBER2024 itself collected samples via VirusTotal and partner antivirus
backends. EMBER-Dyn sampling is a deterministic Python script
(`scripts/sample_ember_dyn.py`). Dynamic detonation uses CAPEv2 v2.5 against
Windows 10 22H2 Enterprise Evaluation guest VMs.

**Who was involved in the collection process?**
The four authors of this paper.

**Over what timeframe was the data collected?**
The underlying EMBER2024 source samples were collected between 2023-09-24 and
2024-12-14. EMBER-Dyn sample curation was finalised ahead of the Q1-2026
re-check; dynamic detonation was performed in 2025–2026.

**Were any ethical review processes conducted?**
The dataset contains only published malware/benign hashes. No human-subject
research is involved. PTIT does not require ethical review for this type of
dataset preparation.

---

## Preprocessing / Cleaning / Labelling

**Was any preprocessing/cleaning/labelling done?**
- EMBER2024 metadata was joined to the 136,611 SHA-256 hashes available on disk.
- Detection_ratio was already pre-parsed to float in clean-week parquet.
- One duplicate SHA-256 in the test parquet was deduplicated.
- CAPE reports were extracted in the Avast-CTU Reduced schema (`behavior.summary` +
  `static.pe`) using a post-processing hook.

**Is the raw data also saved?**
The full EMBER2024 source remains publicly available via Joyce et al.'s release.
Raw CAPE reports (including dropped binaries, memory dumps, pcaps) are not retained
in the release due to storage constraints (see Limitations).

**Is the preprocessing software publicly available?**
Yes, MIT-licensed at `github.com/tvquynh/ember-dyn`.

---

## Uses

**Has the dataset been used for any tasks already?**
Baseline binary detection, family classification, drift, open-set, and capa
multi-label tasks (Technical Validation section).

**What other tasks could the dataset be used for?**
The 15 pre-defined tasks cover binary detection, multi-class family, drift,
open-set detection, few-shot meta-learning, packer fingerprinting, exploit
detection, APT attribution, capa/MITRE/MBC multi-label, per-architecture
detection, capability count regression, and time-distinct benchmarking.
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
Yes, in two tiers. The **open tier** (CC-BY-4.0) holds the 26-column open
manifest, splits, task definitions, static parquets, raw EMBER JSONL,
annotation parquets, and the Small dynamic variant. The **controlled-access
tier** (data-use agreement, not redistributable) holds the three per-sample
re-check columns, the `failed_detonation_samples.json` artefact, and the Full
CAPE report variant. See the per-artefact licence/provenance matrix in the
manuscript Data Records section.

**How will it be distributed?**
- Zenodo: archival copy of record (persistent DOI; open tier + file-level checksums). The open tier is plain Parquet/JSONL and loads directly with pandas/pyarrow after download.
- Controlled tier: access request + affiliation-verified data-use agreement
- GitHub: code mirror + small reference files

**When will it be distributed?**
Open tier deposited on Zenodo at submission (DOI reserved); public on acceptance.

**Is the dataset subject to any IP-based restrictions?**
Open-tier CC-BY-4.0 requires attribution to the EMBER-Dyn authors and the
upstream EMBER2024 corpus. Controlled-tier artefacts (re-check, Full reports,
failed-detonation list) are not redistributable. Raw PE binaries are not
distributed under either tier.

---

## Maintenance

**Who will support, host, and maintain the dataset?**
The PTIT research team at the contact `quynhtv@ptithcm.edu.vn` and
`thuaht@ptit.edu.vn`. Zenodo provides persistent archival hosting; GitHub
provides version control.

**Will the dataset be updated?**
The initial release is v1.0.0. We will issue patch releases for verified errors;
substantive updates (e.g., extending to longer time windows or alternative sandbox
configurations) will be versioned as v2.x.

**If the dataset relates to people, will their data be retained or removed?**
The dataset contains no personal information.

**Will older versions be supported?**
Older versions remain accessible through the Zenodo DOI versioning system.
