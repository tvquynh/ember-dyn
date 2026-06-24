# EMBER-Dyn — code

MIT-licensed build pipeline and analysis utilities for the EMBER-Dyn dataset.

## Layout

```
code/
├── README.md              this file
├── RUNBOOK.md             end-to-end rebuild guide
├── requirements.txt
└── scripts/
    ├── sample_ember_dyn.py            purposive curation entry point
    ├── copy_pe.py                     parallel PE-file copier
    ├── filter_parquet.py              per-split parquet filter
    ├── filter_jsonl.py                streaming JSONL router
    ├── rebuild_parquet_v2.py          schema extension to v2
    ├── verify_staging.py              build-staging structural verifier
    ├── verify_release.py              downloader-side released-bundle verifier
    ├── full_inventory.py              descriptive statistics + plots
    ├── dataset_statistics.py
    ├── compress_jsonl.py              JSONL → JSONL.zst
    ├── verify_pe_hashes.py
    ├── verify_consolidation.py
    ├── dedupe_jsonl_richer.py
    ├── cleanup_manifest.py
    ├── generate_splits_json.py
    ├── generate_task_definitions.py
    └── generate_notebooks.py
```

The five Colab quickstart notebooks (`01_quickstart_lightgbm.ipynb` …
`05_quickstart_textcape_lgbm.ipynb`) ship at the repository root
`notebooks/` directory. The LightGBM baseline training script that
produced the reported baseline tables is at
`../baselines/run_baselines_final.py`.

## Dependencies

```
numpy>=1.24
pandas>=2.0
pyarrow>=14.0
scikit-learn>=1.3
scipy>=1.10
lightgbm>=4.0
xgboost>=2.0
torch>=2.0
zstandard>=0.22
tqdm>=4.65
matplotlib>=3.7
```

For reproducing the dynamic Small / Full variants, an additional CAPEv2
+ KVM + libvirt + MongoDB + INetSim sandbox stack is required. We
follow the upstream CAPEv2 deployment recipe and refer operators to the
project documentation at <https://github.com/kevoreilly/CAPEv2>.

## Quick run

```bash
pip install -r requirements.txt
# sample_ember_dyn.py reads the EMBER2024 clean-week parquet splits from
# ./data/ember2024/parquet/ (ember2024_{train,test,challenge}.parquet) and
# writes the purposive-selection manifest. It documents the initial selection
# criteria only; the released dataset/manifest_open.csv is the canonical record of
# the final 10,000-sample list (including subset F and the TLSH re-split).
python scripts/sample_ember_dyn.py --output manifest.csv --seed 42
```

See `RUNBOOK.md` for the complete step-by-step rebuild pipeline.
