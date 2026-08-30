# Recorded results environment

The stored targeted results for T5, T9, T15, and the dynamic-missingness
analyses were produced with Python 3.11.9 (64-bit) on Windows 11 Pro build
26100. The exact recorded package versions are in
`requirements-results-environment.txt`.

The production runner uses train-only dynamic vocabulary fitting, validation
early stopping, validation-only selection among four declared fusion rules,
and a frozen `0.7 * static + 0.3 * dynamic` rule for test and T15. It represents
`no_behavior` rows as all-zero dynamic vectors. The learned representation has
19,152 TF-IDF features and 322 API-count features, for 19,474 columns total;
the fitted vocabulary is stored in
`../baselines/artifacts/dynamic_vocabularies.json`.

This lock records the environment of the stored result artifacts. A fresh
environment reproduction was not performed while assembling the submission
candidate. CAPEv2 redetonation is environment-dependent and is not expected to
be bitwise reproducible from the released reduced reports.
