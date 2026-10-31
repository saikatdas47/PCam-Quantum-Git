# Compression and paired learning-rate analyses

These analyses were planned after the original official-test results were inspected. They remain post hoc. Existing training-only CV selects rates and regularization; it does not make the sensitivity analyses confirmatory.

Download `pcam_sensitivity_evidence.zip` from the `analysis-2026-10-31` GitHub release. From the repository root:

```
python run_sensitivity.py verify --evidence-zip /path/to/pcam_sensitivity_evidence.zip
python run_sensitivity.py analyze --evidence-zip /path/to/pcam_sensitivity_evidence.zip
python run_sensitivity.py models --evidence-zip /path/to/pcam_sensitivity_evidence.zip
python run_sensitivity.py figures --evidence-zip /path/to/pcam_sensitivity_evidence.zip
```

The runner verifies the release SHA256, archive CRC, 1,394 file manifest entries and 182 fitted-artifact lock entries. It uses an ignored `08_workspace/sensitivity_project` reconstruction. All scientific paths stay inside that workspace. Reanalysis writes there and does not train or replace the archived evidence.

The analysis uses 20,000 paired WSI bootstrap draws, seed 71913. The same slide-count vector applies to every model and fixed optimization seed. A tie-aware slide-pair matrix equals pooled patch-weighted AUROC. Eight draws per model were checked directly using weighted scikit-learn AUROC. Compression secondary family size is two, and each learning-rate/selected-head family size is four. Corrections operate within those declared families, not across all paper analyses. Seed SD is reported separately.

The model audit replays 24 neural models on saved validation/test matrices and three compressed logistic test models. The full 2048-dimensional test matrix is absent, so its score replay cannot run from this asset. The complete 2048-feature cache is also needed for new training. All four final logistic fits stopped on parameter change with gradient above the gradient threshold. Three unselected full-feature C=10 CV fits reached the iteration limit.

The saved engine and source versions preserve original provenance and internal filenames. Five saved versions have identical circuit/MLP definitions; operational staging and resume code changed. A recorded numerical input transition checked a 1,024-row cohort, not equality of the unavailable whole previous matrix. No full cross-device bitwise equality, independent researcher replication, quantum speedup, hardware validation or clinical readiness is claimed.

The existing `run.py` remains the entry point for the original locked study. `run_sensitivity.py` handles the post hoc stages separately. Manuscript files are maintained outside this repository.
