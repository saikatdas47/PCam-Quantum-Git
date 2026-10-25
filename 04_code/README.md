# Code map

| File | Role |
| --- | --- |
| training/study.py | Ordered training stages and recovery |
| training/source_snapshot | Original scientific scripts, configs and helpers |
| analysis/verify_evidence.py | Export hashes and prediction metrics |
| analysis/validate_circuit.py | Dense matrix, finite difference and analytical checks |
| analysis/analyze_predictions.py | Contrasts, saved intervals and slide jackknife |
| analysis/summarize_saved_evidence.py | PCA, selection, traces and slide influence |
| analysis/audit_images.py | Raw labels and exact pixel hashes |
| analysis/overlap_sensitivity.py | Descriptive overlap exclusion |
| verification/verify_study.py | Scope, metrics, intervals, tests and test lock |
| verification/unpack_evidence.py | Checked archive extraction |
| visualization/plot_results.py | Curves, paired contrasts and diagnostics |

Run through `run.py` from the repository root. Raw audits require `02_data/raw`. Analysis reads original evidence and writes derived outputs under `05_results`. Source checksums are in `07_evidence/source_snapshot_manifest.json`.
