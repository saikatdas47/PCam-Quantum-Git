# Result figures

- `01_roc`: Test ROC curves, three neural seeds and the shared classical reference.
- `02_precision_recall`: Corresponding precision recall curves and prevalence.
- `03_paired_comparisons`: Primary and secondary mean AUROC differences and paired slide intervals.
- `04_training_and_time`: Training diagnostics and measured neural training compute time.

PNG, PDF and 400 DPI RGB TIFF are provided. The methodology diagram is in `03_methodology`. Stored exports retain the reviewed layout. `python run.py figures` rebuilds plots from measured predictions, bootstrap analysis and training records. Fonts can affect appearance. Curves are empirical and unsmoothed.
