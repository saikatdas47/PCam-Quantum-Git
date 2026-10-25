# Results and statistical records

`validation` and `test` retain original metrics and model summaries. `tables` contains per model metrics, selection scores, PCA variance, slide counts, confusion counts, traces and comparisons. `analysis/results.json` retains detailed contrasts. `checks` holds circuit validation, pixel hashes, audits and consistency checks.

Neural summaries are arithmetic means of three seed specific metrics, not a pooled score ensemble. The selected classical reference is one fitted model. Seed SD and slide bootstrap uncertainty describe different variation.

Primary AUROC difference: -0.0119111373. Conditional percentile bootstrap 95% interval: [-0.0235935701, -0.0003790422]. Separate post hoc slide jackknife p value: 0.0526137. Four secondary bootstrap intervals use family adjustment, while secondary post hoc p values use Holm adjustment.

The later pixel audit found 33 shared hashes, affecting 35 training and 34 test rows. Excluding those test rows gives a descriptive difference of -0.01187925. Original primary results are retained. This sensitivity does not establish independence.

Predictions, models, bootstrap arrays and job logs are in the evidence archive. Extract with `python run.py unpack`. Recomputed checks do not overwrite that original evidence or tune test thresholds.
