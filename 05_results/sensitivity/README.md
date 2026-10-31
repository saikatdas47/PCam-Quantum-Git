# Post hoc sensitivity results

Compression logistic test AUROC at 8/32/128/2048 dimensions is 0.869531/0.902468/0.909897/0.910823. The 2048-minus-8 difference is 0.041292, with conditional 95% WSI interval [0.022349, 0.059071]. Adjusted 32-minus-8 and 128-minus-8 intervals also exclude zero. No 128-versus-2048 equivalence claim is made.

Head-specific learning rates were selected using the original CPU training CV: 0.001 for MLPs and 0.01 for circuits. CNOT-on selected-minus-shared mean change is 0.003488, with four-comparison-adjusted conditional interval [0.000115, 0.007056]. Seed changes are -0.000414, 0.003265 and 0.007613. Other rate-change adjusted intervals and all selected-rate head-comparison adjusted intervals include zero. The original CPU fits and new shared-rate fits remain distinct.

Full-precision results, bootstrap arrays, per-seed metrics, timing and numerical/model audits are in this folder. All conclusions condition on fixed fitted models and retained seeds. The analyses are post hoc after original test inspection. Immutable raw evidence is a release asset with checksums, rather than a Git working copy. No new training was performed during reanalysis.
