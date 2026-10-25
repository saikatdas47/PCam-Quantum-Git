# Manuscript consistency audit

Date: 24 October 2026

## Decision and scope

No remaining numerical contradiction was identified between the checked current manuscript and retained study outputs after the reporting and figure reference corrections below. This is a bounded consistency check, not a guarantee that every possible error is absent or that the paper is ready for submission.

The current title is **A comparison of simulated quantum and classical classification heads on frozen ResNet50 features for PatchCamelyon**. The title paragraph, document metadata and canonical Markdown agree. The word controlled was removed from the title. Controlled NOT remains where it names the scientific gate.

## Implemented corrections

- Added the existing methodology as Figure 1. Updated all five figure captions, prose references and image sources. Result figures are Figures 2 to 5.
- Removed the stale statement that the methodology figure was disabled. Matched the source Markdown Table 2 columns to the Word table.
- Separated probability metrics from threshold dependent metrics in Methods and the S1 Table caption and note. Brier score and log loss do not use a classification threshold.
- Specified zero indexed epochs for batch order seeds. Made the recorded CPU float64, two thread head training setting explicit.
- Clarified that fourteen classical selection candidates are nonneural, alongside two MLP candidates. Added the saved classical reference mean cross validation AUROC of 0.884710.
- Retained author and acknowledgment placeholders without inventing identities or declarations.

No fitted model, threshold, sample membership or original scientific table value was changed.

## Executed and verified

| Check | Finding |
| --- | --- |
| Original export manifest | All 1,207 recorded file sizes and SHA256 digests verified |
| Validation and test predictions | 26 archives agree with original saved metric records |
| Official test metrics | All 13 model entries checked for AUROC, average precision, Brier score, log loss, accuracy, sensitivity, specificity, precision, F1 and confusion counts |
| Fixed model scope | Twelve neural fits and one selected classical reference, eight components, three neural seeds |
| Training records | Neural parameter counts, 100 epochs, 204,800 updates, learning rate and recorded runtime agree |
| Uncertainty | Five nominal and four family adjusted intervals reconstructed from saved bootstrap arrays |
| Post hoc tests | Jackknife p values reconstructed from saved standard errors and statistics. Four secondary Holm adjustments agree |
| Test lock | 676 recorded code and fitted artifact hashes agree |
| Manuscript tables | All eleven Word table payloads exactly match the pre-edit document |
| Cross references | Five ordered figure captions and all five figures cited in prose. All 25 references cited |
| Abstract | 247 words |
| Final Word layout | All 26 pages visually reviewed. After the final S1 note correction only page 22 changed and was reviewed again |

The machine readable checks are document_checks.json and scientific_checks.json. The full export verification record is ../../major_revision/results/evidence_reverification.json. Existing circuit validation and raw image audit records were read and checked against the reported scope. This pass did not rerun the raw pixel scan, full bootstrap sampling or all leave one slide calculations. No new training was performed. No fresh external full text or external code audit was performed in this pass.

## Differences that require explanation, not numerical correction

The primary AUROC difference is -0.0119111373. Its saved conditional bootstrap 95% interval is [-0.0235935701, -0.0003790422]. The separate post hoc slide jackknife p value is 0.0526137. These are different inference procedures and are not an interval and p value from one identical test. The manuscript retains this method dependence and avoids a method independent significance claim.

The final shared learning rate is 0.003. Head specific training fold optima differ. The question concerns a matched shared schedule and does not establish that every head is individually optimized.

Official slide identifiers are split disjoint, while the later raw pixel audit found 33 identical train/test image hashes affecting 35 training and 34 test rows. Both statements can hold. The original official test analysis remains primary and the overlap exclusion is explicitly post hoc sensitivity analysis.

Historical protocols include more dimensions, seeds and backbones. The retained final scope records the resource amendment to one backbone, eight components and three seeds before official test scoring. Historical files are provenance, not the executed final schedule.

The original code uses calibration terminology for threshold selection. The manuscript describes threshold selection and does not claim fitted probability calibration. Training fold selection uses 16,384 training rows and final head fitting uses the full 262,144 training rows.

## Known limits and remaining work

Eight principal components explain 58.8005% of training feature variance. This does not establish that eight components are sufficient for every classifier. The study uses one dataset, one frozen backbone and three neural seeds. Parameter counts and optimization budgets differ between model families. A full 2,048 feature classical baseline has not been executed. No quantum hardware or speedup was demonstrated. Patient independence and absence of near duplicates remain unproven. Bootstrap uncertainty is conditional on the retained fitted models and analysis design.

Author details, declarations and a public persistent repository remain pending. Existing literature access limits remain in the citation audit. This audit does not certify absence of prior art or acceptance by PLOS ONE. No submission package was recreated and no online submission was made.

## Current outputs

- ../PCam_ResNet50_Quantum_Manuscript.docx
- ../MANUSCRIPT.md
- ../../PLOS_ONE_submission/figures/Fig1_methodology.drawio
- ../../PLOS_ONE_submission/figures/Fig1.tif through Fig5.tif

Rebuild the Word manuscript with ../build_manuscript_docx.py. revise.py records the one time revision and must not be used as a repeatable rebuild command.

## Superseding author and repository update 25 October 2026

Author details and declarations are now supplied. Acknowledgments removed. Dataset and GitHub links added. The prepared GitHub checkout runner passed saved evidence checks. Public upload and reuse licensing remain pending. See ../../repository_release/REPORT.md for the current packaging record. The original historical audit above remains preserved.
