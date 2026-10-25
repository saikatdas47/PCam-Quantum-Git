# PCam ResNet50 head comparison

I compare small simulated quantum and classical classification heads using the same frozen ResNet50 representation of PatchCamelyon images. The question is whether the tested circuit improves patch classification under an eight component constraint and a shared neural training schedule.

**Saikat Das**  
Department of Computer Science and Engineering  
Ahsanullah University of Science and Technology, Dhaka, Bangladesh  
Contact: saikatdasmain47@gmail.com

## Read the folders in order

| Folder | Contents |
| --- | --- |
| [01_plan](01_plan) | Research question, final scope, comparisons and execution order |
| [02_data](02_data) | Official dataset links, checksums and input instructions |
| [03_methodology](03_methodology) | Method description and editable workflow diagram |
| [04_code](04_code) | Training, analysis, verification and plotting code |
| [05_results](05_results) | Selection tables, validation/test metrics and statistical records |
| [06_figures](06_figures) | ROC, precision recall, paired contrasts and training/timing figures |
| [07_evidence](07_evidence) | Original evidence archive, source checksums and environments |

`08_workspace` holds extracted evidence, training outputs, logs and caches when code runs. It is excluded from Git. Downloaded images belong in `02_data/raw`, also excluded.

## Check the saved study

Run from the repository root:

```bash
python -m pip install -r requirements-analysis.txt
python run.py verify
```

This verifies seven archive parts, extracts evidence, checks the original 1,207 manifest records, recomputes metrics for 26 prediction archives, validates the circuit and checks 676 test lock hashes. It does not fit models.

## Repeat analysis and figures

```bash
python run.py analyze
python run.py figures
```

Analysis reads saved predictions, transforms and training records. It reconstructs intervals from retained bootstrap arrays and recomputes slide jackknife, PCA/selection summaries and descriptive overlap sensitivity. It does not repeat the entire bootstrap experiment. Results go to `05_results` and figures to `06_figures`. Original evidence remains unchanged. Figures are PNG, PDF and 400 DPI RGB TIFF. Font availability may change rebuilt layouts.

## Run training separately

Read [04_code/training/README.md](04_code/training/README.md) first.

```bash
python 04_code/training/study.py --stage plan
python 04_code/training/study.py --stage all
```

Official data are required in `02_data/raw`. Training uses its own workspace, preserving the completed evidence. Individual stages can be run by name. Interrupted feature chunks and epochs resume through the retained checkpoint code. Completed stages are skipped. A changed source snapshot stops the run.

Original head fitting used CPU float64 and two threads. Frozen feature extraction selects CUDA when available and falls back to CPU. Exact original versions are in `07_evidence/environments`. The stage entry point and plan have been checked. A new full dataset training run through this reorganized entry point has not been executed.

## Main findings

CNOT on mean test AUROC is 0.8657 and selected classical AUROC is 0.8776. The difference is -0.011911, with conditional paired slide bootstrap 95% interval [-0.023594, -0.000379]. The separate post hoc jackknife gives p=0.052614. These are different procedures, so exclusion of zero is method dependent. Adjusted secondary intervals include zero.

The scope is one dataset, one frozen backbone, eight PCA components and three neural seeds. PCA retains 58.8005% of training variance. A later audit found 33 identical train/test image hashes affecting 34 test rows. The original primary analysis is retained. Overlap exclusion is descriptive. Patient independence, quantum speedup, hardware performance and clinical readiness are not established.

## Dataset and provenance

Dataset: https://github.com/basveeling/pcam#download  
Mirror listed by maintainers: https://zenodo.org/records/2546921

Keep the official splits. Raw images and complete frozen feature arrays are not in the original evidence export. The final reduced scope is recorded in `01_plan/final_scope.json`. Older protocols are provenance, not the final executed schedule. See [07_evidence/PROVENANCE.md](07_evidence/PROVENANCE.md) for source terms and limitations.
