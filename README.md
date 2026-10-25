# Simulated quantum and classical heads on frozen ResNet50 features

Reproducibility materials for **A comparison of simulated quantum and classical classification heads on frozen ResNet50 features for PatchCamelyon**.

**Author:** Saikat Das  
Department of Computer Science and Engineering, Ahsanullah University of Science and Technology, Dhaka, Bangladesh  
**Correspondence:** saikatdasmain47@gmail.com  
**Repository:** https://github.com/saikatdas47/PCam-Quantum-Git

This is a patch level classification study on the official PatchCamelyon splits. It is not a patient diagnosis system. A frozen ImageNet ResNet50 produces 2,048 features, reduced to eight training fitted PCA components. Standard and arctangent MLP heads, classically simulated RY circuits with CNOT off and on, and a training selected classical reference are compared. Neural fits use seeds 101, 202 and 303 under a shared schedule. The final executed scope is one backbone and eight components. Historical source protocols are provenance and do not replace the final SCOPE.json.

The CNOT on head has mean test AUROC 0.8657 and the selected reference 0.8776. The primary difference is -0.011911, with conditional paired slide bootstrap 95% interval [-0.023594, -0.000379]. The separate post hoc jackknife p value is 0.052614. These are different uncertainty procedures. No quantum advantage, hardware validation, speedup or equivalence claim is supported.

## Dataset

Obtain the original images, labels and slide metadata from the [official PCam download page](https://github.com/basveeling/pcam#download). A [Zenodo mirror](https://zenodo.org/records/2546921) is listed by the maintainers. Use the original training, validation and test splits, with 262,144, 32,768 and 32,768 rows. The official README provides original MD5 checksums. PCam maintainers state that the dataset is CC0. Do not substitute the Kaggle training subset for the official splits.

Raw images are not copied here. Optional raw image checks expect original HDF5 image and label files under data/raw using official filenames. Images may be extracted or gzipped. Label checks expect the original y.h5.gz files. The original train image gzip was absent in the supplied environment, limiting its provenance claim. A later pixel hash audit found 33 train/test identical image hashes affecting 34 test rows. The official primary analysis is retained and exclusion is a post hoc descriptive sensitivity check.

## Contents

| Location | Contents |
| --- | --- |
| evidence_archive | Seven ordered archive parts and SHA256 index. They contain all 1,208 supplied evidence files, including the original 1,207 entry manifest |
| resnet50_evidence after extraction | Original sources/configurations, IDs, transforms, CV records, model states, 26 prediction archives, bootstrap arrays, timing and test lock |
| paper/major_revision/analysis | Circuit validation, evidence verification, reporting summaries and optional raw image audits |
| paper/major_revision/results and tables | Executed checks, image hashes, selection/PCA records and tables |
| paper/plos_one_initial/analysis | Saved prediction analysis and post hoc slide jackknife |
| paper/plos_one_word | Current Word/Markdown manuscript, author details and figure/document builders |
| paper/PLOS_ONE_submission/figures | Editable draw.io methodology and five figure exports, including 400 DPI TIFF |

Raw images and complete frozen feature arrays are absent from the original export. Download or regenerate them for new training. Reconstructed initialization tensors are not archived original initial tensors. Fitted models, final transforms and measured predictions are retained.

## Reproduce without training

```bash
python -m pip install -r requirements-analysis.txt
python reproduce.py
```

The runner verifies archive checksums, safely extracts evidence, verifies all 1,207 original manifest entries, recomputes metrics for 26 prediction archives, validates circuit outputs/gradients and checks final scope, saved intervals, p values and 676 test lock hashes. It does not train or tune models. CPU is sufficient. PennyLane and JAX are not required.

For the additional saved prediction jackknife and reporting tables:

```bash
python reproduce.py --full-analysis
```

With the original dataset present, optional image checks are:

```bash
python paper/major_revision/analysis/audit_images.py
python paper/major_revision/analysis/overlap_sensitivity.py
```

These scripts update derived reports only. Original evidence and test thresholds remain unchanged. Repeating unpack_evidence.py verifies existing files and stops if any differ.

## Environments and training

Original and reporting environments and exact requirement snapshots are recorded separately under paper/major_revision/release. requirements-analysis.txt provides practical dependency ranges, not a promise of identical training on every version. Pickled scikit-learn models are version sensitive. Load trusted model states in the recorded environment when reproducing fitted model inference.

Original neural heads used CPU float64 and two PyTorch threads. Frozen feature extraction used recorded CPU/CUDA sessions. Unpacked experiments and shared sources preserve the executed implementation and guards. They are not a one command promise to retrain everything on an arbitrary runtime. The default runner validates saved evidence without repeating training. No independent investigator replication is claimed.

## Manuscript and release

The current manuscript has five figures, eleven tables and twenty five references. The availability paragraph gives the intended repository URL and accurately notes pending upload. Replace that pending statement only after the complete release is publicly accessible. No publication DOI has been assigned.

Funding: no specific funding. Competing interests: none declared. Saikat Das is the sole author. Methods briefly discloses AI assistance with code and language work. The author retains responsibility.

Version 0.3.0 is prepared locally. Preparation does not constitute a GitHub upload or DOI deposit. All archive parts are required for extraction. Extracted evidence is ignored by Git because its exact bytes are stored in the verified archive. Raw images, credentials, dependency directories and caches must not be uploaded. Third party terms are described in THIRD_PARTY_NOTICES.md. A new project wide reuse license has not been assigned by this packaging step. Code and derived evidence licensing should be selected by the author before journal deposition.
