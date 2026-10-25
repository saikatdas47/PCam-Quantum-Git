# Training and recovery

The source snapshot contains the scientific implementation of the completed study. `study.py` exposes thirteen named stages without embedding source inside a notebook cell.

Use a compatible original environment recorded under `07_evidence/environments`. Training requires matching torch and torchvision. The retained implementation uses native PyTorch and downloads pinned model weights with a checksum. PennyLane, JAX and timm are not required.

Place official files in `02_data/raw`, then run:

```bash
python 04_code/training/study.py --stage plan
python 04_code/training/study.py --stage all
```

A named stage can be run separately:

```bash
python 04_code/training/study.py --stage 04_train_features
```

The separate workspace is `08_workspace/training`. Original internal relative paths preserve imports and scientific identities. Raw data are linked from `02_data/raw`. Existing workspace source files are compared before reuse. A mismatch stops execution and preserves work.

State and logs are in the workspace workflow directory. Completed stages are skipped. Interrupted chunks and epochs resume using the scientific checkpoint code. Retain the workspace after interruption. Do not run two processes against it. A test lock prevents further training changes in that scientific run.

Feature extraction selects CUDA when present, otherwise CPU. Head fitting retains matched CPU float64 with two threads. Moving heads to GPU would change the recorded comparison.

Historical selection definitions remain in the snapshot. The final helper applies ResNet50, eight components and three final seeds. Raw images and complete features must be downloaded or computed for new training. Source inspection and plan execution are checked. A new full dataset run through this reorganized entry point has not been executed. Runtime compatibility must be resolved without weakening identity guards.
