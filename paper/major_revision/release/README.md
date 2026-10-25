Historical local release specification 0.2.0. Superseded by repository README.md and RELEASE_STATUS.json version0.3.0. Original environment records remain authoritative.

# Local release specification version0.2.0

Status: prepared locally, not publicly deposited. This directory does not claim a public DOI or independently reproduced original training environment.

The release_manifest.json file lists the complete original ResNet50 evidence and the new reporting scripts, tables and outputs with SHA256 checksums. Paths are relative to the project root. Files are referenced in their existing locations to avoid a second large duplicate copy. A public deposit must include those files with the same relative layout and state the source licenses. Original raw images and complete frozen encoder feature arrays are excluded. Raw PCam can be obtained from its maintainers.

## Recompute the added checks

Use Python with requirements-revision.txt and run from the project root:

    python paper/major_revision/release/reproduce_checks.py

To additionally check saved image hash arrays, raw label order and the descriptive overlap exclusion:

    python paper/major_revision/release/reproduce_checks.py --raw-images

For a fresh pixel hash pass, remove only paper/major_revision/results/train_image_hashes.npz, valid_image_hashes.npz and test_image_hashes.npz before that command. Original raw images are never deleted.

Original run versions are recorded separately in environment_original.json and requirements-original.txt. The saved per-session records retain exact runtime changes. The original Python3.13 scientific environment has not been reinstalled or freshly tested by this revision. The local reporting environment has been executed and recorded. Circuit checks use only the retained torch_circuit.py and independent dense matrices. Metric verification uses saved prediction arrays. Full bootstrap arrays remain supplied. The first32 weighted bootstrap draws per model were previously checked directly, while the full20,000 draws were reused for interval reconstruction. No independent investigator replication is claimed.

## Included evidence

All original model states, transforms, configurations, split IDs and slide metadata, 78 CV scores and corresponding predictions, 26 validation/test predictions, bootstrap arrays and original export manifest are listed. New tables give the exact learning rate grid, PCA variance, slide counts, influence and train/test pixel overlap. New scripts validate circuits, gradients and the gate free cosine expression. Reconstructed initial tensors are clearly labeled with the local runtime. Original initial tensors were not separately archived.

## Still required before submission

Deposit the manifest-listed files in a public versioned repository, confirm licenses, obtain its DOI, and replace the manuscript availability placeholder. Confirm authors, affiliations, funding, conflicts, contributions, and ethics determination. Recover supplemental workflow logs from the original Drive project if available. A full-feature baseline is not included and remains an optional exploratory extension. Do not present older pilot full-feature results as the final official-test baseline.
