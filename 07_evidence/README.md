# Evidence and environments

The seven archive parts form one compressed tar stream. `archive/INDEX.json` records order, sizes and SHA256 digests. All parts are required. Run `python run.py unpack`. Existing extracted files are verified before reuse. A mismatch stops extraction and preserves the differing file.

The archive retains all 1,208 supplied evidence files. The original manifest lists 1,207 entries. Contents include source/configuration snapshots, split and slide IDs, fitted transforms/models, selection records, 26 prediction archives, bootstrap arrays, logs and test lock. Raw images and complete frozen feature arrays were absent from the original export.

Extraction writes to `08_workspace/resnet50_evidence`. Original internal paths remain intact for checksum and lock verification. Historic path strings do not override the numbered workflow or final scope.

`source_snapshot_manifest.json` checks visible training sources. `environments` distinguishes the original training and later analysis environments, with exact version snapshots. Arbitrary changed environments are not promised identical floating point results.
