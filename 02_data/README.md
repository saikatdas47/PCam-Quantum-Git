# Dataset inputs

Official download page: https://github.com/basveeling/pcam#download

Maintainers' mirror: https://zenodo.org/records/2546921

Use original image, label and slide metadata files. Place them in `02_data/raw`. Keep official train, validation and test splits. Do not substitute the Kaggle subset. Image files may be gzipped or extracted HDF5. Keep original `y.h5.gz` labels for the retained raw label audit.

The training split contains 262,144 patches. Validation and test each contain 32,768. Images are 96 by 96 RGB. A positive label indicates tumor in the central 32 by 32 region. Slide metadata are required for grouped selection and uncertainty.

The maintainers state that data are CC0. Cite PCam and the original CAMELYON16 source. Raw images are not included here. The retained run lacked the original compressed train image file, limiting its compressed file provenance. Saved pixel hashes do not establish patient independence.

## Original file checksums

Compare compressed image and label file MD5 values before extraction. Metadata values apply directly to CSV files.

| File | MD5 |
| --- | --- |
| camelyonpatch_level_2_split_train_x.h5.gz | 1571f514728f59376b705fc836ff4b63 |
| camelyonpatch_level_2_split_train_y.h5.gz | 35c2d7259d906cfc8143347bb8e05be7 |
| camelyonpatch_level_2_split_valid_x.h5.gz | d8c2d60d490dbd479f8199bdfa0cf6ec |
| camelyonpatch_level_2_split_valid_y.h5.gz | 60a7035772fbdb7f34eb86d4420cf66a |
| camelyonpatch_level_2_split_test_x.h5.gz | d5b63470df7cfa627aeec8b9dc0c066e |
| camelyonpatch_level_2_split_test_y.h5.gz | 2b85f58b927af9964a4c15b8f7e8f179 |
| camelyonpatch_level_2_split_train_meta.csv | 5a3dd671e465cfd74b5b822125e65b0a |
| camelyonpatch_level_2_split_valid_meta.csv | 3455fd69135b66734e1008f3af684566 |
| camelyonpatch_level_2_split_test_meta.csv | 67589e00a4a37ec317f2d1932c7502ca |
