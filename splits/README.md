# Split files

`splits_seed_<seed>.json` (seeds 42, 1024, 2026) contain the validation indices of the five stratified folds
(`val_idx`, list of five lists). Indices refer to the order of `torchvision.datasets.ImageFolder` on the dataset
(classes and files sorted alphabetically; Zenodo record, see `data/README.md`). Each split covers all 996 images exactly
once; the training set of a fold is the complement of its validation set. The same splits were used for all models.
The field `fingerprint` is a hash of the image list (relative paths and file sizes) written by the training script.
