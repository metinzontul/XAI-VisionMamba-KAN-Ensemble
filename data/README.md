# Dataset layout

The training scripts read the images with `torchvision.datasets.ImageFolder`, so the dataset root must contain
one sub-folder per class:

```
data/Marble/
├── Medium1/   (176 images)
├── Medium2/   (170)
├── Walnut1/   (199)
├── Walnut2/   (137)
├── White1/    (227)
└── White2/    (87)
```

996 JPEG photographs (6016 × 4000 pixels, Nikon D3200) of natural-stone tiles, sorted into six quality classes by factory experts
(class 1 = first quality, class 2 = second quality within each colour family).

**Availability:** Zenodo, https://doi.org/10.5281/zenodo.24053101 (licence CC BY 4.0). The Zenodo record contains `Marble.zip`, `md5sums.txt` and a `README.txt`.

No tile or slab identifiers exist, so all splits are made at image level. If you have such identifiers, pass a CSV
with the columns `filename,group` to `master_train.py --groups_csv` to get group-wise folds (`StratifiedGroupKFold`).
