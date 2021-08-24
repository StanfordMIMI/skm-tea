# Versioning

Medical data continues to grow incrementally and it is useful to have fixed versions to look back 
on for benchmarking and communicating our results.

If you want to make a new version of the dataset, please follow the rules for versioning detailed
here (link to be added). Please also run your versioning by both Arjun Desai (arjundd at stanford dot edu) and Akshay Chaudhari (akshaysc at stanford dot edu).

The most up-to-date version of the qDESS dataset is v1.0.0.

Version history is described below. If new versions are made, add details appropriately

### v1.0.0
--------------------------------
Date Released: August 23, 2021

This is the first stable release of the qDESS dataset.

Changelist:
  * Bumping `v0.1.0 -> v1.0.0` to indicate stable release.
  * No changes were made; this version is synonymous with `v0.1.0`.

### v0.1.0
--------------------------------
Date Released: March 8, 2021

Splits are the same as v0.0.2. Image data can be found at:

```
/bmrNAS/people/arjun/data/qdess_knee_2020/image_files_v0.1
```

Changelist:
  * Segmentation masks in the test set were corrected by two additional annotators.
  The following segmentations were changed: `MTR_005`, `MTR_030`, `MTR_066`, `MTR_134`, `MTR_219`, `MTR_248`.

  * Additional bounding box annotations

Data Distribution:

            # Scans    % Male    % Female  Age (mean +/- std)      # BBoxes
    -----  ---------  --------  ----------  --------------------  ----------
    train         86      53.5        46.5  43.9 +/- 18.2                242
    val           33      54.5        45.5  44.5 +/- 18.1                104
    test          36      72.2        27.8  42.1 +/- 15.8                130

### v0.0.2
--------------------------------
Date Released: November 24, 2020

Splits are the same as v0.0.1. Additional bounding box annotations were added.

            # Scans    % Male    % Female  Age (mean +/- std)      # BBoxes
    -----  ---------  --------  ----------  --------------------  ----------
    train         86      53.5        46.5  43.9 +/- 18.2                239
    val           33      54.5        45.5  44.5 +/- 18.1                102
    test          36      72.2        27.8  42.1 +/- 15.8                130

### v0.0.1
--------------------------------
Date Released: October 16, 2020

This is the first version of the qDESS dataset.

Examples in the test split have received arthroscopic surgery (to establish gold standard injury reports) 
and were evaluated by 2 msk rads. Training and validation splits were pseudo randomly determined from the remaining scans
while approximiately balancing age and sex. Details are shown below:

            # Scans    % Male    % Female  Age (mean +/- std)      # BBoxes
    -----  ---------  --------  ----------  --------------------  ----------
    train         86      53.5        46.5  43.9 +/- 18.2                207
    val           33      54.5        45.5  44.5 +/- 18.1                 88
    test          36      72.2        27.8  42.1 +/- 15.8                 23
