# 2020 Stanford qDESS Knee Dataset
authors: Arjun Desai (arjundd at stanford dot edu), Elka Rubin, Andrew Schmidt, Akshay Chaudhari

This folder contains the data and annotations for the 2020 Stanford qDESS knee dataset.

## Directories

- `image_files/`: Contains images files (echo1, echo2, root-mean-square, segmentations).
- `annotations/`: Versioned splits of train/val/test data. See "Versioning" for more info.

For details on these, see sections Image Files and Annotations below

## Image Files
We loosely define image data as all data derived from the GE DICOMS.
This includes the original dicom data (echo 1, echo 2), derived entities (root-mean-square of 2 echos),
as well as segmentations performed by the 3D Lab on the dicoms.

All image data are stored in the `image_files/` directory.
Each scan is represented by an HDF5 file with the keys (and shape, if applicable) below:

    | "echo1" (X x Y x Z): Echo 1 of the qDESS scan
    | "echo2" (X x Y x Z): Echo 2 of the qDESS scan
    | "rms" (X x Y x Z): Root-mean-square of the two echos (sqrt(echo1.^2 + echo2.^2))
    | "seg" (X x Y x Z x 6): One-hot encoded segmentations for 6 classes. See order below.
    | "stats": A dictionary with statistics (mean, standard dev., min, max) of the corresponding volume
        | "echo1"
            | "mean": A scalar for the mean - mean(echo1)
            | "std": A scalar for standard deviation - std(echo1)
            | "min": A scalar - min(echo1)
            | "max": A scalar - max(echo1)
        | "echo2"
            ... Like structure for echo1 stats
        | "rms"
            ... Like structure for echo1 stats

*Note*: Shapes are given in the `(X,Y,Z)` medical format, where `X` corresponds to the rows,
`Y` corresponds to the columns, `Z` corresponds to the depth, etc.

### Segmentations
As mentioned above, the `seg` key holds one-hot encoded segmentations of key soft tissues
in the knee. The order of these segmentations is as follows:

1. Patellar Cartilage
2. Femoral Cartilage
3. Tibial Cartilage - Medial
4. Tibial Cartilage - Lateral
5. Meniscus - Medial
6. Meniscus - Lateral

You may find it useful to combine the medial/lateral components of tibial cartilage and the meniscus.

## Reconstructions
While the DICOM image data is useful for image-based tasks, we may also want to use this dataset for training and evaluating reconstruction algorithms. This dataset also exposes the raw kspace data and potential target images which have been quality checked. The details are provided below.

All qDESS data was acquired with 2x1 parallel imaging using 8 coils with elliptical sampling. Missing data was subsequently estimated using ARC (GE) with the GE Orchestra MATLAB SDK (version TBD). This data is considered the fully-sampled kspace.

### 2D Hybrid Dimension Reconstructions
qDESS is a 3D sequence, but many reconstruction algorithms reconstruct 2D slices to reduce the computation overload of 3D operations.

The 3D kspace `(kx,ky,kz)` is partially reconstructed using the 1D IFFT along the readout direction (`kx`). The resulting hybrid kspace is of dimensions `(x, ky, kz)`. 2D slices to reconstruct are in the `(ky, kz)` plane.

The image is estimated using 2D SENSE reconstruction. Sensitivity maps were estimated using the JSENSE algorithm (sigpy) with a kernel width of 6 and a 24x24 calibration region per slice.

We have added the data for this scenario in the `files_recon_calib-24` folder. Data for each scan is stored as an HDF% file with the following keys:

    | "kspace" (Nx x Ny x Nz x # echos x # coils): Hybrid kspace (x,ky,kz) 
    | "maps" (Nx x Ny x Nz x # coils x # maps): Sensitivity maps
    | "target" (Nx x Ny x Nz x # echos x # maps): Reconstructed image

where `# echos = 2` and `# maps = 1`. `# coils` is typically 8 or 16. All values are complex (np.complex64).


## Annotations and Dataset Splits
Information for all dataset splits can be found in the annotation files, which are json files stored
in a similar manner to the [COCO annotation format](https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch).
These annotation files are also versioned manually (see Versioning section below). Files are named
`{train, val, test}.json`, corresponding to the respective splits.

We break down the different components of the dictionary below:

```
{
    "info": {...},
    "categories": [...], <-- Only detection categories (not segmentation)
    "images": [...],
    "annotations": [...], <-- Only detection annotations (not segmentation)
}
```

### Info
The “info” section contains high level information about the split.
```
  "info": {
    "contributor": "Arjun Desai & Akshay Chaudhari",
    "description": "2020 Stanford qDESS Dataset - test",
    "year": "2020",
    "date_created": "2020-10-16 22:51:12 PDT",
    "version": "v0.0.1"
  },
```

### Images
The "images" section contains the complete list of scans in this split. This is simply a list of the 
scans and useful scan metadata. Note that image ids (`id`) are unique. A description of the keys and structure are shown below:

- `id`: The numeric image id
- `file_name`: The file holding this scan's information
- `msp_id`: The MedSegPy id (only useful for those using MedSegPy)
- `scan_id`: The scan id. This is the universal string used to track this scan.
- `subject_id`: The subject id
- `timepoint`: The timepoint of the scan (zero-indexed). i.e. The first scan, second scan, etc.
- `voxel_spacing`: The spacing for each voxel in mm. In same orientation as `orientation`
- `matrix_shape`: The shape of the matrix
- `orientation`: The orientation of the scan. `SI`- superior to inferior, `AP` - anterior to posterior, `LR` - left to right.
- `num_echoes`: The number of echoes. Should be 2 for all qDESS data
- `inspected`: If `True`, at least one labeler has looked at the image for labeling detection annotations.


```
"images": [
    {
      "id": 1,
      "file_name": "MTR_005.h5",
      "msp_id": "0000099_V00",
      "msp_file_name": "0000099_V00.h5",
      "scan_id": "MTR_005",
      "subject_id": 99,
      "timepoint": 0,
      "voxel_spacing": [0.3125, 0.3125, 0.8],
      "matrix_shape": [512, 512, 160],
      "orientation": ["SI", "AP", "LR"],
      "num_echoes": 2,
      "inspected": true
    },
    {...},
    ...
]
```

### Categories
The "categories" object contains a list of categories and each of those belongs to a supercategory.
The category is a combination of the pathology type and pathology subtype (e.g. Meniscus Tear - Myxoid).
There may be plans in the future to make tissue type an additional stratification level. This would
primarily affect the "Cartilage Lesion" categories, which are currently only separated by grade, but not
by tissue. If you would like to do tissues as well, you will have to reindex your categories (and annotations).

```
    "categories": [
        {
          "supercategory": "Meniscal Tear",
          "supercategory_id": 1,
          "id": 1,
          "name": "Meniscal Tear (Myxoid)"
        },
        {...},
    ]
```

### Tissues
The "tissues" object contains a list of tissues that are referenced in each annotation label

```
"tissues": [
    {
      "id": 1,
      "name": "Meniscus"
    },
    {...},
]
```

### Annotations
The "annotations" section has several components, which makes it a bit trickier to understand. It contains
a list of every individual detection (object) annotation from every scan in the dataset. For example,
if there are 10 Grade 2A cartilage lesions in a single scan, there will be 10 annotations corresponding to
these individual lesions for that scan alone.

The image id corresponds to a specific scan in the dataset.

The bounding box (bbox) format is [top left X position, top left Y position, top left Z position, deltaX, deltaY, deltaZ]. NOTE: X corresponds to row (SI), Y to column (AP), and Z (RL/LR) to depth.

The category id corresponds to a single category specified in the categories section.

Each annotation also has an id (unique to all other annotations in the dataset).

The confidence is on a scale of 0-5, where 0 is not confident at all and 5 is extremely confident. Feel free to filter labels by this scale.

The tissue id corresponds to a single tissue specified in the tissues section.

```
"annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 6,
      "tissue_id": 1,
      "bbox": [
        304.0,
        297.0,
        87.0,
        20.0,
        35.0,
        20.0
      ],
      "confidence": 2.0,
    },
    {...},
    ...
]
```

### Example Code

```python
import json

# Load version v0.0.1 train annotation file
train_ann_file = "annotations/v0.0.1/train.json"

with open(train_ann_file, "r") as f:
    annotations = json.load(f)

# ====================
# Detection Categories
# ====================
categories: List[Dict] = annotations["categories"]
categories[0]

# ====================
# Scan information
# ====================
scans: List[Dict]= annotations["images"]

# ====================
# Detection annotations (bounding boxes, etc)
# ====================
anns: List[Dict] = annotations["annotations"]
```

## Versioning
Medical data continues to grow incrementally and it is useful to have fixed versions to look back 
on for benchmarking and communicating our results.

If you want to make a new version of the dataset, please follow the rules for versioning detailed
here (link to be added). Please also run your versioning by both Arjun Desai (arjundd at stanford dot edu) and Akshay Chaudhari (akshaysc at stanford dot edu).

The most up-to-date version of the qDESS dataset is v0.0.2.

Version history is described below. If new versions are made, add details appropriately

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
