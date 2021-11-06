# Stanford Knee MRI Multi-Task Evaluation (SKM-TEA) Dataset
[Dataset Download](https://stanfordaimi.azurewebsites.net/datasets/4aaeafb9-c6e6-4e3c-9188-3aaaf0e0a9e7) | [Paper](https://openreview.net/forum?id=YDMFgD_qJuA)

Authors: Arjun Desai (arjundd at stanford dot edu), Andrew Schmidt, Elka Rubin, Akshay Chaudhari & collaborators

Latest Version: v1.0.0

This repository contains the annotations for the 2021 Stanford Knee MRI Multi-Task Evaluation (SKM-TEA) dataset, a dataset that provides paired raw quantitative MRI (qMRI) data, image data, and dense labels of tissues and pathology.

This dataset enables two benchmarking tracks:

  1. `Raw Data` Track: Benchmarking related to MRI reconstruction and analysis on images generated from raw MRI data (i.e. k-space)
  2. `DICOM` Track: Benchmarking related to image analysis on scanner-generated DICOM images.

Details on the data to use for each track are provided below.

## Access and Setup
To download the dataset, follow the instructions below. Note the dataset is 900GB in a compressed format and 1.6TB in the expanded format. Please ensure there is sufficient disk space to download and uncompress the data. Currently, the download does not support downloading different directories for different tracks separately. We are actively working to resolve this issue.

1. Navigate to [this page](https://stanfordaimi.azurewebsites.net/datasets/4aaeafb9-c6e6-4e3c-9188-3aaaf0e0a9e7).
1. On the top left corner, click the login button. Create a new account or log into an existing account.
1. Navigate back to the link above
1. Follow instructions for downloading the dataset.
1. To use this directory with this codebase, download the dataset under `<dataset-path>/skm-tea/`, where `<dataset-path>` corresponds to the datasets directory configured in Meddlr.


## Data Overview
This dataset consists of raw k-space and image data acquired using quantitative double-echo-steady-state (qDESS) MRI knee scans of patients at Stanford Healthcare. Images were manually segmented for 4 tissues: (1) Patellar Cartilage, (2) Femoral Cartilage, (3) Tibial Cartilage, and (4) Meniscus. Images were also manually annotated with bounding boxes for pathology documented in radiologist reports.

## Directories

All data and annotations are stored in the directory structure shown below. Details of how data in each folder should be used are provided in the Tracks below.

- `annotations/`: Versioned splits of train/val/test data. See "Versioning" for more info.
- `files_recon_calib-24/`: Data related to the `Raw Data` track in HDF5 format
- `image_files/`: Data related to the `DICOM` track in HDF5 format
- `dicoms/`: Scanner-generated DICOM files. This should be used for visualization purposes only.
- `segmentation_masks/raw-data-track`: Ground truth segmentations (in Nifti format) for `Raw Data` track
- `segmentation_masks/dicom-track`: Ground truth segmentations (in Nifti format) for `DICOM` track
- `all_metadata.csv`: De-identified DICOM metadata for each scan

All `.tar.gz` files should be extracted using 
```bash
tar -xvzf tar-file.tar.gz
```

For details on how data in each directory is used, see [Benchmarking Tracks](#benchmarking-tracks).

## Benchmarking Tracks
### `DICOM` Track
The `DICOM` benchmarking track uses scanner-generated DICOM images as the input for image segmentation and detection tasks.

All DICOM pixel data has been extracted are stored in the `image_files/` directory.
Each scan is represented by an HDF5 file with the schema (and shape, if applicable) below:

    | "echo1" (X x Y x Z): Echo 1 of the qDESS scan
    | "echo2" (X x Y x Z): Echo 2 of the qDESS scan
    | "seg" (X x Y x Z x 6): One-hot encoded segmentations for 6 classes. See order below.
    | "stats": A dictionary with statistics (mean, standard dev., min, max) of the corresponding volume
        | "echo1"
            | "mean": A scalar for the mean - mean(echo1)
            | "std": A scalar for standard deviation - std(echo1)
            | "min": A scalar - min(echo1)
            | "max": A scalar - max(echo1)
        | "echo2"
            ... Like structure for echo1 stats
        | "rss"
            ... Like structure for echo1 stats, but for root-sum-of-squares of the two echos
            ... (i.e. sqrt(echo1**2 + echo2**2))

*Note*: Shapes are given in the `(X,Y,Z)` medical format, where `X` corresponds to the rows,
`Y` corresponds to the columns, `Z` corresponds to the depth, etc.

#### Segmentations
As mentioned above, the `seg` key holds one-hot encoded segmentations of key soft tissues
in the knee. The order of these segmentations is as follows:

1. Patellar Cartilage
2. Femoral Cartilage
3. Tibial Cartilage - Medial
4. Tibial Cartilage - Lateral
5. Meniscus - Medial
6. Meniscus - Lateral

You may find it useful to combine the medial/lateral components of tibial cartilage and the meniscus.

These segmentations are also available in the Nifti format in `segmentation_masks/dicom-track/`.

#### Bounding Boxes
See [Annotations and Dataset Splits](#annotations-and-dataset-splits).

### `Raw Data` Track
While the DICOM image data is useful for image-based tasks, we may also want to use this dataset for training and evaluating reconstruction algorithms and image analysis on reconstructed outputs. This dataset also exposes the raw kspace data and SENSE-generated target images which have been quality checked.

All qDESS data was acquired with 2x1 parallel imaging with elliptical sampling. Missing k-space data was subsequently estimated using ARC (GE) with the GE Orchestra MATLAB SDK (version TBD). This data is considered the fully-sampled kspace.

#### 2D Hybrid Dimension Reconstructions
qDESS is a 3D sequence, but many reconstruction algorithms reconstruct 2D slices to reduce the computation overload of 3D operations.

The 3D kspace `(kx,ky,kz)` is partially reconstructed using the 1D IFFT along the readout direction (`kx`). The resulting hybrid kspace is of dimensions `(x, ky, kz)`. 2D slices to reconstruct are in the `(ky, kz)` plane.

The image is estimated using 2D SENSE reconstruction. K-space was zero-padded following the ZIP2 protocol prior to reconstruction to align SENSE-reconstructed images with dimensions of the DICOM images. Sensitivity maps were estimated using the JSENSE algorithm (sigpy) with a kernel width of 6 and a 24x24 calibration region per slice.

Data for this track can be found in the `files_recon_calib-24` folder. Data for each scan is stored as an HDF5 file with the following schema:

    | "kspace" (Nx x Ny x Nz x # echos x # coils): Hybrid kspace (x,ky,kz) 
    | "maps" (Nx x Ny x Nz x # coils x # maps): Sensitivity maps
    | "target" (Nx x Ny x Nz x # echos x # maps): Reconstructed image

where `# echos = 2` and `# maps = 1`. `# coils` is typically 8 or 16. All values are complex (np.complex64).

#### Segmentations
Segmentations corresponding to SENSE-based image reconstructions can be found in `segmentation_masks/raw-data-track`. The order of the segmentation masks are the same as in the DICOM track.

#### Bounding Boxes
See [Annotations and Dataset Splits](#annotations-and-dataset-splits).
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
    "contributor": "Arjun Desai, Elka Rubin, Andrew Schmidt, Akshay Chaudhari",
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
