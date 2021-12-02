# Stanford Knee MRI Multi-Task Evaluation (SKM-TEA) Dataset
[**Paper**](https://openreview.net/forum?id=YDMFgD_qJuA)
| [**Dataset Download**](https://stanfordaimi.azurewebsites.net/datasets/4aaeafb9-c6e6-4e3c-9188-3aaaf0e0a9e7)
| [**Dataset Docs**](./DATASET.md)
| [**About**](#about)

The 2021 Stanford Knee MRI with Multi-Task Evaluation (SKM-TEA) dataset pairs raw quantitative MRI (qMRI) data, image data, and dense labels of tissues and pathology for end-to-end exploration and evaluation of the MR imaging pipeline.

This repository contains the building blocks for training and benchmarking models with the SKM-TEA dataset, such as PyTorch dataloaders, evaluation metrics, and baselines. It also contains tutorials for using the dataset and codebase. It utilizes [Meddlr](https://github.com/ad12/meddlr) and [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) for training, evaluation, and machine utilities.

## ⚡ QuickStart
```bash
pip install skm-tea
```
> _Installing locally_: For local development, fork and clone the repo and run `pip install -e ".[dev]"`

> _Installing from main: For most up-to-date code without a local install, run `pip install "skm-tea @ git+https://github.com/StanfordMIMI/skm-tea@main"`

Configure your paths and get going!
```python
import meddlr as mr
import os

# (Optional) Configure and save machine/cluster preferences.
# This only has to be done once and will persist across sessions.
cluster = mr.Cluster()
cluster.set(results_dir="/path/to/save/results", data_dir="/path/to/datasets")
cluster.save()
# OR set these as environment variables.
os.environ["MEDDLR_RESULTS_DIR"] = "/path/to/save/results"
os.environ["MEDDLR_DATASETS_DIR"] = "/path/to/datasets"

# TODO: Add how to easily fetch dataset.
```

## 📝 Documentation
Documentation for downloading and using the SKM-TEA dataset can be found in [DATASET.md](./DATASET.md). Benchmarks are constantly evolving - check this repository for up-to-date baselines.

## 🐘 Model Zoo
A list of pre-trained models can be found [here](MODEL_ZOO.md) and in [Google Drive](https://drive.google.com/drive/folders/156cyINgx-x4uJasMBA6YPipdfOhg7cG5?usp=sharing). 

To use them, pass the google drive urls for the config and weights (model) files to `st.build_deployment_model`:

```python
import skm_tea as st

# Make sure to add "download://" before the url!
model = st.get_model_from_zoo(
  cfg_or_file="download://https://drive.google.com/file/d/1DTSfmaGu2X9CpE5qW52ux63QrIs9L0oa/view?usp=sharing",
  weights_path="download://https://drive.google.com/file/d/1no9-COhdT2Ai3yuxXpSYMpE76hbqZTWn/view?usp=sharing",
)
```

## ✉️ About
<a name="about"></a> 
This repository is being developed at the Stanford's MIMI Lab. Please reach out to `arjundd [at] stanford [dot] edu` if you would like to use or contribute to SKM-TEA. 

If you use the SKM-TEA dataset or code, please use the following BibTex:

```
@inproceedings{desai2021skm,
  title={SKM-TEA: A Dataset for Accelerated MRI Reconstruction with Dense Image Labels for Quantitative Clinical Evaluation},
  author={Desai, Arjun D and Schmidt, Andrew M and Rubin, Elka B and Sandino, Christopher Michael and Black, Marianne Susan and Mazzoli, Valentina and Stevens, Kathryn J and Boutin, Robert and Re, Christopher and Gold, Garry E and others},
  booktitle={Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2)},
  year={2021}
}
```
