.. _api_modeling:

skm_tea.data
================

SKM-TEA Dataset Catalog
^^^^^^^^^^^^^^^^^^^^^^^
All SKM-TEA dataset splits are accessible from the :cls:`meddlr.data.DatasetCatalog`,
which will return a list of dictionaries, where each dictionary corresponds to a single
scan in that dataset. We refer to this list of dictionaries as the ``dataset_dicts``.
To get the ``dataset_dicts``, see the code below:

.. code-block:: python
   :caption: Getting the dataset_dicts

    from meddlr.data import DatasetCatalog
    import skm_tea as st  # this is import is required load skm-tea into the DatasetCatalog
    
    # Get SKM-TEA dataset dictionaries corresponding to the 'train' split.
    dataset_dicts_train = DatasetCatalog.get("skmtea_v1_train")
    # Val split.
    dataset_dicts_val = DatasetCatalog.get("skmtea_v1_val")
    # Test split.
    dataset_dicts_val = DatasetCatalog.get("skmtea_v1_test")


Each dictionary will contain the following information for the corresponding scan:
    * ``'scan_id'`` (str): The scan name.
    * ``'subject_id'`` (int): The subject id.
    * ``'timepoint'`` (int): The timepoint the subject was scanned.
    * ``'voxel_spacing'`` (List[float, float, float]): The voxel spacing for each dimension of the scan. e.g. ``[0.3125, 0.3125, 0.8]``.
    * ``'matrix_shape'`` (List[int, int, int]): The shape of the image matrix. This is the shape after zero-padding.
    * ``'orientation'`` (List[str, str, str]): The human readable orientation in DOSMA convention - e.g. ``['SI', 'AP', 'LR']``
      where ``SI`` - superior -> inferior, ``AP`` - anterior -> posterior, ``LR`` - left -> right.
    * ``'num_echoes'`` (int): The number of echoes. This will always be 2.
      This mask should be used for segmentation in the DICOM Track.
    * ``'num_coils'`` (int): The number of coils.
    * ``'filename'`` (str): The base name of the HDF5 file.
    * ``'recon_file'`` (str): The HDF5 file containing the Raw Data Track data.
    * ``'image_file'`` (str): The HDF5 file containing the DICOM Track data.
    * ``'gw_corr_mask_file'`` (str): The NIfTI file corresponding to the gradient warp corrected segmentation mask.
      This mask should be used for segmentation in the Raw Data Track.
    * ``'dicom_mask_file'`` (str): The NIfTI file corresponding to the DICOM segmentation mask.
    * ``'dicom_dir'`` (str): The directory containing raw DICOM files.



Torch Datasets
^^^^^^^^^^^^^^
SKM-TEA provides standard torch ``Dataset`` classes for loading data from
Raw Data Track (:class:`skm_tea.data.dataset.SkmTeaRawDataset`) and the DICOM track (:class:`skm_tea.data.dataset.SkmTeaRawDataset`).

Each dataset takes a list of scan dictionaries (i.e. ``dataset_dicts``) and other parameters
that indicate what data should be loaded.

.. autosummary::
    :toctree: generated
    :nosignatures:

    skm_tea.data.dataset.SkmTeaRawDataset
    skm_tea.data.dataset.SkmTeaDicomDataset
