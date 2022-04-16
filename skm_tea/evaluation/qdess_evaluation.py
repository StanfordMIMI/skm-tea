import copy
import logging
import os
import time
from typing import Dict

import dosma as dm
import h5py
import meddlr.ops as oF
import meddlr.utils.comm as comm
import numpy as np
import torch
from dosma.core import MedicalVolume
from dosma.core.device import get_array_module
from dosma.scan_sequences.mri import QDess
from meddlr.data import MetadataCatalog
from meddlr.data.catalog import DatasetCatalog
from meddlr.data.data_utils import collect_mask
from meddlr.evaluation.evaluator import DatasetEvaluators
from meddlr.evaluation.recon_evaluation import ReconEvaluator
from meddlr.evaluation.scan_evaluator import ScanEvaluator
from meddlr.evaluation.seg_evaluation import SemSegEvaluator
from meddlr.metrics.collection import MetricCollection
from meddlr.ops import complex as cplx
from meddlr.utils.general import move_to_device
from pytorch_lightning.utilities.distributed import rank_zero_only
from tqdm import tqdm

from skm_tea.data.register import seg_categories_to_idxs
from skm_tea.metrics.qmri import QuantitativeKneeMRI
from skm_tea.utils import env

__all__ = ["SkmTeaEvaluator"]


class SkmTeaEvaluator(ScanEvaluator):
    """SKM-TEA evaluator for reconstruction, segmentation, and qMRI metrics."""

    def __init__(
        self,
        dataset_name,
        cfg,
        distributed=False,
        sync_outputs=False,
        aggregate_scans=True,
        output_dir=None,
        group_by_scan=False,
        skip_rescale=False,
        save_scans=False,
        recon_metrics=None,
        sem_seg_metrics=None,
        flush_period: int = None,
        to_cpu=True,
        tasks=("recon",),
        use_qmri: bool = False,
    ):
        """
        Args:
            dataset_name (str): The name of the dataset (e.g. 'skm_tea_v1_test').
            cfg (CfgNode): The config.
            distributed (bool, optional): Set to ``True`` if program is being run
                in distributed mode (e.g. using DistributedDataParallel).
            sync_outputs (bool, optional): If ``True`` and ``distributed=True``,
                synchronize outputs between processes.
            aggregate_scans (bool, optional): If ``True``, aggregate slices/patches
                into a scan to compute scan metrics.
            output_dir (str, optional): The directory to save the output files.
            group_by_scan (bool, optional): If ``True``, ``.evaluate()`` will
                return a dictionary of scan_id -> {metric1: value1, metric2: value2, ...}.
            skip_rescale (bool, optional): If ``True``, do not undo the normalization
                done to the kspace data. This should be ``False`` when computing metrics
                for comparable results.
            save_scans (bool, optional): If ``True``, save the scan predictions to disk.
            recon_metrics (Sequence[str], optional): A list of metrics for reconstruction.
                E.g. ``'psnr'``, ``'ssim'``, ``'ms_ssim'``, ``'ms_ssim_full'``,
        """
        # self._tasks = self._tasks_from_config(cfg)
        self._output_dir = output_dir
        if self._output_dir:
            os.makedirs(self._output_dir, exist_ok=True)

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)
        self._distributed = distributed
        self._sync_outputs = sync_outputs
        self._aggregate_scans = aggregate_scans
        self._group_by_scan = group_by_scan
        self._skip_rescale = skip_rescale
        self._save_scans = save_scans
        self._metadata = MetadataCatalog.get(dataset_name)
        dataset_dicts = DatasetCatalog.get(dataset_name)
        self._scan_metadata = {ex["scan_id"]: ex for ex in dataset_dicts}

        # The directory of images to load from.
        self._image_dir = (
            self._metadata.image_dir
            if cfg.DATASETS.QDESS.DATASET_TYPE == "qDESSImageDataset"
            else self._metadata.recon_dir
        )

        paths = cfg.TEST.QDESS_EVALUATOR.ADDITIONAL_PATHS
        additional_paths = {k: v for k, v in zip(paths[0::2], paths[1::2])}
        self.additional_paths = additional_paths

        if use_qmri and not aggregate_scans:
            raise ValueError(
                "qMRI analysis is only valid for the full scan. Set `aggregate_scans=True`"
            )
        if use_qmri and skip_rescale:
            raise ValueError("Cannot skip rescaling when computing quantitative metrics")
        if use_qmri and (tasks == ("recon",)) and cfg.DATASETS.QDESS.ECHO_KIND in ("echo2",):
            raise ValueError(
                "Cannot compute qMRI metrics with only echo1 or echo2. "
                "Both echos must be present."
            )
        if (
            use_qmri
            and ("recon" in tasks)
            and cfg.DATASETS.QDESS.ECHO_KIND == "echo1"
            and "recon/echo2" not in self.additional_paths
        ):
            raise ValueError(
                "Cannot compute qMRI merics with only echo1. Must specify additional path to echo2"
            )
        self._use_qmri = use_qmri

        unknown_tasks = set(tasks) - {"recon", "sem_seg"}
        if len(unknown_tasks) != 0:
            raise ValueError(f"Unknown tasks for qDESS evaluation: {unknown_tasks}")
        self._tasks = tasks

        evaluators = {}
        evaluator_kwargs = dict(
            dataset_name=dataset_name,
            cfg=cfg,
            distributed=distributed,
            sync_outputs=sync_outputs,
            aggregate_scans=aggregate_scans,
            group_by_scan=group_by_scan,
            flush_period=0,  # flushing handled by this evaluator
            to_cpu=to_cpu,
        )
        if "recon" in tasks:
            evaluators["recon"] = ReconEvaluator(
                output_dir=os.path.join(output_dir, "recon") if output_dir else None,
                metrics=recon_metrics,
                save_scans=save_scans,
                skip_rescale=False,
                eval_in_process=True,
                structure_channel_by=(
                    "echo" if cfg.DATASETS.QDESS.ECHO_KIND == "echo1+echo2" else None
                ),
                **evaluator_kwargs,
            )
        if "sem_seg" in tasks:
            evaluators["sem_seg"] = SemSegEvaluator(
                output_dir=os.path.join(output_dir, "sem_seg") if output_dir else None,
                metrics=sem_seg_metrics,
                save_seg=save_scans,
                **evaluator_kwargs,
            )
        self.evaluators = DatasetEvaluators(evaluators)
        self.seg_classes = (
            evaluators["sem_seg"]._class_names
            if "sem_seg" in evaluators
            else ("pc", "fc", "tc", "men")
        )
        # TODO: Change this to use subregions
        self.subregions = tuple({"fc", "tc", "pc", "men"} & set(self.seg_classes))

        if flush_period is None:
            flush_period = cfg.TEST.FLUSH_PERIOD
        if distributed and flush_period != 0:
            raise ValueError("Result flushing is not enabled in distributed mode.")
        self.flush_period = flush_period
        self.to_cpu = to_cpu

        self._remaining_state = None
        self._is_flushing = False

    def reset(self):
        self._remaining_state = None
        self._is_flushing = False
        self.evaluators.reset()
        if not self._use_qmri:
            self.scan_metrics = None
            return

        subregions = self.subregions
        use_cpu = True

        scan_metrics = {}
        if "sem_seg" in self._tasks:
            metric_names = ["t2_seg_gt", "t2_seg_pred"]
            qmri_metrics = [
                QuantitativeKneeMRI(
                    subregions=subregions,
                    channel_names=self.seg_classes,
                    use_cpu=use_cpu,
                    output_dir=os.path.join(self._output_dir, odir) if self._output_dir else None,
                )
                for odir in metric_names
            ]

            # TODO: this will need to be fixed for multi-task models
            # Ground truth
            gt_metric = qmri_metrics[0]
            gt_metric.register_update_aliases(
                qmap_sem_seg_gt="quantitative_map", sem_seg_gt="sem_seg"
            )
            # Prediction
            pred_metric = qmri_metrics[1]
            pred_metric.register_update_aliases(
                qmap_sem_seg_pred="quantitative_map", sem_seg_pred="sem_seg"
            )

            scan_metrics.update({k: v for k, v in zip(metric_names, qmri_metrics)})

        if "recon" in self._tasks:
            metric_names = ["t2_recon_gt", "t2_recon_pred"]
            qmri_metrics = [
                QuantitativeKneeMRI(
                    subregions=subregions,
                    channel_names=self.seg_classes,
                    use_cpu=use_cpu,
                    output_dir=os.path.join(self._output_dir, odir) if self._output_dir else None,
                )
                for odir in metric_names
            ]

            # Ground truth
            gt_metric = qmri_metrics[0]
            gt_metric.register_update_aliases(
                qmap_recon_gt="quantitative_map", sem_seg_gt="sem_seg"
            )
            # Prediction
            pred_metric = qmri_metrics[1]
            pred_metric.register_update_aliases(
                qmap_recon_pred="quantitative_map", sem_seg_gt="sem_seg"
            )

            scan_metrics.update({k: v for k, v in zip(metric_names, qmri_metrics)})

        self.scan_metrics = MetricCollection(scan_metrics)
        self.scan_metrics.eval()

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a recon model (e.g., GeneralizedRCNN).
                Currently this should be an empty dictionary.
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.

        Note:
            All elements in ``inputs`` and ``outputs`` should already
            be detached from the computational graph.
        """
        self.evaluators.process(inputs, outputs)

        predictions = self.evaluators[0]._predictions
        has_num_examples = self.flush_period > 0 and len(predictions) >= self.flush_period
        has_num_scans = self.flush_period < 0 and len(
            {x["metadata"]["scan_id"] for x in predictions}
        ) > abs(self.flush_period)
        if has_num_examples or has_num_scans:
            self.flush(skip_last_scan=True)

    def flush(self, skip_last_scan: bool = True):
        e: ScanEvaluator
        outs = []
        for e in self.evaluators:
            outs.append(e.enter_prediction_scope(skip_last_scan=skip_last_scan))
        if any(x is False for x in outs):
            return False

        # Evaluate metrics for base evaluators.
        for e in self.evaluators:
            e.flush(enter_prediction_scope=False)

        # Have to empty cache between evaluations because of large memory footprint
        # of specific reconstruction metrics.
        if "recon" in self._tasks:
            self.clear_cache()

        # Do evaluation here.
        self._is_flushing = True
        self.evaluate(skip_evaluators=True)
        self._is_flushing = False

        for e in self.evaluators:
            e.exit_prediction_scope()

    def exit_prediction_scope(self):
        ret_val = super().exit_prediction_scope()
        self.clear_cache()
        return ret_val

    def clear_cache(self):
        if env.supports_cupy():
            import cupy as cp

            cp._default_memory_pool.free_all_blocks()
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.empty_cache()

    def structure_scans(self):
        scans = {}
        e: ScanEvaluator
        for k, e in self.evaluators.items():
            scans[k] = e.structure_scans(verbose=False)

        # This is a hacky solution to allow qMRI inference with models trained on individual echos.
        # TODO: Clean up
        if "recon/echo2" in self.additional_paths:
            for scan_id, out in scans["recon"].items():
                for k, dirpath, sl in zip(
                    ["pred", "target"],
                    [self.additional_paths["recon/echo2"], self._metadata.recon_dir],
                    [(), (Ellipsis, slice(1, 2), 0)],
                ):
                    with h5py.File(os.path.join(dirpath, scan_id + ".h5"), "r") as f:
                        echo2 = f[k][sl]
                    out[k] = torch.cat(
                        [out[k], torch.from_numpy(echo2).to(out[k].device)], dim=-1
                    ).contiguous()

        # Temporary fix for determining the device that the metrics should be computed on.
        # TODO: Find a better method for determining which device to compute on.
        if not env.supports_cupy():
            scans = move_to_device(scans, "cpu")
        k = list(scans.keys())[0]
        device = scans[k][list(scans[k].keys())[0]]["pred"].device

        key = list(scans.keys())[0]
        scan_ids = scans[key].keys()
        for k, v in scans.items():
            assert v.keys() == scan_ids, f"Mismatched scan ids {k}"

        # Compute T2 scans
        outputs = {}
        for scan_id in tqdm(scan_ids, desc="Structuring T2 map"):
            seg_kwargs = self._get_segmentations(scans, scan_id)
            qmap_kwargs = self._get_t2(scans, scan_id, seg_kwargs, device)

            metadata = self._metadata.scan_metadata
            metadata = metadata[metadata["MTR_ID"] == scan_id]
            assert len(metadata) == 1
            metadata = metadata.iloc[0]

            out = {
                **seg_kwargs,
                **qmap_kwargs,
                "ids": scan_id,
                "medial_direction": str(metadata["MedialDirection"]),
            }
            out = {k: [v] for k, v in out.items()}
            outputs[scan_id] = out

        return outputs

    def _get_segmentations(self, scans, scan_id) -> Dict[str, MedicalVolume]:
        if "sem_seg" not in scans:
            scan_metadata = self._scan_metadata[scan_id]
            orientation, spacing = scan_metadata["orientation"], scan_metadata["voxel_spacing"]
            affine = dm.to_affine(orientation, spacing)

            # seg_file = path_manager.get_local_path(
            #     os.path.join(self._metadata.image_dir, scan_id + ".h5")
            # )
            # assert self.seg_classes == ("pc", "fc", "tc", "men")
            # with h5py.File(seg_file, "r") as f:
            #     gt_seg = f["seg"][()]
            gt_seg = dm.NiftiReader().load(
                os.path.join(self._metadata.mask_gradwarp_corrected_dir, scan_id + ".nii.gz")
            )
            gt_seg = oF.categorical_to_one_hot(
                gt_seg.A.astype(np.int64), background=0, channel_dim=-1
            )
            seg_idxs = seg_categories_to_idxs(self.seg_classes)
            gt_seg = collect_mask(gt_seg, index=seg_idxs, out_channel_first=False)
            gt_seg = dm.MedicalVolume(gt_seg, affine=affine)
            return {"sem_seg_gt": gt_seg}

        gt_seg = scans["sem_seg"][scan_id]["target"].permute((1, 2, 3, 0))
        pred_seg = scans["sem_seg"][scan_id]["pred"].permute((1, 2, 3, 0))
        affine = scans["sem_seg"][scan_id]["affine"]
        gt_seg = dm.MedicalVolume.from_torch(gt_seg, affine=affine)
        pred_seg = dm.MedicalVolume.from_torch(pred_seg, affine=affine)
        return {"sem_seg_gt": gt_seg, "sem_seg_pred": pred_seg}

    def _get_t2(self, scans, scan_id, segmentations, device=None):
        path_manager = env.get_path_manager()

        echos = {}
        if "recon" not in scans:
            scan_metadata = self._scan_metadata[scan_id]
            orientation, spacing = scan_metadata["orientation"], scan_metadata["voxel_spacing"]
            affine = dm.to_affine(orientation, spacing)

            image_file = path_manager.get_local_path(os.path.join(self._image_dir, scan_id + ".h5"))
            with h5py.File(image_file, "r") as f:
                if all(k in f for k in ("echo1", "echo2")):
                    echo1, echo2 = f["echo1"][()], f["echo2"][()]
                elif "target" in f:
                    img = np.abs(f["target"][()])
                    echo1, echo2 = img[..., 0, 0], img[..., 1, 0]
                else:
                    raise ValueError(f"Unknown data format in {image_file}")
            echo1, echo2 = torch.as_tensor(echo1), torch.as_tensor(echo2)
            echos["gt"] = (echo1, echo2)
        else:
            # TODO: Figure out what the permutation here should be.
            # This will likely depend on if it is multi channel or single channel.
            recons = scans["recon"][scan_id]
            affine = recons["affine"]

            recon_keys = ["pred", "target"]
            recons.update({k: cplx.abs(recons[k]) for k in recon_keys})
            echos = {
                k: tuple(recons[v][..., i] for i in range(2))
                for k, v in zip(["pred", "gt"], recon_keys)
            }

        segmentations = move_to_device(segmentations, device, base_types=(MedicalVolume,))
        echos = move_to_device(echos, device, base_types=(MedicalVolume,))

        if self._tasks == ("sem_seg",):
            gt_seg, pred_seg = segmentations["sem_seg_gt"], segmentations["sem_seg_pred"]
            t2map_gt = self.compute_t2_map(echos["gt"], scan_id, affine, gt_seg)
            t2map_pred = self.compute_t2_map(echos["gt"], scan_id, affine, pred_seg)
            out = {"qmap_sem_seg_gt": t2map_gt, "qmap_sem_seg_pred": t2map_pred}
        elif self._tasks == ("recon",):
            gt_seg = segmentations["sem_seg_gt"]
            t2map_gt = self.compute_t2_map(echos["gt"], scan_id, affine, gt_seg)
            t2map_pred = self.compute_t2_map(echos["pred"], scan_id, affine, gt_seg)
            out = {"qmap_recon_gt": t2map_gt, "qmap_recon_pred": t2map_pred}
        else:
            raise ValueError(f"{self._tasks} not supported")

        return out

    def evaluate(self, skip_evaluators=False):
        # Sync predictions (if applicable)
        if not skip_evaluators:
            self.evaluators.evaluate()

        # Compute metrics per scan.
        if self._use_qmri:
            scans = self.structure_scans()
            scans = scans.values()
            start_time = time.perf_counter()
            for pred in tqdm(scans, desc="Scan Metrics", disable=not comm.is_main_process()):
                self.evaluate_prediction(pred, self.scan_metrics)
            self._logger.info(
                f"qDESS T2 Evaluation too {time.perf_counter() - start_time:0.2f} seconds"
            )

        slice_metrics = self.aggregate_metrics("slice_metrics")
        scan_metrics = self.aggregate_metrics("scan_metrics")
        pred_vals = slice_metrics.to_dict()
        pred_vals.update(scan_metrics.to_dict())
        self._results = pred_vals

        self.log_summary(save_output=not self._is_flushing)

        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)

    def evaluate_prediction(self, prediction, metrics: MetricCollection):
        metrics(**prediction)
        return metrics.to_dict()

    def aggregate_metrics(self, kind="slice_metrics"):
        """Aggregates metrics from all of the sub-evaluators."""
        metrics = [getattr(e, kind) for e in self.evaluators.values() if hasattr(e, kind)]
        if hasattr(self, kind):
            metrics.append(getattr(self, kind))
        metrics = [m for m in metrics if m is not None]
        assert all(isinstance(m, MetricCollection) for m in metrics)

        agg_metrics = {}
        for m in metrics:
            agg_metrics.update(m)
        return MetricCollection(agg_metrics)

    @rank_zero_only
    def log_summary(self, save_output=None):
        output_dir = self._output_dir
        if self.scan_metrics is not None:
            self._logger.info(
                "[{}] Scan metrics summary:\n{}".format(
                    type(self).__name__, self.scan_metrics.summary()
                )
            )

        if not output_dir or save_output is False:
            return

        # Combine slice and scan metrics for sub-evaluators.
        slice_metrics: MetricCollection = self.aggregate_metrics("slice_metrics")
        scan_metrics: MetricCollection = self.aggregate_metrics("scan_metrics")

        dirpath = output_dir
        os.makedirs(dirpath, exist_ok=True)
        test_results_summary_path = os.path.join(dirpath, "results.txt")
        slice_metrics_path = os.path.join(dirpath, "slice_metrics.csv")
        scan_metrics_path = os.path.join(dirpath, "scan_metrics.csv")

        # Write details to test file
        with open(test_results_summary_path, "w+") as f:
            f.write("Results generated on %s\n" % time.strftime("%X %x %Z"))
            # f.write("Weights Loaded: %s\n" % os.path.basename(self._config.TEST_WEIGHT_PATH))

            f.write("--" * 40)
            f.write("\n")
            f.write("Slice Metrics:\n")
            f.write(slice_metrics.summary())
            f.write("--" * 40)
            f.write("\n")
            f.write("Scan Metrics:\n")
            f.write(scan_metrics.summary())
            f.write("--" * 40)
            f.write("\n")

        df = slice_metrics.to_pandas()
        df.to_csv(slice_metrics_path, header=True, index=True)

        df = scan_metrics.to_pandas()
        df.to_csv(scan_metrics_path, header=True, index=True)

    def compute_t2_map(self, echos, scan_id, affine, seg):
        """Computes T2 qMRI parameter map."""
        metadata = self._metadata.scan_metadata
        metadata = metadata[metadata["MTR_ID"] == scan_id]
        assert len(metadata) == 1
        metadata = metadata.iloc[0]

        if isinstance(echos, torch.Tensor):
            echos = [echos]
        if len(echos) == 1:
            echos = cplx.abs(echos[0])
            # Check this
            echo1, echo2 = echos[0], echos[1]
        else:
            assert len(echos) == 2
            if cplx.is_complex(echos[0]) or cplx.is_complex_as_real(echos[1]):
                echo1, echo2 = cplx.abs(echos[0]), cplx.abs(echos[1])
            else:
                echo1, echo2 = echos[0], echos[1]

        # Compute t1 map based on the segmentation.
        # T1 for all relevant tissues is assumed to be 1.2 seconds (1200 ms)
        # except for meniscus, which is 1 second.
        xp = get_array_module(seg.A)
        t1 = 1200.0 * MedicalVolume(xp.ones_like(seg.A[..., 0]), affine=seg.affine)
        men_idx = self.seg_classes.index("men")
        t1[seg[..., men_idx].A == 1] = 1000.0

        echo1 = MedicalVolume.from_torch(echo1, affine=affine)
        echo2 = MedicalVolume.from_torch(echo2, affine=affine)
        t1 = t1.reformat_as(echo1)
        qdess = QDess([echo1, echo2])
        t2map = qdess.generate_t2_map(
            suppress_fat=True,
            suppress_fluid=True,
            gl_area=float(metadata["SpoilerGradientArea"]),
            tg=float(metadata["SpoilerGradientTime"]),
            tr=float(metadata["RepetitionTime"]),
            te=float(metadata["EchoTime1"]),
            alpha=float(metadata["FlipAngle"]),
            t1=t1,
            nan_bounds=(0, 100),
            nan_to_num=True,
        )
        return t2map.volumetric_map
