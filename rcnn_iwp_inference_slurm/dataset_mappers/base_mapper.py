# Original code comes from the detectron2 DatasetMapper class
# https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/dataset_mapper.py
import copy
import logging
import numpy as np
from typing import List, Optional, Union, Dict
import torch
import rasterio

from detectron2.config import configurable

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from enum import Enum

"""
This file contains the default mapping that's applied to "dataset dicts".
"""

__all__ = ["DatasetMapper"]

class Normalize(Enum):
    _16_BIT = 1
    _16_BIT_MIN_MAX = 2
    _16_BIT_MIN_MAX_NIR = 3

class BaseMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.

    This is the default callable to be used to map your dataset dict into training data.
    You may need to follow it to implement your own one for customized logic,
    such as a different way to read or transform images.
    See :doc:`/tutorials/data_loading` for details.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    @configurable
    def __init__(
        self,
        is_train: bool,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
        use_instance_mask: bool = False,
        use_keypoint: bool = False,
        instance_mask_format: str = "polygon",
        keypoint_hflip_indices: Optional[np.ndarray] = None,
        recompute_boxes: bool = False,
        load_lat_long: bool = False,
        normalize: Normalize = None,
    ):
        """

        Args:
            is_train: whether it's used in training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            use_instance_mask: whether to process instance segmentation annotations, if available
            use_keypoint: whether to process keypoint annotations if available
            instance_mask_format: one of "polygon" or "bitmask". Process instance segmentation
                masks into this format.
            keypoint_hflip_indices: see :func:`detection_utils.create_keypoint_hflip_indices`
            recompute_boxes: whether to overwrite bounding box annotations
                by computing tight bounding boxes from instance mask annotations.
            load_lat_long: whether to load lat,long data from the image source file
            normalize: What type of normalization to apply to the source image before
                the corresponding transforms. Mainly used to make high bit depth images
                compatible with the transformation libraries.
        """
        if recompute_boxes:
            assert use_instance_mask, "recompute_boxes requires instance masks"
        # fmt: off
        self.is_train               = is_train
        self.augmentations          = T.AugmentationList(augmentations)
        self.image_format           = image_format
        self.use_instance_mask      = use_instance_mask
        self.instance_mask_format   = instance_mask_format
        self.use_keypoint           = use_keypoint
        self.keypoint_hflip_indices = keypoint_hflip_indices
        self.recompute_boxes        = recompute_boxes
        self.load_lat_long          = load_lat_long
        self.normalize              = normalize
        # fmt: on
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        augs = utils.build_augmentation(cfg, is_train)
        if cfg.INPUT.CROP.ENABLED and is_train:
            augs.insert(0, T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))
            recompute_boxes = cfg.MODEL.MASK_ON
        else:
            recompute_boxes = False

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "use_instance_mask": cfg.MODEL.MASK_ON,
            "instance_mask_format": cfg.INPUT.MASK_FORMAT,
            "use_keypoint": cfg.MODEL.KEYPOINT_ON,
            "recompute_boxes": recompute_boxes,
        }

        if cfg.MODEL.KEYPOINT_ON:
            ret["keypoint_hflip_indices"] = utils.create_keypoint_hflip_indices(
                cfg.DATASETS.TRAIN
            )

        if cfg.MODEL.LOAD_PROPOSALS:
            ret["precomputed_proposal_topk"] = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )
        return ret

    def load_coordinate_data(self, dataset_dict: Dict):
        file_name = dataset_dict["file_name"]
        assert (
            file_name.split(".")[1].lower() == "tif"
        ), "Images must be supplied in TIF format in order to load lat, long information"
        with rasterio.open(file_name) as src:
            self.load_coordinate_data(dataset_dict, src)

    def load_coordinate_data(self, dataset_dict: Dict, src: rasterio.DatasetReader):
        long, lat = src.lnglat()
        dataset_dict["long"] = long
        dataset_dict["lat"] = lat

    def _transform_annotations(self, dataset_dict, transforms, image_shape):
        for anno in dataset_dict["annotations"]:
            if not self.use_instance_mask:
                anno.pop("segmentation", None)
            if not self.use_keypoint:
                anno.pop("keypoints", None)

        annos = [
            utils.transform_instance_annotations(
                obj,
                transforms,
                image_shape,
                keypoint_hflip_indices=self.keypoint_hflip_indices,
            )
            for obj in dataset_dict.pop("annotations")
            if obj.get("iscrowd", 0) == 0
        ]
        instances = utils.annotations_to_instances(
            annos, image_shape, mask_format=self.instance_mask_format
        )

        # After transforms such as cropping are applied, the bounding box may no longer
        # tightly bound the object. As an example, imagine a triangle object
        # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
        # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
        # the intersection of original bounding box and the cropping box.
        if self.recompute_boxes:
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
        dataset_dict["instances"] = utils.filter_empty_instances(instances)

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        file_name = dataset_dict["file_name"]
        image = None
        # Use rasterio to load tif images
        if file_name.split(".")[1].lower() == "tif":
            with rasterio.open(dataset_dict["file_name"]) as src:
                image = src.read()
                # Rasterio reads images in [C, H, W] order but the augmentations
                # expects it in [H, W, C] order.
                image = image.transpose((1, 2, 0))
                if self.load_lat_long:
                    self.load_coordinate_data(dataset_dict, src)
        else:
            assert (
                self.load_lat_long == False
            ), f"Input image format should be TIF when loading lat,long data: {file_name}"
            image = utils.read_image(file_name, format=self.image_format)

        if self.normalize:
            image =  self._normalize_image(image)

        utils.check_image_size(dataset_dict, image)

        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(
                dataset_dict.pop("sem_seg_file_name"), "L"
            ).squeeze(2)
        else:
            sem_seg_gt = None

        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        transforms = self.augmentations(aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            self._transform_annotations(dataset_dict, transforms, image_shape)

        return dataset_dict

    def _normalize_image(self, image: np.ndarray):
        image = image.astype(np.float32)
        match self.normalize:
            case Normalize._16_BIT:
                image /= 65535
            case Normalize._16_BIT_MIN_MAX:
                # Apply min-max normalization
                min_val = np.min(image)
                max_val = np.max(image)
                image = (image - min_val) / (max_val - min_val)
            case Normalize._16_BIT_MIN_MAX_NIR:
                # Apply min-max normalization                                                                                                               
                min_val = np.min(image)
                max_val = np.max(image)
                image = (image - min_val) / (max_val - min_val)
                # Reorder bands from R G B NIR to NIR G R B 
                if image.shape[2] == 4:  # Ensure the image has 4 bands
                    image = image[:, :, [3, 1, 0, 2]]
            case _:
                raise ValueError(f"Normalization type not supported: {self.normalize}")
        return image
