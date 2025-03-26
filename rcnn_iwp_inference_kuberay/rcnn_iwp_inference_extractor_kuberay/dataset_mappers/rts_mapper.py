import torch
import copy
import numpy as np
import rasterio
from dataset_mappers.base_mapper import BaseMapper
from detectron2.data import detection_utils
from typing import List, Union
import detectron2.data.transforms as T

class RtsDatasetMapper(BaseMapper):
    """
    Customized mapper for RTS dataset.
    """

    def __init__(
        self,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        is_train,
        **kwargs,
    ):
        super().__init__(
            is_train=is_train,
            augmentations=augmentations,
            # This arugment is required so we provide it but its value is irrelevant
            # since we read the image data in our own way.
            image_format="RGB",
            **kwargs
        )

    def __call__(self, dataset_dict):
        """
        Given the 16-bit multiband nature of the RTS dataset, we can't use the default
        datamapper in order to prepare our data for training. The default mapper relies
        on PIL for image loading which doesn't support 16-bit multiband images. The
        implementation below takes a lot of inspiration from the default dataset mapper
        with the biggest difference being the use of rasterio for loading the image data.

        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        with rasterio.open(dataset_dict["file_name"]) as src:
            image = src.read()

        # The raw image read from the .tif file is a 16-bit multiband image. When
        # read into a numpy array it uses a uint16 dtype. Pytorch doesn't support
        # this data type and most image transformation functions assume [0, 1] floats
        # or [0, 255] uint8. To comply with the expected data types while avoiding
        # lossing bit information we normalize our uint16 bands to [0, 1] floats.
        # This is done by dividing the input image by the max value of uint16.
        image = image.astype(np.float32)
        image /= 65535

        # Rasterio reads images in [C, H, W] order but the augmentations
        # expects it in [H, W, C] order.
        image = image.transpose((1, 2, 0))

        try:
            sem_seg_gt = detection_utils.read_image(
                dataset_dict.pop("sem_seg_file_name"), "L"
            ).squeeze(2)
        except IndexError:
            raise ValueError(
                "The provided input dataset_dict must contain 'sem_seg_file_name' information"
            )

        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        # The augmentations are applied in place.
        self.augmentations(aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        # Image is currently in form [H, W, C] but models expects [C, H, W] thus
        # we apply the transpose below.
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image).transpose((2, 0, 1))
        )
        dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        if self.load_lat_long:
            self.load_coordinate_data(dataset_dict)

        return dataset_dict

    def verify_dimensions(image, dataset_dict):
        """
        Args:
            image: ndarray represending image data. Expected to have shape [H, W, C]
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        image_wh = (image.shape[1], image.shape[0])
        expected_wh = (dataset_dict["width"], dataset_dict["height"])
        if not image_wh == expected_wh:
            raise detection_utils.SizeMismatchError(
                "Mismatched image shape{}, got {}, expect {}.".format(
                    (
                        " for image " + dataset_dict["file_name"]
                        if "file_name" in dataset_dict
                        else ""
                    ),
                    image_wh,
                    expected_wh,
                )
                + " Please check the width/height in your annotation."
            )
