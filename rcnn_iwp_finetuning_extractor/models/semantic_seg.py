from typing import Tuple
from enum import Enum
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.structures import ImageList
from detectron2.modeling import Backbone, build_backbone, SEM_SEG_HEADS_REGISTRY


# only difference between this and the built in detectron2 SemanticSegmentor
# is that postprocessing is applied based on the post_process argument to the
# forward function.
class SemanticSegmentor(nn.Module):
    """
    Main class for semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
        """
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.register_buffer(
            "pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False
        )
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())
        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs, post_process=True):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.

                For now, each item in the list is a dict that contains:

                   * "image": Tensor, image in (C, H, W) format.
                   * "sem_seg": semantic segmentation ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.


        Returns:
            list[dict]:
              Each dict is the output for one input image.
              The dict contains one key "sem_seg" whose value is a
              Tensor that represents the
              per-pixel segmentation prediced by the head.
              The prediction has shape KxHxW that represents the logits of
              each class for each pixel.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(
            images,
            self.backbone.size_divisibility,
            padding_constraints=self.backbone.padding_constraints,
        )

        if self.backbone.location_embedding:
            # Create a batch of coordinates based on the input batch
            batched_coordinates = torch.zeros((len(batched_inputs), 2))
            for i, input in enumerate(batched_inputs):
                batched_coordinates[i] = torch.Tensor([input["long"], input["lat"]])
            batched_coordinates = batched_coordinates.to(self.device).double()
            features = self.backbone(images.tensor, batched_coordinates)
        else:
            features = self.backbone(images.tensor)

        if "sem_seg" in batched_inputs[0]:
            targets = [x["sem_seg"].to(self.device) for x in batched_inputs]
            targets = ImageList.from_tensors(
                targets,
                self.backbone.size_divisibility,
                self.sem_seg_head.ignore_value,
                self.backbone.padding_constraints,
            ).tensor
        else:
            targets = None
        results, losses = self.sem_seg_head(features, targets)

        if self.training:
            return losses

        if post_process:
            processed_results = []
            for result, input_per_image, image_size in zip(
                results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = sem_seg_postprocess(result, image_size, height, width)
                processed_results.append({"sem_seg": r})
            return processed_results

        return [{"sem_seg": result} for result in results]


def build_sem_seg_head(cfg, input_shape):
    """
    Build a semantic segmentation head from `cfg.MODEL.SEM_SEG_HEAD.NAME`.
    """
    name = cfg.MODEL.SEM_SEG_HEAD.NAME
    return SEM_SEG_HEADS_REGISTRY.get(name)(cfg, input_shape)


# We use this instead of the built in detetron2 post process function so that
# we can control the type of interpolation that is used to resize the mask.
def sem_seg_postprocess(result, img_size, output_height, output_width, mode="bilinear"):
    """
    Return semantic segmentation predictions in the original resolution.

    The input images are often resized when entering semantic segmentor. Moreover, in same
    cases, they also padded inside segmentor to be divisible by maximum network stride.
    As a result, we often need the predictions of the segmentor in a different
    resolution from its inputs.

    Args:
        result (Tensor): semantic segmentation prediction logits. A tensor of shape (C, H, W),
            where C is the number of classes, and H, W are the height and width of the prediction.
        img_size (tuple): image size that segmentor is taking as input.
        output_height, output_width: the desired output resolution.

    Returns:
        semantic segmentation prediction (Tensor): A tensor of the shape
            (C, output_height, output_width) that contains per-pixel soft predictions.
    """
    result = result[:, : img_size[0], : img_size[1]].expand(1, -1, -1, -1)
    if mode == "nearest":
        return F.interpolate(result, size=(output_height, output_width), mode=mode)[0]
    return F.interpolate(
        result, size=(output_height, output_width), mode=mode, align_corners=False
    )[0]
