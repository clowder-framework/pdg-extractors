from detectron2.structures import Boxes, Instances
from typing import Any, Dict, List
import numpy as np
import torch
from segment_anything.modeling import Sam
from detectron2.structures.masks import polygons_to_bitmask


class CustomSam(Sam):
    def __init__(self, input_format="RGB", *args, **kwargs):
        self.input_format = input_format
        super().__init__(*args, **kwargs)

    """
    custom SAM model for training purposes
    add loss function to the forward method
    image_encoder and prompt_encoder are not trained in default (with torch.no_grad())
    """

    def dice_loss(self, pred, target):
        smooth = 1e-6
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()

        return 1 - (2 * intersection + smooth) / (union + smooth)

    def forward(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool = False,
        required_grad: bool = False,
    ) -> List[Dict[str, torch.Tensor]]:
        # move batched_input to device
        batched_input = [
            {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in x.items()
            }
            for x in batched_input
        ]
        input_images = torch.stack(
            [self.preprocess(x["image"]) for x in batched_input], dim=0
        )
        with torch.no_grad():
            image_embeddings = self.image_encoder(input_images)

        outputs = []
        for image_record, curr_embedding in zip(batched_input, image_embeddings):

            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None
            with torch.no_grad():
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=points,
                    boxes=image_record.get("boxes", None),
                    masks=image_record.get("mask_inputs", None),
                )

            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )

            masks = self.postprocess_masks(
                low_res_masks,
                input_size=image_record["image"].shape[-2:],
                original_size=image_record["original_size"],
            )
            # TODO: remove this

            if self.training:
                m = torch.nn.Sigmoid()
                masks = m(masks)
            else:
                masks = masks > self.mask_threshold

            outputs.append(
                {
                    "masks": masks,
                    "iou_predictions": iou_predictions,
                    "low_res_logits": low_res_masks,
                }
            )

        if not self.training:
            # transferm to standard detectron2 model output format for evaluation
            # the default batch_size for inference is 1
            transformed_outputs = []
            transformed_outputs.append(
                {
                    "instances": Instances(
                        image_size=batched_input[0]["instances"].image_size
                    )
                }
            )
            transformed_outputs[0]["instances"].set(
                "pred_masks", outputs[0]["masks"].squeeze(1)
            )
            # resize boxes to the original size
            rh, rw = (
                batched_input[0]["height"]
                / batched_input[0]["instances"].image_size[0],
                batched_input[0]["width"] / batched_input[0]["instances"].image_size[1],
            )
            gt_boxes = batched_input[0]["instances"].gt_boxes.tensor
            gt_boxes = gt_boxes * torch.tensor([rw, rh, rw, rh], device=gt_boxes.device)

            transformed_outputs[0]["instances"].set("pred_boxes", Boxes(gt_boxes))
            transformed_outputs[0]["instances"].set(
                "pred_classes", batched_input[0]["instances"].gt_classes
            )
            transformed_outputs[0]["instances"].set(
                "scores", outputs[0]["iou_predictions"].squeeze(1)
            )

            return transformed_outputs

        # return loss if training
        # calculate loss for training
        loss_dict = torch.zeros(1, requires_grad=True).to(self.device)
        for image_record, output in zip(batched_input, outputs):
            pred_masks = output["masks"]
            pred_masks = pred_masks.squeeze(1)
            gt_masks = image_record["instances"].gt_masks.polygons
            gt_masks = [
                polygons_to_bitmask(
                    p,
                    image_record["instances"].image_size[0],
                    image_record["instances"].image_size[1],
                )
                for p in gt_masks
            ]
            gt_masks = np.stack(gt_masks, axis=0)
            gt_masks = torch.as_tensor(
                gt_masks, dtype=torch.float32, device=pred_masks.device
            )
            # resize gt_masks to the same size as pred_masks
            gt_masks = torch.nn.functional.interpolate(
                gt_masks.unsqueeze(0), size=pred_masks.shape[-2:], mode="nearest"
            ).squeeze(0)
            loss_dict += self.dice_loss(pred_masks, gt_masks)

        loss_dict /= len(batched_input)

        return loss_dict
