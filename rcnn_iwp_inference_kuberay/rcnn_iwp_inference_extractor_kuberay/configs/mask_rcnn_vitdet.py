import torch.nn as nn

from configs.dataloaders.coco_loader import dataloader
from functools import partial
from detectron2 import model_zoo
from detectron2.config import LazyCall as L
from detectron2.layers import ShapeSpec
from detectron2.modeling import ViT
from detectron2.modeling.anchor_generator import DefaultAnchorGenerator
from detectron2.modeling.backbone.fpn import LastLevelMaxPool
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.proposal_generator import RPN, StandardRPNHead
from detectron2.modeling.roi_heads import (
    StandardROIHeads,
    FastRCNNOutputLayers,
    MaskRCNNConvUpsampleHead,
    FastRCNNConvFCHead,
)
from detectron2.modeling.backbone.vit import get_vit_lr_decay_rate
from detectron2.solver import WarmupParamScheduler
from fvcore.common.param_scheduler import MultiStepParamScheduler
from detectron2.model_zoo import get_config
from models.simple_feature_pyramid import SimpleFeaturePyramid
from models.rcnn import GeneralizedRCNN

# Base
embed_dim, depth, num_heads, dp = 768, 12, 12, 0.1
constants = get_config("common/data/constants.py").constants

model = L(GeneralizedRCNN)(
    backbone=L(SimpleFeaturePyramid)(
        img_size="${.net.img_size}",
        patch_size="${.net.patch_size}",
        net=L(ViT)(  # Single-scale ViT backbone
            img_size=1024,
            patch_size=16,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            drop_path_rate=dp,
            window_size=14,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            window_block_indexes=[
                # 2, 5, 8 11 for global attention
                0,
                1,
                3,
                4,
                6,
                7,
                9,
                10,
            ],
            residual_block_indexes=[],
            out_feature="last_feat",
        ),
        in_feature="${.net.out_feature}",
        out_channels=256,
        scale_factors=(4.0, 2.0, 1.0, 0.5),
        top_block=L(LastLevelMaxPool)(),
        norm="LN",
        square_pad=1024,
    ),
    proposal_generator=L(RPN)(
        in_features=["p2", "p3", "p4", "p5", "p6"],
        head=L(StandardRPNHead)(in_channels=256, num_anchors=3, conv_dims=[-1, -1]),
        anchor_generator=L(DefaultAnchorGenerator)(
            sizes=[[32], [64], [128], [256], [512]],
            aspect_ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64],
            offset=0.0,
        ),
        anchor_matcher=L(Matcher)(
            thresholds=[0.3, 0.7], labels=[0, -1, 1], allow_low_quality_matches=True
        ),
        box2box_transform=L(Box2BoxTransform)(weights=[1.0, 1.0, 1.0, 1.0]),
        batch_size_per_image=256,
        positive_fraction=0.5,
        pre_nms_topk=(2000, 1000),
        post_nms_topk=(1000, 1000),
        nms_thresh=0.7,
    ),
    roi_heads=L(StandardROIHeads)(
        num_classes=80,
        batch_size_per_image=512,
        positive_fraction=0.25,
        proposal_matcher=L(Matcher)(
            thresholds=[0.5], labels=[0, 1], allow_low_quality_matches=False
        ),
        box_in_features=["p2", "p3", "p4", "p5"],
        box_pooler=L(ROIPooler)(
            output_size=7,
            scales=(1.0 / 4, 1.0 / 8, 1.0 / 16, 1.0 / 32),
            sampling_ratio=0,
            pooler_type="ROIAlignV2",
        ),
        box_head=L(FastRCNNConvFCHead)(
            input_shape=ShapeSpec(channels=256, height=7, width=7),
            conv_dims=[256, 256, 256, 256],
            fc_dims=[1024],
            conv_norm="LN",
        ),
        box_predictor=L(FastRCNNOutputLayers)(
            input_shape=ShapeSpec(channels=1024),
            test_score_thresh=0.05,
            box2box_transform=L(Box2BoxTransform)(weights=(10, 10, 5, 5)),
            num_classes="${..num_classes}",
        ),
        mask_in_features=["p2", "p3", "p4", "p5"],
        mask_pooler=L(ROIPooler)(
            output_size=14,
            scales=(1.0 / 4, 1.0 / 8, 1.0 / 16, 1.0 / 32),
            sampling_ratio=0,
            pooler_type="ROIAlignV2",
        ),
        mask_head=L(MaskRCNNConvUpsampleHead)(
            input_shape=ShapeSpec(channels=256, width=14, height=14),
            num_classes="${..num_classes}",
            conv_dims=[256, 256, 256, 256, 256],
            conv_norm="LN",
        ),
    ),
    pixel_mean=constants.imagenet_rgb256_mean,
    pixel_std=constants.imagenet_rgb256_mean,
    input_format="RGB",
)

# Initialization and trainer settings
train = model_zoo.get_config("common/train.py").train
train.amp.enabled = True
train.ddp.fp16_compression = True
train.init_checkpoint = "detectron2://ImageNetPretrained/MAE/mae_pretrain_vit_base.pth?matching_heuristics=True"
train.checkpointer.period = 1000
train.eval_period = 500
train.seed = 1129

# This should be updated on a per dataset basis depending on how many images can
# fit in a SINGLE GPU at once.
dataloader.train.total_batch_size = 1

# Schedule
# This should be updated on a per dataset basis depending on how large your
# dataset is and how many epochs you want to train for.
train.max_iter = 1005
lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1, 0.05],
        milestones=[train.max_iter * 0.889, train.max_iter * 0.963],
        num_updates=train.max_iter,
    ),
    warmup_length=250 / train.max_iter,
    warmup_factor=0.001,
)

# Optimizer
optimizer = model_zoo.get_config("common/optim.py").AdamW
optimizer.params.lr_factor_func = partial(
    get_vit_lr_decay_rate, num_layers=12, lr_decay_rate=0.7
)
optimizer.params.overrides = {"pos_embed": {"weight_decay": 0.0}}