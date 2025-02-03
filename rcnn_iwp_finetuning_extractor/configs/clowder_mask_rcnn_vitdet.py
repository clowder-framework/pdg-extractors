import json
import os

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

def load_model_config():
    """Load model configuration from JSON file"""
    # with open('/home/mohanar2/vit_sandbox/model_config.json', 'r') as f:
    #     config = json.load(f)
    # return config
    
    # Check if environment variable is set
    if 'MODEL_CONFIG_FILE_PATH' in os.environ:
        with open(os.environ['MODEL_CONFIG_FILE_PATH'], 'r') as f:
            config = json.load(f)
        return config
    else:
        raise ValueError("MODEL_CONFIG_FILE_PATH environment variable not set")

# Load configuration from JSON
cfg = load_model_config()
constants = get_config("common/data/constants.py").constants

# Model configuration using values from JSON
model = L(GeneralizedRCNN)(
    backbone=L(SimpleFeaturePyramid)(
        img_size="${.net.img_size}",
        patch_size="${.net.patch_size}",
        net=L(ViT)(
            img_size=cfg['vit']['img_size'],
            patch_size=cfg['vit']['patch_size'],
            embed_dim=cfg['vit']['embed_dim'],
            depth=cfg['vit']['depth'],
            num_heads=cfg['vit']['num_heads'],
            drop_path_rate=cfg['vit']['drop_path_rate'],
            window_size=cfg['vit']['window_size'],
            mlp_ratio=cfg['vit']['mlp_ratio'],
            qkv_bias=cfg['vit']['qkv_bias'],
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            window_block_indexes=cfg['vit']['window_block_indexes'],
            residual_block_indexes=cfg['vit']['residual_block_indexes'],
            out_feature=cfg['vit']['out_feature'],
        ),
        in_feature="${.net.out_feature}",
        out_channels=cfg['backbone']['out_channels'],
        scale_factors=tuple(cfg['backbone']['scale_factors']),
        top_block=L(LastLevelMaxPool)(),
        norm=cfg['backbone']['norm'],
        square_pad=cfg['backbone']['square_pad'],
    ),
    proposal_generator=L(RPN)(
        in_features=cfg['rpn']['in_features'],
        head=L(StandardRPNHead)(
            in_channels=cfg['rpn']['head']['in_channels'],
            num_anchors=cfg['rpn']['head']['num_anchors'],
            conv_dims=cfg['rpn']['head']['conv_dims']
        ),
        anchor_generator=L(DefaultAnchorGenerator)(
            sizes=cfg['rpn']['anchor_generator']['sizes'],
            aspect_ratios=cfg['rpn']['anchor_generator']['aspect_ratios'],
            strides=cfg['rpn']['anchor_generator']['strides'],
            offset=cfg['rpn']['anchor_generator']['offset'],
        ),
        anchor_matcher=L(Matcher)(
            thresholds=cfg['rpn']['matcher']['thresholds'],
            labels=cfg['rpn']['matcher']['labels'],
            allow_low_quality_matches=cfg['rpn']['matcher']['allow_low_quality_matches']
        ),
        box2box_transform=L(Box2BoxTransform)(
            weights=cfg['rpn']['box2box_transform']['weights']
        ),
        batch_size_per_image=cfg['rpn']['batch_size_per_image'],
        positive_fraction=cfg['rpn']['positive_fraction'],
        pre_nms_topk=tuple(cfg['rpn']['pre_nms_topk']),
        post_nms_topk=tuple(cfg['rpn']['post_nms_topk']),
        nms_thresh=cfg['rpn']['nms_thresh'],
    ),
    roi_heads=L(StandardROIHeads)(
        num_classes=cfg['roi_heads']['num_classes'],
        batch_size_per_image=cfg['roi_heads']['batch_size_per_image'],
        positive_fraction=cfg['roi_heads']['positive_fraction'],
        proposal_matcher=L(Matcher)(
            thresholds=cfg['roi_heads']['proposal_matcher']['thresholds'],
            labels=cfg['roi_heads']['proposal_matcher']['labels'],
            allow_low_quality_matches=cfg['roi_heads']['proposal_matcher']['allow_low_quality_matches']
        ),
        box_in_features=cfg['roi_heads']['box_in_features'],
        box_pooler=L(ROIPooler)(
            output_size=cfg['roi_heads']['box_pooler']['output_size'],
            scales=tuple(cfg['roi_heads']['box_pooler']['scales']),
            sampling_ratio=cfg['roi_heads']['box_pooler']['sampling_ratio'],
            pooler_type=cfg['roi_heads']['box_pooler']['pooler_type'],
        ),
        box_head=L(FastRCNNConvFCHead)(
            input_shape=ShapeSpec(
                channels=cfg['roi_heads']['box_head']['input_shape']['channels'],
                height=cfg['roi_heads']['box_head']['input_shape']['height'],
                width=cfg['roi_heads']['box_head']['input_shape']['width']
            ),
            conv_dims=cfg['roi_heads']['box_head']['conv_dims'],
            fc_dims=cfg['roi_heads']['box_head']['fc_dims'],
            conv_norm=cfg['roi_heads']['box_head']['conv_norm'],
        ),
        box_predictor=L(FastRCNNOutputLayers)(
            input_shape=ShapeSpec(channels=cfg['roi_heads']['box_predictor']['input_shape']['channels']),
            test_score_thresh=cfg['roi_heads']['box_predictor']['test_score_thresh'],
            box2box_transform=L(Box2BoxTransform)(weights=cfg['roi_heads']['box_predictor']['box2box_weights']),
            num_classes="${..num_classes}",
        ),
        mask_in_features=cfg['roi_heads']['mask_in_features'],
        mask_pooler=L(ROIPooler)(
            output_size=cfg['roi_heads']['mask_pooler']['output_size'],
            scales=tuple(cfg['roi_heads']['mask_pooler']['scales']),
            sampling_ratio=cfg['roi_heads']['mask_pooler']['sampling_ratio'],
            pooler_type=cfg['roi_heads']['mask_pooler']['pooler_type'],
        ),
        mask_head=L(MaskRCNNConvUpsampleHead)(
            input_shape=ShapeSpec(
                channels=cfg['roi_heads']['mask_head']['input_shape']['channels'],
                width=cfg['roi_heads']['mask_head']['input_shape']['width'],
                height=cfg['roi_heads']['mask_head']['input_shape']['height']
            ),
            num_classes="${..num_classes}",
            conv_dims=cfg['roi_heads']['mask_head']['conv_dims'],
            conv_norm=cfg['roi_heads']['mask_head']['conv_norm'],
        ),
    ),
    pixel_mean=constants.imagenet_rgb256_mean,
    pixel_std=constants.imagenet_rgb256_mean,
    input_format=cfg['input_format'],
)

# Training configuration
train = model_zoo.get_config("common/train.py").train
train.amp.enabled = cfg['train']['amp']['enabled']
train.ddp.fp16_compression = cfg['train']['ddp']['fp16_compression']
train.init_checkpoint = cfg['train']['init_checkpoint']
train.checkpointer.period = cfg['train']['checkpointer_period']
train.eval_period = cfg['train']['eval_period']
train.max_iter = cfg['train']['max_iter']
train.seed = cfg['train']['seed']

# Dataloader configuration
dataloader.train.total_batch_size = cfg['dataloader']['train']['total_batch_size']

# Schedule
lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=cfg['scheduler']['values'],
        milestones=[
            train.max_iter * cfg['scheduler']['milestone_factors'][0],
            train.max_iter * cfg['scheduler']['milestone_factors'][1]
        ],
        num_updates=train.max_iter,
    ),
    warmup_length=250 / train.max_iter,
    warmup_factor=cfg['scheduler']['warmup_factor'],
)

# Optimizer
optimizer = model_zoo.get_config("common/optim.py").AdamW
optimizer.params.lr_factor_func = partial(
    get_vit_lr_decay_rate,
    num_layers=cfg['optimizer']['num_layers'],
    lr_decay_rate=cfg['optimizer']['lr_decay_rate']
)
optimizer.params.overrides = cfg['optimizer']['params_overrides']