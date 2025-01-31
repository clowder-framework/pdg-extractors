#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Training script using the new "LazyConfig" python config files.

This scripts reads a given python config file and runs the training or evaluation.
It can be used to train any models or dataset as long as they can be
instantiated by the recursive construction defined in the given config file.

Besides lazy construction of models, dataloader, etc., this scripts expects a
few common configuration parameters currently defined in "configs/common/train.py".
To add more complicated training logic, you can easily add other configs
in the config file and implement a new train_net.py to handle them.
"""

import logging
import datetime
import rasterio

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
from detectron2.data.datasets.coco import register_coco_instances
from detectron2.engine import (
    AMPTrainer,
    SimpleTrainer,
    default_argument_parser,
    default_setup,
    default_writers,
    hooks,
    launch,
)
from detectron2.engine.defaults import create_ddp_model
from detectron2.evaluation import inference_on_dataset, print_csv_format
from detectron2.utils import comm
from omegaconf import OmegaConf
from PIL import Image, UnidentifiedImageError

from utils.custom_hooks import (
    InstanceSegFigureHook,
    SemSegFigureHook,
    AdditionalEvalHook,
)
from models.location_embedding import LocationEmbedding

import warnings

# Prevents excesive warnings coming from detectron2 package regarding their
# code using deprecated functions.
warnings.filterwarnings("ignore")

logger = logging.getLogger("detectron2")

# Global Constants
INFRASTRUCTURE_STUFF_CLASSES = ["background", "building", "road"]
INFRASTRUCTURE_STUFF_MXR_CLASSES = ["background", "building", "road", "tanks"]
RTS_STUFF_CLASSES = ["background", "rts"]


def do_test(cfg, model):
    if "evaluator" in cfg.dataloader:
        ret = inference_on_dataset(
            model,
            instantiate(cfg.dataloader.test),
            instantiate(cfg.dataloader.evaluator),
        )
        print_csv_format(ret)
        return ret


def do_train(args, cfg):
    """
    Args:
        cfg: an object with the following attributes:
            model: instantiate to a module
            dataloader.{train,test}: instantiate to dataloaders
            dataloader.evaluator: instantiate to evaluator for test set
            optimizer: instantaite to an optimizer
            lr_multiplier: instantiate to a fvcore scheduler
            train: other misc config defined in `configs/common/train.py`, including:
                output_dir (str)
                init_checkpoint (str)
                amp.enabled (bool)
                max_iter (int)
                eval_period, log_period (int)
                device (str)
                checkpointer (dict)
                ddp (dict)
    """
    model = instantiate(cfg.model)

    logger = logging.getLogger("detectron2")
    logger.info("Model:\n{}".format(model))
    model.to(cfg.train.device)

    cfg.optimizer.params.model = model
    optim = instantiate(cfg.optimizer)

    train_loader = instantiate(cfg.dataloader.train)

    model = create_ddp_model(model, **cfg.train.ddp)
    trainer = (AMPTrainer if cfg.train.amp.enabled else SimpleTrainer)(
        model, train_loader, optim
    )

    checkpointer = DetectionCheckpointer(
        model,
        cfg.train.output_dir,
        trainer=trainer,
    )
    trainer.register_hooks(
        [
            hooks.IterationTimer(),
            hooks.LRScheduler(scheduler=instantiate(cfg.lr_multiplier)),
            (
                hooks.PeriodicCheckpointer(checkpointer, **cfg.train.checkpointer)
                if comm.is_main_process()
                else None
            ),
            hooks.EvalHook(cfg.train.eval_period, lambda: do_test(cfg, model)),
            (
                AdditionalEvalHook(
                    cfg.train.eval_period,
                    model,
                    instantiate(cfg.dataloader.test),
                    args['compute_f1'],
                )
                if comm.is_main_process()
                else None
            ),
            (
                hooks.PeriodicWriter(
                    default_writers(cfg.train.output_dir, cfg.train.max_iter),
                    period=cfg.train.log_period,
                )
                if comm.is_main_process()
                else None
            ),
            (get_figure_hook(args, cfg) if comm.is_main_process() else None),
        ]
    )

    checkpointer.resume_or_load(cfg.train.init_checkpoint, resume=args['resume'])
    if args['resume'] and checkpointer.has_checkpoint():
        # The checkpoint stores the training iteration that just finished, thus we start
        # at the next iteration
        start_iter = trainer.iter + 1
    else:
        start_iter = 0
    trainer.train(start_iter, cfg.train.max_iter)


def get_figure_hook(args, cfg):
    """
    Determines what hook should be attached to the trainer for generating metric
    figures.
    """
    if (
        (args['dataset'] == "iwp")
        or (args['dataset'] == "iwp_multi")
        or (args['dataset'] == "iwp_mxr_multi")
        or (args['dataset'] == "iwp_mxr")
    ):
        return InstanceSegFigureHook(cfg.train.output_dir, cfg.train.eval_period)
    elif args['dataset'] == "infrastructure":
        return SemSegFigureHook(
            cfg.train.output_dir, cfg.train.eval_period, INFRASTRUCTURE_STUFF_CLASSES
        )
    elif args['dataset'] == "infrastructure_mxr":
        return SemSegFigureHook(
            cfg.train.output_dir,
            cfg.train.eval_period,
            INFRASTRUCTURE_STUFF_MXR_CLASSES,
        )
    elif args['dataset'] == "rts" or args['dataset'] == "rts_mxr":
        return SemSegFigureHook(
            cfg.train.output_dir, cfg.train.eval_period, RTS_STUFF_CLASSES
        )
    else:
        raise ValueError(f"{args['dataset']} is not a valid dataset.")


def custom_cfg(cfg, args):
    """
    Custom logic to setup the config and the environment.
    """
    # merge dataset specific config
    dataset_cfg = LazyConfig.load("configs/datasets/{}.py".format(args['dataset']))
    cfg = OmegaConf.merge(cfg, dataset_cfg)

    # set output directory
    args['model'] = args['config_file'].split("/")[-1].split(".")[0]
    cfg.train.output_dir = "results/{}/{}".format( args['model'], args['dataset'])
    cfg.dataloader.evaluator.output_dir = cfg.train.output_dir

    # set the training config based on the number of gpus
    cfg.dataloader.train.total_batch_size = (
        cfg.dataloader.train.total_batch_size * args['num_gpus']
    )
    cfg.optimizer.lr = (
        cfg.optimizer.lr / 8 * args['num_gpus']
    )  # linear scaling rule, 8 gpus in the pre-training phase

    cfg.model.backbone.freeze_backbone = args['vit_freeze']
    if args['vit_freeze']:
        # If the backbone is frozen we shouldn't be using any dropout
        # on those layers.
        cfg.model.backbone.net.drop_path_rate = 0.0

    # If location embeddings are part of the model, we need to update
    # the shape of the input into the downstream heads.
    if "location_embedding_config" in cfg.model.backbone.keys():
        embedding_config = cfg.model.backbone.location_embedding_config
        # Downstream head channels only need to be updated if the location embeddings
        # are merged with the image embeddings after passing through the SFP.
        if not embedding_config.pre_pyramid:
            extra_channels = LocationEmbedding.additional_channels(
                embedding_config.merge_strategy
            )
            # Accounts for segmentation tasks
            if "sem_seg_head" in cfg.model.keys():
                for feature in cfg.model.sem_seg_head.input_shape.keys():
                    cfg.model.sem_seg_head.input_shape[
                        feature
                    ].channels += extra_channels
            # Accounts for instance segmentation tasks
            else:
                cfg.model.proposal_generator.head.in_channels += extra_channels
                cfg.model.roi_heads.box_head.input_shape.channels += extra_channels
                cfg.model.roi_heads.mask_head.input_shape.channels += extra_channels

    return cfg


def register_dataset(args):
    """
    Each training dataset to be registered with rel. json
    """
    if args['dataset'] == "iwp_multi":
        for split in ["train", "val"]:
            register_coco_instances(
                f"iwp_multi_{split}",
                {},
                f"data/iwp_multi/iwp_multi_{split}.json",
                "",
            )
    elif args['dataset'] == "iwp":
        for split in ["train", "val"]:
            register_coco_instances(
                f"iwp_{split}",
                {},
                f"data/iwp/iwp_{split}.json",
                "",
            )
    elif args['dataset'] == "iwp_mxr_multi":
        for split in ["train", "val"]:
            register_coco_instances(
                f"iwp_mxr_multi_{split}",
                {},
                f"data/iwp_mxr_multi/iwp_mxr_multi_{split}.json",
                "",
            )
    elif args['dataset'] == "iwp_mxr":
        for split in ["train", "val"]:
            register_coco_instances(
                f"iwp_mxr_{split}",
                {},
                f"data/iwp_mxr/iwp_mxr_{split}.json",
                "",
            )
    elif args['dataset'] == "infrastructure":
        DatasetCatalog.register(
            f"infrastructure_train",
            lambda: register_semantic_dataset(
                "data/infrastructure/train_mask",
                "data/infrastructure/train_img",
                ".TIF",
                ".TIF",
            ),
        )
        MetadataCatalog.get(f"infrastructure_train").set(
            stuff_classes=INFRASTRUCTURE_STUFF_CLASSES, ignore_label=-1
        )
        DatasetCatalog.register(
            f"infrastructure_val",
            lambda: register_semantic_dataset(
                "data/infrastructure/val_mask",
                "data/infrastructure/val_img",
                ".TIF",
                ".TIF",
            ),
        )
        MetadataCatalog.get(f"infrastructure_val").set(
            stuff_classes=INFRASTRUCTURE_STUFF_CLASSES, ignore_label=-1
        )

    elif args['dataset'] == "infrastructure_mxr":
        DatasetCatalog.register(
            f"infrastructure_mxr_train",
            lambda: register_semantic_dataset(
                "data/infrastructure_mxr/train_mask",
                "data/infrastructure_mxr/train_img",
                ".TIF",
                ".TIF",
            ),
        )
        MetadataCatalog.get(f"infrastructure_mxr_train").set(
            stuff_classes=INFRASTRUCTURE_STUFF_MXR_CLASSES, ignore_label=-1
        )
        DatasetCatalog.register(
            f"infrastructure_mxr_val",
            lambda: register_semantic_dataset(
                "data/infrastructure_mxr/val_mask",
                "data/infrastructure_mxr/val_img",
                ".TIF",
                ".TIF",
            ),
        )
        MetadataCatalog.get(f"infrastructure_mxr_val").set(
            stuff_classes=INFRASTRUCTURE_STUFF_MXR_CLASSES, ignore_label=-1
        )
    elif args['dataset'] == "rts":
        DatasetCatalog.register(
            f"rts_train",
            lambda: register_semantic_dataset(
                f"data/rts/train_mask", f"data/rts/train_img", ".tif", ".tif"
            ),
        )
        MetadataCatalog.get(f"rts_train").set(
            stuff_classes=RTS_STUFF_CLASSES, ignore_label=-1
        )
        DatasetCatalog.register(
            f"rts_val",
            lambda: register_semantic_dataset(
                f"data/rts/val_mask", f"data/rts/val_img", ".tif", ".tif"
            ),
        )
        MetadataCatalog.get(f"rts_val").set(
            stuff_classes=RTS_STUFF_CLASSES, ignore_label=-1
        )

    elif args['dataset'] == "rts_mxr":
        DatasetCatalog.register(
            f"rts_mxr_train",
            lambda: register_semantic_dataset(
                f"data/rts_mxr/train_mask", f"data/rts_mxr/train_img", ".tif", ".tif"
            ),
        )
        MetadataCatalog.get(f"rts_mxr_train").set(
            stuff_classes=RTS_STUFF_CLASSES, ignore_label=-1
        )
        DatasetCatalog.register(
            f"rts_mxr_val",
            lambda: register_semantic_dataset(
                f"data/rts_mxr/val_mask", f"data/rts_mxr/val_img", ".tif", ".tif"
            ),
        )
        MetadataCatalog.get(f"rts_mxr_val").set(
            stuff_classes=RTS_STUFF_CLASSES, ignore_label=-1
        )
    else:
        raise ValueError(f"{args['dataset']} is not a valid dataset.")


def register_semantic_dataset(
    mask_root: str, img_root: str, mask_extension: str, img_extension: str
):
    dataset = load_sem_seg(mask_root, img_root, mask_extension, img_extension)
    for count, img_dict in enumerate(dataset):
        file_name = img_dict["file_name"]
        try:
            img = Image.open(file_name)
            img_dict["height"] = img.height
            img_dict["width"] = img.width
        except UnidentifiedImageError:
            # This branch is likely reached because we are trying to read raster
            # data in a format not supported by PIL. Since we are working with
            # Geospatial data it is common to have 16-bit multiband images which
            # are not supported by PIL, in that case we use rasterio.
            with rasterio.open(file_name) as src:
                # Image data is read in [C, H, W] order hence the indexing
                # decisions below.
                img_dict["height"] = src.height
                img_dict["width"] = src.width
        img_dict["image_id"] = count
    return dataset


def main(args):
    cfg = LazyConfig.load(args['config_file'])
    cfg = LazyConfig.apply_overrides(cfg, args['opts'])
    cfg = custom_cfg(cfg, args)
    default_setup(cfg, args)

    register_dataset(args)

    if args['eval_only']:
        model = instantiate(cfg.model)
        model.to(cfg.train.device)
        model = create_ddp_model(model)
        DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
        print(do_test(cfg, model))
    else:
        do_train(args, cfg)


def invoke_main() -> None:
    parser = default_argument_parser()
    parser.add_argument("--dataset", type=str, required=True, help="dataset name")
    parser.add_argument(
        "--compute-f1",
        type=bool,
        help="Whether or not to compute metrics associated with F1 score (recall, precision, F1)"
        "during training. This is currently only supported for semantic segmentation tasks.",
    )
    parser.add_argument(
        "--vit-freeze",
        type=bool,
        default=True,
        help="Whether or not to freeze the vit backbone for trianing "
        "To freeze vit-backbone chnage to True",
    )
    args =  { 
        'config_file':'configs/clowder_mask_rcnn_vitdet.py',
        'resume':False,
        'eval_only':False,
        'num_gpus':1,
        'num_machines':1, 
        'machine_rank':0, 
        'dist_url':'tcp://127.0.0.1:50815', 
        'opts':[], 
        'dataset':'iwp', 
        'compute_f1':None, 
        'vit_freeze':True 
        }
    print(args)
    launch(
        main,
        args['num_gpus'],
        num_machines=args['num_machines'],
        machine_rank=args['machine_rank'],
        dist_url=args['dist_url'],
        args=(args,),
    )
    return None


if __name__ == "__main__":
    invoke_main()  # pragma: no cover
