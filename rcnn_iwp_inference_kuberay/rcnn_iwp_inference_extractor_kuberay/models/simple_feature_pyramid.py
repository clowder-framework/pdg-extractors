import math
import torch
import torch.nn as nn

from models.location_embedding import LocationEmbedding, LocationEmbeddingConfig
from detectron2.layers import Conv2d, get_norm
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.backbone.fpn import _assert_strides_are_log2_contiguous
from typing import List, Dict


class SimpleFeaturePyramid(Backbone):
    """
    This module implements SimpleFeaturePyramid in :paper:`vitdet`.
    It creates pyramid features built on top of the input feature map.
    """

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        net: Backbone,
        in_feature: str,
        out_channels: int,
        scale_factors: List[float],
        top_block: nn.Module = None,
        norm: str = "LN",
        square_pad: int = 0,
        location_embedding_config: LocationEmbedding = None,
        freeze_backbone=True,
    ):
        """
        Args:
            net (Backbone): module representing the subnetwork backbone.
                Must be a subclass of :class:`Backbone`.
            in_feature (str): names of the input feature maps coming
                from the net.
            out_channels (int): number of channels in the output feature maps.
            scale_factors (list[float]): list of scaling factors to upsample or downsample
                the input features for creating pyramid features.
            top_block (nn.Module or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                pyramid output, and the result will extend the result list. The top_block
                further downsamples the feature map. It must have an attribute
                "num_levels", meaning the number of extra pyramid levels added by
                this block, and "in_feature", which is a string representing
                its input feature (e.g., p5).
            norm (str): the normalization to use.
            square_pad (int): If > 0, require input images to be padded to specific square size.
        """
        super(SimpleFeaturePyramid, self).__init__()
        assert isinstance(net, Backbone)

        self.scale_factors = scale_factors

        input_shapes = net.output_shape()
        strides = [
            int(input_shapes[in_feature].stride / scale) for scale in scale_factors
        ]
        _assert_strides_are_log2_contiguous(strides)

        dim = input_shapes[in_feature].channels
        if location_embedding_config and location_embedding_config.pre_pyramid:
            dim += LocationEmbedding.additional_channels(
                location_embedding_config.merge_strategy
            )

        self.stages = []
        use_bias = norm == ""
        for idx, scale in enumerate(scale_factors):
            out_dim = dim
            if scale == 4.0:
                layers = [
                    nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
                    get_norm(norm, dim // 2),
                    nn.GELU(),
                    nn.ConvTranspose2d(dim // 2, dim // 4, kernel_size=2, stride=2),
                ]
                out_dim = dim // 4
            elif scale == 2.0:
                layers = [nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2)]
                out_dim = dim // 2
            elif scale == 1.0:
                layers = []
            elif scale == 0.5:
                layers = [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                raise NotImplementedError(f"scale_factor={scale} is not supported yet.")

            layers.extend(
                [
                    Conv2d(
                        out_dim,
                        out_channels,
                        kernel_size=1,
                        bias=use_bias,
                        norm=get_norm(norm, out_channels),
                    ),
                    Conv2d(
                        out_channels,
                        out_channels,
                        kernel_size=3,
                        padding=1,
                        bias=use_bias,
                        norm=get_norm(norm, out_channels),
                    ),
                ]
            )
            layers = nn.Sequential(*layers)

            stage = int(math.log2(strides[idx]))
            self.add_module(f"simfp_{stage}", layers)
            self.stages.append(layers)

        self.net = net
        if freeze_backbone:
            # Freeze backbone weights
            for param in self.net.parameters():
                param.requires_grad = False

        self.in_feature = in_feature
        self.top_block = top_block
        # Return feature names are "p<stage>", like ["p2", "p3", ..., "p6"]
        self._out_feature_strides = {
            "p{}".format(int(math.log2(s))): s for s in strides
        }
        # top block output feature maps.
        if self.top_block is not None:
            for s in range(stage, stage + self.top_block.num_levels):
                self._out_feature_strides["p{}".format(s + 1)] = 2 ** (s + 1)

        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {k: out_channels for k in self._out_features}
        self._size_divisibility = strides[-1]
        self._square_pad = square_pad
        self.location_embedding = None
        if location_embedding_config:
            self.location_embedding = self._get_location_embedding(
                location_embedding_config,
                img_size,
                patch_size,
                (
                    input_shapes[in_feature].channels
                    if location_embedding_config.pre_pyramid
                    else out_channels
                ),
                scale_factors,
            )

    @property
    def padding_constraints(self):
        return {
            "size_divisiblity": self._size_divisibility,
            "square_size": self._square_pad,
        }

    @property
    def device(self):
        return (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

    def forward(self, x, coordinates=None):
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.
            coordinates: Tensor of shape (N, 2) representing the coordinates of
            each image in the batch.

        Returns:
            dict[str->Tensor]:
                mapping from feature map name to pyramid feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["p2", "p3", ..., "p6"].
        """
        if self.location_embedding and coordinates is None:
            raise ValueError(
                "Model has location embedding module configured but a coordinate tensor"
                "was not provided to the forward() call."
            )

        bottom_up_features = self.net(x)
        if self.location_embedding and self.location_embedding.pre_pyramid:
            merged_features = self.location_embedding.merge_embeddings(
                bottom_up_features, coordinates
            )
            features = merged_features[self.in_feature]
        else:
            features = bottom_up_features[self.in_feature]
        results = []

        for stage in self.stages:
            results.append(stage(features))

        if self.top_block is not None:
            if self.top_block.in_feature in bottom_up_features:
                top_block_in_feature = bottom_up_features[self.top_block.in_feature]
            else:
                top_block_in_feature = results[
                    self._out_features.index(self.top_block.in_feature)
                ]
            results.extend(self.top_block(top_block_in_feature))
        assert len(self._out_features) == len(results)
        out_features = {f: res for f, res in zip(self._out_features, results)}

        if self.location_embedding and not self.location_embedding.pre_pyramid:
            return self.location_embedding.merge_embeddings(out_features, coordinates)

        return out_features

    def _get_location_embedding(
        self,
        config: LocationEmbeddingConfig,
        img_size: int,
        patch_size: int,
        feature_channels: int,
        scale_factors: List[float],
    ) -> LocationEmbedding:
        if config.pre_pyramid:
            return LocationEmbedding(
                config,
                img_size,
                patch_size,
                feature_channels,
                [1.0],
                [self.in_feature],
            )

        if self.top_block:
            scale_factors.append(0.25)
        return LocationEmbedding(
            config,
            img_size,
            patch_size,
            feature_channels,
            scale_factors,
            self._out_features,
        )
