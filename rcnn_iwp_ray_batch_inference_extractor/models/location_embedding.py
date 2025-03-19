import torch
from dataclasses import dataclass
from enum import Enum
from huggingface_hub import hf_hub_download
from satclip.load import get_satclip
from torch import nn, Tensor, cat
from typing import Dict, List
from models.utils import CrossAttentionBlock


class SatCLIPConfig(Enum):
    L10 = 1
    L40 = 2


class MergeStrategy(Enum):
    ADD = 1
    CONCAT = 2
    NORM_ADD = 3
    NORM_CONCAT = 4
    CONCAT_NORM = 5
    PROJ_CONCAT = 6
    PROJ_ADD = 7
    CROSS_ATTN = 8


@dataclass
class LocationEmbeddingConfig:
    """
    Properties:
        sat_clip_config: What type of SatCLIP model to use for computing embeddings
        merge_strategy: What strategy to use when merging location embeddings with
        the image embeddings
        pre_pyramid: Whether the location embeddings should be merged with the image
        embeddings before passing through the simple feature pyramid.
    """

    sat_clip_config: SatCLIPConfig
    merge_strategy: MergeStrategy
    pre_pyramid: bool


class LocationEmbedding(nn.Module):
    """
    This module facilitates the use of location embeddings in the context of
    a larger Simple Feature Pyramid (SFP) network structure. It allows computing location
    embeddings for a set of coordinates as well as merging location embeddings with
    other features.
    """

    SAT_CLIP_DIMS = 256

    def __init__(
        self,
        config: LocationEmbeddingConfig,
        img_size: int,
        patch_size: int,
        feature_channels: int,
        scale_factors: List[float],
        pyramid_features: List[str],
    ):
        """
        Args:
            config (LocationEmbeddingConfig): Config paramters for setting up embedding
                model and embedding merge strategy.
            img_size (int): the size of the image associated with the location.
                Used for determening parameters of the merging strategies.
            patch_size (int): patch size used by the associated transformer.
                Used for determening parameters of the merging strategies.
            feature_channels (int): number of channels in the input feature maps.
            scale_factors (list[float]): list of scaling factors used by the associated
                SFP. If the embeddings are being merged before going through the SFP,
                this should have a single entry.
            pyramid_features (list[str]): list of feature names for each stage of the
                SFP. If embeddings are being merged before going through the SFP, this
                should have a single entry.
        """
        super().__init__()
        self.sat_clip_config = config.sat_clip_config
        self.merge_strategy = config.merge_strategy
        self.pre_pyramid = config.pre_pyramid
        if self.pre_pyramid and (
            self.merge_strategy == MergeStrategy.ADD
            or self.merge_strategy == MergeStrategy.NORM_ADD
        ):
            raise ValueError(
                "Merging the locations embeddings before the SFP is not compatible "
                "with the ADD or NORM_ADD merging strategies."
            )
        self.feature_names = pyramid_features
        self.location_embedding = self._build_location_embedding()
        self.embedding_transform = self._build_transform(
            img_size, patch_size, scale_factors, feature_channels
        )
        self.strategy_funcs = {
            MergeStrategy.ADD: self._add_embeddings,
            MergeStrategy.NORM_ADD: self._norm_add_embeddings,
            MergeStrategy.CONCAT: self._concat_embeddings,
            MergeStrategy.CONCAT_NORM: self._concat_norm_embeddings,
            MergeStrategy.NORM_CONCAT: self._norm_concat_embeddings,
            MergeStrategy.PROJ_ADD: self._proj_add_embeddings,
            MergeStrategy.PROJ_CONCAT: self._proj_concat_embeddings,
            MergeStrategy.CROSS_ATTN: self._cross_attn_embeddings,
        }

    @property
    def device(self):
        return (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

    @staticmethod
    def additional_channels(merge_strategy: MergeStrategy):
        """
        Returns the number of additional channels produced by the given strategy
        """
        match merge_strategy:
            case (
                MergeStrategy.CONCAT
                | MergeStrategy.NORM_CONCAT
                | MergeStrategy.CONCAT_NORM
            ):
                return 256
            case MergeStrategy.PROJ_CONCAT:
                return 1
            case _:
                return 0

    def compute_embeddings(self, coordinates: Tensor):
        """
        Computes the location embddings of the input coordinates
        """
        return self.location_embedding(coordinates)

    def merge_embeddings(self, features: Dict[str, Tensor], coordinates: torch.Tensor):
        """
        Merges the provided features with the location embeddings of the produced
        by the input coordinates.
        """
        embeddings = self.compute_embeddings(coordinates)
        # The rest of the model expects to work with torch.float32 so we convert
        # the output. This does result in a loss of information and the way around
        # it would be to train a location encoder natively on torch.float32.
        embeddings = embeddings.float()
        assert (
            list(features.keys()) == self.feature_names
        ), f"Input features don't match those used when constructing the class {features.keys()} vs {self.feature_names}"
        # Apply the correct merging strategy
        merged_features = self.strategy_funcs[self.merge_strategy](features, embeddings)
        return merged_features

    def _build_location_embedding(self):
        satclip = None
        if self.sat_clip_config == SatCLIPConfig.L10:
            satclip = get_satclip(
                hf_hub_download(
                    "microsoft/SatCLIP-ViT16-L10", "satclip-vit16-l10.ckpt"
                ),
                device=self.device,
            )  # Only loads location encoder by default
        elif self.sat_clip_config == SatCLIPConfig.L40:
            satclip = get_satclip(
                hf_hub_download(
                    "microsoft/SatCLIP-ViT16-L40", "satclip-vit16-l40.ckpt"
                ),
                device=self.device,
            )  # Only loads location encoder by default

        # Freeze the layers in the location encoder as we don't intend on fine tuning
        # the weights.
        for param in satclip.parameters():
            param.requires_grad = False

        return satclip

    def _build_transform(
        self,
        img_size: int,
        patch_size: int,
        scale_factors: List[float],
        feature_channels: int,
    ):
        image_embedding_shape = img_size // patch_size
        image_embedding_shapes = [
            int(scale * image_embedding_shape) for scale in scale_factors
        ]

        transforms = {}
        match self.merge_strategy:
            case MergeStrategy.PROJ_ADD | MergeStrategy.PROJ_CONCAT:
                for feature, shape in zip(
                    self.feature_names,
                    image_embedding_shapes,
                ):
                    transforms[feature] = nn.Sequential(
                        nn.Linear(
                            self.SAT_CLIP_DIMS, shape**2, bias=False, device=self.device
                        ),
                        nn.LayerNorm(shape**2, device=self.device),
                    )
                return transforms
            case MergeStrategy.NORM_CONCAT | MergeStrategy.NORM_ADD:
                for feature, shape in zip(
                    self.feature_names,
                    image_embedding_shapes,
                ):
                    transforms[feature] = nn.LayerNorm(
                        [self.SAT_CLIP_DIMS, shape, shape], device=self.device
                    )
                return transforms
            case MergeStrategy.CONCAT_NORM:
                for feature, shape in zip(
                    self.feature_names,
                    image_embedding_shapes,
                ):
                    transforms[feature] = nn.LayerNorm(
                        [self.SAT_CLIP_DIMS + feature_channels, shape, shape],
                        device=self.device,
                    )
                return transforms
            case MergeStrategy.CROSS_ATTN:
                for feature, shape in zip(
                    self.feature_names,
                    image_embedding_shapes,
                ):
                    transforms[feature] = CrossAttentionBlock(
                        embed_dim=512,
                        dim_source=feature_channels,
                        dim_context=self.SAT_CLIP_DIMS,
                        num_heads=8,
                        device=self.device,
                    )
                return transforms
            case _:
                return None

    def _add_embeddings(self, features: Dict[str, Tensor], location_embeddings: Tensor):
        # Add the location embeddings to each feature maps of the outputted
        # backbone feature pyramid.
        for name, feature_map in features.items():
            # This produces a location embedding with shape (Hf, Wf, B, embed)
            # where Hf and Wf are the height and width of the feature map respectively,
            # B is the batch size and embed is the location embedding dimensions.
            expanded_embedding = location_embeddings.repeat(
                feature_map.shape[2], feature_map.shape[3], 1, 1
            )
            # (Hf, Wf, B, embed) -> (B, embed, Hf, Wf) this enables us to
            # concatenate the location embeddings with the image embeddings.
            expanded_embedding = expanded_embedding.permute(2, 3, 0, 1)
            features[name] = feature_map.add(expanded_embedding)
        return features

    def _norm_add_embeddings(
        self, features: Dict[str, Tensor], location_embeddings: Tensor
    ):
        # Add the normalized location embeddings to each feature map of the outputted
        # backbone feature pyramid.
        for name, feature_map in features.items():
            # This produces a location embedding with shape (Hf, Wf, B, embed)
            # where Hf and Wf are the height and width of the feature map respectively,
            # B is the batch size and embed is the location embedding dimensions.
            expanded_embedding = location_embeddings.repeat(
                feature_map.shape[2], feature_map.shape[3], 1, 1
            )
            # (Hf, Wf, B, embed) -> (B, embed, Hf, Wf) this enables us to
            # concatenate the location embeddings with the image embeddings.
            expanded_embedding = expanded_embedding.permute(2, 3, 0, 1)
            expanded_embedding = self.embedding_transform[name](expanded_embedding)
            features[name] = feature_map.add(expanded_embedding)
        return features

    def _concat_embeddings(
        self, features: Dict[str, Tensor], location_embeddings: Tensor
    ):
        # Concatenate the location embeddings to each feature maps of the outputted
        # backbone feature pyramid.
        for name, feature_map in features.items():
            # This produces a location embedding with shape (Hf, Wf, B, embed)
            # where Hf and Wf are the height and width of the feature map respectively,
            # B is the batch size and embed is the location embedding dimensions.
            expanded_embedding = location_embeddings.repeat(
                feature_map.shape[2], feature_map.shape[3], 1, 1
            )
            # (Hf, Wf, B, embed) -> (B, embed, Hf, Wf) this enables us to
            # concatenate the location embeddings with the image embeddings.
            expanded_embedding = expanded_embedding.permute(2, 3, 0, 1)
            features[name] = cat((feature_map, expanded_embedding), dim=1)
        return features

    def _concat_norm_embeddings(
        self, features: Dict[str, Tensor], location_embeddings: Tensor
    ):
        # Concatenate the location embeddings to each feature maps of the outputted
        # backbone feature pyramid.
        for name, feature_map in features.items():
            # This produces a location embedding with shape (Hf, Wf, B, embed)
            # where Hf and Wf are the height and width of the feature map respectively,
            # B is the batch size and embed is the location embedding dimensions.
            expanded_embedding = location_embeddings.repeat(
                feature_map.shape[2], feature_map.shape[3], 1, 1
            )
            # (Hf, Wf, B, embed) -> (B, embed, Hf, Wf) this enables us to
            # concatenate the location embeddings with the image embeddings.
            expanded_embedding = expanded_embedding.permute(2, 3, 0, 1)
            combined_embeddings = cat((feature_map, expanded_embedding), dim=1)
            features[name] = self.embedding_transform[name](combined_embeddings)
        return features

    def _norm_concat_embeddings(
        self, features: Dict[str, Tensor], location_embeddings: Tensor
    ):
        # Concatenate the normalized location embeddings to each feature map of the
        # outputted backbone feature pyramid.
        for name, feature_map in features.items():
            # This produces a location embedding with shape (Hf, Wf, B, embed)
            # where Hf and Wf are the height and width of the feature map respectively,
            # B is the batch size and embed is the location embedding dimensions.
            expanded_embedding = location_embeddings.repeat(
                feature_map.shape[2], feature_map.shape[3], 1, 1
            )
            # (Hf, Wf, B, embed) -> (B, embed, Hf, Wf) this enables us to
            # concatenate the location embeddings with the image embeddings.
            expanded_embedding = expanded_embedding.permute(2, 3, 0, 1)
            expanded_embedding = self.embedding_transform[name](expanded_embedding)
            features[name] = cat((feature_map, expanded_embedding), dim=1)
        return features

    def _proj_concat_embeddings(
        self, features: Dict[str, Tensor], location_embeddings: Tensor
    ):
        # Concatenate the projected location embeddings to each feature map of the
        # outputted backbone feature pyramid.
        for name, feature_map in features.items():
            B, _, H, W = feature_map.shape
            projected_embedding = self.embedding_transform[name](location_embeddings)
            projected_embedding = projected_embedding.reshape(B, 1, H, W)
            features[name] = cat((feature_map, projected_embedding), dim=1)
        return features

    def _proj_add_embeddings(
        self, features: Dict[str, Tensor], location_embeddings: Tensor
    ):
        # Add the projected location embeddings to each feature map of the outputted
        # backbone feature pyramid.
        for name, feature_map in features.items():
            B, _, H, W = feature_map.shape
            projected_embedding = self.embedding_transform[name](location_embeddings)
            projected_embedding = projected_embedding.reshape(B, 1, H, W)
            features[name] = feature_map.add(projected_embedding)
        return features

    def _cross_attn_embeddings(
        self, features: Dict[str, Tensor], location_embeddings: Tensor
    ):
        # New added dimension represents the length of the location embedding
        # sequence.
        location_embeddings = location_embeddings.unsqueeze(1)
        for name, feature_map in features.items():
            B, C, H, W = feature_map.shape
            # Change feature_map shape from (B, C, H, W) -> (B, H, W, C) -> (B, H*W, C)
            feature_map = feature_map.permute(0, 2, 3, 1).view(B, H * W, C)
            merged_embedding = self.embedding_transform[name](
                feature_map, location_embeddings
            )
            # Change merged embeddings from (B, H*W, C) -> (B, H, W, C) -> (B, C, H, W)
            merged_embedding = merged_embedding.view(B, H, W, C).permute(0, 3, 1, 2)
            features[name] = merged_embedding
        return features
