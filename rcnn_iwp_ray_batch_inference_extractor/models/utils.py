import torch.nn as nn


class CrossAttentionBlock(nn.Module):
    """Transformer block with support for cross attention"""

    def __init__(
        self,
        dim_source,
        dim_context,
        num_heads,
        embed_dim=None,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        device=None,
    ):
        """
        Args:
            dim_source (int): Number of input channels in source sequence.
            dim_context (int): Number of input channels in context sequence.
            num_heads (int): Number of attention heads in each ViT block.
            embed_dim (int): Number of internal embedding dimensions used by attention block.
            If not provided, set to dim_source by default.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            drop_path (float): Stochastic depth rate.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
        """
        super().__init__()
        self.norm1 = norm_layer(dim_source)
        embed_dim = embed_dim if embed_dim is not None else dim_source
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads=num_heads, batch_first=True
        )
        self.query = nn.Linear(dim_source, embed_dim, bias=qkv_bias)
        self.key = nn.Linear(dim_context, embed_dim, bias=qkv_bias)
        self.value = nn.Linear(dim_context, embed_dim, bias=qkv_bias)

        self.back_to_source = None
        if embed_dim != dim_source:
            # We only need this if embed_dim is different from dim_source
            self.back_to_source = nn.Linear(embed_dim, dim_source)

        from timm.models.layers import DropPath, Mlp

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim_source)
        self.mlp = Mlp(
            in_features=dim_source,
            hidden_features=int(dim_source * mlp_ratio),
            act_layer=act_layer,
        )
        if device:
            self.to(device)

    def forward(self, x, context):
        """
        Input tensors x and context should have shape (B, L, C) where B is the
        batch size, L is the sequence length, and C is the channels in each token.
        """
        shortcut = x

        x = self.norm1(x)

        query, key, value = (
            self.query(x),
            self.key(context),
            self.value(context),
        )
        x = self.attn(query, key, value, need_weights=False)[0]

        if self.back_to_source:
            x = self.back_to_source(x)

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x
