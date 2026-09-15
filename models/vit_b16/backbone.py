import os

import numpy as np
import torch
import torch.nn as nn

from third_party.vit_pytorch.models.modeling import VisionTransformer
from third_party.vit_pytorch.models.configs import get_b16_config
from ..transformer_backbone import TransformerBackbone


class ViTB16Backbone(TransformerBackbone):
    """
    Loads the original Google ViT-B/16 ImageNet-21k checkpoint
    into a compatible PyTorch implementation.
    """

    def __init__(self, img_size: int = 224):
        super().__init__()

        config = get_b16_config()

        vit = VisionTransformer(
            config,
            img_size=img_size,
            num_classes=21843,
            zero_head=False,
            vis=False,
        )

        ckpt_path = os.path.join(
            "third_party",
            "vit_pytorch",
            "ckpts",
            "ViT-B_16.npz",
        )

        weights = np.load(ckpt_path)
        vit.load_from(weights)

        # Remove classification head so this acts as a backbone.
        vit.head = nn.Identity()

        self.vit = vit

        self._embed_dim = config.hidden_size

        self.img_size = img_size
        self.patch_size = 16
        self.grid_size = img_size // self.patch_size
        self.num_patches = self.grid_size * self.grid_size

    # ======================================================================
    # TransformerBackbone interface
    # ======================================================================

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def num_layers(self) -> int:
        return len(self.blocks)

    @property
    def num_heads(self) -> int:
        return self.blocks[0].attn.num_attention_heads

    @property
    def head_dim(self) -> int:
        return self.blocks[0].attn.attention_head_size

    @property
    def mlp_dim(self) -> int:
        return self.blocks[0].ffn.fc1.out_features

    def get_patch_coordinates(self) -> torch.Tensor:
        """
        Return spatial patch coordinates in patch-token order.

        Shape:
            [num_patches, 2]
        """
        return torch.stack(
            torch.meshgrid(
                torch.arange(self.grid_size),
                torch.arange(self.grid_size),
                indexing="ij",
            ),
            dim=-1,
        ).reshape(-1, 2)

    def get_attention_output_weight(
        self,
        layer_idx: int,
    ) -> torch.Tensor:
        return self.blocks[layer_idx].attn.out.weight

    def get_mlp_output_weight(
        self,
        layer_idx: int,
    ) -> torch.Tensor:
        return self.blocks[layer_idx].ffn.fc2.weight

    def freeze(self) -> None:
        for parameter in self.vit.parameters():
            parameter.requires_grad = False

    def unfreeze(self) -> None:
        for parameter in self.vit.parameters():
            parameter.requires_grad = True

    # ======================================================================
    # ViT-specific interface
    # ======================================================================

    @property
    def embeddings(self):
        return self.vit.transformer.embeddings

    @property
    def blocks(self):
        return self.vit.transformer.encoder.layer

    @property
    def norm(self):
        return self.vit.transformer.encoder.encoder_norm

    def get_patch_tokens(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return patch tokens before CLS and positional embeddings.

        Shape:
            [B, num_patches, embed_dim]
        """
        x = self.embeddings.patch_embeddings(x)
        return x.flatten(2).transpose(-1, -2)

    def get_cls_token(
        self,
        batch_size: int,
    ) -> torch.Tensor:
        """
        Return the CLS token expanded to the batch size.

        Shape:
            [B, 1, embed_dim]
        """
        return self.embeddings.cls_token.expand(
            batch_size,
            -1,
            -1,
        )

    def get_positional_embeddings(self):
        """
        Return:
            cls_pos:   [1, 1, embed_dim]
            patch_pos: [1, num_patches, embed_dim]
        """
        pos = self.embeddings.position_embeddings

        return (
            pos[:, :1],
            pos[:, 1:],
        )

    def input_dropout(
        self,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        return self.embeddings.dropout(tokens)

    def encode_tokens(
        self,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Run Transformer blocks followed by the final normalization."""
        hidden = tokens

        for block in self.blocks:
            hidden, _ = block(hidden)

        return self.norm(hidden)

    def forward_features(
        self,
        x: torch.Tensor,
    ):
        """
        Standard backbone forward.

        Returns:
            cls:
                [B, embed_dim]

            patch_tokens:
                [B, num_patches, embed_dim]
        """
        batch_size = x.shape[0]

        patch_tokens = self.get_patch_tokens(x)
        cls_token = self.get_cls_token(batch_size)

        cls_pos, patch_pos = self.get_positional_embeddings()

        tokens = torch.cat(
            (
                cls_token + cls_pos,
                patch_tokens + patch_pos,
            ),
            dim=1,
        )

        tokens = self.input_dropout(tokens)
        encoded = self.encode_tokens(tokens)

        return {
            "cls": encoded[:, 0],
            "patch_tokens": encoded[:, 1:],
        }

    def get_layerwise_features(
        self,
        x: torch.Tensor,
    ):
        """
        Return CLS and patch representations after every Transformer layer.

        Returns:
            cls:
                [num_layers, B, embed_dim]
            patch_tokens:
                [num_layers, B, num_patches, embed_dim]
        """
        batch_size = x.shape[0]

        patch_tokens = self.get_patch_tokens(x)
        cls_token = self.get_cls_token(batch_size)

        cls_pos, patch_pos = self.get_positional_embeddings()

        hidden = torch.cat(
            (
                cls_token + cls_pos,
                patch_tokens + patch_pos,
            ),
            dim=1,
        )
        hidden = self.input_dropout(hidden)

        layer_outputs = []
        for block in self.blocks:
            hidden, _ = block(hidden)
            layer_outputs.append(hidden)

        layer_outputs = torch.stack(layer_outputs, dim=0)

        return {
            "cls": layer_outputs[:, :, 0],
            "patch_tokens": layer_outputs[:, :, 1:],
        }
