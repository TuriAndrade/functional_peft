import os

import numpy as np
import torch
import torch.nn as nn

from third_party.vit_pytorch.models.modeling import VisionTransformer
from third_party.vit_pytorch.models.configs import get_b16_config


class ViTB16Backbone(nn.Module):
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
            num_classes=21843,  # matches the default in the implementation
            zero_head=False,
            vis=False,
        )

        _CKPT_PATH = os.path.join("third_party", "vit_pytorch", "ckpts", "ViT-B_16.npz")
        weights = np.load(_CKPT_PATH)
        vit.load_from(weights)

        # Remove classification head so this acts like a backbone
        vit.head = nn.Identity()

        self.vit = vit
        self.embed_dim = config.hidden_size
        self.img_size = img_size
        self.patch_size = 16
        self.grid_size = img_size // self.patch_size
        self.num_patches = self.grid_size * self.grid_size

    @property
    def embeddings(self):
        return self.vit.transformer.embeddings

    @property
    def blocks(self):
        return self.vit.transformer.encoder.layer

    @property
    def norm(self):
        return self.vit.transformer.encoder.encoder_norm

    def freeze(self):
        for p in self.vit.parameters():
            p.requires_grad = False

    def unfreeze(self):
        for p in self.vit.parameters():
            p.requires_grad = True

    def get_patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns patch tokens before CLS and before positional embeddings.
        Shape: [B, 196, 768] for 224x224 input.
        """
        x = self.embeddings.patch_embeddings(x)  # [B, 768, 14, 14]
        x = x.flatten(2).transpose(-1, -2)  # [B, 196, 768]
        return x

    def get_cls_token(self, batch_size: int) -> torch.Tensor:
        """
        Returns the CLS token expanded to batch size.
        Shape: [B, 1, 768]
        """
        return self.embeddings.cls_token.expand(batch_size, -1, -1)

    def get_positional_embeddings(self):
        """
        Returns:
          cls_pos:   [1, 1, 768]
          patch_pos: [1, 196, 768]
        """
        pos = self.embeddings.position_embeddings
        cls_pos = pos[:, :1]
        patch_pos = pos[:, 1:]
        return cls_pos, patch_pos

    def input_dropout(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.embeddings.dropout(tokens)

    def encode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Runs transformer blocks + final norm on a token sequence.
        """
        h = tokens
        for blk in self.blocks:
            h, _ = blk(h)
        h = self.norm(h)
        return h

    def forward_features(self, x: torch.Tensor):
        """
        Standard backbone forward.
        Returns final CLS and patch tokens.
        """
        B = x.shape[0]
        patch_tokens = self.get_patch_tokens(x)
        cls_token = self.get_cls_token(B)
        cls_pos, patch_pos = self.get_positional_embeddings()

        tokens = torch.cat(
            [cls_token + cls_pos, patch_tokens + patch_pos],
            dim=1,
        )
        tokens = self.input_dropout(tokens)
        encoded = self.encode_tokens(tokens)

        return {
            "cls": encoded[:, 0],  # [B, 768]
            "patch_tokens": encoded[:, 1:],  # [B, 196, 768]
        }
