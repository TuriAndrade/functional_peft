import os

import numpy as np
import torch
import torch.nn as nn

from ...third_party.vit_pytorch.models.configs import get_b16_config
from ...third_party.vit_pytorch.models.modeling import VisionTransformer
from ..transformer_backbone import TransformerBackbone


class ViTB16Backbone(TransformerBackbone):
    """Original Google ViT-B/16 ImageNet-21k backbone."""

    def __init__(self, img_size: int = 224):
        super().__init__()
        config = get_b16_config()
        vit = VisionTransformer(
            config, img_size=img_size, num_classes=21843, zero_head=False, vis=False
        )
        ckpt_path = os.path.join(
            "src", "third_party", "vit_pytorch", "ckpts", "ViT-B_16.npz"
        )
        vit.load_from(np.load(ckpt_path))
        vit.head = nn.Identity()
        self.vit = vit
        self._embed_dim = config.hidden_size
        self.img_size = img_size
        self.patch_size = 16
        self.grid_size = img_size // self.patch_size
        self.num_patches = self.grid_size * self.grid_size

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

    def freeze(self) -> None:
        for parameter in self.vit.parameters():
            parameter.requires_grad = False

    def unfreeze(self) -> None:
        for parameter in self.vit.parameters():
            parameter.requires_grad = True

    @property
    def embeddings(self):
        return self.vit.transformer.embeddings

    @property
    def blocks(self):
        return self.vit.transformer.encoder.layer

    @property
    def norm(self):
        return self.vit.transformer.encoder.encoder_norm

    def get_patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        return self.embeddings.patch_embeddings(x).flatten(2).transpose(-1, -2)

    def get_cls_token(self, batch_size: int) -> torch.Tensor:
        return self.embeddings.cls_token.expand(batch_size, -1, -1)

    def get_positional_embeddings(self):
        pos = self.embeddings.position_embeddings
        return pos[:, :1], pos[:, 1:]

    def input_dropout(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.embeddings.dropout(tokens)

    def encode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden = tokens
        for block in self.blocks:
            hidden, _ = block(hidden)
        return self.norm(hidden)

    def forward_features(self, x: torch.Tensor):
        batch_size = x.shape[0]
        patch_tokens = self.get_patch_tokens(x)
        cls_token = self.get_cls_token(batch_size)
        cls_pos, patch_pos = self.get_positional_embeddings()
        tokens = torch.cat((cls_token + cls_pos, patch_tokens + patch_pos), dim=1)
        encoded = self.encode_tokens(self.input_dropout(tokens))
        return {"cls": encoded[:, 0], "patch_tokens": encoded[:, 1:]}
