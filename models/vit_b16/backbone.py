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

    def get_param_group(self, name: str):
        """
        Return parameter group specification for FIM aggregation.

        Exact supported names:
          - "block"
          - "attention_qkvo"
          - "mlp_layer"
          - "patch_embed_channel"
        """
        if name == "block":
            return self._get_block_param_group()

        if name == "attention_qkvo":
            return self._get_attention_qkvo_param_group()

        if name == "mlp_layer":
            return self._get_mlp_layer_param_group()

        if name == "patch_embed_channel":
            return self._get_patch_embed_channel_param_group()

        raise ValueError(
            f"Unknown param group: {name}. "
            "Expected exactly one of: "
            "'block', 'attention_qkvo', 'mlp_layer', 'patch_embed_channel'."
        )

    def _get_block_param_group(self):
        groups = []

        for block_idx, block in enumerate(self.blocks):
            prefix = f"vit.transformer.encoder.layer.{block_idx}"

            items = []
            for param_name, param in block.named_parameters(
                prefix=prefix,
                recurse=True,
            ):
                items.append(
                    {
                        "name": param_name,
                        "param": param,
                        "index": None,
                    }
                )

            groups.append(
                {
                    "name": f"block_{block_idx}",
                    "index": (block_idx,),
                    "items": items,
                }
            )

        return {
            "name": "block",
            "shape": (len(self.blocks),),
            "groups": groups,
        }

    def _get_attention_qkvo_param_group(self):
        groups = []

        role_names = ["query", "key", "value", "out"]

        for block_idx, block in enumerate(self.blocks):
            attn = block.attn
            n_heads = attn.num_attention_heads
            head_dim = attn.attention_head_size

            prefix = f"vit.transformer.encoder.layer.{block_idx}.attn"

            for head_idx in range(n_heads):
                start = head_idx * head_dim
                end = (head_idx + 1) * head_dim

                row_slice = slice(start, end)
                col_slice = slice(start, end)

                groups.append(
                    {
                        "name": f"attention_q_{block_idx}_{head_idx}",
                        "index": (block_idx, head_idx, 0),
                        "items": [
                            {
                                "name": f"{prefix}.query.weight",
                                "param": attn.query.weight,
                                "index": (row_slice, slice(None)),
                            }
                        ],
                    }
                )

                groups.append(
                    {
                        "name": f"attention_k_{block_idx}_{head_idx}",
                        "index": (block_idx, head_idx, 1),
                        "items": [
                            {
                                "name": f"{prefix}.key.weight",
                                "param": attn.key.weight,
                                "index": (row_slice, slice(None)),
                            }
                        ],
                    }
                )

                groups.append(
                    {
                        "name": f"attention_v_{block_idx}_{head_idx}",
                        "index": (block_idx, head_idx, 2),
                        "items": [
                            {
                                "name": f"{prefix}.value.weight",
                                "param": attn.value.weight,
                                "index": (row_slice, slice(None)),
                            }
                        ],
                    }
                )

                groups.append(
                    {
                        "name": f"attention_o_{block_idx}_{head_idx}",
                        "index": (block_idx, head_idx, 3),
                        "items": [
                            {
                                "name": f"{prefix}.out.weight",
                                "param": attn.out.weight,
                                "index": (slice(None), col_slice),
                            }
                        ],
                    }
                )

        n_blocks = len(self.blocks)
        n_heads = self.blocks[0].attn.num_attention_heads

        return {
            "name": "attention_qkvo",
            "shape": (n_blocks, n_heads, len(role_names)),
            "roles": role_names,
            "groups": groups,
        }

    def _get_mlp_layer_param_group(self):
        groups = []

        for block_idx, block in enumerate(self.blocks):
            prefix = f"vit.transformer.encoder.layer.{block_idx}.ffn"

            groups.append(
                {
                    "name": f"mlp_layer_{block_idx}",
                    "index": (block_idx,),
                    "items": [
                        {
                            "name": f"{prefix}.fc1.weight",
                            "param": block.ffn.fc1.weight,
                            "index": None,
                        },
                        {
                            "name": f"{prefix}.fc2.weight",
                            "param": block.ffn.fc2.weight,
                            "index": None,
                        },
                    ],
                }
            )

        return {
            "name": "mlp_layer",
            "shape": (len(self.blocks),),
            "groups": groups,
        }

    def _get_patch_embed_channel_param_group(self):
        groups = []

        patch_embed = self.embeddings.patch_embeddings
        n_channels = patch_embed.weight.shape[0]

        prefix = "vit.transformer.embeddings.patch_embeddings"

        for channel_idx in range(n_channels):
            groups.append(
                {
                    "name": f"patch_embed_channel_{channel_idx}",
                    "index": (channel_idx,),
                    "items": [
                        {
                            "name": f"{prefix}.weight",
                            "param": patch_embed.weight,
                            "index": (
                                channel_idx,
                                slice(None),
                                slice(None),
                                slice(None),
                            ),
                        }
                    ],
                }
            )

        return {
            "name": "patch_embed_channel",
            "shape": (n_channels,),
            "groups": groups,
        }
