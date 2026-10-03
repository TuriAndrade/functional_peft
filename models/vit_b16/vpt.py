import math

import torch
import torch.nn as nn

from ..peft_model import PEFTModel
from .backbone import ViTB16Backbone


class ViTB16VPT(PEFTModel):
    """Visual Prompt Tuning for ViT-B/16."""

    def __init__(
        self,
        img_size: int,
        n_classes: int,
        freeze_backbone: bool = True,
        prompt_length: int = 0,
        prompt_mode: str = "shallow",
        prompt_dropout: float | None = None,
    ):
        if prompt_length < 0:
            raise ValueError("prompt_length must be non-negative")
        if prompt_mode not in {"shallow", "deep"}:
            raise ValueError("prompt_mode must be 'shallow' or 'deep'")
        if prompt_dropout is not None and not 0.0 <= prompt_dropout < 1.0:
            raise ValueError("prompt_dropout must be in [0, 1)")

        super().__init__(
            backbone=ViTB16Backbone(img_size=img_size),
            n_classes=n_classes,
            freeze_backbone=freeze_backbone,
        )

        self.img_size = img_size
        self.prompt_length = prompt_length
        self.prompt_mode = prompt_mode
        self.prompt_dropout = (
            (0.0 if prompt_mode == "shallow" else 0.1)
            if prompt_dropout is None
            else float(prompt_dropout)
        )

        self.prompt = nn.Parameter(
            torch.zeros(
                self.backbone.num_layers if prompt_mode == "deep" else 1,
                prompt_length,
                self.backbone.embed_dim,
            )
        )
        self.prompt_dropout_layer = nn.Dropout(self.prompt_dropout)

        self.init_peft()

    def init_peft(
        self,
        init_from: torch.Tensor | None = None,
    ) -> None:
        if self.prompt_length == 0:
            return

        if init_from is not None:
            if init_from.shape != self.prompt.shape:
                raise ValueError(
                    f"Prompt shape mismatch: expected {tuple(self.prompt.shape)}, "
                    f"got {tuple(init_from.shape)}."
                )

            with torch.no_grad():
                self.prompt.copy_(init_from)

            return

        val = math.sqrt(
            6.0 / (3 * self.backbone.patch_size**2 + self.backbone.embed_dim)
        )
        nn.init.uniform_(self.prompt, -val, val)

    def get_prompt_tokens(
        self,
        batch_size: int,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        prompt = self.prompt[layer_idx if self.prompt_mode == "deep" else 0]

        return self.prompt_dropout_layer(
            prompt.unsqueeze(0).expand(
                batch_size,
                -1,
                -1,
            )
        )

    def encode_tokens(
        self,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Run the encoder with shallow or deep prompt injection."""
        if self.prompt_length == 0:
            return self.backbone.encode_tokens(tokens)

        hidden = tokens
        batch_size = hidden.shape[0]

        if self.prompt_mode == "shallow":
            hidden = torch.cat(
                (
                    hidden[:, :1],
                    self.get_prompt_tokens(batch_size),
                    hidden[:, 1:],
                ),
                dim=1,
            )

            for block in self.backbone.blocks:
                hidden, _ = block(hidden)

        else:
            for layer_idx, block in enumerate(self.backbone.blocks):
                patch_start = 1 if layer_idx == 0 else 1 + self.prompt_length

                hidden = torch.cat(
                    (
                        hidden[:, :1],
                        self.get_prompt_tokens(
                            batch_size,
                            layer_idx,
                        ),
                        hidden[:, patch_start:],
                    ),
                    dim=1,
                )

                hidden, _ = block(hidden)

        return self.backbone.norm(hidden)

    def forward_features(
        self,
        x: torch.Tensor,
    ):
        batch_size = x.shape[0]

        patch_tokens = self.backbone.get_patch_tokens(x)
        cls_token = self.backbone.get_cls_token(batch_size)
        cls_pos, patch_pos = self.backbone.get_positional_embeddings()

        tokens = torch.cat(
            (
                cls_token + cls_pos,
                patch_tokens + patch_pos,
            ),
            dim=1,
        )

        encoded = self.encode_tokens(self.backbone.input_dropout(tokens))

        prompt_end = 1 + self.prompt_length

        return {
            "cls": encoded[:, 0],
            "prompt_tokens": encoded[:, 1:prompt_end],
            "patch_tokens": encoded[:, prompt_end:],
        }

    def get_layerwise_features(
        self,
        x: torch.Tensor,
    ):
        """
        Return CLS, prompt, and patch representations after every Transformer layer.

        Returns:
            cls:
                [num_layers, B, embed_dim]

            prompt_tokens:
                [num_layers, B, prompt_length, embed_dim]

            patch_tokens:
                [num_layers, B, num_patches, embed_dim]
        """
        batch_size = x.shape[0]

        patch_tokens = self.backbone.get_patch_tokens(x)
        cls_token = self.backbone.get_cls_token(batch_size)
        cls_pos, patch_pos = self.backbone.get_positional_embeddings()

        hidden = torch.cat(
            (
                cls_token + cls_pos,
                patch_tokens + patch_pos,
            ),
            dim=1,
        )
        hidden = self.backbone.input_dropout(hidden)

        cls_features = []
        prompt_features = []
        patch_features = []

        if self.prompt_mode == "shallow":
            hidden = torch.cat(
                (
                    hidden[:, :1],
                    self.get_prompt_tokens(batch_size),
                    hidden[:, 1:],
                ),
                dim=1,
            )

            for block in self.backbone.blocks:
                hidden, _ = block(hidden)

                prompt_end = 1 + self.prompt_length

                cls_features.append(hidden[:, 0])
                prompt_features.append(hidden[:, 1:prompt_end])
                patch_features.append(hidden[:, prompt_end:])

        else:
            for layer_idx, block in enumerate(self.backbone.blocks):
                patch_start = 1 if layer_idx == 0 else 1 + self.prompt_length

                hidden = torch.cat(
                    (
                        hidden[:, :1],
                        self.get_prompt_tokens(batch_size, layer_idx),
                        hidden[:, patch_start:],
                    ),
                    dim=1,
                )

                hidden, _ = block(hidden)

                prompt_end = 1 + self.prompt_length

                cls_features.append(hidden[:, 0])
                prompt_features.append(hidden[:, 1:prompt_end])
                patch_features.append(hidden[:, prompt_end:])

        return {
            "cls": torch.stack(cls_features, dim=0),
            "prompt_tokens": torch.stack(prompt_features, dim=0),
            "patch_tokens": torch.stack(patch_features, dim=0),
        }

    def save(
        self,
        path: str,
    ) -> None:
        state_dict = {
            key: value.detach().cpu()
            for key, value in self.state_dict().items()
            if (
                key == "prompt"
                or key.startswith("head.")
                or (not self.is_backbone_frozen and key.startswith("backbone."))
            )
        }

        torch.save(
            state_dict,
            path,
        )
