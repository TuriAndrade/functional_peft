import math

import torch
import torch.nn as nn

from ...peft_model import PEFTModel
from ..backbone import ViTB16Backbone


class AdaptFormerAdapter(nn.Module):
    """Parallel bottleneck adapter used by AdaptFormer."""

    def __init__(
        self,
        embed_dim: int,
        bottleneck_dim: int,
        scale: float = 0.1,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.scale = float(scale)
        self.down = nn.Linear(embed_dim, bottleneck_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(bottleneck_dim, embed_dim)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.act(self.down(x)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.forward_features(x)) * self.scale


class ViTB16AdaptFormer(PEFTModel):
    """AdaptFormer for ViT-B/16."""

    def __init__(
        self,
        backbone: ViTB16Backbone,
        n_classes: int,
        bottleneck_dim: int = 64,
        adapter_scale: float = 0.1,
        adapter_dropout: float = 0.1,
    ):
        if bottleneck_dim < 0:
            raise ValueError("bottleneck_dim must be non-negative")
        if bottleneck_dim > 0 and adapter_scale <= 0:
            raise ValueError("adapter_scale must be positive when bottleneck_dim > 0")
        if not 0.0 <= adapter_dropout < 1.0:
            raise ValueError("adapter_dropout must be in [0, 1)")

        super().__init__(backbone=backbone, n_classes=n_classes)

        self.bottleneck_dim = bottleneck_dim
        self.adapter_scale = float(adapter_scale)
        self.adapter_dropout = float(adapter_dropout)
        self.adapters = nn.ModuleList(
            [
                AdaptFormerAdapter(
                    self.backbone.embed_dim,
                    bottleneck_dim,
                    self.adapter_scale,
                    self.adapter_dropout,
                )
                for _ in range(self.backbone.num_layers)
            ]
            if bottleneck_dim > 0
            else []
        )

        self.init_peft()

    def init_peft(self, init_from: nn.ModuleList | None = None) -> None:
        if self.bottleneck_dim == 0:
            return
        if init_from is not None:
            try:
                self.adapters.load_state_dict(init_from.state_dict(), strict=True)
            except RuntimeError as exc:
                raise ValueError("AdaptFormer adapter shape mismatch.") from exc
            return
        for adapter in self.adapters:
            adapter.reset_parameters()

    def forward_features(self, x: torch.Tensor):
        if self.bottleneck_dim == 0:
            return self.backbone.forward_features(x)

        batch_size = x.shape[0]
        patch_tokens = self.backbone.get_patch_tokens(x)
        cls_token = self.backbone.get_cls_token(batch_size)
        cls_pos, patch_pos = self.backbone.get_positional_embeddings()
        hidden = self.backbone.input_dropout(
            torch.cat((cls_token + cls_pos, patch_tokens + patch_pos), dim=1)
        )

        for layer_idx, block in enumerate(self.backbone.blocks):
            residual = hidden
            hidden = block.attention_norm(hidden)
            hidden, _ = block.attn(hidden)
            hidden = residual + hidden

            residual = hidden
            hidden = block.ffn_norm(hidden)
            hidden = residual + block.ffn(hidden) + self.adapters[layer_idx](hidden)

        encoded = self.backbone.norm(hidden)
        return {"cls": encoded[:, 0], "patch_tokens": encoded[:, 1:]}

    def save(self, path: str) -> None:
        state_dict = {
            key: value.detach().cpu()
            for key, value in self.state_dict().items()
            if key.startswith("head.") or key.startswith("adapters.")
        }
        torch.save(state_dict, path)
