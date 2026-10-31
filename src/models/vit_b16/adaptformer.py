import math
from types import MethodType

import torch
import torch.nn as nn

from ..peft_model import PEFTModel
from .backbone import ViTB16Backbone


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
        img_size: int,
        n_classes: int,
        freeze_backbone: bool = True,
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

        super().__init__(
            backbone=ViTB16Backbone(img_size=img_size),
            n_classes=n_classes,
            freeze_backbone=freeze_backbone,
        )

        self.img_size = img_size
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

        if bottleneck_dim > 0:
            self._patch_adaptformer_ffns()

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

    def _patch_adaptformer_ffns(self) -> None:
        def make_forward(layer_idx, original_forward):
            def forward(_, x):
                return original_forward(x) + self.adapters[layer_idx](x)

            return forward

        for layer_idx, block in enumerate(self.backbone.blocks):
            block.ffn.forward = MethodType(
                make_forward(layer_idx, block.ffn.forward), block.ffn
            )

    def save(self, path: str) -> None:
        state_dict = {
            key: value.detach().cpu()
            for key, value in self.state_dict().items()
            if key.startswith("head.")
            or key.startswith("adapters.")
            or (not self.is_backbone_frozen and key.startswith("backbone."))
        }
        torch.save(state_dict, path)
