import math
from types import MethodType

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..peft_model import PEFTModel
from .backbone import ViTB16Backbone


class ViTB16LoRA(PEFTModel):
    """LoRA for ViT-B/16."""

    def __init__(
        self,
        img_size: int,
        n_classes: int,
        freeze_backbone: bool = True,
        rank: int = 8,
        lora_alpha: float | None = None,
        lora_targets=("query", "value"),
    ):
        if rank < 0:
            raise ValueError("rank must be non-negative")

        lora_targets = tuple(lora_targets)

        if len(set(lora_targets)) != len(lora_targets):
            raise ValueError("lora_targets must not contain duplicates")

        if any(name not in {"query", "value"} for name in lora_targets):
            raise ValueError("lora_targets may contain only 'query' and 'value'")

        if rank > 0 and not lora_targets:
            raise ValueError("lora_targets must be non-empty when rank > 0")

        lora_alpha = float(rank if lora_alpha is None else lora_alpha)

        if rank > 0 and lora_alpha <= 0:
            raise ValueError("lora_alpha must be positive when rank > 0")

        super().__init__(
            backbone=ViTB16Backbone(img_size=img_size),
            n_classes=n_classes,
            freeze_backbone=freeze_backbone,
        )

        self.img_size = img_size
        self.rank = rank
        self.lora_targets = lora_targets
        self.lora_alpha = lora_alpha
        self.lora_scaling = lora_alpha / rank if rank > 0 else 0.0

        if rank > 0:
            reference = next(self.backbone.parameters())

            self.lora_A = nn.Parameter(
                reference.new_empty(
                    self.backbone.num_layers,
                    len(lora_targets),
                    rank,
                    self.backbone.embed_dim,
                )
            )
            self.lora_B = nn.Parameter(
                reference.new_empty(
                    self.backbone.num_layers,
                    len(lora_targets),
                    self.backbone.embed_dim,
                    rank,
                )
            )
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        self.init_peft()

        if rank > 0:
            self._patch_lora_projections()

    def init_peft(
        self,
        init_from: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> None:
        if self.rank == 0:
            return

        if init_from is not None:
            lora_A, lora_B = init_from

            if lora_A.shape != self.lora_A.shape or lora_B.shape != self.lora_B.shape:
                raise ValueError(
                    f"LoRA shape mismatch: expected "
                    f"A={tuple(self.lora_A.shape)}, B={tuple(self.lora_B.shape)}, "
                    f"got A={tuple(lora_A.shape)}, B={tuple(lora_B.shape)}."
                )

            with torch.no_grad():
                self.lora_A.copy_(lora_A)
                self.lora_B.copy_(lora_B)

            return

        for value in self.lora_A.flatten(0, 1):
            nn.init.kaiming_uniform_(value, a=math.sqrt(5))

        nn.init.zeros_(self.lora_B)

    def _lora_delta(
        self,
        x: torch.Tensor,
        layer_idx: int,
        target_idx: int,
    ) -> torch.Tensor:
        return (
            F.linear(
                F.linear(
                    x,
                    self.lora_A[layer_idx, target_idx],
                ),
                self.lora_B[layer_idx, target_idx],
            )
            * self.lora_scaling
        )

    def _patch_lora_projections(self) -> None:
        def make_forward(layer_idx, target_idx, original_forward):
            def forward(_, x):
                return original_forward(x) + self._lora_delta(
                    x,
                    layer_idx,
                    target_idx,
                )

            return forward

        for layer_idx, block in enumerate(self.backbone.blocks):
            for target_idx, target_name in enumerate(self.lora_targets):
                projection = getattr(block.attn, target_name)

                projection.forward = MethodType(
                    make_forward(
                        layer_idx,
                        target_idx,
                        projection.forward,
                    ),
                    projection,
                )

    def save(
        self,
        path: str,
    ) -> None:
        state_dict = {
            key: value.detach().cpu()
            for key, value in self.state_dict().items()
            if (
                key.startswith("head.")
                or key in {"lora_A", "lora_B"}
                or (not self.is_backbone_frozen and key.startswith("backbone."))
            )
        }

        torch.save(state_dict, path)
