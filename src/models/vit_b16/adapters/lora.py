import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...peft_model import PEFTModel
from ..backbone import ViTB16Backbone


class ViTB16LoRA(PEFTModel):
    """LoRA for ViT-B/16."""

    def __init__(
        self,
        backbone: ViTB16Backbone,
        n_classes: int,
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

        super().__init__(backbone=backbone, n_classes=n_classes)

        self.rank = rank
        self.lora_targets = lora_targets
        self.lora_alpha = lora_alpha
        self.lora_scaling = lora_alpha / rank if rank > 0 else 0.0

        if rank > 0:
            self.lora_A = nn.Parameter(
                torch.empty(
                    self.backbone.num_layers,
                    len(lora_targets),
                    rank,
                    self.backbone.embed_dim,
                )
            )
            self.lora_B = nn.Parameter(
                torch.empty(
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

    def init_peft(
        self, init_from: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> None:
        if self.rank == 0:
            return
        if init_from is not None:
            lora_A, lora_B = init_from
            if lora_A.shape != self.lora_A.shape or lora_B.shape != self.lora_B.shape:
                raise ValueError(
                    f"LoRA shape mismatch: expected A={tuple(self.lora_A.shape)}, "
                    f"B={tuple(self.lora_B.shape)}, got A={tuple(lora_A.shape)}, "
                    f"B={tuple(lora_B.shape)}."
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
                F.linear(x, self.lora_A[layer_idx, target_idx]),
                self.lora_B[layer_idx, target_idx],
            )
            * self.lora_scaling
        )

    def forward_features(self, x: torch.Tensor):
        if self.rank == 0:
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
            x = block.attention_norm(hidden)
            attn = block.attn

            projections = {
                "query": attn.query(x),
                "key": attn.key(x),
                "value": attn.value(x),
            }
            for target_idx, name in enumerate(self.lora_targets):
                projections[name] = projections[name] + self._lora_delta(
                    x, layer_idx, target_idx
                )

            query = attn.transpose_for_scores(projections["query"])
            key = attn.transpose_for_scores(projections["key"])
            value = attn.transpose_for_scores(projections["value"])

            scores = torch.matmul(query, key.transpose(-1, -2))
            scores = scores / math.sqrt(attn.attention_head_size)
            probs = attn.attn_dropout(attn.softmax(scores))

            context = torch.matmul(probs, value)
            context = context.permute(0, 2, 1, 3).contiguous()
            context = context.view(batch_size, -1, attn.all_head_size)

            hidden = residual + attn.proj_dropout(attn.out(context))
            hidden = hidden + block.ffn(block.ffn_norm(hidden))

        encoded = self.backbone.norm(hidden)
        return {"cls": encoded[:, 0], "patch_tokens": encoded[:, 1:]}

    def save(self, path: str) -> None:
        state_dict = {
            key: value.detach().cpu()
            for key, value in self.state_dict().items()
            if key.startswith("head.") or key in {"lora_A", "lora_B"}
        }
        torch.save(state_dict, path)
