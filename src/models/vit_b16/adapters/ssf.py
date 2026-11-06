import math

import torch
import torch.nn as nn

from ...peft_model import PEFTModel
from ..backbone import ViTB16Backbone


class ViTB16SSF(PEFTModel):
    """Scale-and-Shift Fine-Tuning for ViT-B/16."""

    _ALL_SSF_TARGETS = (
        "patch",
        "attention_norm",
        "query",
        "key",
        "value",
        "attention_out",
        "ffn_norm",
        "fc1",
        "fc2",
        "final_norm",
    )

    def __init__(
        self,
        backbone: ViTB16Backbone,
        n_classes: int,
        ssf_targets=_ALL_SSF_TARGETS,
    ):
        ssf_targets = tuple(ssf_targets)
        if len(set(ssf_targets)) != len(ssf_targets):
            raise ValueError("ssf_targets must not contain duplicates")
        if any(name not in self._ALL_SSF_TARGETS for name in ssf_targets):
            raise ValueError(
                "ssf_targets may contain only "
                + ", ".join(repr(name) for name in self._ALL_SSF_TARGETS)
            )

        super().__init__(backbone=backbone, n_classes=n_classes)

        self.ssf_targets = ssf_targets
        self.ssf_scale = nn.ParameterDict()
        self.ssf_shift = nn.ParameterDict()

        shapes = {
            "patch": (self.backbone.embed_dim,),
            "attention_norm": (self.backbone.num_layers, self.backbone.embed_dim),
            "query": (self.backbone.num_layers, self.backbone.embed_dim),
            "key": (self.backbone.num_layers, self.backbone.embed_dim),
            "value": (self.backbone.num_layers, self.backbone.embed_dim),
            "attention_out": (self.backbone.num_layers, self.backbone.embed_dim),
            "ffn_norm": (self.backbone.num_layers, self.backbone.embed_dim),
            "fc1": (self.backbone.num_layers, self.backbone.mlp_dim),
            "fc2": (self.backbone.num_layers, self.backbone.embed_dim),
            "final_norm": (self.backbone.embed_dim,),
        }
        for name in ssf_targets:
            self.ssf_scale[name] = nn.Parameter(torch.empty(shapes[name]))
            self.ssf_shift[name] = nn.Parameter(torch.empty(shapes[name]))

        self.init_peft()

    def init_peft(
        self,
        init_from: tuple[nn.ParameterDict, nn.ParameterDict] | None = None,
    ) -> None:
        if not self.ssf_targets:
            return
        if init_from is not None:
            scales, shifts = init_from
            if set(scales) != set(self.ssf_scale) or set(shifts) != set(self.ssf_shift):
                raise ValueError(
                    f"SSF target mismatch: expected {self.ssf_targets}, "
                    f"got scales={tuple(scales)}, shifts={tuple(shifts)}."
                )
            for name in self.ssf_targets:
                if (
                    scales[name].shape != self.ssf_scale[name].shape
                    or shifts[name].shape != self.ssf_shift[name].shape
                ):
                    raise ValueError(
                        f"SSF shape mismatch for target {name!r}: expected "
                        f"scale={tuple(self.ssf_scale[name].shape)}, "
                        f"shift={tuple(self.ssf_shift[name].shape)}, got "
                        f"scale={tuple(scales[name].shape)}, shift={tuple(shifts[name].shape)}."
                    )
            with torch.no_grad():
                for name in self.ssf_targets:
                    self.ssf_scale[name].copy_(scales[name])
                    self.ssf_shift[name].copy_(shifts[name])
            return
        with torch.no_grad():
            for scale in self.ssf_scale.values():
                nn.init.normal_(scale, mean=1.0, std=0.02)
            for shift in self.ssf_shift.values():
                nn.init.normal_(shift, mean=0.0, std=0.02)

    @staticmethod
    def _apply_affine(
        x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor
    ) -> torch.Tensor:
        if x.shape[-1] == scale.shape[0]:
            return x * scale + shift
        shape = (1, -1, *(1 for _ in range(x.ndim - 2)))
        return x * scale.view(shape) + shift.view(shape)

    def _apply_ssf(
        self,
        x: torch.Tensor,
        name: str,
        layer_idx: int | None = None,
    ) -> torch.Tensor:
        if name not in self.ssf_scale:
            return x
        scale, shift = self.ssf_scale[name], self.ssf_shift[name]
        if layer_idx is not None:
            scale, shift = scale[layer_idx], shift[layer_idx]
        return self._apply_affine(x, scale, shift)

    def forward_features(self, x: torch.Tensor):
        if not self.ssf_targets:
            return self.backbone.forward_features(x)

        batch_size = x.shape[0]
        patch_tokens = self.backbone.embeddings.patch_embeddings(x)
        patch_tokens = self._apply_ssf(patch_tokens, "patch")
        patch_tokens = patch_tokens.flatten(2).transpose(-1, -2)

        cls_token = self.backbone.get_cls_token(batch_size)
        cls_pos, patch_pos = self.backbone.get_positional_embeddings()
        hidden = self.backbone.input_dropout(
            torch.cat((cls_token + cls_pos, patch_tokens + patch_pos), dim=1)
        )

        for layer_idx, block in enumerate(self.backbone.blocks):
            residual = hidden
            x = self._apply_ssf(
                block.attention_norm(hidden), "attention_norm", layer_idx
            )
            attn = block.attn

            query = self._apply_ssf(attn.query(x), "query", layer_idx)
            key = self._apply_ssf(attn.key(x), "key", layer_idx)
            value = self._apply_ssf(attn.value(x), "value", layer_idx)

            query = attn.transpose_for_scores(query)
            key = attn.transpose_for_scores(key)
            value = attn.transpose_for_scores(value)

            scores = torch.matmul(query, key.transpose(-1, -2))
            scores = scores / math.sqrt(attn.attention_head_size)
            probs = attn.attn_dropout(attn.softmax(scores))

            context = torch.matmul(probs, value)
            context = context.permute(0, 2, 1, 3).contiguous()
            context = context.view(batch_size, -1, attn.all_head_size)

            hidden = self._apply_ssf(attn.out(context), "attention_out", layer_idx)
            hidden = residual + attn.proj_dropout(hidden)

            residual = hidden
            x = self._apply_ssf(block.ffn_norm(hidden), "ffn_norm", layer_idx)
            ffn = block.ffn

            x = self._apply_ssf(ffn.fc1(x), "fc1", layer_idx)
            x = ffn.dropout(ffn.act_fn(x))
            x = self._apply_ssf(ffn.fc2(x), "fc2", layer_idx)
            hidden = residual + ffn.dropout(x)

        encoded = self._apply_ssf(self.backbone.norm(hidden), "final_norm")
        return {"cls": encoded[:, 0], "patch_tokens": encoded[:, 1:]}

    def save(self, path: str) -> None:
        state_dict = {
            key: value.detach().cpu()
            for key, value in self.state_dict().items()
            if key.startswith(("head.", "ssf_scale.", "ssf_shift."))
        }
        torch.save(state_dict, path)
