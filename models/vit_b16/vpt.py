import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import reduce
from operator import mul
from types import MethodType
from typing import Optional, List
from .backbone import ViTB16Backbone


class ViTB16VPT(nn.Module):
    """
    VPT model implemented for a ViT-B/16 backbone.
    """

    def __init__(
        self,
        img_size: int,
        n_classes: int,
        prompt_length: int = 0,
        prompt_mode: str = "shallow",
        freeze_backbone: bool = True,
    ):
        super().__init__()

        self.img_size = img_size
        self.n_classes = n_classes
        self.prompt_length = prompt_length
        self.prompt_mode = prompt_mode
        self.is_backbone_frozen = freeze_backbone

        # Backbone
        self.backbone = ViTB16Backbone(img_size=img_size)
        if freeze_backbone:
            self.freeze_backbone()

        # Head
        self.head = nn.Linear(self.backbone.embed_dim, n_classes)
        self.init_head()

        # Prompt
        self.prompt = nn.Parameter(
            torch.zeros(
                len(self.backbone.blocks) if prompt_mode == "deep" else 1,
                self.prompt_length,
                self.backbone.embed_dim,
            )
        )
        self.prompt_dropout_layer = nn.Dropout(0.0 if prompt_mode == "shallow" else 0.1)
        self.init_prompt()

    def freeze_backbone(self):
        self.backbone.freeze()
        self.backbone.eval()
        self.is_backbone_frozen = True

    def unfreeze_backbone(self):
        self.backbone.unfreeze()
        self.backbone.train(self.training)
        self.is_backbone_frozen = False

    def train(self, mode: bool = True):
        super().train(mode)

        if self.is_backbone_frozen:
            self.backbone.eval()

        return self

    def eval(self):
        return self.train(False)

    def init_head(self, init_from: Optional[nn.Linear] = None):
        if init_from is not None:
            if (
                init_from.weight.shape != self.head.weight.shape
                or init_from.bias.shape != self.head.bias.shape
            ):
                raise ValueError(
                    f"Head shape mismatch: expected {tuple(self.head.weight.shape)}, "
                    f"got {tuple(init_from.weight.shape)}."
                )

            with torch.no_grad():
                self.head.weight.copy_(init_from.weight)
                self.head.bias.copy_(init_from.bias)
            return

        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def init_prompt(self, init_from: Optional[nn.Parameter] = None):
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

        patch_size = (self.backbone.patch_size, self.backbone.patch_size)
        prompt_dim = self.backbone.embed_dim
        val = math.sqrt(6.0 / float(3 * reduce(mul, patch_size, 1) + prompt_dim))
        nn.init.uniform_(self.prompt.data, -val, val)

    def get_prompt_tokens(self, batch_size: int, layer_idx: int = 0) -> torch.Tensor:
        prompt = self.prompt[layer_idx if self.prompt_mode == "deep" else 0]
        prompt = prompt.unsqueeze(0).expand(batch_size, -1, -1)
        return self.prompt_dropout_layer(prompt)

    def encode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Runs transformer blocks + final norm with VPT prompt injection.

        Input tokens must contain only:
        [CLS, PATCH_1, ..., PATCH_N]

        Output tokens contain:
        [CLS, PROMPT_1, ..., PROMPT_P, PATCH_1, ..., PATCH_N]
        """
        h = tokens
        batch_size = h.shape[0]

        if self.prompt_length == 0:
            return self.backbone.encode_tokens(h)

        if self.prompt_mode == "shallow":
            prompt = self.get_prompt_tokens(batch_size)
            h = torch.cat([h[:, :1], prompt, h[:, 1:]], dim=1)

            for blk in self.backbone.blocks:
                h, _ = blk(h)

            return self.backbone.norm(h)

        if self.prompt_mode == "deep":
            for layer_idx, blk in enumerate(self.backbone.blocks):
                prompt = self.get_prompt_tokens(batch_size, layer_idx)

                if layer_idx == 0:
                    h = torch.cat([h[:, :1], prompt, h[:, 1:]], dim=1)
                else:
                    h = torch.cat(
                        [h[:, :1], prompt, h[:, 1 + self.prompt_length :]],
                        dim=1,
                    )

                h, _ = blk(h)

            return self.backbone.norm(h)

        raise ValueError(f"Invalid prompt_mode: {self.prompt_mode}")

    def forward_features(self, x: torch.Tensor):
        """
        Standard VPT forward.
        Returns final CLS and patch tokens.
        """
        batch_size = x.shape[0]

        patch_tokens = self.backbone.get_patch_tokens(x)
        cls_token = self.backbone.get_cls_token(batch_size)
        cls_pos, patch_pos = self.backbone.get_positional_embeddings()

        tokens = torch.cat(
            [cls_token + cls_pos, patch_tokens + patch_pos],
            dim=1,
        )
        tokens = self.backbone.input_dropout(tokens)
        encoded = self.encode_tokens(tokens)

        prompt_offset = self.prompt_length if self.prompt_length > 0 else 0

        return {
            "cls": encoded[:, 0],
            "prompt_tokens": encoded[:, 1 : 1 + prompt_offset],
            "patch_tokens": encoded[:, 1 + prompt_offset :],
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        return self.head(features["cls"])

    def compute_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ):
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)

        loss_components = {
            "cross_entropy_loss": loss.item(),
        }

        return logits, loss, loss_components

    def get_param_count(self):
        total_count = sum(p.numel() for p in self.parameters())
        backbone_count = sum(p.numel() for p in self.backbone.parameters())
        trainable_count = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "total_count": total_count,
            "backbone_count": backbone_count,
            "trainable_count": trainable_count,
            "trainable_over_total": (
                trainable_count / total_count if total_count > 0 else 0.0
            ),
            "trainable_over_backbone": (
                trainable_count / backbone_count if backbone_count > 0 else 0.0
            ),
        }

    def save(self, path: str):
        """
        Save only VPT trainable parameters from model state_dict:
        prompt tokens and classification head.
        """
        trainable_state_dict = {
            key: value.detach().cpu()
            for key, value in self.state_dict().items()
            if key == "prompt" or key.startswith("head.")
        }

        torch.save(trainable_state_dict, path)

    def load_from_path(self, path: str, map_location="cpu"):
        """
        Load only prompt tokens and classification head.
        """
        trainable_state_dict = torch.load(path, map_location=map_location)

        self.load_state_dict(trainable_state_dict, strict=False)

    def _create_component_gates(self):
        """Create zero log-gates and instrument each transformer block once."""
        if hasattr(self, "fim_gates") and hasattr(self, "prompt_gates"):
            return

        blocks = self.backbone.blocks
        n_layers = len(blocks)
        n_heads = blocks[0].attn.num_attention_heads
        n_prompts = self.prompt_length
        n_patches = self.backbone.num_patches
        prompt_end = 1 + n_prompts
        n_tokens = prompt_end + n_patches
        prompt_slice = slice(1, prompt_end)
        patch_slice = slice(prompt_end, None)
        reference = next(self.backbone.parameters())

        def gate(shape):
            return nn.Parameter(reference.new_zeros(shape), requires_grad=False)

        self.fim_gates = nn.ParameterDict(
            {
                # [layer, branch], branch order: attention, MLP
                "residual_path": gate((n_layers, 2)),
                # [layer, attention head]
                "attention_head": gate((n_layers, n_heads)),
                # [layer, destination, source]
                # destination/source order: CLS, PATCH
                "attention_transport": gate((n_layers, 2, 2)),
                # [layer, token type], token type order: CLS, PATCH
                "mlp_token_output": gate((n_layers, 2)),
                # [layer, prompt token, direction, backbone token type]
                # direction 0: backbone <- prompt
                # direction 1: prompt <- backbone
                # backbone token type order: CLS, PATCH
                "tokenwise_prompt_backbone_transport": gate(
                    (n_layers, n_prompts, 2, 2)
                ),
            }
        )
        # [layer, prompt token]
        self.prompt_gates = gate((n_layers, n_prompts))

        def scale(value, tensor):
            return value.to(tensor).exp()

        def make_forward(layer_idx):
            def forward(block, hidden):
                gates, attn = self.fim_gates, block.attn
                residual = hidden
                x = block.attention_norm(hidden)

                # Prompt participation is gated after LayerNorm and before QKV.
                if n_prompts:
                    prompt_scale = scale(self.prompt_gates[layer_idx], x)
                    x = torch.cat(
                        (
                            x[:, :1],
                            x[:, prompt_slice] * prompt_scale[None, :, None],
                            x[:, patch_slice],
                        ),
                        dim=1,
                    )

                query = attn.transpose_for_scores(attn.query(x))
                key = attn.transpose_for_scores(attn.key(x))
                value = attn.transpose_for_scores(attn.value(x))

                attention_probs = attn.softmax(
                    query @ key.transpose(-1, -2) / math.sqrt(attn.attention_head_size)
                )
                weights = attention_probs if attn.vis else None
                attention_probs = attn.attn_dropout(attention_probs)

                backbone_transport = scale(
                    gates["attention_transport"][layer_idx],
                    attention_probs,
                )
                token_transport = scale(
                    gates["tokenwise_prompt_backbone_transport"][layer_idx],
                    attention_probs,
                )

                # transport[destination, source]; prompt-to-prompt stays ungated.
                transport = attention_probs.new_ones((n_tokens, n_tokens))
                transport[0, 0] = backbone_transport[0, 0]
                transport[0, prompt_slice] = token_transport[:, 0, 0]
                transport[0, patch_slice] = backbone_transport[0, 1]
                transport[prompt_slice, 0] = token_transport[:, 1, 0]
                transport[prompt_slice, patch_slice] = token_transport[:, 1, 1, None]
                transport[patch_slice, 0] = backbone_transport[1, 0]
                transport[patch_slice, prompt_slice] = token_transport[:, 0, 1]
                transport[patch_slice, patch_slice] = backbone_transport[1, 1]

                context = (attention_probs * transport[None, None]) @ value
                context = (
                    context
                    * scale(
                        gates["attention_head"][layer_idx],
                        context,
                    )[None, :, None, None]
                )

                batch_size, _, current_tokens, _ = context.shape
                context = context.transpose(1, 2).reshape(
                    batch_size,
                    current_tokens,
                    attn.all_head_size,
                )
                attention_output = attn.proj_dropout(attn.out(context))
                hidden = (
                    residual
                    + scale(
                        gates["residual_path"][layer_idx, 0],
                        attention_output,
                    )
                    * attention_output
                )

                residual = hidden
                mlp_output = block.ffn(block.ffn_norm(hidden))
                token_scale = scale(
                    gates["mlp_token_output"][layer_idx],
                    mlp_output,
                )
                mlp_output = torch.cat(
                    (
                        mlp_output[:, :1] * token_scale[0],
                        mlp_output[:, prompt_slice],
                        mlp_output[:, patch_slice] * token_scale[1],
                    ),
                    dim=1,
                )
                hidden = (
                    residual
                    + scale(
                        gates["residual_path"][layer_idx, 1],
                        mlp_output,
                    )
                    * mlp_output
                )

                return hidden, weights

            return forward

        for layer_idx, block in enumerate(blocks):
            block.forward = MethodType(make_forward(layer_idx), block)

    def _set_gates_grad(
        self,
        fim: bool = True,
        prompt: bool = False,
    ):
        """Set gradient mode for the FIM and prompt diagnostic gates."""
        for gate in self.fim_gates.values():
            gate.grad = None
            gate.requires_grad_(fim)

        self.prompt_gates.grad = None
        self.prompt_gates.requires_grad_(prompt)

    def estimate_sensitivity_profiles(
        self,
        loader,
        prompt: bool = False,
        mc_samples: int = 20,
        component_chunk_size: int = 32,
        eps: float = 1e-12,
    ):
        """Estimate F and, optionally, T, L and G for every gate family."""
        if prompt and self.prompt_length == 0:
            raise ValueError("Prompt sensitivity requires prompt_length > 0.")

        self._create_component_gates()
        self.eval()

        gate_names, fim_gates = zip(*self.fim_gates.items())
        gate_shapes = tuple(gate.shape for gate in fim_gates)
        gate_sizes = tuple(gate.numel() for gate in fim_gates)
        tokenwise_idx = gate_names.index("tokenwise_prompt_backbone_transport")

        aggregate_shape = (len(self.backbone.blocks), 2, 2)
        names = (*gate_names, "prompt_backbone_transport")
        shapes = (*gate_shapes, aggregate_shape)
        sizes = (*gate_sizes, math.prod(aggregate_shape))
        n_components = sum(sizes)

        self._set_gates_grad(fim=True, prompt=prompt)

        with torch.no_grad():
            for gate in (*fim_gates, self.prompt_gates):
                gate.zero_()

        device = self.prompt_gates.device
        dtype = self.prompt_gates.dtype
        fisher_sum = self.prompt_gates.new_zeros(n_components)
        derivative_sum = (
            self.prompt_gates.new_zeros((n_components, *self.prompt_gates.shape))
            if prompt
            else None
        )
        component_basis = (
            torch.eye(n_components, device=device, dtype=dtype) if prompt else None
        )
        sample_bases = {}
        sample_count = 0

        for batch in loader:
            x = batch[0].to(device, non_blocking=True)
            log_probs = self(x).log_softmax(dim=-1)
            labels = torch.multinomial(
                log_probs.exp(),
                mc_samples,
                replacement=True,
            )
            selected = log_probs.gather(1, labels).flatten()
            n_samples = selected.numel()

            sample_basis = sample_bases.get(n_samples)
            if sample_basis is None:
                sample_basis = sample_bases[n_samples] = torch.eye(
                    n_samples,
                    device=device,
                    dtype=selected.dtype,
                )

            gate_scores = torch.autograd.grad(
                selected,
                fim_gates,
                grad_outputs=sample_basis,
                create_graph=prompt,
                retain_graph=prompt,
                is_grads_batched=True,
            )

            # Summing scores before squaring exactly recovers the shared gate.
            aggregate_score = gate_scores[tokenwise_idx].sum(dim=2).flatten(1)
            fisher_samples = torch.cat(
                [score.flatten(1) for score in gate_scores] + [aggregate_score],
                dim=1,
            ).square()

            fisher_sum.add_(fisher_samples.detach().sum(dim=0))
            sample_count += n_samples

            if prompt:
                # The detached term adds the predictive-distribution derivative.
                surrogate = (
                    fisher_samples + fisher_samples.detach() * selected[:, None]
                ).sum(dim=0)

                for start in range(
                    0,
                    n_components,
                    component_chunk_size,
                ):
                    stop = min(
                        start + component_chunk_size,
                        n_components,
                    )
                    derivative_sum[start:stop].add_(
                        torch.autograd.grad(
                            surrogate,
                            self.prompt_gates,
                            grad_outputs=component_basis[start:stop],
                            retain_graph=stop < n_components,
                            is_grads_batched=True,
                        )[0].detach()
                    )

        fisher = fisher_sum / sample_count

        def unpack(profile):
            return {
                name: part.reshape(
                    *profile.shape[:-1],
                    *shape,
                )
                for name, shape, part in zip(
                    names,
                    shapes,
                    profile.split(sizes, dim=-1),
                )
            }

        profiles = {"F": unpack(fisher)}

        if prompt:
            derivative = (derivative_sum / sample_count).movedim(0, -1)
            token = derivative / (fisher + eps)
            layer = token.sum(dim=1)

            profiles.update(
                T=unpack(token),
                L=unpack(layer),
                G=unpack(layer.sum(dim=0)),
            )

        self._set_gates_grad(fim=False, prompt=False)
        self.sensitivity_profiles = profiles
        return profiles
