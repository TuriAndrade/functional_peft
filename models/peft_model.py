import math
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from types import MethodType
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from .utils import load_or_create_spectral_basis
from .transformer_backbone import TransformerBackbone


class PEFTModel(nn.Module, ABC):
    """Shared PEFT model and functional-Fisher framework."""

    def __init__(
        self,
        backbone: TransformerBackbone,
        n_classes: int,
        freeze_backbone: bool = True,
    ):
        if n_classes < 2:
            raise ValueError("n_classes must be >= 2")

        super().__init__()

        self.backbone = backbone
        self.n_classes = int(n_classes)
        self.is_backbone_frozen = bool(freeze_backbone)

        if freeze_backbone:
            self.freeze_backbone()

        self.head = nn.Linear(self.backbone.embed_dim, n_classes)
        self.init_head()

    # ======================================================================
    # Model interface
    # ======================================================================

    def freeze_backbone(self) -> None:
        self.backbone.freeze()
        self.backbone.eval()
        self.is_backbone_frozen = True

    def unfreeze_backbone(self) -> None:
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

    def init_head(
        self,
        init_from: nn.Linear | None = None,
    ) -> None:
        if init_from is None:
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

            return

        if (
            init_from.weight.shape != self.head.weight.shape
            or init_from.bias is None
            or init_from.bias.shape != self.head.bias.shape
        ):

            raise ValueError(
                f"Head shape mismatch: expected "
                f"weight={tuple(self.head.weight.shape)}, "
                f"bias={tuple(self.head.bias.shape)}."
            )

        with torch.no_grad():
            self.head.weight.copy_(init_from.weight)
            self.head.bias.copy_(init_from.bias)

    @abstractmethod
    def init_peft(
        self,
        init_from: Any = None,
    ) -> None: ...

    def forward_features(
        self,
        x: torch.Tensor,
    ) -> Mapping[str, torch.Tensor]:

        return self.backbone.forward_features(x)

    def get_layerwise_features(
        self,
        x: torch.Tensor,
    ) -> Mapping[str, torch.Tensor]:

        return self.backbone.get_layerwise_features(x)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return self.head(self.forward_features(x)["cls"])

    def compute_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ):

        logits = self.forward(x)

        loss = F.cross_entropy(logits, y)

        return (
            logits,
            loss,
            {"cross_entropy_loss": loss.item()},
        )

    def get_param_count(self):
        params = [
            parameter
            for name, parameter in self.named_parameters()
            if not name.startswith("functional_gates.")
        ]

        total_count = sum(parameter.numel() for parameter in params)

        backbone_count = sum(
            parameter.numel() for parameter in self.backbone.parameters()
        )

        trainable_count = sum(
            parameter.numel() for parameter in params if parameter.requires_grad
        )

        return {
            "total_count": total_count,
            "backbone_count": backbone_count,
            "trainable_count": trainable_count,
            "trainable_over_total": (
                trainable_count / total_count if total_count else 0.0
            ),
            "trainable_over_backbone": (
                trainable_count / backbone_count if backbone_count else 0.0
            ),
        }

    @abstractmethod
    def save(
        self,
        path: str,
    ) -> None: ...

    def load_from_path(
        self,
        path: str,
        map_location="cpu",
    ) -> None:

        self.load_state_dict(
            torch.load(
                path,
                map_location=map_location,
            ),
            strict=False,
        )

    # ======================================================================
    # Functional spectral analysis
    # ======================================================================

    def _make_gate(self, shape: tuple[int, ...]) -> nn.Parameter:
        reference = next(self.backbone.parameters())
        return nn.Parameter(reference.new_ones(shape), requires_grad=False)

    def _functional_extra_mlp_update(
        self,
        layer_idx: int,
        value: torch.Tensor,
    ) -> torch.Tensor | None:
        """Optional parallel MLP-side update. AdaptFormer overrides this method."""
        return None

    def _functional_shapes(self) -> dict[str, tuple[int, ...]]:
        layers = self.backbone.num_layers
        return {
            "attention_qk": (layers, self.backbone.num_heads, self.backbone.head_dim),
            "attention_ov": (layers, self.backbone.num_heads, self.backbone.head_dim),
            "mlp_in": (layers, self.backbone.embed_dim),
            "mlp_out": (layers, self.backbone.embed_dim),
        }

    def _basis(self, name: str, layer_idx: int) -> torch.Tensor:
        return getattr(self, f"_functional_basis_{name}")[layer_idx]

    def prepare_functional_analysis(
        self,
        basis_cache_path: str = "functional_spectral_basis.pt",
    ) -> None:
        """Lazily create the shared spectral basis, gates, and instrumented forward."""
        config = str(basis_cache_path)
        if getattr(self, "_functional_config", None) == config:
            return

        layers = range(self.backbone.num_layers)
        basis = load_or_create_spectral_basis(
            tuple(self.backbone.get_attention_input_weight(i) for i in layers),
            tuple(self.backbone.get_attention_output_weight(i) for i in layers),
            tuple(self.backbone.get_mlp_input_weight(i) for i in layers),
            tuple(self.backbone.get_mlp_output_weight(i) for i in layers),
            num_heads=self.backbone.num_heads,
            head_dim=self.backbone.head_dim,
            cache_path=basis_cache_path,
            allow_create=self.is_backbone_frozen,
        )

        reference = next(self.backbone.parameters())
        for name, value in basis.items():
            buffer_name = f"_functional_basis_{name}"
            value = value.to(device=reference.device, dtype=reference.dtype)
            if buffer_name in self._buffers:
                setattr(self, buffer_name, value)
            else:
                self.register_buffer(buffer_name, value, persistent=False)

        if not hasattr(self, "functional_gates"):
            self.functional_gates = nn.ParameterDict(
                {
                    name: self._make_gate(shape)
                    for name, shape in self._functional_shapes().items()
                }
            )

        if not getattr(self, "_functional_forward_installed", False):
            self.install_functional_forward()
            self._functional_forward_installed = True

        self._functional_config = config

    def _record_functional_participation(
        self,
        name: str,
        value: torch.Tensor,
    ) -> None:
        participation = getattr(self, "_functional_participation", None)
        if participation is not None:
            participation[name].append(value.detach())

    # ======================================================================
    # Functional interventions
    # ======================================================================

    def _apply_attention_qk(
        self,
        scores: torch.Tensor,
        value: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        u = self._basis("attention_qk_u", layer_idx)
        s = self._basis("attention_qk_s", layer_idx)
        v = self._basis("attention_qk_v", layer_idx)
        gates = self.functional_gates["attention_qk"][layer_idx]

        left = torch.einsum("btd,hdr->bhtr", value, u)
        right = torch.einsum("btd,hdr->bhtr", value, v)
        scale = s / math.sqrt(self.backbone.head_dim)

        self._record_functional_participation(
            "attention_qk",
            scale.square()[None] * left.square().sum(dim=2) * right.square().sum(dim=2),
        )

        delta = torch.einsum(
            "bhtr,bhsr,hr->bhts",
            left,
            right,
            scale * (gates - 1),
        )
        return scores + delta

    def _apply_attention_ov(
        self,
        output: torch.Tensor,
        value: torch.Tensor,
        attention_probs: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        u = self._basis("attention_ov_u", layer_idx)
        s = self._basis("attention_ov_s", layer_idx)
        v = self._basis("attention_ov_v", layer_idx)
        gates = self.functional_gates["attention_ov"][layer_idx]

        read = torch.einsum("btd,hdr->bhtr", value, v)
        transported = torch.einsum("bhts,bhsr->bhtr", attention_probs, read)

        self._record_functional_participation(
            "attention_ov",
            s.square()[None] * transported.square().sum(dim=2),
        )

        delta = torch.einsum(
            "bhtr,hr,hdr->btd",
            transported,
            s * (gates - 1),
            u,
        )
        return output + delta

    def _apply_mlp_in(
        self,
        preactivation: torch.Tensor,
        value: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        u = self._basis("mlp_in_u", layer_idx)
        s = self._basis("mlp_in_s", layer_idx)
        v = self._basis("mlp_in_v", layer_idx)
        gates = self.functional_gates["mlp_in"][layer_idx]

        read = value @ v
        self._record_functional_participation(
            "mlp_in",
            s.square()[None] * read.square().sum(dim=1),
        )

        delta = (read * (s * (gates - 1))) @ u.T
        return preactivation + delta

    def _apply_mlp_out(
        self,
        output: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        u = self._basis("mlp_out_u", layer_idx)
        gates = self.functional_gates["mlp_out"][layer_idx]
        coordinates = output @ u

        self._record_functional_participation(
            "mlp_out",
            coordinates.square().sum(dim=1),
        )

        return output + (coordinates * (gates - 1)) @ u.T

    # ======================================================================
    # Shared instrumented Transformer forward
    # ======================================================================

    def install_functional_forward(self) -> None:
        n_heads = self.backbone.num_heads
        head_dim = self.backbone.head_dim

        def make_forward(layer_idx):
            def forward(block, hidden):
                residual = hidden
                attention = block.attn
                value = block.attention_norm(hidden)

                query = attention.transpose_for_scores(attention.query(value))
                key = attention.transpose_for_scores(attention.key(value))
                attention_value = attention.transpose_for_scores(attention.value(value))

                scores = query @ key.transpose(-1, -2) / math.sqrt(head_dim)
                scores = self._apply_attention_qk(scores, value, layer_idx)
                attention_weights = attention.softmax(scores)
                weights = attention_weights if attention.vis else None
                attention_probs = attention.attn_dropout(attention_weights)

                context = attention_probs @ attention_value
                context = context.transpose(1, 2).reshape(
                    value.shape[0],
                    value.shape[1],
                    n_heads * head_dim,
                )
                attention_output = attention.out(context)
                attention_output = self._apply_attention_ov(
                    attention_output,
                    value,
                    attention_probs,
                    layer_idx,
                )
                hidden = residual + attention.proj_dropout(attention_output)

                residual = hidden
                feed_forward = block.ffn
                value = block.ffn_norm(hidden)

                preactivation = feed_forward.fc1(value)
                preactivation = self._apply_mlp_in(preactivation, value, layer_idx)
                mlp_hidden = feed_forward.dropout(feed_forward.act_fn(preactivation))
                mlp_output = feed_forward.dropout(feed_forward.fc2(mlp_hidden))

                extra_update = self._functional_extra_mlp_update(layer_idx, value)
                if extra_update is not None:
                    mlp_output = mlp_output + extra_update

                mlp_output = self._apply_mlp_out(mlp_output, layer_idx)
                hidden = residual + mlp_output
                return hidden, weights

            return forward

        for layer_idx, block in enumerate(self.backbone.blocks):
            block.forward = MethodType(make_forward(layer_idx), block)

    # ======================================================================
    # Functional gate lifecycle
    # ======================================================================

    def reset_functional_gates(self) -> None:
        with torch.no_grad():
            for gate in self.functional_gates.values():
                gate.fill_(1)

    def set_functional_gate_grads(self, mode: bool = True) -> None:
        for gate in self.functional_gates.values():
            gate.grad = None
            gate.requires_grad_(mode)

    # ======================================================================
    # Exact diagonal Fisher estimator
    # ======================================================================

    def _exact_diagonal_fisher(
        self,
        log_probs: torch.Tensor,
        grad_chunk_size: int,
    ) -> dict[str, torch.Tensor]:
        gate_names, gates = zip(*self.functional_gates.items())
        batch_size, n_classes = log_probs.shape
        flat_log_probs = log_probs.reshape(-1)
        probabilities = flat_log_probs.detach().exp()
        n_outputs = flat_log_probs.numel()

        fisher = {
            name: gate.new_zeros(batch_size, gate.numel())
            for name, gate in self.functional_gates.items()
        }

        for start in range(0, n_outputs, grad_chunk_size):
            stop = min(start + grad_chunk_size, n_outputs)
            indices = torch.arange(start, stop, device=log_probs.device)
            grad_outputs = log_probs.new_zeros(stop - start, n_outputs)
            grad_outputs[
                torch.arange(stop - start, device=log_probs.device), indices
            ] = 1

            gradients = torch.autograd.grad(
                flat_log_probs,
                gates,
                grad_outputs=grad_outputs,
                retain_graph=stop < n_outputs,
                is_grads_batched=True,
            )

            sample_indices = torch.div(indices, n_classes, rounding_mode="floor")
            weights = probabilities[indices, None]
            for name, gradient in zip(gate_names, gradients, strict=True):
                contribution = gradient.reshape(stop - start, -1).square() * weights
                fisher[name].index_add_(0, sample_indices, contribution)

        return {
            name: value.reshape(batch_size, *self.functional_gates[name].shape)
            for name, value in fisher.items()
        }

    def estimate_functional_signature(
        self,
        loader,
        basis_cache_path: str = "functional_spectral_basis.pt",
        eps: float = 1e-12,
        grad_chunk_size: int = 64,
        verbose: bool = False,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Estimate exact diagonal F, participation P, and sample-normalized S."""
        if grad_chunk_size < 1:
            raise ValueError("grad_chunk_size must be positive")

        self.prepare_functional_analysis(basis_cache_path)
        gate_state = {
            name: (gate.detach().clone(), gate.requires_grad)
            for name, gate in self.functional_gates.items()
        }
        was_training = self.training
        self.reset_functional_gates()
        self.set_functional_gate_grads(True)
        self.eval()

        sums = {
            statistic: {
                name: torch.zeros(gate.shape, dtype=torch.float64)
                for name, gate in self.functional_gates.items()
            }
            for statistic in ("F", "P", "S")
        }
        count = 0
        iterator = tqdm(loader, desc="Functional signature") if verbose else loader

        try:
            for batch in iterator:
                x = batch[0] if isinstance(batch, Sequence) else batch
                x = x.to(next(self.parameters()).device, non_blocking=True)

                self._functional_participation = {
                    name: [] for name in self.functional_gates
                }
                log_probs = self(x).log_softmax(dim=-1)
                participation = {
                    name: torch.stack(values, dim=1)
                    for name, values in self._functional_participation.items()
                }
                fisher = self._exact_diagonal_fisher(log_probs, grad_chunk_size)
                self._functional_participation = None

                for name in self.functional_gates:
                    p = participation[name].double().cpu()
                    f = fisher[name].detach().double().cpu()
                    s = f / (p + eps)
                    sums["P"][name].add_(p.sum(dim=0))
                    sums["F"][name].add_(f.sum(dim=0))
                    sums["S"][name].add_(s.sum(dim=0))

                count += x.shape[0]

            if count == 0:
                raise ValueError("loader is empty")

            return {
                statistic: {name: value / count for name, value in components.items()}
                for statistic, components in sums.items()
            }
        finally:
            self._functional_participation = None
            with torch.no_grad():
                for name, gate in self.functional_gates.items():
                    value, requires_grad = gate_state[name]
                    gate.copy_(value)
                    gate.grad = None
                    gate.requires_grad_(requires_grad)
            self.train(was_training)
