"""Immutable, inference-only modifications to selected model computations."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True)
class DisableAttentionHead:
    layer_index: int
    head_index: int


@dataclass(frozen=True)
class ScaleMLPActivation:
    layer_index: int
    unit_index: int
    scale: float


Intervention = DisableAttentionHead | ScaleMLPActivation


@dataclass(frozen=True, init=False)
class InterventionSet:
    """Small composable collection of temporary forward-pass interventions."""

    items: tuple[Intervention, ...]

    def __init__(self, *items: Intervention) -> None:
        object.__setattr__(self, "items", tuple(items))

    @property
    def enabled(self) -> bool:
        return bool(self.items)

    def validate(self, *, num_layers: int, num_heads: int, mlp_dim: int) -> None:
        for item in self.items:
            if not 0 <= item.layer_index < num_layers:
                raise ValueError(f"Intervention layer must be between 0 and {num_layers - 1}")
            if isinstance(item, DisableAttentionHead) and not 0 <= item.head_index < num_heads:
                raise ValueError(f"Attention head must be between 0 and {num_heads - 1}")
            if isinstance(item, ScaleMLPActivation):
                if not 0 <= item.unit_index < mlp_dim:
                    raise ValueError(f"MLP unit must be between 0 and {mlp_dim - 1}")
                if not float("-inf") < item.scale < float("inf"):
                    raise ValueError("MLP activation scale must be finite")

    def apply_attention_output(self, layer_index: int, attended: Tensor) -> Tensor:
        """Disable selected heads before concatenation and output projection."""
        factors = attended.new_ones(attended.shape[1])
        for item in self.items:
            if isinstance(item, DisableAttentionHead) and item.layer_index == layer_index:
                factors[item.head_index] = 0.0
        return attended * factors.view(1, -1, 1, 1)

    def apply_mlp_activations(self, layer_index: int, activations: Tensor) -> Tensor:
        """Scale selected post-GELU MLP units before the output projection."""
        factors = activations.new_ones(activations.shape[-1])
        for item in self.items:
            if isinstance(item, ScaleMLPActivation) and item.layer_index == layer_index:
                factors[item.unit_index] *= item.scale
        return activations * factors.view(1, 1, -1)
