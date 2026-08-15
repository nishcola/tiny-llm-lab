"""A small pre-normalization decoder-only transformer."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as functional

from tiny_llm_lab.config import ModelConfig
from tiny_llm_lab.interventions import InterventionSet
from tiny_llm_lab.model.attention import CausalSelfAttention


@dataclass(frozen=True)
class InstrumentationRequest:
    """Select semantic model representations to capture during a forward pass."""

    attention_weights: bool = False
    hidden_states: bool = False
    attention_outputs: bool = False
    mlp_activations: bool = False
    mlp_activation_layer: int | None = None

    @property
    def enabled(self) -> bool:
        return any(
            (
                self.attention_weights,
                self.hidden_states,
                self.attention_outputs,
                self.mlp_activations,
                self.mlp_activation_layer is not None,
            )
        )


@dataclass(frozen=True)
class ModelInstrumentation:
    """Detached, caller-requested semantic representations from a model pass."""

    attention_weights: tuple[Tensor, ...] | None = None
    hidden_states: tuple[Tensor, ...] | None = None
    attention_outputs: tuple[Tensor, ...] | None = None
    mlp_activations: tuple[Tensor, ...] | None = None
    selected_mlp_activation: Tensor | None = None
    selected_mlp_activation_layer: int | None = None


@dataclass(frozen=True)
class ModelOutput:
    logits: Tensor
    loss: Tensor | None = None
    instrumentation: ModelInstrumentation | None = None

    @property
    def attentions(self) -> tuple[Tensor, ...] | None:
        """Compatibility alias for attention captures requested by older callers."""
        if self.instrumentation is None:
            return None
        return self.instrumentation.attention_weights


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config.embedding_dim, config.mlp_dim),
            nn.GELU(),
            nn.Linear(config.mlp_dim, config.embedding_dim),
            nn.Dropout(config.dropout),
        )

    def forward(
        self,
        inputs: Tensor,
        return_activation: bool = False,
        interventions: InterventionSet | None = None,
        layer_index: int | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        activations = self.layers[1](self.layers[0](inputs))
        if interventions is not None and layer_index is not None and interventions.enabled:
            activations = interventions.apply_mlp_activations(layer_index, activations)
        outputs = self.layers[3](self.layers[2](activations))
        return outputs, activations if return_activation else None


class DecoderBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.embedding_dim)
        self.attention = CausalSelfAttention(config)
        self.mlp_norm = nn.LayerNorm(config.embedding_dim)
        self.mlp = FeedForward(config)

    def forward(
        self,
        inputs: Tensor,
        instrumentation: InstrumentationRequest | None,
        interventions: InterventionSet | None = None,
        layer_index: int | None = None,
    ) -> tuple[Tensor, Tensor | None, Tensor | None, Tensor | None]:
        capture_attention_weights = instrumentation is not None and instrumentation.attention_weights
        capture_attention_outputs = instrumentation is not None and instrumentation.attention_outputs
        capture_mlp_activations = instrumentation is not None and instrumentation.mlp_activations
        attention_output, weights = self.attention(
            self.attention_norm(inputs),
            return_attention=capture_attention_weights,
            interventions=interventions,
            layer_index=layer_index,
        )
        hidden_states = inputs + attention_output
        mlp_output, mlp_activations = self.mlp(
            self.mlp_norm(hidden_states),
            return_activation=capture_mlp_activations,
            interventions=interventions,
            layer_index=layer_index,
        )
        hidden_states = hidden_states + mlp_output
        return (
            hidden_states,
            weights,
            attention_output if capture_attention_outputs else None,
            mlp_activations,
        )


class DecoderOnlyTransformer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.vocabulary_size is None:
            raise ValueError("vocabulary_size must be set before constructing the model")
        self.config = config
        self.token_embeddings = nn.Embedding(config.vocabulary_size, config.embedding_dim)
        self.position_embeddings = nn.Embedding(config.context_length, config.embedding_dim)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.num_layers)])
        self.final_norm = nn.LayerNorm(config.embedding_dim)
        self.language_model_head = nn.Linear(config.embedding_dim, config.vocabulary_size, bias=False)
        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: Tensor,
        targets: Tensor | None = None,
        instrumentation: InstrumentationRequest | None = None,
        return_attentions: bool | None = None,
        interventions: InterventionSet | None = None,
    ) -> ModelOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch, sequence)")
        batch_size, sequence_length = input_ids.shape
        if sequence_length > self.config.context_length:
            raise ValueError("sequence length exceeds the configured context length")
        if targets is not None and targets.shape != input_ids.shape:
            raise ValueError("targets must have the same shape as input_ids")
        if return_attentions is not None:
            if instrumentation is not None:
                raise ValueError("use either instrumentation or return_attentions, not both")
            instrumentation = InstrumentationRequest(attention_weights=return_attentions)
        if interventions is not None and interventions.enabled:
            interventions.validate(
                num_layers=self.config.num_layers,
                num_heads=self.config.num_heads,
                mlp_dim=self.config.mlp_dim,
            )

        positions = torch.arange(sequence_length, device=input_ids.device)
        hidden_states = self.token_embeddings(input_ids) + self.position_embeddings(positions)
        hidden_states = self.embedding_dropout(hidden_states)
        capture = instrumentation if instrumentation is not None and instrumentation.enabled else None
        if capture is not None and capture.mlp_activation_layer is not None:
            if not 0 <= capture.mlp_activation_layer < self.config.num_layers:
                raise ValueError(
                    f"MLP activation layer must be between 0 and {self.config.num_layers - 1}"
                )
        captured_attentions: list[Tensor] = []
        captured_hidden_states: list[Tensor] = []
        captured_attention_outputs: list[Tensor] = []
        captured_mlp_activations: list[Tensor] = []
        selected_mlp_activation: Tensor | None = None
        if capture is not None and capture.hidden_states:
            captured_hidden_states.append(hidden_states.detach())
        for layer_index, block in enumerate(self.blocks):
            block_capture = capture
            if capture is not None and capture.mlp_activation_layer is not None:
                block_capture = InstrumentationRequest(
                    attention_weights=capture.attention_weights,
                    hidden_states=capture.hidden_states,
                    attention_outputs=capture.attention_outputs,
                    mlp_activations=capture.mlp_activations,
                    mlp_activation_layer=None,
                )
                if layer_index == capture.mlp_activation_layer:
                    block_capture = InstrumentationRequest(
                        attention_weights=capture.attention_weights,
                        hidden_states=capture.hidden_states,
                        attention_outputs=capture.attention_outputs,
                        mlp_activations=True,
                    )
            hidden_states, weights, attention_output, mlp_activations = block(
                hidden_states,
                block_capture,
                interventions=interventions,
                layer_index=layer_index,
            )
            if weights is not None:
                captured_attentions.append(weights.detach())
            if capture is not None and capture.hidden_states:
                captured_hidden_states.append(hidden_states.detach())
            if attention_output is not None:
                captured_attention_outputs.append(attention_output.detach())
            if mlp_activations is not None:
                if capture is not None and capture.mlp_activation_layer is not None:
                    if layer_index == capture.mlp_activation_layer:
                        selected_mlp_activation = mlp_activations.detach()
                else:
                    captured_mlp_activations.append(mlp_activations.detach())
        logits = self.language_model_head(self.final_norm(hidden_states))
        loss = None
        if targets is not None:
            loss = functional.cross_entropy(
                logits.reshape(batch_size * sequence_length, -1),
                targets.reshape(batch_size * sequence_length),
            )
        captured = None
        if capture is not None:
            captured = ModelInstrumentation(
                attention_weights=tuple(captured_attentions) if capture.attention_weights else None,
                hidden_states=tuple(captured_hidden_states) if capture.hidden_states else None,
                attention_outputs=tuple(captured_attention_outputs) if capture.attention_outputs else None,
                mlp_activations=tuple(captured_mlp_activations) if capture.mlp_activations else None,
                selected_mlp_activation=selected_mlp_activation,
                selected_mlp_activation_layer=capture.mlp_activation_layer,
            )
        return ModelOutput(logits=logits, loss=loss, instrumentation=captured)
