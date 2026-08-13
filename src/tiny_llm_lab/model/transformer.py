"""A small pre-normalization decoder-only transformer."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as functional

from tiny_llm_lab.config import ModelConfig
from tiny_llm_lab.model.attention import CausalSelfAttention


@dataclass(frozen=True)
class ModelOutput:
    logits: Tensor
    loss: Tensor | None = None
    attentions: tuple[Tensor, ...] | None = None


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config.embedding_dim, config.mlp_dim),
            nn.GELU(),
            nn.Linear(config.mlp_dim, config.embedding_dim),
            nn.Dropout(config.dropout),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.layers(inputs)


class DecoderBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.embedding_dim)
        self.attention = CausalSelfAttention(config)
        self.mlp_norm = nn.LayerNorm(config.embedding_dim)
        self.mlp = FeedForward(config)

    def forward(self, inputs: Tensor, return_attention: bool) -> tuple[Tensor, Tensor | None]:
        attention_output, weights = self.attention(
            self.attention_norm(inputs),
            return_attention=return_attention,
        )
        hidden_states = inputs + attention_output
        hidden_states = hidden_states + self.mlp(self.mlp_norm(hidden_states))
        return hidden_states, weights


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
        return_attentions: bool = False,
    ) -> ModelOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch, sequence)")
        batch_size, sequence_length = input_ids.shape
        if sequence_length > self.config.context_length:
            raise ValueError("sequence length exceeds the configured context length")
        if targets is not None and targets.shape != input_ids.shape:
            raise ValueError("targets must have the same shape as input_ids")

        positions = torch.arange(sequence_length, device=input_ids.device)
        hidden_states = self.token_embeddings(input_ids) + self.position_embeddings(positions)
        hidden_states = self.embedding_dropout(hidden_states)
        captured_attentions: list[Tensor] = []
        for block in self.blocks:
            hidden_states, weights = block(hidden_states, return_attention=return_attentions)
            if weights is not None:
                captured_attentions.append(weights)
        logits = self.language_model_head(self.final_norm(hidden_states))
        loss = None
        if targets is not None:
            loss = functional.cross_entropy(
                logits.reshape(batch_size * sequence_length, -1),
                targets.reshape(batch_size * sequence_length),
            )
        return ModelOutput(
            logits=logits,
            loss=loss,
            attentions=tuple(captured_attentions) if return_attentions else None,
        )

