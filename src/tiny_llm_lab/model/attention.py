"""Explicit causal multi-head self-attention."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from tiny_llm_lab.config import ModelConfig
from tiny_llm_lab.interventions import InterventionSet


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.embedding_dim // config.num_heads
        self.query_key_value = nn.Linear(config.embedding_dim, 3 * config.embedding_dim)
        self.output_projection = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.attention_dropout = nn.Dropout(config.dropout)
        self.output_dropout = nn.Dropout(config.dropout)
        mask = torch.tril(torch.ones(config.context_length, config.context_length, dtype=torch.bool))
        self.register_buffer("causal_mask", mask.view(1, 1, config.context_length, config.context_length))

    def forward(
        self,
        inputs: Tensor,
        return_attention: bool = False,
        interventions: InterventionSet | None = None,
        layer_index: int | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        batch_size, sequence_length, embedding_dim = inputs.shape
        query, key, value = self.query_key_value(inputs).chunk(3, dim=-1)

        def split_heads(tensor: Tensor) -> Tensor:
            return tensor.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)

        query = split_heads(query)
        key = split_heads(key)
        value = split_heads(value)
        scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~self.causal_mask[:, :, :sequence_length, :sequence_length], float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        dropped_weights = self.attention_dropout(weights)
        attended = dropped_weights @ value
        if interventions is not None and layer_index is not None and interventions.enabled:
            attended = interventions.apply_attention_output(layer_index, attended)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, sequence_length, embedding_dim)
        output = self.output_dropout(self.output_projection(attended))
        return output, weights if return_attention else None
