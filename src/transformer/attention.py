"""Attention primitives used across the transformer architecture."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute attention weights and the resulting value combination."""

    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    attention_weights = torch.softmax(scores, dim=-1)
    weighted_sum = torch.matmul(attention_weights, value)
    return weighted_sum, attention_weights


class MultiHeadAttention(nn.Module):
    """Standard multi-head attention over token representations."""

    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape tensor to expose the head dimension."""

        batch_size, seq_len, d_model = x.size()
        d_k = d_model // self.num_heads
        return x.view(batch_size, seq_len, self.num_heads, d_k).transpose(1, 2)

    def _prepare_mask(
        self,
        mask: torch.Tensor,
        target_shape: Tuple[int, int, int, int],
    ) -> torch.Tensor:
        """Broadcast masks so they align with attention score tensors."""

        batch, heads, seq_len_q, seq_len_k = target_shape

        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        if mask.dim() == 4 and mask.size(1) == 1 and heads > 1:
            mask = mask.expand(-1, heads, -1, -1)
        if mask.size(-2) == 1 and seq_len_q > 1:
            mask = mask.expand(-1, -1, seq_len_q, -1)
        if mask.size(-1) == 1 and seq_len_k > 1:
            mask = mask.expand(-1, -1, -1, seq_len_k)
        return mask

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = self.split_heads(self.W_q(query))
        k = self.split_heads(self.W_k(key))
        v = self.split_heads(self.W_v(value))

        if mask is not None:
            mask = self._prepare_mask(mask, (q.size(0), q.size(1), q.size(2), k.size(2)))

        attn_output, _ = scaled_dot_product_attention(q, k, v, mask)

        batch_size, _, seq_len, _ = attn_output.size()
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.W_o(attn_output)
