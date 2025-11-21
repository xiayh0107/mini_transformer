"""Transformer encoder stack broken down by layer."""

from __future__ import annotations

from typing import Iterable, Optional

import torch
import torch.nn as nn

from .attention import MultiHeadAttention


class TransformerEncoderLayer(nn.Module):
    """Single encoder layer: self-attention followed by feed-forward."""

    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attn_out = self.self_attn(x, x, x, mask=padding_mask)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        return self.norm2(x + ffn_out)


class TransformerEncoderStack(nn.Module):
    """Group multiple encoder layers into a stack."""

    def __init__(self, layers: Iterable[TransformerEncoderLayer]) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, padding_mask=padding_mask)
        return x
