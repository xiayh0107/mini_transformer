"""Transformer decoder stack broken down by layer."""

from __future__ import annotations

from typing import Iterable, Optional

import torch
import torch.nn as nn

from .attention import MultiHeadAttention


class TransformerDecoderLayer(nn.Module):
    """Single decoder layer with masked self-attn and encoder cross-attn."""

    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        self.masked_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.enc_dec_attn = self.cross_attn  # backward-compatible name
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        enc_output: torch.Tensor,
        look_ahead_mask: Optional[torch.Tensor] = None,
        enc_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        masked = self.masked_attn(x, x, x, mask=look_ahead_mask)
        x = self.norm1(x + masked)

        attended = self.cross_attn(x, enc_output, enc_output, mask=enc_padding_mask)
        x = self.norm2(x + attended)

        ffn_out = self.ffn(x)
        return self.norm3(x + ffn_out)


class TransformerDecoderStack(nn.Module):
    """Group multiple decoder layers into a stack."""

    def __init__(self, layers: Iterable[TransformerDecoderLayer]) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        x: torch.Tensor,
        enc_output: torch.Tensor,
        look_ahead_mask: Optional[torch.Tensor] = None,
        enc_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(
                x,
                enc_output,
                look_ahead_mask=look_ahead_mask,
                enc_padding_mask=enc_padding_mask,
            )
        return x
