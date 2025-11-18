"""High-level sequence-to-sequence transformer assembly."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from config.config import config

from .decoder import TransformerDecoderLayer, TransformerDecoderStack
from .encoder import TransformerEncoderLayer, TransformerEncoderStack
from .positional_encoding import SinusoidalPositionalEncoding


class Seq2SeqTransformer(nn.Module):
    """Full encoder-decoder transformer with configurable depth."""

    def __init__(
        self,
        vocab_size: Optional[int] = None,
        d_model: Optional[int] = None,
        num_heads: Optional[int] = None,
        num_encoder_layers: Optional[int] = None,
        num_decoder_layers: Optional[int] = None,
    ) -> None:
        super().__init__()
        vocab_size = vocab_size or len(config.VOCAB)
        d_model = d_model or config.D_MODEL
        num_heads = num_heads or config.NUM_HEADS
        num_encoder_layers = num_encoder_layers or getattr(config, "NUM_ENCODER_LAYERS", 1)
        num_decoder_layers = num_decoder_layers or getattr(config, "NUM_DECODER_LAYERS", 1)

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len=100)

        encoder_layers = [
            TransformerEncoderLayer(d_model, num_heads) for _ in range(num_encoder_layers)
        ]
        decoder_layers = [
            TransformerDecoderLayer(d_model, num_heads) for _ in range(num_decoder_layers)
        ]

        self.encoder = TransformerEncoderStack(encoder_layers)
        self.decoder = TransformerDecoderStack(decoder_layers)

        self.final_proj = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        encoder_input: torch.Tensor,
        decoder_input: torch.Tensor,
        look_ahead_mask: Optional[torch.Tensor] = None,
        enc_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        enc_emb = self.embedding(encoder_input)
        dec_emb = self.embedding(decoder_input)

        enc_emb = self.pos_encoding(enc_emb)
        dec_emb = self.pos_encoding(dec_emb)

        enc_output = self.encoder(enc_emb, padding_mask=enc_padding_mask)

        if enc_padding_mask is not None:
            batch_size, _, _, enc_len = enc_padding_mask.shape
            dec_len = decoder_input.size(1)
            enc_dec_mask = enc_padding_mask.expand(batch_size, 1, dec_len, enc_len)
        else:
            enc_dec_mask = None

        dec_output = self.decoder(
            dec_emb,
            enc_output,
            look_ahead_mask=look_ahead_mask,
            enc_padding_mask=enc_dec_mask,
        )

        return self.final_proj(dec_output)
