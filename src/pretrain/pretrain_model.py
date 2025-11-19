"""Model wrappers focused on pretraining workloads."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from config.config import config
from src.transformer.encoder import TransformerEncoderLayer, TransformerEncoderStack
from src.transformer.positional_encoding import SinusoidalPositionalEncoding


class MaskedLanguageModel(nn.Module):
    """Encoder-only architecture with an MLM classification head."""

    def __init__(
        self,
        vocab_size: Optional[int] = None,
        d_model: Optional[int] = None,
        num_heads: Optional[int] = None,
        num_layers: Optional[int] = None,
    ) -> None:
        super().__init__()
        vocab_size = vocab_size or len(config.VOCAB)
        d_model = d_model or config.D_MODEL
        num_heads = num_heads or config.NUM_HEADS
        num_layers = num_layers or getattr(config, "NUM_ENCODER_LAYERS", 2)

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len=512)
        encoder_layers = [TransformerEncoderLayer(d_model, num_heads) for _ in range(num_layers)]
        self.encoder = TransformerEncoderStack(encoder_layers)
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        embeddings = self.embedding(input_ids)
        embeddings = self.pos_encoding(embeddings)
        hidden_states = self.encoder(embeddings, padding_mask=attention_mask)
        hidden_states = self.norm(hidden_states)
        return self.lm_head(hidden_states)


class CausalLanguageModel(nn.Module):
    """Decoder-only transformer implemented via masked self-attention blocks."""

    def __init__(
        self,
        vocab_size: Optional[int] = None,
        d_model: Optional[int] = None,
        num_heads: Optional[int] = None,
        num_layers: Optional[int] = None,
    ) -> None:
        super().__init__()
        vocab_size = vocab_size or len(config.VOCAB)
        d_model = d_model or config.D_MODEL
        num_heads = num_heads or config.NUM_HEADS
        num_layers = num_layers or getattr(config, "NUM_DECODER_LAYERS", 2)

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len=512)
        decoder_layers = [TransformerEncoderLayer(d_model, num_heads) for _ in range(num_layers)]
        self.decoder = TransformerEncoderStack(decoder_layers)
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor, look_ahead_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        embeddings = self.embedding(input_ids)
        embeddings = self.pos_encoding(embeddings)
        hidden_states = self.decoder(embeddings, padding_mask=look_ahead_mask)
        hidden_states = self.norm(hidden_states)
        return self.lm_head(hidden_states)


MODEL_REGISTRY = {
    "mlm": MaskedLanguageModel,
    "clm": CausalLanguageModel,
}


def build_pretrain_model(task: str, **kwargs) -> nn.Module:
    task = task.lower()
    if task not in MODEL_REGISTRY:
        raise ValueError("task must be 'mlm' or 'clm'")
    return MODEL_REGISTRY[task](**kwargs)
