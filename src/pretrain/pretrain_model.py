"""Model wrappers focused on pretraining workloads."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from config.config import config
from src.transformer.encoder import TransformerEncoderLayer, TransformerEncoderStack
from src.transformer.positional_encoding import SinusoidalPositionalEncoding
from src.transformer.model import Seq2SeqTransformer
from src.mask_utils import create_padding_mask, create_decoder_mask


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


class Seq2SeqLanguageModel(nn.Module):
    """Encoder-Decoder architecture for Seq2Seq tasks (e.g. translation, denoising)."""

    def __init__(
        self,
        vocab_size: Optional[int] = None,
        d_model: Optional[int] = None,
        num_heads: Optional[int] = None,
        num_encoder_layers: Optional[int] = None,
        num_decoder_layers: Optional[int] = None,
        num_layers: Optional[int] = None,
    ) -> None:
        super().__init__()
        if num_layers is not None:
            if num_encoder_layers is None:
                num_encoder_layers = num_layers
            if num_decoder_layers is None:
                num_decoder_layers = num_layers

        self.model = Seq2SeqTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # input_ids -> encoder_input
        # decoder_input_ids -> decoder_input
        # attention_mask -> enc_padding_mask
        # decoder_attention_mask -> look_ahead_mask (usually generated inside or passed)

        # Note: Seq2SeqTransformer.forward takes:
        # encoder_input, decoder_input, look_ahead_mask, enc_padding_mask

        return self.model(
            encoder_input=input_ids,
            decoder_input=decoder_input_ids,
            look_ahead_mask=decoder_attention_mask,
            enc_padding_mask=attention_mask,
        )


MODEL_REGISTRY = {
    "mlm": MaskedLanguageModel,
    "clm": CausalLanguageModel,
    "seq2seq": Seq2SeqLanguageModel,
}


def build_pretrain_model(task: str, **kwargs) -> nn.Module:
    task = task.lower()
    if task not in MODEL_REGISTRY:
        raise ValueError("task must be 'mlm' or 'clm'")
    return MODEL_REGISTRY[task](**kwargs)
