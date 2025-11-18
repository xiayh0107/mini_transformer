"""Re-export transformer components from the modular implementation.

The full architecture now lives under ``src.transformer`` so each building
block has its own file (attention, encoder, decoder, etc.).  This lightweight
module keeps the historical import path ``src.model`` intact while pointing
readers to the reorganised code structure.
"""

from .transformer import (
    DecoderLayer,
    MiniTransformer,
    Seq2SeqTransformer,
    MultiHeadAttention,
    PositionalEncoding,
    TransformerDecoderStack,
    TransformerEncoderStack,
    EncoderLayer,
    scaled_dot_product_attention,
)

DecoderStack = TransformerDecoderStack
EncoderStack = TransformerEncoderStack

__all__ = [
    "PositionalEncoding",
    "scaled_dot_product_attention",
    "MultiHeadAttention",
    "EncoderLayer",
    "DecoderLayer",
    "EncoderStack",
    "DecoderStack",
    "TransformerEncoderStack",
    "TransformerDecoderStack",
    "MiniTransformer",
    "Seq2SeqTransformer",
]

