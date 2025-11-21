"""Transformer building blocks broken into clear modules.

This package surfaces the pieces of the sequence-to-sequence transformer so that
beginners can navigate the architecture step by step:

positional_encoding  -> sinusoidal position information
attention            -> scaled dot-product + multi-head attention logic
encoder              -> encoder layer stack
decoder              -> decoder layer stack
model                -> full seq2seq assembly

The public symbols mirror the original names so existing imports keep working,
while newcomers can import the highlighted components directly.
"""

from .positional_encoding import SinusoidalPositionalEncoding
from .attention import scaled_dot_product_attention, MultiHeadAttention
from .encoder import TransformerEncoderLayer, TransformerEncoderStack
from .decoder import TransformerDecoderLayer, TransformerDecoderStack
from .model import Seq2SeqTransformer

PositionalEncoding = SinusoidalPositionalEncoding
EncoderLayer = TransformerEncoderLayer
DecoderLayer = TransformerDecoderLayer
MiniTransformer = Seq2SeqTransformer

__all__ = [
    "SinusoidalPositionalEncoding",
    "PositionalEncoding",
    "scaled_dot_product_attention",
    "MultiHeadAttention",
    "TransformerEncoderLayer",
    "TransformerEncoderStack",
    "EncoderLayer",
    "TransformerDecoderLayer",
    "TransformerDecoderStack",
    "DecoderLayer",
    "Seq2SeqTransformer",
    "MiniTransformer",
]
