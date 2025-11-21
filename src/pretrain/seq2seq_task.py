"""Seq2Seq (Encoder-Decoder) pretraining task helpers."""

from __future__ import annotations

from typing import Sequence

import torch

from src.mask_utils import create_decoder_mask, create_padding_mask
from .clm_task import IGNORE_INDEX
from .mlm_task import MaskingParams, mask_tokens


class Seq2SeqModelingTask:
    """
    Prepare Seq2Seq batches (BART-style Denoising Autoencoder).
    
    Encoder input: Noised/Masked sequence.
    Decoder input: Original sequence (shifted).
    Labels: Original sequence.
    """

    def __init__(
        self,
        vocab_size: int,
        mask_token_id: int,
        pad_token_id: int,
        special_token_ids: Sequence[int],
        mask_prob: float = 0.15,
    ) -> None:
        self.params = MaskingParams(
            vocab_size=vocab_size,
            mask_token_id=mask_token_id,
            pad_token_id=pad_token_id,
            special_token_ids=special_token_ids,
            mask_prob=mask_prob,
        )

    def build_batch(self, input_ids: torch.Tensor):
        # 1. Prepare Encoder Input (Masked)
        # We ignore the labels returned by mask_tokens because for Seq2Seq 
        # we want to reconstruct the WHOLE sequence, not just masked parts.
        masked_input_ids, _ = mask_tokens(input_ids, self.params)
        
        # 2. Prepare Decoder Input and Labels (Shifted)
        # Assuming input_ids has [BOS] ... [EOS] structure or similar.
        # decoder_input: [BOS] t1 t2 ... tn
        # labels:        t1 t2 ... tn [EOS]
        decoder_input_ids = input_ids[:, :-1]
        labels = input_ids[:, 1:].clone()
        
        # Mask padding in labels
        labels = labels.masked_fill(labels == self.params.pad_token_id, IGNORE_INDEX)

        # 3. Create Masks
        # Encoder padding mask
        enc_padding_mask = create_padding_mask(masked_input_ids, self.params.pad_token_id)
        
        # Decoder look-ahead mask (and padding mask combined usually in create_decoder_mask)
        # Note: create_decoder_mask usually handles both look-ahead and padding for the decoder input
        decoder_attention_mask = create_decoder_mask(decoder_input_ids, self.params.pad_token_id)

        return {
            "input_ids": masked_input_ids,              # Encoder input
            "decoder_input_ids": decoder_input_ids,     # Decoder input
            "labels": labels,                           # Target
            "attention_mask": enc_padding_mask,         # Encoder mask
            "decoder_attention_mask": decoder_attention_mask # Decoder mask
        }
