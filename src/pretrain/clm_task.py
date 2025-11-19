"""Causal language modeling helpers."""

from __future__ import annotations

from typing import Dict

import torch

from src.mask_utils import create_decoder_mask
from .mlm_task import IGNORE_INDEX


def shift_tokens_for_clm(batch: torch.Tensor, pad_token_id: int) -> Dict[str, torch.Tensor]:
    """Create decoder inputs and labels by shifting to the right."""

    decoder_inputs = batch[:, :-1]
    labels = batch[:, 1:].clone()
    labels = labels.masked_fill(labels == pad_token_id, IGNORE_INDEX)
    return {"input_ids": decoder_inputs, "labels": labels}


class CausalLanguageModelingTask:
    """Prepare Causal LM batches (GPT-style)."""

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def build_batch(self, input_ids: torch.Tensor):
        batch = shift_tokens_for_clm(input_ids, self.pad_token_id)
        look_ahead_mask = create_decoder_mask(batch["input_ids"], pad_token_id=self.pad_token_id)
        batch["look_ahead_mask"] = look_ahead_mask
        return batch
