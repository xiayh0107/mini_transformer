"""Masked language modeling utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import torch

from src.mask_utils import create_padding_mask

IGNORE_INDEX = -100


@dataclass
class MaskingParams:
    """Container for probabilities and vocabulary ids used during masking."""

    vocab_size: int
    mask_token_id: int
    pad_token_id: int
    special_token_ids: Sequence[int]
    mask_prob: float = 0.15


def _build_special_mask(input_ids: torch.Tensor, special_token_ids: Iterable[int]) -> torch.Tensor:
    special_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for token_id in special_token_ids:
        special_mask |= input_ids == token_id
    return special_mask


def mask_tokens(input_ids: torch.Tensor, params: MaskingParams) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply BERT-style masking strategy (80/10/10) to a batch of token ids."""

    if params.mask_prob <= 0:
        attention_mask = create_padding_mask(input_ids, params.pad_token_id)
        labels = torch.full_like(input_ids, IGNORE_INDEX)
        return input_ids, labels

    device = input_ids.device
    labels = input_ids.clone()

    special_mask = _build_special_mask(input_ids, params.special_token_ids)
    sampling_weights = torch.full_like(input_ids, params.mask_prob, dtype=torch.float)
    sampling_weights = sampling_weights.masked_fill(special_mask, 0.0)

    mask_choices = torch.bernoulli(sampling_weights).bool()
    labels = labels.masked_fill(~mask_choices, IGNORE_INDEX)

    masked_inputs = input_ids.clone()
    replacement_probs = torch.rand_like(input_ids, dtype=torch.float)

    mask_mask = mask_choices & (replacement_probs < 0.8)
    random_mask = mask_choices & (replacement_probs >= 0.8) & (replacement_probs < 0.9)
    # remaining 10% keep original token

    masked_inputs[mask_mask] = params.mask_token_id

    random_tokens = torch.randint(low=0, high=params.vocab_size, size=input_ids.size(), device=device)
    masked_inputs[random_mask] = random_tokens[random_mask]

    return masked_inputs, labels


class MaskedLanguageModelingTask:
    """Convenience helper that prepares MLM batches for the pretraining script."""

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
        masked_inputs, labels = mask_tokens(input_ids, self.params)
        attention_mask = create_padding_mask(masked_inputs, self.params.pad_token_id)
        return {
            "input_ids": masked_inputs,
            "labels": labels,
            "attention_mask": attention_mask,
        }
