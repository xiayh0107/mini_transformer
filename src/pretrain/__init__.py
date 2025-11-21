"""Utility modules that turn the mini transformer into a small pretraining lab."""

from .mlm_task import mask_tokens, MaskedLanguageModelingTask
from .clm_task import shift_tokens_for_clm, CausalLanguageModelingTask
from .seq2seq_task import Seq2SeqModelingTask
from .pretrain_model import (
    MaskedLanguageModel,
    CausalLanguageModel,
    Seq2SeqLanguageModel,
    build_pretrain_model,
)
from .data_collator import LineByLineTextDataset, PretrainDataCollator, FlexibleTextDataset

__all__ = [
    "mask_tokens",
    "MaskedLanguageModelingTask",
    "shift_tokens_for_clm",
    "CausalLanguageModelingTask",
    "Seq2SeqModelingTask",
    "MaskedLanguageModel",
    "CausalLanguageModel",
    "Seq2SeqLanguageModel",
    "build_pretrain_model",
    "LineByLineTextDataset",
    "PretrainDataCollator",
    "FlexibleTextDataset",
]
