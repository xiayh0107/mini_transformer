import torch

from config.config import config
from src.pretrain.mlm_task import MaskingParams, mask_tokens, IGNORE_INDEX
from src.pretrain.clm_task import shift_tokens_for_clm


def test_mask_tokens_keep_special_symbols():
    vocab = config.VOCAB
    sample = torch.tensor([
        [vocab["<sos>"], vocab["a"], vocab["b"], vocab["<eos>"]],
    ])
    params = MaskingParams(
        vocab_size=len(vocab),
        mask_token_id=vocab["[MASK]"],
        pad_token_id=vocab["<pad>"],
        special_token_ids=[vocab["<sos>"], vocab["<eos>"]],
        mask_prob=1.0,
    )
    masked, labels = mask_tokens(sample, params)

    assert masked[0, 0].item() == vocab["<sos>"]
    assert masked[0, -1].item() == vocab["<eos>"]
    assert labels[0, 0].item() == IGNORE_INDEX
    assert labels[0, -1].item() == IGNORE_INDEX
    assert (labels == IGNORE_INDEX).sum() < labels.numel(), "middle tokens should be supervised"


def test_shift_tokens_for_clm_produces_offset_sequences():
    vocab = config.VOCAB
    sample = torch.tensor([
        [vocab["<sos>"], vocab["a"], vocab["b"], vocab["<eos>"], vocab["<pad>"]]
    ])
    batch = shift_tokens_for_clm(sample, pad_token_id=vocab["<pad>"])

    assert batch["input_ids"].shape[-1] == sample.shape[-1] - 1
    assert batch["labels"].shape == batch["input_ids"].shape
    assert batch["input_ids"][0, 0].item() == vocab["<sos>"]
    assert batch["labels"][0, -1].item() == IGNORE_INDEX
