import torch

from config.config import config
from src.pretrain.mlm_task import MaskingParams, mask_tokens, IGNORE_INDEX
from src.pretrain.clm_task import shift_tokens_for_clm
from src.pretrain.seq2seq_task import Seq2SeqModelingTask


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


def test_seq2seq_task_build_batch():
    vocab = config.VOCAB
    vocab_size = len(vocab)
    mask_token_id = vocab["[MASK]"]
    pad_token_id = vocab["<pad>"]
    special_token_ids = [vocab["<sos>"], vocab["<eos>"], vocab["<pad>"]]

    task = Seq2SeqModelingTask(
        vocab_size=vocab_size,
        mask_token_id=mask_token_id,
        pad_token_id=pad_token_id,
        special_token_ids=special_token_ids,
        mask_prob=0.5
    )

    # Create dummy input: batch_size=2, seq_len=10
    # Ensure we use valid token ids from vocab
    input_ids = torch.randint(3, vocab_size, (2, 10))

    batch = task.build_batch(input_ids)

    assert "input_ids" in batch
    assert "decoder_input_ids" in batch
    assert "labels" in batch
    assert "attention_mask" in batch
    assert "decoder_attention_mask" in batch

    # Check shapes
    assert batch["input_ids"].shape == (2, 10)
    assert batch["decoder_input_ids"].shape == (2, 9)  # shifted
    assert batch["labels"].shape == (2, 9)  # shifted

    # Check decoder input is shifted input_ids
    assert torch.equal(batch["decoder_input_ids"], input_ids[:, :-1])
    
    # Check labels are shifted input_ids (ignoring masked padding)
    expected_labels = input_ids[:, 1:].clone()
    expected_labels[expected_labels == pad_token_id] = -100
    assert torch.equal(batch["labels"], expected_labels)