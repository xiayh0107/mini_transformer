import torch
import pytest
from src.pretrain.seq2seq_task import Seq2SeqModelingTask
from src.pretrain.pretrain_model import Seq2SeqLanguageModel

def test_seq2seq_task_build_batch():
    vocab_size = 100
    mask_token_id = 99
    pad_token_id = 0
    special_token_ids = [0, 1, 2]
    
    task = Seq2SeqModelingTask(
        vocab_size=vocab_size,
        mask_token_id=mask_token_id,
        pad_token_id=pad_token_id,
        special_token_ids=special_token_ids,
        mask_prob=0.5 # High prob to ensure masking happens
    )
    
    # Create dummy input: batch_size=2, seq_len=10
    input_ids = torch.randint(3, vocab_size, (2, 10))
    # Ensure no special tokens in random part for simplicity, though task handles it
    
    batch = task.build_batch(input_ids)
    
    assert "input_ids" in batch
    assert "decoder_input_ids" in batch
    assert "labels" in batch
    assert "attention_mask" in batch
    assert "decoder_attention_mask" in batch
    
    # Check shapes
    assert batch["input_ids"].shape == (2, 10)
    assert batch["decoder_input_ids"].shape == (2, 9) # shifted
    assert batch["labels"].shape == (2, 9) # shifted
    
    # Check masking happened in encoder input
    # It's probabilistic, but with 0.5 it should likely happen. 
    # We can check if at least one token is different or mask token
    # But strictly, input_ids should be modified (masked)
    
    # Check decoder input is shifted input_ids
    assert torch.equal(batch["decoder_input_ids"], input_ids[:, :-1])
    
    # Check labels are shifted input_ids (ignoring padding logic for now as we didn't put padding)
    assert torch.equal(batch["labels"], input_ids[:, 1:])

def test_seq2seq_model_forward():
    vocab_size = 100
    d_model = 32
    model = Seq2SeqLanguageModel(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2
    )
    
    batch_size = 2
    enc_len = 10
    dec_len = 9
    
    input_ids = torch.randint(0, vocab_size, (batch_size, enc_len))
    decoder_input_ids = torch.randint(0, vocab_size, (batch_size, dec_len))
    
    # Dummy masks
    attention_mask = torch.ones(batch_size, 1, 1, enc_len) # 1 means valid
    # Wait, let's check mask_utils. create_padding_mask returns (batch, 1, 1, seq_len) where 1 is masked (padding) usually?
    # Let's check mask_utils.py to be sure.
    
    # For now just run forward pass
    output = model(
        input_ids=input_ids,
        decoder_input_ids=decoder_input_ids
    )
    
    assert output.shape == (batch_size, dec_len, vocab_size)
