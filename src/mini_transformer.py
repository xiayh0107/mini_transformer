"""Hands-on mini example that reuses the modular transformer components."""

import torch
import torch.nn as nn
import torch.optim as optim

from config.config import config, reverse_vocab
from .transformer import MiniTransformer
from .mask_utils import create_decoder_mask


VOCAB = config.VOCAB


def _toy_batch():
    """Create a batch for the letter-to-digit translation toy task."""

    encoder_inputs = torch.tensor(
        [
            [VOCAB["a"], VOCAB["b"], VOCAB["c"]],
            [VOCAB["b"], VOCAB["a"], VOCAB["c"]],
        ]
    )

    decoder_inputs = torch.tensor(
        [
            [VOCAB["<sos>"], VOCAB["1"], VOCAB["2"], VOCAB["3"]],
            [VOCAB["<sos>"], VOCAB["2"], VOCAB["1"], VOCAB["3"]],
        ]
    )

    targets = torch.tensor(
        [
            [VOCAB["1"], VOCAB["2"], VOCAB["3"], VOCAB["<eos>"]],
            [VOCAB["2"], VOCAB["1"], VOCAB["3"], VOCAB["<eos>"]],
        ]
    )

    return encoder_inputs, decoder_inputs, targets


def train_toy_model(epochs: int = 100) -> MiniTransformer:
    """Tiny training loop to illustrate how the pieces connect."""

    enc_inputs, dec_inputs, targets = _toy_batch()
    look_ahead_mask = create_decoder_mask(dec_inputs)

    model = MiniTransformer(vocab_size=len(VOCAB))
    criterion = nn.CrossEntropyLoss(ignore_index=VOCAB["<pad>"])
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = model(enc_inputs, dec_inputs, look_ahead_mask=look_ahead_mask)
        loss = criterion(logits.view(-1, len(VOCAB)), targets.view(-1))
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"epoch={epoch:03d}  loss={loss.item():.4f}")

    return model


def autoregressive_decode(model: MiniTransformer, src_tokens: torch.Tensor, max_len: int = 3):
    """Greedy decode loop for the toy dataset."""

    generated = [VOCAB["<sos>"]]
    for _ in range(max_len):
        dec_input = torch.tensor([generated])
        with torch.no_grad():
            logits = model(src_tokens, dec_input)
        next_token = torch.argmax(logits[0, -1], dim=-1).item()
        generated.append(next_token)
    return generated[1:]


if __name__ == "__main__":
    print("\n=== Mini Transformer Playground ===")
    model = train_toy_model()

    test_src = torch.tensor([[VOCAB["a"], VOCAB["b"], VOCAB["c"]]])
    output_ids = autoregressive_decode(model, test_src)
    output_tokens = [reverse_vocab[idx] for idx in output_ids]
    print(f"\nInput : ['a', 'b', 'c']")
    print(f"Output: {output_tokens}")