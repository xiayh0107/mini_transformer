import argparse
import sys
from pathlib import Path
from typing import List, Optional

import torch

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.config import config, reverse_vocab
from config.pretrain_config import pretrain_config
from src.device_utils import get_device
from src.mask_utils import create_decoder_mask
from src.pretrain import build_pretrain_model
from src.pretrain.data_collator import LineByLineTextDataset


def load_model(model_path: Path, device: torch.device) -> torch.nn.Module:
    """Load the pretrained CLM model."""
    model = build_pretrain_model(
        task="clm",
        vocab_size=len(config.VOCAB),
        num_layers=pretrain_config.NUM_LAYERS,
    )
    if model_path.exists():
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"✅ Loaded model from {model_path}")
    else:
        print(f"⚠️ Model file not found at {model_path}, using random weights.")
    
    model.to(device)
    model.eval()
    return model


def generate(
    model: torch.nn.Module,
    input_text: str,
    device: torch.device,
    max_length: int = 20,
) -> str:
    """Generate text autoregressively."""
    vocab = config.VOCAB
    
    # Tokenize input
    # Always use character-level tokenization to handle "ab c", "abc", "a b c" consistently.
    # This matches the logic in FlexibleTextDataset used for fine-tuning.
    input_tokens = [c for c in input_text if c.strip()]

    if not input_tokens:
        return ""

    # Add <sos> and input tokens
    tokens = ["<sos>"] + input_tokens
    
    # Add <sep> if it exists in vocab
    if "<sep>" in vocab:
        tokens.append("<sep>")
    
    input_ids = [vocab.get(t, vocab.get("<unk>")) for t in tokens]
    
    # Start generation
    current_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    
    with torch.inference_mode():
        for _ in range(max_length):
            # Create mask for decoder
            # For CLM, we just need the look_ahead_mask for the current sequence length
            # The model expects: input_ids, look_ahead_mask
            
            # Note: The CausalLanguageModel.forward takes (input_ids, look_ahead_mask)
            # and internally handles embeddings and pos encoding.
            
            look_ahead_mask = create_decoder_mask(current_ids, pad_token_id=vocab["<pad>"])
            
            logits = model(current_ids, look_ahead_mask=look_ahead_mask)
            
            # Get next token from the last position
            next_token_logits = logits[0, -1, :]
            next_token_id = torch.argmax(next_token_logits).item()
            
            # Stop if <eos>
            if next_token_id == vocab["<eos>"]:
                break
                
            # Append to sequence
            current_ids = torch.cat(
                [current_ids, torch.tensor([[next_token_id]], device=device)], dim=1
            )
    
    # Decode output
    # We only want the part AFTER the input (and <sep>)
    full_ids = current_ids[0].tolist()
    
    # Find where generation started (after input_ids)
    generated_ids = full_ids[len(input_ids):]
    
    output_tokens = [reverse_vocab.get(i, "<unk>") for i in generated_ids]
    return " ".join(output_tokens)


def main():
    parser = argparse.ArgumentParser(description="Generate text using pretrained CLM model")
    parser.add_argument("--input", type=str, required=True, help="Input text (e.g. 'a b c')")
    parser.add_argument("--model", type=Path, default=pretrain_config.OUTPUT_PATH, help="Path to model checkpoint")
    parser.add_argument("--max-len", type=int, default=20, help="Maximum generation length")
    
    args = parser.parse_args()
    
    device, _ = get_device()
    model = load_model(args.model, device)
    
    output = generate(model, args.input, device, args.max_len)
    print(f"\nInput: {args.input}")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
