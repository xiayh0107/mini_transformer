"""Fine-tuning script for mini-transformer."""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.config import config
from config.pretrain_config import pretrain_config
from src.device_utils import get_device, verify_gpu_usage
from src.pretrain import (
    FlexibleTextDataset,
    PretrainDataCollator,
    build_pretrain_model,
)

IGNORE_INDEX = -100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mini-transformer fine-tuning script")
    parser.add_argument("--corpus", type=Path, default=ROOT / "data" / "finetune.txt")
    parser.add_argument("--model-path", type=Path, default=pretrain_config.OUTPUT_PATH)
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "finetuned_model.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-len", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    return parser.parse_args()


def train() -> None:
    args = parse_args()

    # Use FlexibleTextDataset for fine-tuning
    dataset = FlexibleTextDataset(args.corpus, max_seq_len=args.max_len)
    
    # We use CLM task for fine-tuning
    collator = PretrainDataCollator(task="clm", mask_prob=0.0)
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=collator,
    )

    device, device_info = get_device()
    print(f"\n[Fine-tune] Using device: {device_info}")

    # Load pretrained model
    model = build_pretrain_model(
        task="clm",
        vocab_size=len(config.VOCAB),
        num_layers=pretrain_config.NUM_LAYERS,
    )
    
    if args.model_path.exists():
        state_dict = torch.load(args.model_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"✅ Loaded pretrained model from {args.model_path}")
    else:
        print(f"⚠️ Pretrained model not found at {args.model_path}, starting from scratch.")

    model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(1, args.epochs + 1):
        running_loss = 0.0
        for step, batch in enumerate(dataloader, start=1):
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            look_ahead_mask = batch["look_ahead_mask"].to(device)

            logits = model(input_ids, look_ahead_mask=look_ahead_mask)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Loss {avg_loss:.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output)
    print(f"\n✅ Saved fine-tuned model to {args.output}")


if __name__ == "__main__":
    train()
