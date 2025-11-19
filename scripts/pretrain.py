"""Minimal pretraining loop that supports MLM and CLM tasks."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config.config import config
from config.pretrain_config import pretrain_config
from src.device_utils import get_device, verify_gpu_usage
from src.pretrain import LineByLineTextDataset, PretrainDataCollator, build_pretrain_model

IGNORE_INDEX = -100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mini-transformer pretraining script")
    parser.add_argument("--task", choices=["mlm", "clm"], default=pretrain_config.TASK)
    parser.add_argument("--corpus", type=Path, default=pretrain_config.CORPUS_PATH)
    parser.add_argument("--output", type=Path, default=pretrain_config.OUTPUT_PATH)
    parser.add_argument("--epochs", type=int, default=pretrain_config.NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=pretrain_config.BATCH_SIZE)
    parser.add_argument("--max-len", type=int, default=pretrain_config.MAX_SEQ_LEN)
    parser.add_argument("--lr", type=float, default=pretrain_config.LEARNING_RATE)
    parser.add_argument("--mask-prob", type=float, default=pretrain_config.MASK_PROB)
    parser.add_argument("--num-layers", type=int, default=pretrain_config.NUM_LAYERS)
    parser.add_argument("--log-interval", type=int, default=pretrain_config.LOG_INTERVAL)
    parser.add_argument("--grad-clip", type=float, default=pretrain_config.GRAD_CLIP)
    return parser.parse_args()


def train() -> None:
    args = parse_args()

    dataset = LineByLineTextDataset(args.corpus, max_seq_len=args.max_len)
    collator = PretrainDataCollator(task=args.task, mask_prob=args.mask_prob)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=collator,
    )

    device, device_info = get_device()
    print(f"\n[Pretrain] Using device: {device_info}")

    model = build_pretrain_model(
        task=args.task,
        vocab_size=len(config.VOCAB),
        num_layers=args.num_layers,
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    sample_batch = next(iter(dataloader))
    sample_input = sample_batch["input_ids"]
    verify_gpu_usage(model, sample_input.to(device))

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for step, batch in enumerate(dataloader, start=1):
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            if args.task == "mlm":
                attention_mask = batch["attention_mask"].to(device)
                logits = model(input_ids, attention_mask=attention_mask)
            else:
                look_ahead_mask = batch["look_ahead_mask"].to(device)
                logits = model(input_ids, look_ahead_mask=look_ahead_mask)

            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss.backward()
            if args.grad_clip is not None and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            running_loss += loss.item()

            if step % args.log_interval == 0:
                avg = running_loss / args.log_interval
                print(
                    f"Epoch {epoch:03d} | Step {step:04d} | Loss {avg:.4f} | Task {args.task.upper()}"
                )
                running_loss = 0.0

        if running_loss:
            avg = running_loss / max(1, step % args.log_interval)
            print(f"Epoch {epoch:03d} | Step {step:04d} | Loss {avg:.4f} | (final chunk)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output)
    print(f"\n✅ Saved pretraining checkpoint to {args.output}")


if __name__ == "__main__":
    train()
