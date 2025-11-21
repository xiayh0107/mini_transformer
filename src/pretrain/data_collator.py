"""Dataset and collator utilities for pretraining tasks."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import torch
from torch.utils.data import Dataset

from config.config import config
from .mlm_task import MaskedLanguageModelingTask
from .clm_task import CausalLanguageModelingTask
from .seq2seq_task import Seq2SeqModelingTask


class LineByLineTextDataset(Dataset):
    """Load whitespace separated tokens from a text file and convert them to ids."""

    def __init__(
        self,
        file_path: Path,
        max_seq_len: int,
        vocab: Dict[str, int] | None = None,
        add_special_tokens: bool = True,
    ) -> None:
        super().__init__()
        self.vocab = vocab or config.VOCAB
        self.file_path = Path(file_path)
        self.max_seq_len = max_seq_len
        self.pad_token_id = self.vocab["<pad>"]
        self.sos_token_id = self.vocab["<sos>"]
        self.eos_token_id = self.vocab["<eos>"]
        self.sep_token_id = self.vocab.get("<sep>")
        self.add_special_tokens = add_special_tokens
        self.samples = self._load_corpus()

    def _token_to_id(self, token: str) -> int:
        return self.vocab.get(token, self.vocab.get("<unk>", 0))

    def _encode_line(self, line: str) -> List[int] | None:
        line = line.strip()
        if not line or line.startswith("#"):
            return None
        parts = line.split("\t")
        if not parts:
            return None

        input_text = parts[0].strip()
        output_text = parts[1].strip() if len(parts) > 1 else ""
        
        input_tokens = input_text.split()
        output_tokens = output_text.split()

        if not input_tokens:
            return None

        if self.add_special_tokens:
            # <sos> input <sep> output <eos>
            tokens = ["<sos>"] + input_tokens + ["<sep>"] + output_tokens + ["<eos>"]
        else:
            tokens = input_tokens + output_tokens

        token_ids = [self._token_to_id(token) for token in tokens]
        token_ids = token_ids[: self.max_seq_len]

        if len(token_ids) < self.max_seq_len:
            token_ids.extend([self.pad_token_id] * (self.max_seq_len - len(token_ids)))
        return token_ids

    def _load_corpus(self) -> List[torch.Tensor]:
        if not self.file_path.exists():
            raise FileNotFoundError(f"Corpus file not found: {self.file_path}")

        encoded: List[torch.Tensor] = []
        with self.file_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                token_ids = self._encode_line(line)
                if token_ids is None:
                    continue
                encoded.append(torch.tensor(token_ids, dtype=torch.long))
        if not encoded:
            raise ValueError("Corpus is empty after filtering comments/blank lines.")
        return encoded

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:  # type: ignore[override]
        return self.samples[idx]


class PretrainDataCollator:
    """Switch between MLM, CLM, and Seq2Seq batch preparation logic."""

    def __init__(self, task: str, mask_prob: float = 0.15, vocab: Dict[str, int] | None = None) -> None:
        vocab = vocab or config.VOCAB
        pad_token_id = vocab["<pad>"]
        mask_token_id = vocab["[MASK]"]
        special_token_ids = [vocab["<pad>"], vocab["<sos>"], vocab["<eos>"]]

        task = task.lower()
        if task not in {"mlm", "clm", "seq2seq"}:
            raise ValueError("task must be 'mlm', 'clm', or 'seq2seq'")
        self.task = task

        if task == "mlm":
            self.impl = MaskedLanguageModelingTask(
                vocab_size=len(vocab),
                mask_token_id=mask_token_id,
                pad_token_id=pad_token_id,
                special_token_ids=special_token_ids,
                mask_prob=mask_prob,
            )
        elif task == "seq2seq":
            self.impl = Seq2SeqModelingTask(
                vocab_size=len(vocab),
                mask_token_id=mask_token_id,
                pad_token_id=pad_token_id,
                special_token_ids=special_token_ids,
                mask_prob=mask_prob,
            )
        else:
            self.impl = CausalLanguageModelingTask(pad_token_id=pad_token_id)

    def __call__(self, batch: Iterable[torch.Tensor]) -> Dict[str, torch.Tensor]:
        stacked = torch.stack(list(batch))
        return self.impl.build_batch(stacked)


class FlexibleTextDataset(Dataset):
    """
    Dataset that handles flexible formatting (e.g. 'abc - 312').
    It treats '-' as the separator and tokenizes characters individually, ignoring spaces.
    """

    def __init__(
        self,
        file_path: Path,
        max_seq_len: int,
        vocab: Dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        self.vocab = vocab or config.VOCAB
        self.file_path = Path(file_path)
        self.max_seq_len = max_seq_len
        self.pad_token_id = self.vocab["<pad>"]
        self.sos_token_id = self.vocab["<sos>"]
        self.eos_token_id = self.vocab["<eos>"]
        self.sep_token_id = self.vocab.get("<sep>")
        self.samples = self._load_corpus()

    def _token_to_id(self, token: str) -> int:
        return self.vocab.get(token, self.vocab.get("<unk>", 0))

    def _encode_line(self, line: str) -> List[int] | None:
        line = line.strip()
        if not line or line.startswith("#"):
            return None
        
        # Split by '-'
        parts = line.split("-")
        if len(parts) != 2:
            return None

        input_raw = parts[0]
        output_raw = parts[1]

        # Tokenize characters, ignoring spaces
        input_tokens = [c for c in input_raw if c.strip()]
        output_tokens = [c for c in output_raw if c.strip()]

        if not input_tokens:
            return None

        # <sos> input <sep> output <eos>
        tokens = ["<sos>"] + input_tokens
        if self.sep_token_id is not None:
             tokens.append("<sep>")
        
        tokens = tokens + output_tokens + ["<eos>"]

        token_ids = [self._token_to_id(token) for token in tokens]
        token_ids = token_ids[: self.max_seq_len]

        if len(token_ids) < self.max_seq_len:
            token_ids.extend([self.pad_token_id] * (self.max_seq_len - len(token_ids)))
        return token_ids

    def _load_corpus(self) -> List[torch.Tensor]:
        if not self.file_path.exists():
            raise FileNotFoundError(f"Corpus file not found: {self.file_path}")

        encoded: List[torch.Tensor] = []
        with self.file_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                token_ids = self._encode_line(line)
                if token_ids is None:
                    continue
                encoded.append(torch.tensor(token_ids, dtype=torch.long))
        if not encoded:
            raise ValueError("Corpus is empty after filtering comments/blank lines.")
        return encoded

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.samples[idx]
