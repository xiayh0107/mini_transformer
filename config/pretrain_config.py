"""Default hyper-parameters for the pretraining scripts."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class PretrainConfig:
    """Configuration bundle focused on MLM/CLM style pretraining."""

    CORPUS_PATH = ROOT / "data" / "data.txt"
    OUTPUT_PATH = ROOT / "models" / "pretrain_model.pt"

    TASK = "mlm"  # mlm or clm
    MASK_PROB = 0.15

    MAX_SEQ_LEN = 64
    BATCH_SIZE = 64
    NUM_EPOCHS = 50
    LEARNING_RATE = 3e-4
    NUM_LAYERS = 4
    GRAD_CLIP = 1.0

    LOG_INTERVAL = 50


pretrain_config = PretrainConfig()
