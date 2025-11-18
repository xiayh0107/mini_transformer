# test_chat.py - 快速测试对话功能
# 人话：测试推理和对话模块是否正常工作

import sys
from pathlib import Path
from typing import Optional

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.config import config
from scripts.inference import load_model, predict, save_model
from src.transformer import MiniTransformer
from scripts.chat import ChatBot


def test_inference(tmp_path):
    """测试推理功能使用临时模型检查加载与预测。"""

    model = MiniTransformer()
    model_path = tmp_path / "test_model.pt"
    save_model(model, str(model_path))

    loaded_model = load_model(str(model_path))
    assert loaded_model is not None, "加载模型失败"

    result = predict(loaded_model, "a b c")
    assert isinstance(result, str)


def test_chat_bot(tmp_path):
    """测试对话机器人，使用临时模型权重。"""

    model = MiniTransformer()
    model_path = tmp_path / "chat_model.pt"
    save_model(model, str(model_path))

    bot = ChatBot(str(model_path))

    response = bot.chat("a b c")
    assert isinstance(response, str)
    assert len(bot.conversation_history) >= 2

    bot.show_history(1)  # 确认不抛出异常
    bot.clear_history()
    assert bot.conversation_history == []

