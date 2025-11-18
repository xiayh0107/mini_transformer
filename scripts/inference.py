# inference.py - 模型推理模块
# 人话：让训练好的模型进行预测

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from typing import List, Optional

from config.config import config, reverse_vocab
from scripts.data import tokenize_sequence, sequence_to_ids
from src.transformer import MiniTransformer
from src.mask_utils import create_decoder_mask, create_padding_mask


def _max_position_length(model: MiniTransformer) -> Optional[int]:
    """Return positional encoding capacity if available."""

    pos_encoding = getattr(model, "pos_encoding", None)
    pe = getattr(pos_encoding, "pe", None)
    if pe is None:
        return None
    return pe.size(1)

def text_to_ids(text: str, vocab: dict, max_len: Optional[int] = None) -> List[int]:
    """
    人话：把文本转换成ID序列
    例如: "a b c" -> [0, 1, 2]
    """
    tokens = tokenize_sequence(text)
    return sequence_to_ids(tokens, vocab, max_len)

def ids_to_text(ids: List[int], vocab: dict) -> str:
    """
    人话：把ID序列转换成文本
    例如: [0, 1, 2] -> "a b c"
    """
    tokens = [reverse_vocab.get(id, '<unk>') for id in ids]
    # 过滤掉特殊符号
    tokens = [t for t in tokens if t not in ['<pad>', '<sos>', '<eos>', '<unk>']]
    return ' '.join(tokens)

def predict(model: MiniTransformer, input_text: str, max_length: int = 10) -> str:
    """
    人话：用模型预测输入文本的输出
    
    输入:
        model: 训练好的模型
        input_text: 输入文本（例如 "a b c"）
        max_length: 最大生成长度
    
    返回: 预测的输出文本
    """
    vocab = config.VOCAB
    
    # 获取模型所在的设备
    device = next(model.parameters()).device

    max_pos_len = _max_position_length(model)
    
    # 1. 把输入文本转换成ID
    input_ids = text_to_ids(input_text, vocab)
    if not input_ids:
        return ""

    if max_pos_len is not None and len(input_ids) > max_pos_len:
        raise ValueError(
            f"输入序列长度为 {len(input_ids)}，超过位置编码最大长度 {max_pos_len}"
        )
    
    # 2. 转换为tensor并移到设备上
    encoder_input = torch.tensor([input_ids], dtype=torch.long, device=device)
    enc_mask = create_padding_mask(encoder_input)
    
    # 3. 生成过程：从<sos>开始，一步步预测
    generated = [vocab['<sos>']]
    
    if max_pos_len is not None:
        max_decode_steps = max(0, min(max_length, max_pos_len - len(generated)))
    else:
        max_decode_steps = max_length

    if max_decode_steps == 0:
        return ""

    with torch.inference_mode():
        for _ in range(max_decode_steps):
            # 准备decoder输入并移到设备上
            dec_input = torch.tensor([generated], dtype=torch.long, device=device)
            
            # 创建掩码（使用正确的工具函数）
            # look_ahead_mask: [batch, dec_len, dec_len] 包含look-ahead和padding信息
            dec_mask = create_decoder_mask(dec_input)  # [1, dec_len, dec_len]
            
            # 模型预测
            logits = model(encoder_input, dec_input, look_ahead_mask=dec_mask, enc_padding_mask=enc_mask)
            
            # 获取下一个token（使用greedy decoding）
            next_token = torch.argmax(logits[0, -1], dim=-1).item()
            
            # 如果遇到<eos>，停止生成
            if next_token == vocab['<eos>']:
                break
            
            generated.append(next_token)
    
    # 4. 转换回文本（跳过<sos>）
    output_ids = generated[1:]  # 跳过<sos>
    output_text = ids_to_text(output_ids, vocab)
    
    return output_text

def load_model(model_path: Optional[str] = None) -> Optional[MiniTransformer]:
    """
    人话：加载保存的模型
    
    输入:
        model_path: 模型文件路径
    
    返回: 加载的模型，如果失败返回None
    """
    try:
        # 自动检测设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        resolved_path = model_path or config.MODEL_SAVE_PATH

        model = MiniTransformer()
        # 加载模型权重（自动检测设备）
        model.load_state_dict(torch.load(resolved_path, map_location=device))
        model = model.to(device)  # 移到GPU（如果可用）
        model.eval()  # 设置为评估模式
        
        device_info = "GPU" if torch.cuda.is_available() else "CPU"
        print(f"✅ 成功加载模型: {resolved_path} (设备: {device_info})")
        return model
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        return None


def save_model(model: MiniTransformer, model_path: Optional[str] = None):
    """
    人话：保存模型到文件
    
    输入:
        model: 要保存的模型
        model_path: 保存路径
    """
    try:
        resolved_path = model_path or config.MODEL_SAVE_PATH
        torch.save(model.state_dict(), resolved_path)
        print(f"✅ 模型已保存到: {resolved_path}")
    except Exception as e:
        print(f"❌ 保存模型失败: {e}")


def _build_arg_parser() -> "argparse.ArgumentParser":
    """Create CLI parser for running inference from the shell."""
    import argparse

    parser = argparse.ArgumentParser(description="Run MiniTransformer inference.")
    parser.add_argument("--input", required=True, help="Input sequence, e.g. 'a b c'.")
    parser.add_argument("--model", default=None, help="Optional custom model checkpoint path.")
    parser.add_argument(
        "--max-length",
        type=int,
        default=10,
        help="Maximum number of tokens to generate (default: 10).",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    model = load_model(args.model)
    if model is None:
        return 1

    try:
        output = predict(model, args.input, max_length=args.max_length)
    except ValueError as exc:
        print(f"❌ 推理失败: {exc}")
        return 1

    if output:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
