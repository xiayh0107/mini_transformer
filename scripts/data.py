# data.py - 数据准备（只做一件事：提供干净数据）
# 人话：专注数据生成，不关心模型怎么用

import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from typing import List, Tuple, Optional
from config.config import config, reverse_vocab  # 只依赖配置，不依赖模型

def get_toy_data():
    """
    人话：生成玩具数据，专注展示Transformer能力
    返回: encoder输入, decoder输入, 目标, 掩码
    """
    vocab = config.VOCAB
    
    # 输入: [batch_size, seq_len]
    encoder_inputs = torch.tensor([
        [vocab['a'], vocab['b'], vocab['c']],  # "a b c"
        [vocab['b'], vocab['a'], vocab['c']],  # "b a c"
    ])
    
    # 输出: 需要加<sos>和<eos> (因为是自回归生成)
    decoder_inputs = torch.tensor([
        [vocab['<sos>'], vocab['1'], vocab['2'], vocab['3']],  # "<sos> 1 2 3"
        [vocab['<sos>'], vocab['2'], vocab['1'], vocab['3']],
    ])
    
    # 目标: 预测下一个词 (所以比decoder_inputs少一个<eos>)
    targets = torch.tensor([
        [vocab['1'], vocab['2'], vocab['3'], vocab['<eos>']],
        [vocab['2'], vocab['1'], vocab['3'], vocab['<eos>']],
    ])
    
    # 创建掩码: Decoder不能看未来词 (比如生成"1"时不能看"2")
    look_ahead_mask = torch.tril(torch.ones(4, 4))  # 4=序列长度
    
    return encoder_inputs, decoder_inputs, targets, look_ahead_mask

def show_sample(enc_input, target):
    """
    人话：把数字ID变回文字，让人看懂
    输入: encoder输入tensor, target tensor
    返回: (输入文字列表, 目标文字列表)
    """
    vocab = config.VOCAB
    enc_text = [reverse_vocab[i.item()] for i in enc_input]
    target_text = [reverse_vocab[i.item()] for i in target if i.item() != vocab['<eos>']]
    return enc_text, target_text

# ==================== 真实数据加载 ====================

def load_data_from_file(file_path: str, separator: str = "\t") -> List[Tuple[str, str]]:
    """
    人话：从文件读取数据对（输入序列，输出序列）
    文件格式：每行一个样本，用分隔符分开输入和输出
    例如：a b c\t1 2 3
    
    返回: [(输入序列, 输出序列), ...]
    """
    data_pairs = []
    if not os.path.exists(file_path):
        print(f"⚠️  数据文件不存在: {file_path}")
        return data_pairs
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:  # 跳过空行
                continue
            if line.startswith('#'):  # 跳过注释行
                continue
            parts = line.split(separator)
            if len(parts) == 2:
                input_seq = parts[0].strip()
                output_seq = parts[1].strip()
                data_pairs.append((input_seq, output_seq))
            else:
                print(f"⚠️  跳过格式错误的行: {line}")
    
    print(f"✅ 从文件加载了 {len(data_pairs)} 个样本")
    return data_pairs

def tokenize_sequence(sequence: str) -> List[str]:
    """
    人话：把字符串序列分割成token列表
    例如: "a b c" -> ["a", "b", "c"]
    """
    return sequence.split()

def sequence_to_ids(sequence: List[str], vocab: dict, max_len: Optional[int] = None) -> List[int]:
    """
    人话：把token列表转换成ID列表
    如果token不在词汇表中，使用<unk>
    如果指定max_len，会进行padding或截断
    """
    ids = []
    for token in sequence:
        if token in vocab:
            ids.append(vocab[token])
        else:
            ids.append(vocab.get('<unk>', 9))  # 未知词
    
    # Padding或截断
    if max_len is not None:
        if len(ids) < max_len:
            ids.extend([vocab['<pad>']] * (max_len - len(ids)))
        else:
            ids = ids[:max_len]
    
    return ids

def get_real_data(file_path: Optional[str] = None):
    """
    人话：从真实文件加载数据并转换为模型需要的格式
    返回: encoder输入, decoder输入, 目标, 掩码
    
    如果文件不存在或未指定，返回None（可以fallback到玩具数据）
    """
    file_path = file_path or config.DATA_FILE
    if file_path is None or not os.path.exists(file_path):
        return None
    
    # 1. 从文件加载原始数据
    data_pairs = load_data_from_file(file_path, config.DATA_SEPARATOR)
    if not data_pairs:
        return None
    
    vocab = config.VOCAB
    
    # 2. 找到最大序列长度（用于padding）
    max_enc_len = max(len(tokenize_sequence(pair[0])) for pair in data_pairs)
    max_dec_len = max(len(tokenize_sequence(pair[1])) for pair in data_pairs) + 1  # +1 for <eos>
    
    # 3. 转换为ID序列
    encoder_inputs_list = []
    decoder_inputs_list = []
    targets_list = []
    
    for input_seq, output_seq in data_pairs:
        # 输入序列
        input_tokens = tokenize_sequence(input_seq)
        enc_ids = sequence_to_ids(input_tokens, vocab, max_enc_len)
        encoder_inputs_list.append(enc_ids)
        
        # 输出序列（需要加<sos>和<eos>）
        output_tokens = tokenize_sequence(output_seq)
        output_ids = sequence_to_ids(output_tokens, vocab)
        
        # decoder输入: <sos> + 输出序列
        dec_input_ids = [vocab['<sos>']] + output_ids
        if len(dec_input_ids) < max_dec_len:
            dec_input_ids.extend([vocab['<pad>']] * (max_dec_len - len(dec_input_ids)))
        decoder_inputs_list.append(dec_input_ids)
        
        # 目标: 输出序列 + <eos>
        target_ids = output_ids + [vocab['<eos>']]
        if len(target_ids) < max_dec_len:
            target_ids.extend([vocab['<pad>']] * (max_dec_len - len(target_ids)))
        targets_list.append(target_ids)
    
    # 4. 转换为tensor
    encoder_inputs = torch.tensor(encoder_inputs_list, dtype=torch.long)
    decoder_inputs = torch.tensor(decoder_inputs_list, dtype=torch.long)
    targets = torch.tensor(targets_list, dtype=torch.long)
    
    # 5. 创建掩码
    # 注意：这里返回的是基础look-ahead mask，完整的掩码在训练时创建
    # 因为需要batch信息，所以在这里只返回形状模板
    look_ahead_mask_template = torch.tril(torch.ones(max_dec_len, max_dec_len))
    
    return encoder_inputs, decoder_inputs, targets, look_ahead_mask_template

def get_data():
    """
    人话：智能数据加载器
    优先尝试从文件加载真实数据，如果失败则使用玩具数据
    返回: encoder输入, decoder输入, 目标, 掩码
    """
    # 尝试加载真实数据
    real_data = get_real_data()
    if real_data is not None:
        print("📂 使用真实数据文件")
        return real_data
    
    # Fallback到玩具数据
    print("🎮 使用玩具数据（数据文件不存在或为空）")
    return get_toy_data()

