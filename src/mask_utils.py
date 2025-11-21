# mask_utils.py - 掩码工具函数
# 人话：创建各种类型的掩码

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from config.config import config

def create_padding_mask(seq: torch.Tensor, pad_token_id: int = None) -> torch.Tensor:
    """
    人话：创建padding掩码（用于自注意力）
    输入:
        seq: 序列tensor [batch_size, seq_len]
        pad_token_id: padding token的ID
    返回: 掩码 [batch_size, 1, 1, seq_len]
          1表示有效位置，0表示padding位置（需要被mask掉）
    
    注意：返回的掩码中，1=有效，0=需要mask
    在注意力中，0会被mask掉（设为-∞）
    对于自注意力，可以广播到 [batch, num_heads, seq_len, seq_len]
    """
    if pad_token_id is None:
        pad_token_id = config.VOCAB['<pad>']
    
    mask = (seq != pad_token_id).float().unsqueeze(1).unsqueeze(1)
    return mask

def create_look_ahead_mask(seq_len: int, device=None) -> torch.Tensor:
    """
    人话：创建look-ahead掩码（下三角矩阵）
    输入:
        seq_len: 序列长度
        device: 设备（如果为None，使用默认设备）
    返回: 掩码 [seq_len, seq_len]
          1表示可以看，0表示不能看（需要被mask掉）
    
    用于Decoder自注意力，防止看到未来的token
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask

def create_decoder_mask(decoder_input: torch.Tensor, pad_token_id: int = None) -> torch.Tensor:
    """
    人话：创建Decoder的完整掩码
    结合look-ahead mask和padding mask
    
    输入:
        decoder_input: Decoder输入 [batch_size, seq_len]
        pad_token_id: padding token的ID
    返回: 掩码 [batch_size, seq_len, seq_len]
    
    这个掩码会同时：
    1. 防止看未来token（look-ahead）
    2. 忽略padding位置（padding）
    """
    if pad_token_id is None:
        pad_token_id = config.VOCAB['<pad>']
    
    batch_size, seq_len = decoder_input.shape
    device = decoder_input.device  # 获取输入tensor的设备
    
    # 1. 创建look-ahead mask（下三角矩阵）
    look_ahead = create_look_ahead_mask(seq_len, device=device)  # [seq_len, seq_len]
    look_ahead = look_ahead.unsqueeze(0)  # [1, seq_len, seq_len]
    
    # 2. 只对key位置做padding mask，避免整行被mask成NaN
    #    如果某个query本身是padding，让它看到的值都设定为0即可，
    #    CrossEntropyLoss 会忽略对应的<pad>目标
    padding_mask = (decoder_input != pad_token_id).float().unsqueeze(1)  # [batch, 1, seq_len]

    # 3. 组合look-ahead和padding（同时满足才可见）
    combined_mask = look_ahead * padding_mask  # [batch, seq_len, seq_len]
    
    return combined_mask

def create_encoder_decoder_mask(encoder_input: torch.Tensor, decoder_input: torch.Tensor, 
                                pad_token_id: int = None) -> torch.Tensor:
    """
    人话：创建Encoder-Decoder注意力的掩码
    只mask掉encoder中的padding位置
    
    输入:
        encoder_input: Encoder输入 [batch_size, enc_len]
        decoder_input: Decoder输入 [batch_size, dec_len]
        pad_token_id: padding token的ID
    返回: 掩码 [batch_size, 1, dec_len, enc_len]
    """
    if pad_token_id is None:
        pad_token_id = config.VOCAB['<pad>']
    
    # 创建encoder的padding mask [batch, 1, 1, enc_len]
    enc_valid = (encoder_input != pad_token_id).float().unsqueeze(1).unsqueeze(1)
    
    # 扩展到decoder长度 -> [batch, 1, dec_len, enc_len]
    dec_len = decoder_input.shape[1]
    enc_mask = enc_valid.expand(-1, 1, dec_len, -1)
    
    return enc_mask

