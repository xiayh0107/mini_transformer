# 测试cross-attention是否真的在工作

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from src.model import MiniTransformer, DecoderLayer, MultiHeadAttention
from config.config import config, reverse_vocab
from src.device_utils import get_device

def test_cross_attention_directly():
    """直接测试cross-attention层"""
    print("=" * 70)
    print("🔍 直接测试Cross-Attention层")
    print("=" * 70)
    
    device, _ = get_device()
    d_model = config.D_MODEL
    num_heads = config.NUM_HEADS
    
    # 创建cross-attention层
    cross_attn = MultiHeadAttention(d_model, num_heads).to(device)
    
    # 创建测试数据
    batch_size = 1
    enc_len = 3
    dec_len = 1
    
    # encoder输出（不同输入应该不同）
    enc_output1 = torch.randn(batch_size, enc_len, d_model, device=device)
    enc_output2 = torch.randn(batch_size, enc_len, d_model, device=device) * 2  # 故意不同
    
    # decoder输入（相同）
    dec_input = torch.randn(batch_size, dec_len, d_model, device=device)
    
    # 创建mask
    enc_mask = torch.ones(batch_size, 1, dec_len, enc_len, device=device)  # 全部可见
    
    print(f"\n测试数据:")
    print(f"  enc_output1均值: {enc_output1.mean().item():.4f}, 标准差: {enc_output1.std().item():.4f}")
    print(f"  enc_output2均值: {enc_output2.mean().item():.4f}, 标准差: {enc_output2.std().item():.4f}")
    print(f"  dec_input均值: {dec_input.mean().item():.4f}, 标准差: {dec_input.std().item():.4f}")
    
    # 运行cross-attention
    with torch.no_grad():
        output1 = cross_attn(dec_input, enc_output1, enc_output1, mask=enc_mask)
        output2 = cross_attn(dec_input, enc_output2, enc_output2, mask=enc_mask)
    
    print(f"\nCross-Attention输出:")
    print(f"  output1均值: {output1.mean().item():.4f}, 标准差: {output1.std().item():.4f}")
    print(f"  output2均值: {output2.mean().item():.4f}, 标准差: {output2.std().item():.4f}")
    
    diff = (output1 - output2).abs().mean().item()
    print(f"  两个输出的差异: {diff:.6f}")
    
    if diff < 0.001:
        print(f"  ❌ 警告：Cross-Attention输出几乎相同，可能有问题")
    else:
        print(f"  ✅ Cross-Attention输出不同，应该正常工作")
    
    # 测试2：检查注意力权重
    print(f"\n测试2：检查注意力权重")
    print("-" * 70)
    
    # 手动运行scaled_dot_product_attention看看注意力权重
    from src.model import scaled_dot_product_attention
    
    # 准备query, key, value
    q = cross_attn.W_q(dec_input)
    k1 = cross_attn.W_k(enc_output1)
    v1 = cross_attn.W_v(enc_output1)
    k2 = cross_attn.W_k(enc_output2)
    v2 = cross_attn.W_v(enc_output2)
    
    # split heads
    def split_heads(x, num_heads):
        batch_size, seq_len, d_model = x.size()
        d_k = d_model // num_heads
        return x.view(batch_size, seq_len, num_heads, d_k).transpose(1, 2)
    
    q_split = split_heads(q, num_heads)
    k1_split = split_heads(k1, num_heads)
    v1_split = split_heads(v1, num_heads)
    k2_split = split_heads(k2, num_heads)
    v2_split = split_heads(v2, num_heads)
    
    # 调整mask形状
    mask_adjusted = enc_mask.unsqueeze(1)  # [batch, 1, 1, dec_len, enc_len] -> 需要调整
    
    # 运行attention
    attn_output1, attn_weights1 = scaled_dot_product_attention(q_split, k1_split, v1_split, mask=enc_mask)
    attn_output2, attn_weights2 = scaled_dot_product_attention(q_split, k2_split, v2_split, mask=enc_mask)
    
    print(f"  注意力权重1形状: {attn_weights1.shape}")
    print(f"  注意力权重1 (第一个head, 第一个query): {attn_weights1[0, 0, 0].detach().cpu().numpy()}")
    print(f"  注意力权重2 (第一个head, 第一个query): {attn_weights2[0, 0, 0].detach().cpu().numpy()}")
    
    attn_diff = (attn_weights1 - attn_weights2).abs().mean().item()
    print(f"  注意力权重差异: {attn_diff:.6f}")
    
    if attn_diff < 0.001:
        print(f"  ❌ 警告：注意力权重几乎相同！")
    else:
        print(f"  ✅ 注意力权重不同，应该正常")
    
    print("\n" + "=" * 70)

def test_full_model_cross_attention():
    """测试完整模型的cross-attention"""
    print("\n" + "=" * 70)
    print("🔍 测试完整模型的Cross-Attention")
    print("=" * 70)
    
    device, _ = get_device()
    model = MiniTransformer()
    model = model.to(device)
    model.eval()
    
    vocab = config.VOCAB
    
    # 两个不同的encoder输入
    enc_input1 = torch.tensor([[vocab['a'], vocab['b'], vocab['c']]], device=device)
    enc_input2 = torch.tensor([[vocab['c'], vocab['b'], vocab['a']]], device=device)
    dec_input = torch.tensor([[vocab['<sos>']]], device=device)
    
    from src.mask_utils import create_decoder_mask, create_padding_mask
    
    # 运行encoder
    enc_emb1 = model.embedding(enc_input1)
    enc_emb2 = model.embedding(enc_input2)
    enc_mask1 = create_padding_mask(enc_input1)
    enc_mask2 = create_padding_mask(enc_input2)
    
    if isinstance(model.encoder, nn.ModuleList):
        enc_output1 = enc_emb1
        enc_output2 = enc_emb2
        for encoder_layer in model.encoder:
            enc_output1 = encoder_layer(enc_output1, padding_mask=enc_mask1)
            enc_output2 = encoder_layer(enc_output2, padding_mask=enc_mask2)
    else:
        enc_output1 = model.encoder(enc_emb1, padding_mask=enc_mask1)
        enc_output2 = model.encoder(enc_emb2, padding_mask=enc_mask2)
    
    print(f"\nEncoder输出:")
    print(f"  enc_output1均值: {enc_output1.mean().item():.6f}")
    print(f"  enc_output2均值: {enc_output2.mean().item():.6f}")
    enc_diff = (enc_output1 - enc_output2).abs().mean().item()
    print(f"  Encoder输出差异: {enc_diff:.6f}")
    
    # 运行decoder的第一层
    dec_emb = model.embedding(dec_input)
    dec_mask = create_decoder_mask(dec_input)
    
    if hasattr(model.decoder, "layers"):
        decoder_layer = model.decoder.layers[0]
    elif isinstance(model.decoder, nn.ModuleList):
        decoder_layer = model.decoder[0]
    else:
        decoder_layer = model.decoder
    
    # 创建enc_dec_mask
    batch_size, _, _, enc_len = enc_mask1.shape
    dec_len = dec_input.shape[1]
    enc_dec_mask1 = enc_mask1.expand(batch_size, 1, dec_len, enc_len)
    enc_dec_mask2 = enc_mask2.expand(batch_size, 1, dec_len, enc_len)
    
    # 第一步：self-attention
    masked_attn_out = decoder_layer.masked_attn(dec_emb, dec_emb, dec_emb, dec_mask)
    x = decoder_layer.norm1(dec_emb + masked_attn_out)
    
    # 第二步：cross-attention（关键！）
    print(f"\nCross-Attention:")
    print(f"  query (x)形状: {x.shape}")
    print(f"  key/value (enc_output1)形状: {enc_output1.shape}")
    print(f"  mask形状: {enc_dec_mask1.shape}")
    
    cross_attn_out1 = decoder_layer.enc_dec_attn(x, enc_output1, enc_output1, mask=enc_dec_mask1)
    cross_attn_out2 = decoder_layer.enc_dec_attn(x, enc_output2, enc_output2, mask=enc_dec_mask2)
    
    print(f"  cross_attn_out1均值: {cross_attn_out1.mean().item():.6f}")
    print(f"  cross_attn_out2均值: {cross_attn_out2.mean().item():.6f}")
    cross_diff = (cross_attn_out1 - cross_attn_out2).abs().mean().item()
    print(f"  Cross-Attention输出差异: {cross_diff:.6f}")
    
    if cross_diff < 0.001:
        print(f"  ❌ 问题确认：Cross-Attention输出几乎相同！")
        print(f"     这说明Cross-Attention没有正确使用Encoder信息")
    else:
        print(f"  ✅ Cross-Attention输出不同，应该正常")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    test_cross_attention_directly()
    test_full_model_cross_attention()

