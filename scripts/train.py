# train.py - 训练/测试逻辑（像导演拍电影）
# 人话：专注训练流程，所有可视化都在这里

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from config.config import config, reverse_vocab
from scripts.data import get_data, show_sample  # 使用智能数据加载器
from scripts.inference import save_model  # 模型保存功能
from src.transformer import MiniTransformer
from src.mask_utils import create_decoder_mask, create_padding_mask
from src.device_utils import get_device, print_device_info, verify_gpu_usage

def train_model():
    """人话：训练流程的总指挥"""
    # 0. 检测并设置设备（GPU/CPU）
    device, device_info = get_device()
    print_device_info()
    
    # 1. 准备数据（智能加载：优先真实数据，fallback到玩具数据）
    enc_inputs, dec_inputs, targets, _ = get_data()
    
    # 将数据移到设备上
    enc_inputs = enc_inputs.to(device)
    dec_inputs = dec_inputs.to(device)
    targets = targets.to(device)
    
    # 显示训练样本（可视化在训练模块）
    sample_enc, sample_tgt = show_sample(enc_inputs[0], targets[0])
    print("\n📚 玩具任务: 把字母序列翻译成数字序列")
    print(f"📊 训练样本: {sample_enc} -> {sample_tgt}")
    
    # 2. 创建模型（只依赖配置）并移到设备上
    model = MiniTransformer()
    model = model.to(device)
    
    # 验证GPU使用
    verify_gpu_usage(model, enc_inputs)
    
    # 3. 定义训练组件（损失/优化器）
    criterion = nn.CrossEntropyLoss(ignore_index=config.VOCAB['<pad>'])
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
    
    # 早停机制参数
    early_stop_patience = 50  # 如果50个epoch损失没有改善，就停止
    min_delta = 1e-4          # 至少降低这么多才算“真正进步”
    best_loss = float('inf')
    no_improve_count = 0
    best_model_state = None
    best_epoch = -1
    
    # 4. 准备批处理
    total_samples = enc_inputs.size(0)
    batch_size = config.BATCH_SIZE
    num_batches = (total_samples + batch_size - 1) // batch_size  # 向上取整
    
    print(f"\n🚀 开始训练 {config.EPOCHS} 轮")
    print(f"📊 数据统计: {total_samples} 个样本, batch_size={batch_size}, {num_batches} 个batch/epoch")
    print("📝 使用完整掩码策略：look-ahead mask + padding mask")
    print(f"🛑 早停机制：如果 {early_stop_patience} 个epoch损失无改善，将自动停止\n")
    
    for epoch in range(config.EPOCHS):
        epoch_loss = 0.0
        
        # 每个epoch打乱数据顺序
        indices = torch.randperm(total_samples)
        enc_inputs_shuffled = enc_inputs[indices]
        dec_inputs_shuffled = dec_inputs[indices]
        targets_shuffled = targets[indices]
        
        # 按batch处理
        for batch_idx in range(num_batches):
            optimizer.zero_grad()
            
            # 获取当前batch的数据
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_samples)
            
            batch_enc_inputs = enc_inputs_shuffled[start_idx:end_idx].to(device)
            batch_dec_inputs = dec_inputs_shuffled[start_idx:end_idx].to(device)
            batch_targets = targets_shuffled[start_idx:end_idx].to(device)
            
            # 创建完整的掩码（每个batch动态创建，支持不同长度）
            # 1. Decoder掩码（look-ahead + padding）
            dec_mask = create_decoder_mask(batch_dec_inputs)  # [batch, dec_len, dec_len]
            
            # 2. Encoder掩码（encoder的padding）- 用于Encoder自注意力
            enc_mask = create_padding_mask(batch_enc_inputs)  # [batch, 1, 1, enc_len]
            
            # 前向传播（传入所有掩码）
            logits = model(
                batch_enc_inputs, 
                batch_dec_inputs, 
                look_ahead_mask=dec_mask,
                enc_padding_mask=enc_mask  # 用于Encoder自注意力和Encoder-Decoder注意力
            )
            
            # 计算损失 (需要reshape)
            loss = criterion(
                logits.view(-1, len(config.VOCAB)),
                batch_targets.view(-1)
            )
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # 计算平均损失
        loss_value = epoch_loss / num_batches
        scheduler.step(loss_value)
        
        # 早停机制：检查损失是否改善
        if loss_value + min_delta < best_loss:
            best_loss = loss_value
            best_epoch = epoch
            no_improve_count = 0
            # 深拷贝权重，确保后续训练不会覆盖最佳状态
            best_model_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
        else:
            no_improve_count += 1
        
        # 打印训练进度
        if epoch % 20 == 0 or epoch == config.EPOCHS - 1:
            lr = optimizer.param_groups[0]['lr']
            print(f"✅ Epoch {epoch}/{config.EPOCHS} Loss: {loss_value:.4f} LR: {lr:.6f} (最佳: {best_loss:.4f})")
        
        # 早停检查
        if no_improve_count >= early_stop_patience:
            print(f"\n🛑 早停触发：损失在 {early_stop_patience} 个epoch内无改善")
            print(f"   最佳损失: {best_loss:.4f} (Epoch {best_epoch})")
            print(f"   当前损失: {loss_value:.4f}")
            print(f"   已训练: {epoch + 1}/{config.EPOCHS} epochs")
            break
    
    # 5. 恢复最佳模型并保存
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n💾 已恢复最佳模型 (损失: {best_loss:.4f} @ Epoch {best_epoch})")
    
    save_model(model, config.MODEL_SAVE_PATH)
    print(f"✅ 模型已保存到: {config.MODEL_SAVE_PATH}")
    
    return model

def test_model(model):
    """人话：考考训练好的AI"""
    print("\n✅ 训练完成! 来测试一下:")
    device = next(model.parameters()).device  # 获取模型所在的设备
    test_input = torch.tensor([[config.VOCAB['a'], config.VOCAB['b'], config.VOCAB['c']]], device=device)  # "a b c"
    
    # 生成过程: 从<sos>开始，一步步预测
    generated = [config.VOCAB['<sos>']]
    for i in range(3):  # 生成3个数字
        dec_input = torch.tensor([generated], device=device)
        with torch.no_grad():
            # 用当前已生成的内容预测下一个词
            logits = model(test_input, dec_input)
            next_token = torch.argmax(logits[0, -1], dim=-1).item()
            generated.append(next_token)
    
    # 转换结果
    result = [reverse_vocab[token] for token in generated[1:]]  # 跳过<sos>
    print(f"\n🎯 测试结果:")
    print(f"输入: [a, b, c]")
    print(f"输出: {result} (应该接近 ['1','2','3'])")
    
    # =============== 总结整个流程 ===============
    print("\n🧠 代码核心逻辑 (人话版):")
    print("1. Encoder (理解部分):")
    print("   - 用'多头注意力'让每个词和其他词'对眼神' (比如'a'和'b'的关系)")
    print("   - 输出对输入的'深度理解'")
    print("2. Decoder (生成部分):")
    print("   - 用'掩码多头注意力'只能看已生成的词 (生成'1'时看不到'2')")
    print("   - 用'Encoder-Decoder注意力'参考Encoder的理解 (把'a'对应到'1')")
    print("3. 多头注意力 = 请多个侦探团队，从不同角度分析句子")
    print("   - 一个团队看语法关系，一个团队看语义关系...")
    print("   - 最后合并报告，得到全面理解")
    print("\n✨ 你成功实现了Transformer的核心! 这就是ChatGPT的'心脏'")

if __name__ == "__main__":
    trained_model = train_model()
    test_model(trained_model)

