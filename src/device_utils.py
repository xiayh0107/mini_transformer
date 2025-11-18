# device_utils.py - 设备管理工具
# 人话：自动检测并使用GPU，确保训练速度

import torch

def get_device():
    """
    自动检测并返回最佳设备
    
    返回:
        device: torch.device对象
        设备信息字符串
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        device_name = torch.cuda.get_device_name(0)
        device_count = torch.cuda.device_count()
        info = f"CUDA ({device_name})"
        if device_count > 1:
            info += f" - {device_count} GPUs available"
        return device, info
    else:
        device = torch.device('cpu')
        return device, "CPU"

def print_device_info():
    """打印设备信息"""
    device, info = get_device()
    
    print("=" * 70)
    print("🖥️  设备信息")
    print("=" * 70)
    print(f"   使用设备: {info}")
    
    if torch.cuda.is_available():
        print(f"   CUDA版本: {torch.version.cuda}")
        print(f"   cuDNN版本: {torch.backends.cudnn.version()}")
        print(f"   GPU数量: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"   GPU {i}: {props.name}")
            print(f"     总内存: {props.total_memory / 1024**3:.2f} GB")
            print(f"     计算能力: {props.major}.{props.minor}")
        
        # 显示当前GPU内存使用
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
            print(f"   当前GPU内存使用:")
            print(f"     已分配: {memory_allocated:.2f} GB")
            print(f"     已保留: {memory_reserved:.2f} GB")
    else:
        print("   ⚠️  CUDA不可用，将使用CPU训练（速度较慢）")
        print("   💡 提示：如果有NVIDIA GPU，请安装CUDA版本的PyTorch")
    
    print("=" * 70 + "\n")
    
    return device

def verify_gpu_usage(model, sample_input):
    """
    验证模型和数据是否在GPU上
    
    输入:
        model: 模型对象
        sample_input: 示例输入tensor
    
    返回: 是否在GPU上
    """
    model_on_gpu = next(model.parameters()).is_cuda
    input_on_gpu = sample_input.is_cuda
    
    if model_on_gpu and input_on_gpu:
        print("✅ 验证通过：模型和数据都在GPU上")
        return True
    else:
        print("⚠️  警告：")
        if not model_on_gpu:
            print("   - 模型不在GPU上")
        if not input_on_gpu:
            print("   - 输入数据不在GPU上")
        return False

