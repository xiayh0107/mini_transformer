# config.py - 所有可配置参数集中管理
# 人话：就像汽车的仪表盘，所有关键参数一目了然

import os

class Config:
    # =============== 词汇表（未来可扩展） ===============
    VOCAB = {
        'a': 0, 'b': 1, 'c': 2,  # 输入字母的ID
        '1': 3, '2': 4, '3': 5,  # 输出数字的ID
        '<pad>': 6, '<sos>': 7, '<eos>': 8, '<unk>': 9,  # 特殊符号
        '<sep>': 22,  # 输入输出分隔符
        '[MASK]': 10,  # 预训练任务（MLM）专用mask标记
        # 噪声符号（用于数据增强，提高模型鲁棒性）
        ',': 11, '.': 12, '|': 13, '-': 14, '_': 15,  # 分隔符
        ':': 16, ';': 17,  # 标点
        '(': 18, ')': 19, '[': 20, ']': 21,  # 括号
    }
    
    # =============== 模型参数（调这些就能玩出花样） ===============
    D_MODEL = 32  # 向量维度 - 越大越强但越慢（必须能被NUM_HEADS整除）
    NUM_HEADS = 2  # 注意力头数 - 侦探团队数量（D_MODEL必须能被NUM_HEADS整除）
    NUM_ENCODER_LAYERS = 6  # Encoder层数 - 越多理解能力越强（建议3-6层）
    NUM_DECODER_LAYERS = 6  # Decoder层数 - 越多生成能力越强（建议3-6层）
    
    # =============== 训练参数 ===============
    LEARNING_RATE = 0.0005
    EPOCHS = 500  # 训练轮数（复杂任务需要更多轮次）
    BATCH_SIZE = 100  # 一次处理几句话
    
    # =============== 数据参数 ===============
    DATA_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'data.txt')  # 真实数据文件路径
    DATA_SEPARATOR = "\t"  # 数据文件分隔符（制表符分隔：输入\t输出）
    
    # =============== 模型保存参数 ===============
    MODEL_SAVE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'model.pt')  # 模型保存路径

# 创建全局配置实例（其他文件直接import）
config = Config()
reverse_vocab = {v: k for k, v in config.VOCAB.items()}

