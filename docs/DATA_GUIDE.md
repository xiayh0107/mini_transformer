# 📂 真实数据集使用指南

## 🎯 快速开始

### 1. 准备数据文件

创建 `data.txt` 文件，格式如下（制表符分隔）：

```
a b c	1 2 3
b a c	2 1 3
c a b	3 1 2
```

**格式说明：**
- 每行一个样本
- 输入序列和输出序列用**制表符（Tab）**分隔
- 序列内的token用**空格**分隔

### 2. 配置数据路径

在 `config.py` 中设置：

```python
DATA_FILE = "data.txt"  # 你的数据文件路径
DATA_SEPARATOR = "\t"   # 分隔符（默认制表符）
```

### 3. 运行训练

```bash
python train.py
```

系统会自动：
- ✅ 优先加载 `data.txt` 中的真实数据
- ✅ 如果文件不存在，自动fallback到玩具数据
- ✅ 自动处理padding、tokenization等

## 📝 数据文件格式详解

### 标准格式（制表符分隔）

```
输入序列1	输出序列1
输入序列2	输出序列2
...
```

**示例：**
```
hello world	你好 世界
good morning	早上好
```

### 自定义分隔符

如果使用其他分隔符（如逗号），在 `config.py` 中修改：

```python
DATA_SEPARATOR = ","  # 使用逗号分隔
```

数据文件格式：
```
a b c,1 2 3
b a c,2 1 3
```

## 🔧 高级用法

### 1. 使用不同的数据文件

```python
# 在 config.py 中
DATA_FILE = "my_custom_data.txt"
```

或者在代码中直接指定：

```python
from data import get_real_data

enc_inputs, dec_inputs, targets, mask = get_real_data("my_custom_data.txt")
```

### 2. 数据预处理

数据加载器会自动：
- ✅ Tokenization（空格分割）
- ✅ 转换为ID（使用词汇表）
- ✅ Padding（统一长度）
- ✅ 添加特殊符号（`<sos>`, `<eos>`）

### 3. 处理未知词

如果数据中有词汇表不存在的词，会自动使用 `<unk>`：

```python
# 词汇表中没有 'd'
# 输入: "a b d" → 会变成 [0, 1, 9]  # 9是<unk>的ID
```

## 📊 数据文件示例

### 示例1：字母到数字翻译

**data.txt:**
```
a b c	1 2 3
b a c	2 1 3
c a b	3 1 2
a c b	1 3 2
b c a	2 3 1
c b a	3 2 1
```

### 示例2：简单翻译任务

**data.txt:**
```
hello	你好
world	世界
good	好
morning	早上
```

**注意：** 需要先在 `config.py` 的 `VOCAB` 中添加对应的词汇！

## ⚠️ 注意事项

### 1. 词汇表限制

**重要：** 数据文件中的token必须在 `config.py` 的 `VOCAB` 中定义！

如果数据中有新词，需要：
1. 在 `config.py` 的 `VOCAB` 中添加
2. 更新 `reverse_vocab`（自动生成）

### 2. 序列长度

- 系统会自动找到最大序列长度
- 较短的序列会自动padding
- 过长的序列会被截断（如果超过max_len）

### 3. 数据量

- 小数据集（<100样本）：适合快速测试
- 中等数据集（100-1000样本）：需要调整训练参数
- 大数据集（>1000样本）：考虑使用DataLoader分批加载

## 🚀 扩展：支持大数据集

如果数据量很大，可以扩展 `data.py` 使用PyTorch的DataLoader：

```python
from torch.utils.data import Dataset, DataLoader

class TranslationDataset(Dataset):
    def __init__(self, file_path):
        # 加载数据
        ...
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

# 在 train.py 中使用
dataset = TranslationDataset("large_data.txt")
dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)
```

## 💡 调试技巧

### 1. 检查数据加载

```python
from data import load_data_from_file

pairs = load_data_from_file("data.txt")
print(f"加载了 {len(pairs)} 个样本")
for i, (input_seq, output_seq) in enumerate(pairs[:3]):
    print(f"样本{i+1}: {input_seq} -> {output_seq}")
```

### 2. 检查数据转换

```python
from data import get_real_data, show_sample

enc, dec, tgt, mask = get_real_data()
print(f"Encoder输入形状: {enc.shape}")
print(f"Decoder输入形状: {dec.shape}")
print(f"目标形状: {tgt.shape}")

# 查看第一个样本
sample_enc, sample_tgt = show_sample(enc[0], tgt[0])
print(f"样本: {sample_enc} -> {sample_tgt}")
```

### 3. 验证数据格式

```python
# 检查文件是否存在
import os
print(f"文件存在: {os.path.exists('data.txt')}")

# 查看文件内容
with open('data.txt', 'r') as f:
    for i, line in enumerate(f):
        if i < 3:  # 只看前3行
            print(f"第{i+1}行: {repr(line)}")
```

## 🎓 最佳实践

1. **数据验证**：加载后检查样本数量和格式
2. **词汇表管理**：确保所有token都在词汇表中
3. **数据清洗**：去除空行、格式错误的行
4. **数据分割**：大数据集考虑train/val/test分割
5. **版本控制**：数据文件不要提交到git（太大）

---

**现在你可以用真实数据训练Transformer了！** 🚀

