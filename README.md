# Mini Transformer - 模块化解构版

🌟 **极简但工业级的 Transformer 教学仓库** —— 用可拆可拼的代码积木理解序列到序列模型。

## 📁 项目结构一览

```
miniTransformer/
├── config/
│   └── config.py          # 全局超参仪表盘
├── data/
│   └── data.txt           # 可选：真实训练数据
├── docs/
│   ├── README.md          # 本文档
│   ├── CHAT_GUIDE.md
│   ├── DATA_GUIDE.md
│   └── QUICK_START.md
├── models/
│   └── model.pt           # 训练后自动保存的权重
├── scripts/
│   ├── train.py           # 训练/评估流程
│   ├── inference.py       # 推理脚本
│   ├── chat.py            # 交互式对话入口
│   └── data.py            # 数据加载与可视化
├── src/
│   ├── __init__.py        # 暴露常用组件
│   ├── device_utils.py    # GPU/CPU 管理
│   ├── mask_utils.py      # 掩码生成工具
│   ├── mini_transformer.py# 玩具示例训练脚本
│   └── transformer/
│       ├── __init__.py
│       ├── positional_encoding.py  # 位置编码
│       ├── attention.py            # 注意力原语
│       ├── encoder.py              # 编码器层与堆栈
│       ├── decoder.py              # 解码器层与堆栈
│       └── model.py                # Seq2Seq 总装
├── tests/
│   ├── test_chat.py
│   └── test_cross_attention.py
└── quick_start.py         # “一步跑通”演示
```

### 模块命名直指架构

- `transformer/positional_encoding.py`：位置“地址”生成器
- `transformer/attention.py`：`scaled_dot_product_attention` 与 `MultiHeadAttention`
- `transformer/encoder.py`：`TransformerEncoderLayer` + `TransformerEncoderStack`
- `transformer/decoder.py`：`TransformerDecoderLayer` + `TransformerDecoderStack`
- `transformer/model.py`：`Seq2SeqTransformer`（对外别名 `MiniTransformer`）
- `src/model.py`：向旧路径提供重导出，保证兼容测试、脚本

这样安排，初学者可以像拆乐高一样逐个文件理解和改造，资深同学也能直接定位到需要拓展的层级。

## 🎯 设计哲学

- **单一职责**：每个文件只关注一个问题，降低认知负担
- **可视化隔离**：日志、可视化放在 `scripts/`，核心模块无副作用
- **配置驱动**：所有超参集中在 `config/config.py`
- **易测易扩展**：每个积木块可独立导入、单测、替换

## 🚀 快速开始

### 1. 安装依赖

```bash
python -m venv .venv
source .venv/bin/activate  # Windows 使用 .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 训练模型

```bash
python scripts/train.py
```

训练脚本会：

- ✅ 自动检测并使用 GPU（若可用）
- ✅ 优先读取 `data/data.txt`，否则回退到玩具样本
- ✅ 训练完成后保存权重到 `models/model.pt`

### 3. 推理 / 对话

```bash
# 单次推理
python scripts/inference.py --input "a b c"

# 简易聊天界面
python scripts/chat.py
```

更多参数说明见 `docs/CHAT_GUIDE.md` 与 `docs/QUICK_START.md`。

### 4. 玩具任务快速体验

```bash
python quick_start.py
```

或在 `src/mini_transformer.py` 中阅读最小复现版的训练流程。

## 🧱 架构拆解

| 层级 | 文件 | 作用 |
| ---- | ---- | ---- |
| 配置 | `config/config.py` | 词表、模型维度、训练超参、数据路径等集中管理 |
| 工具 | `src/device_utils.py` / `src/mask_utils.py` | 设备检测、掩码生成；独立可测试 |
| 积木 | `src/transformer/*.py` | 将 Transformer 拆成位置编码、注意力、编码器、解码器、整模五个层次 |
| 封装 | `src/model.py` | 对外维持 `MiniTransformer` 等旧接口，向新结构转发 |
| 流程 | `scripts/train.py` 等 | 训练、推理、聊天等上层业务逻辑 |

## 🔁 常见练习

1. **观察掩码**：在 `src/mask_utils.py` 中添加临时打印，运行 `pytest tests/test_cross_attention.py`
2. **调参实验**：修改 `config/config.py` 中的 `D_MODEL`、`NUM_HEADS`、`EPOCHS` 观察训练曲线变化
3. **自定义层**：仿照 `TransformerEncoderLayer` 新建带 Dropout 的版本并替换堆栈
4. **可视化注意力**：在 `transformer/attention.py` 保存 `attention_weights`，再写脚本绘图

## 📊 数据 & 任务

`data/data.txt` 使用制表符分隔的输入输出对：

```text
# 注释行
a b c\tc b a
b a c\tc a b
```

当前提供多种混合任务（反转、复杂映射、组合操作等）。细节见 `docs/DATA_GUIDE.md`。

## 🧪 测试

```bash
pytest tests/test_cross_attention.py  # 注意力与解码堆栈检查
pytest tests/test_chat.py             # 聊天脚本冒烟测试
```

## 💡 心得

> “不要一次吞下整个 Transformer。先理解每层的接口与假设，再把积木重新拼装。”

当你能够在 `src/transformer/` 中替换任意一个模块并快速验证行为，你就真正掌握了这套架构的运行方式。祝玩得开心！

