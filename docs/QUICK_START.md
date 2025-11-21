# ⚡ 快速开始 - 多轮对话

## 🎯 3步开始对话

### 步骤1: 训练模型
```bash
python train.py
```
⏱️ 等待训练完成（会自动保存模型到 `model.pt`）

### 步骤2: 启动对话
```bash
python chat.py
```

### 步骤3: 开始对话
```
👤 你: a b c
🤖 模型: c b a

👤 你: b a c
🤖 模型: c a b
```

---

## 📋 常用命令

| 命令 | 功能 |
|------|------|
| `quit` / `exit` / `q` | 退出对话 |
| `history` | 查看对话历史 |
| `clear` | 清空对话历史 |
| `help` | 显示帮助信息 |

---

## 💻 在代码中使用

### 方式1: 交互式对话
```bash
python chat.py
```

### 方式2: 在Python代码中
```python
from chat import ChatBot

bot = ChatBot("model.pt")
response = bot.chat("a b c")
print(response)  # 输出: c b a
```

### 方式3: 直接推理
```python
from inference import load_model, predict

model = load_model("model.pt")
result = predict(model, "a b c")
print(result)  # 输出: c b a
```

---

## 🔍 文件说明

- `train.py` - 训练模型（自动保存）
- `chat.py` - 交互式对话界面
- `inference.py` - 推理功能（模型预测）
- `model.pt` - 训练好的模型（训练后自动生成）

---

## ❓ 遇到问题？

1. **模型文件不存在？**
   ```bash
   python train.py  # 先训练模型
   ```

2. **想了解更多？**
   - 详细文档: [CHAT_GUIDE.md](CHAT_GUIDE.md)
   - 完整文档: [README.md](README.md)

---

**就这么简单！开始对话吧！** 🚀

