# chat.py - 交互式对话模块
# 人话：与训练好的模型进行多轮对话

import os
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.inference import load_model, predict, save_model
from config.config import config

class ChatBot:
    """
    人话：对话机器人
    支持多轮对话，记住上下文
    """
    def __init__(self, model_path: Optional[str] = None):
        """
        初始化对话机器人
        
        输入:
            model_path: 模型文件路径
        """
        print("🤖 正在加载模型...")
        resolved_path = model_path or config.MODEL_SAVE_PATH
        self.model = load_model(resolved_path)
        if self.model is None:
            print("❌ 模型加载失败！请先训练模型。")
            print("   运行: python scripts/train.py")
            sys.exit(1)
        
        self.conversation_history = []  # 对话历史
        print("✅ 模型加载完成！开始对话吧～\n")
    
    def chat(self, user_input: str) -> str:
        """
        处理用户输入，返回模型回复
        
        输入:
            user_input: 用户输入的文本
        
        返回: 模型的回复
        """
        if not user_input.strip():
            return "请输入有效的内容！"
        
        # 记录用户输入
        self.conversation_history.append(("用户", user_input))
        
        # 模型预测
        try:
            response = predict(self.model, user_input)
            if not response:
                response = "<无法生成回复>"
        except Exception as e:
            response = f"<错误: {e}>"
        
        # 记录模型回复
        self.conversation_history.append(("模型", response))
        
        return response
    
    def show_history(self, n: int = 5):
        """
        显示最近的对话历史
        
        输入:
            n: 显示最近n轮对话
        """
        if not self.conversation_history:
            print("📝 暂无对话历史")
            return
        
        print(f"\n📝 最近 {min(n, len(self.conversation_history)//2)} 轮对话:")
        print("-" * 50)
        
        # 只显示最近的n轮（每轮包含用户输入和模型回复）
        recent = self.conversation_history[-n*2:] if n*2 <= len(self.conversation_history) else self.conversation_history
        
        for role, text in recent:
            prefix = "👤" if role == "用户" else "🤖"
            print(f"{prefix} {role}: {text}")
        
        print("-" * 50)
    
    def clear_history(self):
        """清空对话历史"""
        self.conversation_history = []
        print("✅ 对话历史已清空")

def interactive_chat(model_path: Optional[str] = None):
    """
    人话：启动交互式对话界面
    
    输入:
        model_path: 模型文件路径
    """
    bot = ChatBot(model_path)
    
    print("=" * 60)
    print("🎉 欢迎使用 Mini Transformer 对话系统！")
    print("=" * 60)
    print("\n💡 使用说明:")
    print("  - 输入文本，模型会生成回复")
    print("  - 输入 'quit' 或 'exit' 退出")
    print("  - 输入 'history' 查看对话历史")
    print("  - 输入 'clear' 清空对话历史")
    print("  - 输入 'help' 显示帮助信息")
    print("\n" + "=" * 60 + "\n")
    
    while True:
        try:
            # 获取用户输入
            user_input = input("👤 你: ").strip()
            
            # 处理特殊命令
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 再见！")
                break
            
            elif user_input.lower() == 'history':
                bot.show_history()
                continue
            
            elif user_input.lower() == 'clear':
                bot.clear_history()
                continue
            
            elif user_input.lower() == 'help':
                print("\n💡 可用命令:")
                print("  - quit/exit/q: 退出对话")
                print("  - history: 查看对话历史")
                print("  - clear: 清空对话历史")
                print("  - help: 显示帮助信息")
                print()
                continue
            
            # 正常对话
            if user_input:
                response = bot.chat(user_input)
                print(f"🤖 模型: {response}\n")
        
        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            break
        except Exception as e:
            print(f"❌ 发生错误: {e}\n")

def batch_chat(model_path: str, inputs: list):
    """
    人话：批量对话（非交互式）
    
    输入:
        model_path: 模型文件路径
        inputs: 输入文本列表
    """
    bot = ChatBot(model_path)
    
    print("=" * 60)
    print("📊 批量对话模式")
    print("=" * 60 + "\n")
    
    for i, user_input in enumerate(inputs, 1):
        print(f"[{i}/{len(inputs)}] 输入: {user_input}")
        response = bot.chat(user_input)
        print(f"     输出: {response}\n")
    
    print("=" * 60)
    print("✅ 批量对话完成！")

if __name__ == "__main__":
    # 检查模型文件是否存在
    model_path = config.MODEL_SAVE_PATH
    if not os.path.exists(model_path):
        print("❌ 模型文件不存在！")
        print("   请先运行: python scripts/train.py")
        sys.exit(1)
    
    # 启动交互式对话
    interactive_chat(model_path)

