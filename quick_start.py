#!/usr/bin/env python3
"""
快速启动脚本 - 选择要运行的脚本

使用: python quick_start.py
"""

import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def show_menu():
    """显示菜单"""
    menu = """
╔════════════════════════════════════════════════════════════════╗
║           📚 miniTransformer - 快速启动菜单                     ║
╚════════════════════════════════════════════════════════════════╝

🎯 主要功能:
    1. 训练模型              (python scripts/train.py)
    2. 评估模型              (python scripts/evaluate.py)
    3. 推理预测              (python scripts/inference.py)
    4. 交互聊天              (python scripts/chat.py)
    5. 预训练生成            (python scripts/generate.py)
    6. 微调模型              (python scripts/finetune.py)

🧪 测试:
    7. 测试对话功能          (python -m pytest tests/test_chat.py)
    8. 测试交叉注意力        (python -m pytest tests/test_cross_attention.py)

📖 文档:
    9. 显示聊天指南          (cat docs/CHAT_GUIDE.md)
    10. 显示数据指南          (cat docs/DATA_GUIDE.md)
    11. 显示快速开始指南      (cat docs/QUICK_START.md)
    12. 显示项目文档         (cat docs/README.md)

❌ 退出                      (q/exit)
"""
    print(menu)

def run_script(script_path):
    """运行指定的脚本"""
    if script_path.endswith('.md'):
        # 显示文档
        try:
            with open(script_path, 'r', encoding='utf-8') as f:
                print(f.read())
        except FileNotFoundError:
            print(f"❌ 文件未找到: {script_path}")
    else:
        # 运行 Python 脚本
        command = ['python']
        if script_path.startswith('tests/'):
            # 对测试文件使用 pytest，避免直接运行测试模块失败
            command.extend(['-m', 'pytest', script_path])
            display_cmd = f"python -m pytest {script_path}"
        else:
            command.append(script_path)
            display_cmd = f"python {script_path}"

        print(f"\n🚀 正在运行: {display_cmd}\n")
        import subprocess
        try:
            subprocess.run(command, check=False)
        except Exception as e:
            print(f"❌ 运行出错: {e}")

def main():
    """主菜单循环"""
    scripts = {
        '1': 'scripts/train.py',
        '2': 'scripts/evaluate.py',
        '3': 'scripts/inference.py',
        '4': 'scripts/chat.py',
        '5': 'scripts/generate.py',
        '6': 'scripts/finetune.py',
        '7': 'tests/test_chat.py',
        '8': 'tests/test_cross_attention.py',
        '9': 'docs/CHAT_GUIDE.md',
        '10': 'docs/DATA_GUIDE.md',
        '11': 'docs/QUICK_START.md',
        '12': 'docs/README.md',
    }
    
    while True:
        show_menu()
        choice = input("请选择 (1-12 或 q 退出): ").strip().lower()
        
        if choice in ['q', 'exit', 'quit']:
            print("\n👋 再见!")
            break
        
        if choice in scripts:
            script = scripts[choice]
            # 验证文件是否存在
            if os.path.exists(script):
                run_script(script)
            else:
                print(f"❌ 文件不存在: {script}")
            
            input("\n按 Enter 继续...")
        else:
            print("❌ 无效选择，请重试")
            input("\n按 Enter 继续...")

if __name__ == '__main__':
    main()
