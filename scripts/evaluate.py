# evaluate.py - 模型评估模块
# 人话：全面评估模型性能，找出问题，明确优化方向

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import numpy as np
from typing import List, Tuple, Dict, Optional

from config.config import config
from scripts.data import load_data_from_file, tokenize_sequence, sequence_to_ids
from scripts.inference import load_model, predict, text_to_ids
from src.device_utils import print_device_info, verify_gpu_usage
from src.mask_utils import create_decoder_mask, create_padding_mask

class ModelEvaluator:
    """
    人话：模型评估器
    全面评估模型性能，包括准确率、困惑度、生成质量等
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        初始化评估器
        
        输入:
            model_path: 模型文件路径
        """
        print("📊 初始化模型评估器...")
        resolved_path = model_path or config.MODEL_SAVE_PATH
        self.model = load_model(resolved_path)
        if self.model is None:
            raise ValueError("模型加载失败！")
        
        # 显示设备信息
        print_device_info()
        
        # 验证GPU使用
        device = next(self.model.parameters()).device
        sample_input = torch.tensor([[0]], device=device)  # 创建示例输入
        verify_gpu_usage(self.model, sample_input)
        
        self.vocab = config.VOCAB
        self.model.eval()
        print("✅ 评估器初始化完成\n")

    @staticmethod
    def _repetition_metrics(tokens: List[str]) -> Tuple[float, int]:
        if not tokens:
            return 0.0, 0

        unique_tokens = len(set(tokens))
        total_tokens = len(tokens)
        repetition_rate = 1.0 - (unique_tokens / total_tokens)

        max_consecutive = 1
        current = 1
        for i in range(1, total_tokens):
            if tokens[i] == tokens[i - 1]:
                current += 1
                if current > max_consecutive:
                    max_consecutive = current
            else:
                current = 1

        return repetition_rate, max_consecutive
    
    def load_test_data(self, file_path: str = None) -> List[Tuple[str, str]]:
        """
        加载测试数据
        
        输入:
            file_path: 测试数据文件路径（默认使用训练数据文件）
        
        返回: [(输入序列, 目标序列), ...]
        """
        file_path = file_path or config.DATA_FILE
        if file_path is None or not file_path:
            print("⚠️  未指定测试数据文件，使用训练数据")
            file_path = config.DATA_FILE
        
        data_pairs = load_data_from_file(file_path, config.DATA_SEPARATOR)
        if not data_pairs:
            print("⚠️  测试数据为空，创建默认测试集")
            # 创建默认测试集
            data_pairs = [
                ("a b c", "3 1 2"),
                ("b a c", "1 3 2"),
                ("c a b", "2 3 1"),
                ("a c b", "3 2 1"),
                ("b c a", "1 2 3"),
                ("c b a", "2 1 3"),
                ("c b", "2 1"),
                ("a", "3"),
                ("b", "1"),
                ("c", "2"),
            ]
        
        return data_pairs
    
    def exact_match_accuracy(self, test_data: List[Tuple[str, str]]) -> float:
        """
        完全匹配准确率
        
        输入:
            test_data: 测试数据列表
        
        返回: 准确率 (0-1)
        """
        correct = 0
        total = len(test_data)
        
        for input_seq, target_seq in test_data:
            predicted = predict(self.model, input_seq, max_length=20)
            # 标准化：去除多余空格，转换为列表比较
            pred_tokens = predicted.split()
            target_tokens = target_seq.split()
            
            if pred_tokens == target_tokens:
                correct += 1
        
        accuracy = correct / total if total > 0 else 0.0
        return accuracy
    
    def token_accuracy(self, test_data: List[Tuple[str, str]]) -> Tuple[float, Dict]:
        """
        Token级别的准确率
        
        输入:
            test_data: 测试数据列表
        
        返回: (准确率, 详细统计)
        """
        correct_tokens = 0
        total_tokens = 0
        position_errors = []  # 每个位置的错误统计
        
        for input_seq, target_seq in test_data:
            predicted = predict(self.model, input_seq, max_length=20)
            pred_tokens = predicted.split()
            target_tokens = target_seq.split()
            
            # 按位置比较
            max_len = max(len(pred_tokens), len(target_tokens))
            for i in range(max_len):
                if i >= len(position_errors):
                    position_errors.append({"correct": 0, "total": 0, "errors": []})
                
                entry = position_errors[i]
                entry["total"] += 1
                
                if i < len(pred_tokens) and i < len(target_tokens):
                    if pred_tokens[i] == target_tokens[i]:
                        correct_tokens += 1
                        entry["correct"] += 1
                    else:
                        entry["errors"].append({
                            "input": input_seq,
                            "expected": target_tokens[i],
                            "got": pred_tokens[i],
                            "position": i
                        })
                elif i < len(target_tokens):
                    # 预测太短
                    entry["errors"].append({
                        "input": input_seq,
                        "expected": target_tokens[i],
                        "got": "<MISSING>",
                        "position": i
                    })
                else:
                    # 预测太长
                    entry["errors"].append({
                        "input": input_seq,
                        "expected": "<EOS>",
                        "got": pred_tokens[i],
                        "position": i
                    })
                
                if i < len(pred_tokens) and i < len(target_tokens):
                    total_tokens += 1
        
        accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0
        return accuracy, {"position_errors": position_errors}
    
    def sequence_length_accuracy(self, test_data: List[Tuple[str, str]]) -> Dict:
        """
        序列长度准确性分析
        
        输入:
            test_data: 测试数据列表
        
        返回: 长度统计信息
        """
        length_stats = {
            "exact_match": 0,
            "too_short": 0,
            "too_long": 0,
            "avg_pred_len": 0,
            "avg_target_len": 0,
            "length_errors": []
        }
        
        total = len(test_data)
        pred_lengths = []
        target_lengths = []
        
        for input_seq, target_seq in test_data:
            predicted = predict(self.model, input_seq, max_length=20)
            pred_tokens = predicted.split()
            target_tokens = target_seq.split()
            
            pred_len = len(pred_tokens)
            target_len = len(target_tokens)
            
            pred_lengths.append(pred_len)
            target_lengths.append(target_len)
            
            if pred_len == target_len:
                length_stats["exact_match"] += 1
            elif pred_len < target_len:
                length_stats["too_short"] += 1
                length_stats["length_errors"].append({
                    "input": input_seq,
                    "target_len": target_len,
                    "pred_len": pred_len,
                    "type": "too_short"
                })
            else:
                length_stats["too_long"] += 1
                length_stats["length_errors"].append({
                    "input": input_seq,
                    "target_len": target_len,
                    "pred_len": pred_len,
                    "type": "too_long"
                })
        
        length_stats["exact_match"] /= total if total > 0 else 1
        length_stats["too_short"] /= total if total > 0 else 1
        length_stats["too_long"] /= total if total > 0 else 1
        length_stats["avg_pred_len"] = np.mean(pred_lengths) if pred_lengths else 0
        length_stats["avg_target_len"] = np.mean(target_lengths) if target_lengths else 0
        
        return length_stats
    
    def repetition_analysis(self, test_data: List[Tuple[str, str]]) -> Dict:
        """
        重复率分析
        
        输入:
            test_data: 测试数据列表
        
        返回: 重复统计信息
        """
        repetition_stats = {
            "avg_repetition_rate": 0.0,
            "avg_target_repetition_rate": 0.0,
            "avg_repetition_gap": 0.0,
            "max_repetition": 0,
            "max_target_repetition": 0,
            "repetition_examples": [],
        }
        
        total_pred_rate = 0.0
        total_target_rate = 0.0
        total_gap = 0.0
        
        for input_seq, target_seq in test_data:
            predicted = predict(self.model, input_seq, max_length=20)
            pred_tokens = predicted.split()
            target_tokens = target_seq.split()

            pred_rate, pred_max = self._repetition_metrics(pred_tokens)
            target_rate, target_max = self._repetition_metrics(target_tokens)

            rate_gap = max(0.0, pred_rate - target_rate)
            max_gap = max(0, pred_max - target_max)

            total_pred_rate += pred_rate
            total_target_rate += target_rate
            total_gap += rate_gap

            repetition_stats["max_repetition"] = max(repetition_stats["max_repetition"], pred_max)
            repetition_stats["max_target_repetition"] = max(repetition_stats["max_target_repetition"], target_max)

            if rate_gap > 0.15 or max_gap > 1:
                repetition_stats["repetition_examples"].append({
                    "input": input_seq,
                    "target": target_seq,
                    "predicted": predicted,
                    "pred_repetition_rate": pred_rate,
                    "target_repetition_rate": target_rate,
                    "repetition_rate_gap": rate_gap,
                    "max_consecutive": pred_max,
                    "target_max_consecutive": target_max,
                    "max_consecutive_gap": max_gap,
                })
        
        sample_count = len(test_data) if test_data else 1
        repetition_stats["avg_repetition_rate"] = total_pred_rate / sample_count
        repetition_stats["avg_target_repetition_rate"] = total_target_rate / sample_count
        repetition_stats["avg_repetition_gap"] = total_gap / sample_count
        
        return repetition_stats
    
    def perplexity(self, test_data: List[Tuple[str, str]]) -> float:
        """
        计算困惑度（Perplexity）
        
        输入:
            test_data: 测试数据列表
        
        返回: 困惑度值
        """
        total_loss = 0.0
        total_tokens = 0
        
        criterion = torch.nn.CrossEntropyLoss(ignore_index=self.vocab['<pad>'], reduction='sum')
        
        # 获取模型所在的设备
        device = next(self.model.parameters()).device
        
        with torch.no_grad():
            for input_seq, target_seq in test_data:
                # 准备输入
                input_ids = text_to_ids(input_seq, self.vocab)
                if not input_ids:
                    continue
                
                encoder_input = torch.tensor([input_ids], dtype=torch.long, device=device)
                
                # 准备目标
                target_tokens = tokenize_sequence(target_seq)
                target_ids = sequence_to_ids(target_tokens, self.vocab)
                target_ids.append(self.vocab['<eos>'])
                target_tensor = torch.tensor([target_ids], dtype=torch.long, device=device)
                
                # 生成过程并计算损失
                generated = [self.vocab['<sos>']]
                batch_loss = 0.0
                batch_tokens = 0
                
                for step in range(len(target_ids) + 5):  # 最多多生成5个token
                    dec_input = torch.tensor([generated], dtype=torch.long, device=device)
                    
                    # 创建掩码
                    dec_mask = create_decoder_mask(dec_input)
                    enc_mask = create_padding_mask(encoder_input)
                    
                    # 模型预测
                    logits = self.model(encoder_input, dec_input, 
                                      look_ahead_mask=dec_mask, 
                                      enc_padding_mask=enc_mask)
                    
                    # 计算这一步的损失
                    if step < len(target_ids):
                        target_token = target_tensor[0, step].unsqueeze(0)
                        step_loss = criterion(logits[0, -1:], target_token)
                        batch_loss += step_loss.item()
                        batch_tokens += 1
                    
                    # 预测下一个token
                    next_token = torch.argmax(logits[0, -1], dim=-1).item()
                    if next_token == self.vocab['<eos>']:
                        break
                    generated.append(next_token)
                
                total_loss += batch_loss
                total_tokens += batch_tokens
        
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        perplexity = np.exp(avg_loss) if avg_loss != float('inf') else float('inf')
        
        return perplexity
    
    def error_analysis(self, test_data: List[Tuple[str, str]]) -> Dict:
        """
        错误分析
        
        输入:
            test_data: 测试数据列表
        
        返回: 错误统计信息
        """
        error_stats = {
            "total_errors": 0,
            "mapping_errors": [],  # 映射错误（如a应该->3但->1）
            "length_errors": [],
            "repetition_errors": [],
            "all_errors": []
        }
        
        for input_seq, target_seq in test_data:
            predicted = predict(self.model, input_seq, max_length=20)
            pred_tokens = predicted.split()
            target_tokens = target_seq.split()
            
            # 检查映射错误
            input_tokens = tokenize_sequence(input_seq)
            min_len = min(len(input_tokens), len(target_tokens), len(pred_tokens))
            
            for i in range(min_len):
                expected_mapping = f"{input_tokens[i]} -> {target_tokens[i]}"
                actual_mapping = f"{input_tokens[i]} -> {pred_tokens[i] if i < len(pred_tokens) else '<MISSING>'}"
                
                if i < len(pred_tokens) and pred_tokens[i] != target_tokens[i]:
                    error_stats["mapping_errors"].append({
                        "input": input_seq,
                        "position": i,
                        "input_token": input_tokens[i],
                        "expected": target_tokens[i],
                        "got": pred_tokens[i],
                        "mapping": expected_mapping
                    })
                    error_stats["total_errors"] += 1
            
            # 记录所有错误
            if pred_tokens != target_tokens:
                error_stats["all_errors"].append({
                    "input": input_seq,
                    "expected": target_seq,
                    "got": predicted
                })
        
        return error_stats
    
    def comprehensive_evaluation(self, test_data: List[Tuple[str, str]] = None) -> Dict:
        """
        综合评估
        
        输入:
            test_data: 测试数据（如果为None，则自动加载）
        
        返回: 完整的评估报告
        """
        if test_data is None:
            test_data = self.load_test_data()
        
        print("=" * 70)
        print("📊 开始全面评估模型性能...")
        print("=" * 70 + "\n")
        
        results = {}
        
        # 1. 完全匹配准确率
        print("1️⃣  计算完全匹配准确率...")
        results["exact_match_accuracy"] = self.exact_match_accuracy(test_data)
        print(f"   ✅ 完全匹配准确率: {results['exact_match_accuracy']:.2%}\n")
        
        # 2. Token级别准确率
        print("2️⃣  计算Token级别准确率...")
        results["token_accuracy"], token_details = self.token_accuracy(test_data)
        print(f"   ✅ Token准确率: {results['token_accuracy']:.2%}\n")
        
        # 3. 序列长度分析
        print("3️⃣  分析序列长度准确性...")
        results["length_stats"] = self.sequence_length_accuracy(test_data)
        print(f"   ✅ 长度完全匹配率: {results['length_stats']['exact_match']:.2%}")
        print(f"   ✅ 平均预测长度: {results['length_stats']['avg_pred_len']:.2f}")
        print(f"   ✅ 平均目标长度: {results['length_stats']['avg_target_len']:.2f}\n")
        
        # 4. 重复率分析
        print("4️⃣  分析重复率...")
        results["repetition_stats"] = self.repetition_analysis(test_data)
        print(f"   ✅ 平均重复率: {results['repetition_stats']['avg_repetition_rate']:.2%}")
        print(f"   ✅ 最大连续重复: {results['repetition_stats']['max_repetition']}\n")
        
        # 5. 困惑度
        print("5️⃣  计算困惑度...")
        try:
            results["perplexity"] = self.perplexity(test_data)
            print(f"   ✅ 困惑度: {results['perplexity']:.2f}\n")
        except Exception as e:
            print(f"   ⚠️  困惑度计算失败: {e}\n")
            results["perplexity"] = None
        
        # 6. 错误分析
        print("6️⃣  进行错误分析...")
        results["error_stats"] = self.error_analysis(test_data)
        print(f"   ✅ 总错误数: {results['error_stats']['total_errors']}")
        print(f"   ✅ 映射错误数: {len(results['error_stats']['mapping_errors'])}\n")
        
        results["test_data"] = test_data
        results["token_details"] = token_details
        
        return results
    
    def print_detailed_report(self, results: Dict):
        """
        打印详细评估报告
        
        输入:
            results: 评估结果字典
        """
        print("\n" + "=" * 70)
        print("📋 详细评估报告")
        print("=" * 70 + "\n")
        
        # 总体指标
        print("📊 总体性能指标:")
        print(f"   完全匹配准确率: {results['exact_match_accuracy']:.2%}")
        print(f"   Token准确率: {results['token_accuracy']:.2%}")
        if results.get('perplexity') is not None:
            print(f"   困惑度: {results['perplexity']:.2f}")
        print()
        
        # 长度分析
        print("📏 序列长度分析:")
        length_stats = results['length_stats']
        print(f"   长度完全匹配率: {length_stats['exact_match']:.2%}")
        print(f"   预测过短率: {length_stats['too_short']:.2%}")
        print(f"   预测过长率: {length_stats['too_long']:.2%}")
        print(f"   平均预测长度: {length_stats['avg_pred_len']:.2f}")
        print(f"   平均目标长度: {length_stats['avg_target_len']:.2f}")
        print()
        
        # 重复分析
        print("🔄 重复率分析:")
        rep_stats = results['repetition_stats']
        print(f"   平均重复率(预测): {rep_stats['avg_repetition_rate']:.2%}")
        print(f"   平均重复率(目标): {rep_stats['avg_target_repetition_rate']:.2%}")
        print(f"   平均重复率差值: {rep_stats['avg_repetition_gap']:.2%}")
        print(f"   最大连续重复(预测): {rep_stats['max_repetition']}")
        print(f"   最大连续重复(目标): {rep_stats['max_target_repetition']}")
        if rep_stats['repetition_examples']:
            print(f"   ⚠️  发现 {len(rep_stats['repetition_examples'])} 个高重复率样本:")
            for ex in rep_stats['repetition_examples'][:3]:  # 只显示前3个
                print(f"      - 输入: {ex['input']}")
                print(f"        目标: {ex['target']}")
                print(f"        输出: {ex['predicted']}")
                print(
                    f"        重复率差值: {ex['repetition_rate_gap']:.2%}, "
                    f"最大连续差值: {ex['max_consecutive_gap']}"
                )
        print()
        
        # 错误分析
        print("❌ 错误分析:")
        error_stats = results['error_stats']
        print(f"   总错误数: {error_stats['total_errors']}")
        print(f"   映射错误数: {len(error_stats['mapping_errors'])}")
        
        if error_stats['mapping_errors']:
            print(f"   ⚠️  映射错误示例（前5个）:")
            for err in error_stats['mapping_errors'][:5]:
                print(f"      - 位置 {err['position']}: {err['mapping']}")
                print(f"        实际输出: {err['got']}")
        
        if error_stats['all_errors']:
            print(f"\n   ⚠️  所有错误样本（前5个）:")
            for err in error_stats['all_errors'][:5]:
                print(f"      - 输入: {err['input']}")
                print(f"        期望: {err['expected']}")
                print(f"        实际: {err['got']}")
        print()
        
        # 优化建议
        print("=" * 70)
        print("💡 优化建议")
        print("=" * 70 + "\n")
        
        suggestions = []
        
        if results['exact_match_accuracy'] < 0.5:
            suggestions.append("🔴 完全匹配准确率过低，模型基本无法正确完成任务")
        
        if results['token_accuracy'] < 0.7:
            suggestions.append("🟡 Token准确率偏低，需要改进模型学习能力")
        
        if length_stats['too_long'] > 0.3:
            suggestions.append("🟡 模型生成过长，可能是停止条件不当或训练不足")
        
        if length_stats['too_short'] > 0.3:
            suggestions.append("🟡 模型生成过短，可能过早停止或学习不充分")
        
        if rep_stats['avg_repetition_gap'] > 0.15:
            suggestions.append("🔴 重复率过高，模型陷入重复生成，需要:")
            suggestions.append("   - 检查训练数据是否平衡")
            suggestions.append("   - 增加训练轮数")
            suggestions.append("   - 调整学习率或使用学习率调度")
            suggestions.append("   - 检查解码策略（考虑使用temperature或beam search）")
        
        if rep_stats['max_repetition'] - rep_stats['max_target_repetition'] > 1:
            suggestions.append("🔴 存在严重连续重复问题，模型可能陷入局部最优")
        
        if len(error_stats['mapping_errors']) > len(error_stats['all_errors']) * 0.5:
            suggestions.append("🟡 映射错误较多，模型可能:")
            suggestions.append("   - 训练数据不足或质量不高")
            suggestions.append("   - 模型容量不足（d_model或num_heads太小）")
            suggestions.append("   - 需要更多训练轮数")
        
        if results.get('perplexity') is not None and results['perplexity'] > 10:
            suggestions.append("🟡 困惑度较高，模型不确定性大，需要:")
            suggestions.append("   - 增加训练数据")
            suggestions.append("   - 调整模型架构")
            suggestions.append("   - 改进训练策略")
        
        if not suggestions:
            suggestions.append("✅ 模型性能良好，可以尝试:")
            suggestions.append("   - 增加训练数据以提高泛化能力")
            suggestions.append("   - 尝试更复杂的任务")
        
        for suggestion in suggestions:
            print(f"   {suggestion}")
        
        print("\n" + "=" * 70)


def main():
    """主函数：运行完整评估"""
    import sys
    
    model_path = sys.argv[1] if len(sys.argv) > 1 else config.MODEL_SAVE_PATH
    test_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        evaluator = ModelEvaluator(model_path)
        test_data = evaluator.load_test_data(test_file)
        
        print(f"📂 加载了 {len(test_data)} 个测试样本\n")
        
        results = evaluator.comprehensive_evaluation(test_data)
        evaluator.print_detailed_report(results)
        
    except Exception as e:
        print(f"❌ 评估过程出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

