# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import os
import ast
from typing import Dict
import torch
import json
import traceback
from math_verify import verify

from  ..vilavt_utils import parse_dialogue



import re
import json
from typing import Dict, List, Union

def parse_output(text: str):
    """解析 <tool> 或 <answer> 标签内容"""
    if not text:
        return None
    
    # 解析 <tool>
    if "<tool>" in text:
        match = re.search(r'<tool>(.*?)</tool>', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                return None
        return None
    
    # 解析 <answer>
    if "<answer>" in text:
        match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
        if match:
            content = match.group(1).strip()
            return {"text": content}        # <answer> 内容可能是纯文本
        return None
    return None

def validate_tool_call(data: Dict) -> bool:
    """验证 tool call 格式"""
    if not isinstance(data, dict):
        return False
    
    # 检查必需字段
    if "region" not in data or "query" not in data:
        return False
    
    # 检查 region
    regions = data["region"]
    if not isinstance(regions, list) or len(regions) == 0:
        return False
    
    # 检查每个 region
    for region in regions:
        if not isinstance(region, dict):
            return False
        
        # 检查 index 和 bbox_2d
        if "index" not in region or "bbox_2d" not in region:
            return False
        
        bbox = region["bbox_2d"]
        if not isinstance(bbox, list) or len(bbox) != 4:
            return False
        
        # 检查坐标有效性
        try:
            x1, y1, x2, y2 = bbox
            if x1 >= x2 or y1 >= y2:
                return False
        except:
            return False
    
    # 检查 query
    if not isinstance(data["query"], str) or not data["query"].strip():
        return False
    
    return True

def validate_answer(data: Dict) -> bool:
    """验证 answer 格式"""
    if not isinstance(data, dict):
        return False
    
    # answer 只需要非空字典即可
    # 如果有特定要求，可以在这里添加
    return len(data) > 0

def validate_think_tags(text: str) -> tuple[bool, str]:
    """
    检验文本是否包含<think>标签包围的内容
    
    Args:
        text: 要检验的文本内容
        
    Returns:
        tuple[bool, str]: (是否通过检验, 错误原因/成功信息)
    """
    if not text:
        return False, "文本内容为空"
    
    # 检查是否有<think>开始标签
    if "<think>" not in text:
        return False, "缺少<think>开始标签"
    
    # 检查是否有</think>结束标签
    if "</think>" not in text:
        return False, "缺少</think>结束标签"
    
    # 检查<think>和</think>是否成对出现
    think_count = text.count("<think>")
    end_think_count = text.count("</think>")
    
    if think_count != end_think_count:
        return False, f"<think>标签({think_count})与</think>标签({end_think_count})数量不匹配"
    
    # 检查<think>和</think>是否正确配对（按顺序出现）
    # 使用简单的状态检查
    stack = []
    for i, char in enumerate(text):
        if text[i:i+7] == "<think>":
            if i == 0 or text[i-1:i+7] != "<think>":  # 确保不是重复字符
                if i+7 <= len(text) and text[i:i+7] == "<think>":
                    stack.append(i)
        elif text[i:i+8] == "</think>":
            if i == 0 or text[i-1:i+8] != "</think>":  # 确保不是重复字符
                if stack:
                    stack.pop()
                else:
                    return False, f"在位置{i}发现未匹配的</think>标签"
    
    if stack:
        return False, f"存在未闭合的<think>标签，起始位置: {stack}"
    
    return True, "所有<think>标签都正确配对并包围内容"

def cal_format_reward(conversations) -> float:
    """
    计算格式奖励
    
    规则：
    - assistant 回复中的 <tool> 必须能解析且格式正确
    - assistant 回复中的 <answer> 必须能解析且格式正确
    - assistant 回复必须包含<think>标签包围的内容
    - 任何解析失败返回 0.0，全部成功返回 1.0
    
    Args:
        conversations: 对话列表
    
    Returns:
        float: 0.0 或 1.0
    """
    try:
        for idx, item in enumerate(conversations):
            # 只检查 assistant
            if item.get("role") != "assistant":
                continue
            
            # 获取文本内容
            content = item.get("content", "")
            if isinstance(content, list):
                # 处理列表格式: [{"type": "text", "text": "..."}]
                text = "".join([c.get("text", "") for c in content if c.get("type") == "text"])
            else:
                text = content
            
            # 检查<think>标签
            think_valid, think_error = validate_think_tags(text)
            if not think_valid:
                # print(f"{text} - <think>标签: {think_error}")
                return 0.0
            
            # 检查是否有需要验证的标签
            has_tool = "<tool>" in text
            has_answer = "<answer>" in text
            
            if not has_tool and not has_answer:
                continue
            
            # 解析
            parsed = parse_output(text)
            if parsed is None:
                # print(f"{text} - 标签解析失败")
                return 0.0
            
            # 验证格式
            if has_tool:
                if not validate_tool_call(parsed):
                    # print(f"{text}  - <tool>格式不正确")
                    return 0.0
            
            if has_answer:
                if not validate_answer(parsed):
                    # print(f"{text} - <answer>格式不正确")
                    return 0.0
        
        return 1.0
    
    except Exception as e:
        # print(f"Error in cal_format_reward: {e}")
        return 0.0


def extract_answer(text):
    text = str(text).strip()
    pattern = r'<answer>\s*(.*?)\s*</answer>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text

def extract_box_answer(response):
    resp = response.split("\\boxed{")[-1]
    lt = len(resp)
    counter, end = 1, None
    for i in range(lt):
        if resp[i] == "{":
            counter += 1
        elif resp[i] == "}":
            counter -= 1
        if counter == 0:
            end = i
            break
        elif i == lt - 1:
            end = lt
            break
    if end is not None:
        response = resp[:end]
    return response


def normalize_number(num_str):
    try:
        num_str = num_str.replace(',', '')
        return float(num_str)
    except Exception as e:
        return None


def mean_relative_accuracy(pred, target, start=0.5, end=0.95, interval=0.05):

    if not torch.is_tensor(pred):
        pred = torch.tensor(pred, dtype=torch.float32)
    if not torch.is_tensor(target):
        target = torch.tensor(target, dtype=torch.float32)
    
    epsilon = 1e-8
    rel_error = torch.abs(pred - target) / (torch.abs(target) + epsilon)
    
    thresholds = torch.arange(start, end + interval/2, interval, dtype=torch.float32)
    
    conditions = rel_error < (1 - thresholds)  
    mra = conditions.float().mean()  
    return mra.item()


def is_answer_correct(prediction: str, ground_truth: str) -> bool:
    try:
        return verify(prediction, ground_truth)
    except Exception as e:
        # print(f"在验证过程中发生错误: {e}")
        # print(f"  - Prediction: {prediction}")
        # print(f"  - Ground Truth: {ground_truth}")
        return False

def cal_outcome_reward(model_output, ground_truth, question_type):
    """计算准确率奖励"""
    # 输入验证
    if not all([model_output, ground_truth, question_type]):
        return 0.0
    
    # 提取答案
    try:
        output_ans = extract_answer(model_output)
        match = re.search(r"\\boxed\{(.*?)\}", output_ans)
        if match:
            output_ans = extract_box_answer(output_ans)
        gt_ans = extract_answer(ground_truth)
    except Exception as e:
        print(f"Error extracting answer: {e}")
        return 0.0
    
    if not output_ans or not gt_ans:
        return 0.0
    # 标准化
    question_type = question_type.lower().strip()
    print(f"output_ans: {output_ans}, gt_ans: {gt_ans}", end=" ")
    # 评估
    try:
        if question_type == "multiple choice":
            output_stripped = output_ans.strip()
            gt_stripped = gt_ans.strip()
            if not output_stripped or not gt_stripped:
                return 0.0
            return 1.0 if output_stripped[0].lower() == gt_stripped[0].lower() else 0.0
        elif question_type == "math":
            # gt_number = normalize_number(gt_ans)
            # out_number = normalize_number(output_ans)
            # if gt_number is None or out_number is None:
            #     return 0.0
            # return 1.0 if round(gt_number, 2) == round(out_number, 2) else 0.0
            return float(is_answer_correct(output_ans, gt_ans))
        elif question_type in ["regression", "numerical"]:
            gt_number = normalize_number(gt_ans)
            out_number = normalize_number(output_ans)
            if gt_number is None or out_number is None:
                return 0.0
            mra = mean_relative_accuracy(out_number, gt_number)
            return max(0.0, min(1.0, mra))
        
        elif question_type == "judgment":
            output_lower = output_ans.lower().strip()
            gt_lower = gt_ans.lower().strip()
            
            output_has_yes = bool(re.search(r'\byes\b', output_lower))
            output_has_no = bool(re.search(r'\bno\b', output_lower))
            gt_has_yes = bool(re.search(r'\byes\b', gt_lower))
            gt_has_no = bool(re.search(r'\bno\b', gt_lower))
            
            # print("output_has_yes: ", output_has_yes, "output_has_no: ", output_has_no, "gt_has_yes: ", gt_has_yes)
            if (output_has_yes and gt_has_yes) or (output_has_no and gt_has_no):
                return 1.0
            return 0.0
        
        else:
            return 0.0
    
    except Exception as e:
        print(f"Error computing reward: {e}")
        return 0.0



def gcot_compute_score(predict_str: str, ground_truth: str, question_type: str, data_type: str) -> Dict[str, float]:
    """
    Computes the overall reward, as well as the format and accuracy rewards.
    
    Args:
        predict_str (str): The predicted response string.
        ground_truth (str): The correct answer string.
        
    Returns:
        Dict[str, float]: A dictionary containing the overall, format, and accuracy rewards.
    """
    conversations: list = parse_dialogue(predict_str)
    format_reward = cal_format_reward(conversations)
    outcome_reward = cal_outcome_reward(conversations[-1]['content'], ground_truth, question_type)
    # print("Conversations: ", conversations)
    print(f"Question Type: {question_type}, Format: {format_reward}, Outcome Reward: {outcome_reward }")
    return {
        "overall": 0.5 * outcome_reward  + 0.5 * format_reward if outcome_reward  > 0.0 else outcome_reward ,
        "format": format_reward,
        "accuracy": outcome_reward,
        f"{data_type}_accuracy": outcome_reward,
        f"{data_type}_format": format_reward,
    }


# 1. Multiple Choice 测试
multiple_choice_tests = [
    # (model_output, ground_truth, question_type, expected_reward, description)
    ("A", "A", "multiple choice", 1.0, "正确答案 - 单字母匹配"),
    ("B", "B", "multiple choice", 1.0, "正确答案 - B选项"),
    ("A", "B", "multiple choice", 0.0, "错误答案 - A vs B"),
    ("a", "A", "multiple choice", 1.0, "大小写不敏感 - 小写 vs 大写"),
    ("A ", " A", "multiple choice", 1.0, "忽略空格"),
    ("Answer: A", "A", "multiple choice", 1.0, "答案在文本中"),
    ("The answer is A.", "A", "multiple choice", 1.0, "完整句子中的答案"),
    ("\\boxed{A}", "A", "multiple choice", 1.0, "LaTeX boxed 格式"),
    ("A. First option", "A", "multiple choice", 1.0, "带选项描述"),
    ("ABCD", "B", "multiple choice", 0.0, "多个字母（错误）"),
    ("", "A", "multiple choice", 0.0, "空输出"),
    ("X", "A", "multiple choice", 0.0, "无效选项"),
]

# 2. Numerical 测试
numerical_tests = [
    ("42", "42", "numerical", 1.0, "整数完全匹配"),
    ("3.14", "3.14", "numerical", 1.0, "小数完全匹配"),
    ("3.141", "3.142", "numerical", 1.0, "四舍五入到2位匹配"),
    ("3.145", "3.141", "numerical", 0.0, "四舍五入后不匹配"),
    ("1,234.56", "1234.56", "numerical", 1.0, "逗号分隔数字"),
    ("1234.567", "1234.564", "numerical", 1.0, "精度匹配（2位）"),
    ("0.00", "0.00", "numerical", 1.0, "零值"),
    ("-42", "-42", "numerical", 1.0, "负数"),
    ("-3.14", "-3.14", "numerical", 1.0, "负小数"),
    ("100", "100.001", "numerical", 1.0, "小数点后误差在范围内"),
    ("100", "100.01", "numerical", 0.0, "小数点后误差超出范围"),
    ("\\boxed{42}", "42", "numerical", 1.0, "LaTeX boxed 格式数字"),
    ("The answer is 42", "42", "numerical", 1.0, "文本中的数字"),
    ("abc", "42", "numerical", 0.0, "非数字输出"),
    ("", "42", "numerical", 0.0, "空输出"),
]

# 3. Regression 测试
regression_tests = [
    ("100", "100", "regression", 1.0, "完全匹配"),
    ("100", "105", "regression", 0.95, "5%相对误差 - 应该高分"),
    ("100", "110", "regression", 0.90, "10%相对误差"),
    ("100", "150", "regression", 0.5, "50%相对误差 - 中等分"),
    ("100", "200", "regression", 0.0, "100%相对误差 - 低分"),
    ("50", "55", "regression", 0.90, "小值的相对误差"),
    ("1000", "1050", "regression", 0.95, "大值的相对误差"),
    ("0.5", "0.52", "regression", 0.96, "小数的相对误差"),
    ("-100", "-105", "regression", 0.95, "负数的相对误差"),
    ("\\boxed{100}", "105", "regression", 0.95, "LaTeX格式回归"),
]

# 4. Judgment 测试
judgment_tests = [
    ("Yes", "Yes", "judgment", 1.0, "Yes完全匹配"),
    ("No", "No", "judgment", 1.0, "No完全匹配"),
    ("yes", "Yes", "judgment", 1.0, "大小写不敏感 - yes"),
    ("YES", "yes", "judgment", 1.0, "大小写不敏感 - YES"),
    ("Yes", "No", "judgment", 0.0, "Yes vs No"),
    ("No", "Yes", "judgment", 0.0, "No vs Yes"),
    ("The answer is yes.", "Yes", "judgment", 1.0, "句子中的yes"),
    ("I think no.", "No", "judgment", 1.0, "句子中的no"),
    ("Yes, it is correct.", "Yes", "judgment", 1.0, "Yes在句首"),
    ("No, it is wrong.", "No", "judgment", 1.0, "No在句首"),
    ("\\boxed{Yes}", "Yes", "judgment", 1.0, "LaTeX boxed Yes"),
    ("Maybe", "Yes", "judgment", 0.0, "模糊答案"),
    ("", "Yes", "judgment", 0.0, "空输出"),
    ("yesno", "Yes", "judgment", 1.0, "包含yes单词"),
    ("noyes", "No", "judgment", 1.0, "包含no单词"),
]

# 5. 边界情况测试
edge_case_tests = [
    ("", "", "multiple choice", 0.0, "双空输入"),
    ("A", "", "multiple choice", 0.0, "空ground truth"),
    ("", "A", "multiple choice", 0.0, "空model output"),
    (None, "A", "multiple choice", 0.0, "None输入"),
    ("A", "A", "", 0.0, "空question type"),
    ("A", "A", "unknown", 0.0, "未知question type"),
    ("42", "42", "NUMERICAL", 1.0, "大写question type"),
    ("  A  ", "  A  ", "multiple choice", 1.0, "多余空格"),
    ("\\boxed{\\boxed{A}}", "A", "multiple choice", 1.0, "嵌套boxed"),
]

# 6. 复杂格式测试
complex_format_tests = [
    (
        "Let me solve this step by step:\n1. First...\n2. Then...\nFinal answer: A",
        "A",
        "multiple choice",
        1.0,
        "多步骤推理答案"
    ),
    (
        "The calculation yields: 3.14159, so the answer is approximately 3.14",
        "3.14",
        "numerical",
        1.0,
        "推理过程中的数字"
    ),
    (
        "\\boxed{\\text{Yes, this is correct}}",
        "Yes",
        "judgment",
        1.0,
        "LaTeX text中的判断"
    ),
    (
        "Answer: \\boxed{42.5}",
        "42.5",
        "numerical",
        1.0,
        "Answer前缀 + boxed"
    ),
    (
        "The final result is \\boxed{100}",
        "105",
        "regression",
        0.95,
        "回归任务的推理"
    ),
]

# 7. 数值精度测试
precision_tests = [
    ("3.14159", "3.14", "numerical", 1.0, "π近似 - 应该匹配"),
    ("2.718", "2.72", "numerical", 1.0, "e近似 - 应该匹配"),
    ("1.999", "2.001", "numerical", 1.0, "接近2的值"),
    ("0.001", "0.002", "numerical", 1.0, "极小值比较"),
    ("999.99", "1000.01", "numerical", 1.0, "接近1000"),
    ("0.999", "1.001", "numerical", 1.0, "跨越1的边界"),
]

# 8. 特殊字符测试
special_char_tests = [
    ("A)", "A", "multiple choice", 1.0, "带括号的选项"),
    ("(A)", "A", "multiple choice", 1.0, "括号包围"),
    ("$42$", "42", "numerical", 1.0, "LaTeX数学模式"),
    ("\\text{Yes}", "Yes", "judgment", 1.0, "LaTeX text命令"),
    ("1,234,567", "1234567", "numerical", 1.0, "多个逗号"),
]

# 9. 回归任务特殊测试
regression_special_tests = [
    ("0", "0", "regression", 1.0, "零值回归"),
    ("0.001", "0.0", "regression", 0.0, "接近零的误差（除零）"),
    ("-50", "-48", "regression", 0.96, "负值回归"),
    ("1e6", "1000000", "regression", 1.0, "科学计数法"),
]


# ============= 测试工具函数 =============
def run_test_cases(test_cases, test_name):
    """运行测试用例并打印结果"""
    print(f"\n{'='*80}")
    print(f"Testing: {test_name}")
    print(f"{'='*80}")
    
    passed = 0
    failed = 0
    
    for i, (model_output, ground_truth, question_type, expected, description) in enumerate(test_cases, 1):
        result = cal_outcome_reward(model_output, ground_truth, question_type)
        status = "✓ PASS" if abs(result - expected) < 0.01 else "✗ FAIL"
        
        if abs(result - expected) < 0.01:
            passed += 1
        else:
            failed += 1
        
        print(f"\n{status} Test {i}: {description}")
        print(f"  Model Output: '{model_output}'")
        print(f"  Ground Truth: '{ground_truth}'")
        print(f"  Question Type: '{question_type}'")
        print(f"  Expected: {expected:.2f}, Got: {result:.2f}")
    
    print(f"\n{'='*80}")
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed} tests")
    print(f"{'='*80}\n")
    
    return passed, failed
# ========== 示例测试 ==========
if __name__ == "__main__":
    total_passed = 0
    total_failed = 0
    
    test_suites = [
        (multiple_choice_tests, "Multiple Choice Questions"),
        (numerical_tests, "Numerical Questions"),
        (regression_tests, "Regression Questions"),
        (judgment_tests, "Judgment Questions"),
        (edge_case_tests, "Edge Cases"),
        (complex_format_tests, "Complex Format"),
        (precision_tests, "Numerical Precision"),
        (special_char_tests, "Special Characters"),
        (regression_special_tests, "Regression Special Cases"),
    ]
    
    for test_cases, test_name in test_suites:
        passed, failed = run_test_cases(test_cases, test_name)
        total_passed += passed
        total_failed += failed
    
    # 总结
    print(f"\n{'#'*80}")
    print(f"OVERALL SUMMARY")
    print(f"{'#'*80}")
    print(f"Total Tests: {total_passed + total_failed}")
    print(f"Passed: {total_passed} ({100*total_passed/(total_passed+total_failed):.1f}%)")
    print(f"Failed: {total_failed} ({100*total_failed/(total_passed+total_failed):.1f}%)")
    print(f"{'#'*80}\n")
    
    # 如果有失败的测试，返回错误代码
    if total_failed > 0:
        exit(1)
