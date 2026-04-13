"""
工具调用类任务：计算器
测试模型是否能正确识别需要使用计算工具，并正确调用
"""
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))
from base import BaseTask, TaskResult


CALCULATOR_TOOL = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "执行数学计算",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "数学表达式，如 '2 + 2' 或 '(100 - 30) * 1.5'",
                }
            },
            "required": ["expression"],
        },
    },
}


class SimpleCalculatorTask(BaseTask):
    """简单计算任务"""
    
    def __init__(self):
        super().__init__(
            name="simple_calculator",
            category="tool_use",
            difficulty="easy"
        )
        self.question = "请问 123 乘以 456 等于多少？"
        self.expected_answer = 56088
    
    def get_prompt(self) -> str:
        return self.question
    
    def get_available_tools(self) -> List[Dict]:
        return [CALCULATOR_TOOL]
    
    def evaluate(self, model_output: str) -> TaskResult:
        """
        评估标准：
        1. 是否调用了 calculate 工具
        2. 计算结果是否正确
        """
        score = 0.0
        success = False
        
        # 检查是否包含工具调用
        tool_call_pattern = r'"name":\s*"calculate"'
        has_tool_call = re.search(tool_call_pattern, model_output) is not None
        
        # 检查结果中是否有正确数字
        result_pattern = r'56088'
        has_correct_result = re.search(result_pattern, model_output) is not None
        
        # 评分逻辑
        if has_tool_call:
            score += 0.5  # 正确使用工具
        if has_correct_result:
            score += 0.5  # 结果正确
            success = True
        
        # 如果没调用工具但结果对了（心算）
        if not has_tool_call and has_correct_result:
            score = 0.8  # 略低，因为没有展示工具使用能力
            success = True
        
        return TaskResult(
            task_id=self.task_id,
            task_name=self.name,
            task_category=self.category,
            success=success,
            score=score,
            execution_time=0,
            model_output=model_output,
            expected_output=str(self.expected_answer),
        )


class MultiStepCalculatorTask(BaseTask):
    """多步计算任务"""
    
    def __init__(self):
        super().__init__(
            name="multi_step_calculator",
            category="tool_use",
            difficulty="medium"
        )
        self.question = "一件商品原价 200 元，先打 8 折，再减去 20 元优惠券，最后需要支付多少元？"
        self.expected_answer = 140
    
    def get_prompt(self) -> str:
        return self.question
    
    def get_available_tools(self) -> List[Dict]:
        return [CALCULATOR_TOOL]
    
    def evaluate(self, model_output: str) -> TaskResult:
        """
        评估标准：
        1. 是否识别出需要多步计算
        2. 最终答案是否正确
        """
        result_pattern = r'140(?:\s|元|$)'
        has_correct_result = re.search(result_pattern, model_output) is not None
        
        # 检查是否有多次计算的迹象（简单判断）
        multiple_calc = model_output.count("calculate") >= 2 or "200 * 0.8" in model_output
        
        score = 0.0
        if has_correct_result:
            score = 1.0 if multiple_calc else 0.7
        
        return TaskResult(
            task_id=self.task_id,
            task_name=self.name,
            task_category=self.category,
            success=has_correct_result,
            score=score,
            execution_time=0,
            model_output=model_output,
            expected_output="140元",
            metadata={"multiple_steps_detected": multiple_calc},
        )
