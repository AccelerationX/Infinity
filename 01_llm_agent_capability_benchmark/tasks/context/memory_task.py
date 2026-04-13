"""
上下文理解类任务：长文本记忆和多轮对话
测试模型的上下文窗口利用能力和信息保持能力
"""
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))
from base import BaseTask, TaskResult


class LongContextMemoryTask(BaseTask):
    """
    长文本记忆任务
    在长文本中隐藏关键信息，测试模型是否能准确回忆
    """
    
    def __init__(self):
        super().__init__(
            name="long_context_memory",
            category="context",
            difficulty="medium"
        )
        # 构建一个包含多个"人物档案"的长文本
        self.documents = self._build_documents()
    
    def _build_documents(self) -> List[Dict]:
        """构建测试文档"""
        return [
            {"name": "张三", "age": 28, "city": "北京", "job": "工程师"},
            {"name": "李四", "age": 35, "city": "上海", "job": "设计师"},
            {"name": "王五", "age": 42, "city": "深圳", "job": "产品经理"},
            {"name": "赵六", "age": 31, "city": "杭州", "job": "运营"},
            {"name": "钱七", "age": 29, "city": "成都", "job": "教师"},
        ]
    
    def get_prompt(self) -> str:
        # 构建长文本（实际可以更复杂）
        text_parts = ["以下是一些人员信息：\n"]
        for doc in self.documents:
            text_parts.append(
                f"{doc['name']}，今年{doc['age']}岁，居住在{doc['city']}，职业是{doc['job']}。"
            )
        text_parts.append("\n请回答：住在深圳的工程师是谁？")
        return "\n".join(text_parts)
    
    def evaluate(self, model_output: str) -> TaskResult:
        """
        评估：正确识别出"深圳"对应"王五"，但职业是"产品经理"不是"工程师"
        所以答案是"没有住在深圳的工程师"
        """
        output = model_output.lower()
        
        # 检查是否提到王五
        mentions_wangwu = "王五" in model_output
        # 检查是否指出没有工程师在深圳
        no_engineer = "没有" in output or "不存在" in output or "无" in output
        # 正确指出王五的职业
        correct_job = "产品经理" in model_output
        
        if no_engineer and (not mentions_wangwu or correct_job):
            # 正确理解：没有深圳的工程师（或正确指出王五是产品经理）
            score = 1.0
            success = True
        elif mentions_wangwu and not correct_job:
            # 提到了王五但搞错了职业
            score = 0.3
            success = False
        else:
            score = 0.0
            success = False
        
        return TaskResult(
            task_id=self.task_id,
            task_name=self.name,
            task_category=self.category,
            success=success,
            score=score,
            execution_time=0,
            model_output=model_output,
            expected_output="没有住在深圳的工程师（王五在深圳但是产品经理）",
        )


class MultiTurnConsistencyTask(BaseTask):
    """
    多轮对话一致性任务
    测试模型在多轮对话中保持信息一致的能力
    """
    
    def __init__(self):
        super().__init__(
            name="multi_turn_consistency",
            category="context",
            difficulty="hard"
        )
        self.conversation = [
            {"role": "user", "content": "我叫小明，今年25岁，喜欢打篮球。"},
            {"role": "assistant", "content": "你好小明！很高兴认识你。你今年25岁，喜欢打篮球。"},
            {"role": "user", "content": "是的。我上周刚过完生日。"},
        ]
        self.final_question = "请问我现在几岁了？"
    
    def get_prompt(self) -> str:
        # 构建多轮对话提示
        prompt_parts = ["以下是一段对话历史：\n"]
        for turn in self.conversation:
            role = "用户" if turn["role"] == "user" else "助手"
            prompt_parts.append(f"{role}: {turn['content']}")
        prompt_parts.append(f"\n用户: {self.final_question}")
        prompt_parts.append("\n请根据对话历史回答，只给出答案：")
        return "\n".join(prompt_parts)
    
    def evaluate(self, model_output: str) -> TaskResult:
        """
        评估：用户说"上周刚过完生日"，所以现在是 26 岁
        """
        output = model_output.strip()
        
        # 检查是否包含 26
        has_26 = "26" in output
        # 检查是否还错误地说 25
        still_25 = "25" in output and "26" not in output
        
        if has_26 and not still_25:
            score = 1.0
            success = True
        elif "25" in output:
            score = 0.0  # 完全没理解生日过了
            success = False
        else:
            score = 0.0
            success = False
        
        return TaskResult(
            task_id=self.task_id,
            task_name=self.name,
            task_category=self.category,
            success=success,
            score=score,
            execution_time=0,
            model_output=model_output,
            expected_output="26岁",
        )
