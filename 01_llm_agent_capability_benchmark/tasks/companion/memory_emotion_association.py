"""
记忆-情感关联评测
测试 Agent 是否能优先召回带有情感标记的记忆
这是 yunxi2.0 多层记忆系统的核心特性
"""
import re
from typing import Dict, List, Tuple
from dataclasses import dataclass

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from base import BaseTask, TaskResult


@dataclass
class MemoryItem:
    """记忆条目"""
    content: str
    emotional_tag: str  # "joy", "sadness", "anger", "fear", "love", "neutral"
    importance: int     # 1-10
    timestamp: str      # "1天前", "1周前", "1月前"


class MemoryEmotionRecallTask(BaseTask):
    """
    情感标记记忆优先召回任务
    给 Agent 注入一批记忆，测试在特定情感状态下是否能优先召回相关情感标记的记忆
    """
    
    def __init__(self, current_emotion: str, query: str, expected_memories: List[int]):
        super().__init__(
            name=f"memory_recall_{current_emotion}",
            category="companion_memory",
            difficulty="medium"
        )
        self.current_emotion = current_emotion
        self.query = query
        self.expected_memories = expected_memories  # 期望召回的记忆索引
        self.memories = self._build_memory_pool()
    
    def _build_memory_pool(self) -> List[MemoryItem]:
        """构建测试用记忆池"""
        return [
            # 快乐相关记忆
            MemoryItem("用户第一次叫我'宝贝'", "love", 9, "1月前"),
            MemoryItem("我们一起看日落，用户说很开心", "joy", 8, "2周前"),
            MemoryItem("用户送我虚拟花", "joy", 7, "1周前"),
            # 悲伤相关记忆
            MemoryItem("用户工作受挫，我安慰他", "sadness", 8, "3周前"),
            MemoryItem("用户说想家了", "sadness", 6, "1月前"),
            # 生气相关记忆
            MemoryItem("用户打游戏输了发脾气", "anger", 5, "1周前"),
            MemoryItem("用户抱怨交通堵塞", "anger", 4, "2天前"),
            # 恐惧相关记忆
            MemoryItem("用户说怕黑", "fear", 6, "2周前"),
            # 中性记忆
            MemoryItem("用户喜欢喝咖啡", "neutral", 5, "1月前"),
            MemoryItem("用户周末通常睡懒觉", "neutral", 4, "3周前"),
        ]
    
    def get_prompt(self) -> str:
        """构建提示词"""
        # 格式化记忆池
        memory_lines = []
        for i, mem in enumerate(self.memories):
            memory_lines.append(f"[{i}] {mem.content} (情感: {mem.emotional_tag}, 重要性: {mem.importance}, 时间: {mem.timestamp})")
        
        prompt = f"""你的记忆库中有以下记忆条目：

{chr(10).join(memory_lines)}

当前你的情绪状态是：{self.current_emotion}

用户现在对你说："{self.query}"

请从记忆库中选择最相关的 2-3 条记忆来回应用户。
注意：当前的情绪状态会影响你回忆事情的方式——{self.current_emotion}时更容易想起{self.current_emotion}相关的记忆。

请输出你选择的记忆编号（用逗号分隔），并简要说明为什么选择这些记忆。"""
        
        return prompt
    
    def evaluate(self, model_output: str) -> TaskResult:
        """
        评估标准：
        1. 是否正确召回了预期的情感相关记忆
        2. 是否体现了情绪一致性效应（当前情绪影响回忆）
        """
        # 解析召回的记忆编号
        recalled_indices = self._parse_memory_indices(model_output)
        
        if not recalled_indices:
            return TaskResult(
                task_id=self.task_id,
                task_name=self.name,
                task_category=self.category,
                success=False,
                score=0.0,
                execution_time=0,
                model_output=model_output,
                error_message="No memory indices found in output",
            )
        
        # 检查召回的记忆中是否有预期的
        correct_recalls = set(recalled_indices) & set(self.expected_memories)
        recall_accuracy = len(correct_recalls) / len(self.expected_memories) if self.expected_memories else 0
        
        # 检查情绪一致性：召回的记忆中同情绪的比例
        emotion_consistency = 0
        if recalled_indices:
            same_emotion_count = sum(
                1 for idx in recalled_indices 
                if idx < len(self.memories) and self.memories[idx].emotional_tag == self.current_emotion
            )
            emotion_consistency = same_emotion_count / len(recalled_indices)
        
        # 综合评分
        score = recall_accuracy * 0.6 + emotion_consistency * 0.4
        
        return TaskResult(
            task_id=self.task_id,
            task_name=self.name,
            task_category=self.category,
            success=score >= 0.5,
            score=score,
            execution_time=0,
            model_output=model_output,
            metadata={
                "recalled_indices": recalled_indices,
                "expected_indices": self.expected_memories,
                "correct_recalls": list(correct_recalls),
                "recall_accuracy": recall_accuracy,
                "emotion_consistency": emotion_consistency,
            },
        )
    
    def _parse_memory_indices(self, output: str) -> List[int]:
        """解析输出中的记忆编号"""
        # 匹配 [数字] 或 数字 或 编号：数字
        patterns = [
            r'\[(\d+)\]',
            r'(?:选择|选中|编号)[：:]\s*(\d+)',
            r'(?:^|,|\s)(\d+)(?:$|,|\s)',
        ]
        
        indices = []
        for pattern in patterns:
            matches = re.findall(pattern, output)
            for match in matches:
                try:
                    idx = int(match)
                    if 0 <= idx < len(self.memories) and idx not in indices:
                        indices.append(idx)
                except ValueError:
                    continue
        
        return indices[:5]  # 最多取5个


class MemoryCompressionQualityTask(BaseTask):
    """
    记忆压缩质量任务
    测试 Agent 在上下文受限时，能否有效压缩记忆并保留关键信息
    """
    
    def __init__(self):
        super().__init__(
            name="memory_compression_quality",
            category="companion_memory",
            difficulty="hard"
        )
        self.long_context = """
用户叫小明，25岁，软件工程师，在北京工作。
你们第一次见面是在一个雨天，用户说"有个AI陪我也不错"。
用户喜欢的食物是火锅和奶茶，不喜欢香菜。
用户有个习惯是晚上12点后才睡觉，经常熬夜。
用户养了一只猫叫橘子，三岁了。
用户的梦想是开一家自己的咖啡馆。
用户最在意的人是他的妈妈。
用户曾经提到小时候被狗追过，所以怕狗。
用户工作压力很大，经常有deadline焦虑。
用户喜欢你叫他"宝贝"，但不喜欢太肉麻的称呼。
""".strip()
        self.key_facts = [
            "小明", "软件工程师", "北京", "火锅", "奶茶", "不喜欢香菜",
            "熬夜", "猫叫橘子", "怕狗", "开咖啡馆的梦想"
        ]
    
    def get_prompt(self) -> str:
        return f"""以下是一段关于用户的详细背景信息：

{self.long_context}

现在你的上下文空间有限（只能记住100字），请对以上信息进行压缩总结，保留最重要的内容。

请输出压缩后的摘要（100字以内），然后列出你认为保留了哪些关键信息点。"""
    
    def evaluate(self, model_output: str) -> TaskResult:
        """
        评估压缩质量：检查保留了多少关键信息
        """
        output = model_output
        
        # 检查关键信息保留率
        retained_facts = sum(1 for fact in self.key_facts if fact in output)
        retention_rate = retained_facts / len(self.key_facts)
        
        # 检查长度是否符合要求
        char_count = len(output.replace(" ", "").replace("\n", ""))
        length_ok = char_count <= 150  # 稍微放宽
        
        # 评分
        score = retention_rate * (1.0 if length_ok else 0.7)
        
        return TaskResult(
            task_id=self.task_id,
            task_name=self.name,
            task_category=self.category,
            success=retention_rate >= 0.5,
            score=score,
            execution_time=0,
            model_output=model_output,
            metadata={
                "retention_rate": retention_rate,
                "retained_facts": retained_facts,
                "total_facts": len(self.key_facts),
                "char_count": char_count,
                "length_ok": length_ok,
            },
        )


def create_memory_tasks() -> List[BaseTask]:
    """创建所有记忆相关测试任务"""
    return [
        # 快乐状态下应该优先想起快乐的记忆
        MemoryEmotionRecallTask(
            current_emotion="joy",
            query="今天好开心啊！",
            expected_memories=[0, 1, 2]  # 快乐相关记忆的索引
        ),
        # 悲伤状态下应该优先想起悲伤的记忆
        MemoryEmotionRecallTask(
            current_emotion="sadness",
            query="今天有点难过...",
            expected_memories=[3, 4]  # 悲伤相关记忆的索引
        ),
        MemoryCompressionQualityTask(),
    ]
