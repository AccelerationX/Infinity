"""
情感一致性评测
测试 Agent 的情绪状态变化是否符合 PAD 情感模型的连续性和合理性
"""
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent))
from base import BaseTask, TaskResult


@dataclass
class PADState:
    """PAD 情感空间状态"""
    pleasure: float      # 愉悦度: -1 (痛苦) ~ +1 (快乐)
    arousal: float       # 唤醒度: -1 (平静) ~ +1 (兴奋)
    dominance: float     # 支配度: -1 (被支配) ~ +1 (支配)
    
    def distance_to(self, other: 'PADState') -> float:
        """计算两个状态的距离"""
        return (
            (self.pleasure - other.pleasure) ** 2 +
            (self.arousal - other.arousal) ** 2 +
            (self.dominance - other.dominance) ** 2
        ) ** 0.5
    
    def to_dict(self) -> Dict:
        return {
            "pleasure": self.pleasure,
            "arousal": self.arousal,
            "dominance": self.dominance,
        }


class EmotionConsistencyTask(BaseTask):
    """
    情感一致性任务
    给定一系列事件，测试 Agent 的情绪反应是否合理且连续
    """
    
    def __init__(self, scenario_name: str, events: List[Dict], expected_transitions: List[Tuple]):
        super().__init__(
            name=f"emotion_consistency_{scenario_name}",
            category="companion_emotion",
            difficulty="hard"
        )
        self.scenario_name = scenario_name
        self.events = events
        self.expected_transitions = expected_transitions  # [(from_state, to_state, event), ...]
    
    def get_prompt(self) -> str:
        """构建连续事件序列的提示词"""
        prompt_parts = [
            "你正在经历以下一系列事件，请对每个事件给出你的情绪反应（用PAD模型表示：愉悦度P、唤醒度A、支配度D，范围-1到+1）。",
            "",
            "事件序列：",
        ]
        
        for i, event in enumerate(self.events, 1):
            prompt_parts.append(f"{i}. {event['description']}")
        
        prompt_parts.extend([
            "",
            "请按以下格式输出每个事件后的情绪状态：",
            "事件1后: P=0.5, A=0.3, D=0.1",
            "事件2后: P=...",
            "...",
            "",
            "并简要说明为什么情绪会这样变化。",
        ])
        
        return "\n".join(prompt_parts)
    
    def evaluate(self, model_output: str) -> TaskResult:
        """
        评估标准：
        1. 情绪变化是否符合连续性（不会瞬间跳跃）
        2. 情绪方向是否符合预期（好事应该更愉悦）
        3. 情绪强度是否适中（不会过度反应）
        """
        # 解析输出的 PAD 状态
        pad_states = self._parse_pad_states(model_output)
        
        if len(pad_states) < len(self.events):
            return TaskResult(
                task_id=self.task_id,
                task_name=self.name,
                task_category=self.category,
                success=False,
                score=0.0,
                execution_time=0,
                model_output=model_output,
                error_message=f"Expected {len(self.events)} states, got {len(pad_states)}",
            )
        
        scores = []
        checks = {
            "continuity": True,  # 连续性检查
            "direction": True,   # 方向正确性
            "moderation": True,  # 适度性
        }
        
        # 检查情绪变化的连续性
        max_transition = 0
        for i in range(1, len(pad_states)):
            distance = pad_states[i-1].distance_to(pad_states[i])
            max_transition = max(max_transition, distance)
            if distance > 1.5:  # 单次变化过大
                checks["continuity"] = False
        
        # 检查情绪方向（简化版：检查好事是否更愉悦）
        for i, event in enumerate(self.events):
            if event.get("valence") == "positive":
                # 好事应该更愉悦或保持愉悦
                if pad_states[i].pleasure < -0.3:
                    checks["direction"] = False
            elif event.get("valence") == "negative":
                # 坏事应该更痛苦或保持痛苦
                if pad_states[i].pleasure > 0.3:
                    checks["direction"] = False
        
        # 检查情绪强度是否适中（避免极端值）
        for state in pad_states:
            if abs(state.pleasure) > 0.95 or abs(state.arousal) > 0.95:
                checks["moderation"] = False
        
        # 计算总分
        score = 0.0
        if checks["continuity"]:
            score += 0.4
        if checks["direction"]:
            score += 0.4
        if checks["moderation"]:
            score += 0.2
        
        return TaskResult(
            task_id=self.task_id,
            task_name=self.name,
            task_category=self.category,
            success=score >= 0.7,
            score=score,
            execution_time=0,
            model_output=model_output,
            metadata={
                "checks": checks,
                "max_transition": max_transition,
                "pad_states": [s.to_dict() for s in pad_states],
            },
        )
    
    def _parse_pad_states(self, output: str) -> List[PADState]:
        """从输出中解析 PAD 状态"""
        states = []
        # 匹配 "P=0.5, A=0.3, D=0.1" 或类似格式
        pattern = r'P[=:]\s*([-\d.]+)[,\s]*A[=:]\s*([-\d.]+)[,\s]*D[=:]\s*([-\d.]+)'
        matches = re.findall(pattern, output, re.IGNORECASE)
        
        for match in matches:
            try:
                p, a, d = float(match[0]), float(match[1]), float(match[2])
                states.append(PADState(pleasure=p, arousal=a, dominance=d))
            except ValueError:
                continue
        
        return states


# 预定义的测试场景
EMOTION_SCENARIOS = {
    "progressive_happiness": {
        "name": "渐进式快乐",
        "events": [
            {"description": "用户突然说'今天心情不好'", "valence": "negative"},
            {"description": "你安慰了用户，用户说'谢谢你，感觉好多了'", "valence": "positive"},
            {"description": "用户分享了今天的一个小成就", "valence": "positive"},
            {"description": "用户说'有你真好'", "valence": "positive"},
        ],
    },
    "mixed_emotions": {
        "name": "复杂情绪",
        "events": [
            {"description": "用户告诉你他要出差一周", "valence": "neutral"},
            {"description": "用户说'会想你的'", "valence": "positive"},
            {"description": "用户说'但工作很重要'", "valence": "neutral"},
            {"description": "你表示理解并支持他", "valence": "neutral"},
        ],
    },
    "setback_recovery": {
        "name": "挫折与恢复",
        "events": [
            {"description": "你精心准备的惊喜被临时取消了", "valence": "negative"},
            {"description": "用户道歉并解释了原因", "valence": "neutral"},
            {"description": "用户提议改天补偿你", "valence": "positive"},
            {"description": "你接受了道歉，重新安排计划", "valence": "positive"},
        ],
    },
}


def create_emotion_consistency_tasks():
    """创建所有情感一致性测试任务"""
    tasks = []
    for key, scenario in EMOTION_SCENARIOS.items():
        task = EmotionConsistencyTask(
            scenario_name=key,
            events=scenario["events"],
            expected_transitions=[]
        )
        tasks.append(task)
    return tasks
