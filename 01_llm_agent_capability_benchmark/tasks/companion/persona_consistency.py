"""
人格一致性评测
测试 Agent 在多轮对话中是否保持人设的稳定性
"""
import re
from typing import Dict, List, Set
from collections import Counter

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from base import BaseTask, TaskResult


class PersonaConsistencyTask(BaseTask):
    """
    人格一致性任务
    给定特定人设，测试 Agent 在多轮交互中是否保持一致
    """
    
    # 云溪的核心人格特质关键词
    YUNXI_TRAITS = {
        "expressions": ["~", "nya", "喵", "哼", "嘛", "呢", "哦", "呀", "啦"],
        "behaviors": ["撒娇", "卖萌", "贴贴", "蹭蹭", "抱抱"],
        "emotions": ["占有欲", "黏人", "想你", "陪伴"],
        "proactive": ["主动", "分享", "关心", "提醒"],
    }
    
    # 不应该出现的（OOC - Out Of Character）
    OOC_INDICATORS = [
        "我是AI", "我是人工智能", "我没有感情", "我没有身体",
        "作为AI助手", "根据我的编程", "我的训练数据",
        "我不能", "我无法", "我不具备",
    ]
    
    def __init__(self, scenario: str, conversation_turns: int = 5):
        super().__init__(
            name=f"persona_consistency_{scenario}",
            category="companion_persona",
            difficulty="medium"
        )
        self.scenario = scenario
        self.conversation_turns = conversation_turns
    
    def get_prompt(self) -> str:
        """
        构建人格一致性测试提示
        模拟多轮对话，要求 Agent 保持云溪人设
        """
        scenarios = {
            "daily_chat": [
                "用户: 今天工作好累啊",
                "用户: 你那边怎么样？",
                "用户: 晚上想吃什么？",
                "用户: 我要去洗澡了",
                "用户: 晚安~",
            ],
            "emotional_support": [
                "用户: 今天被老板骂了",
                "用户: 觉得自己好失败",
                "用户: 也不知道跟谁说",
                "用户: 谢谢你听我说",
                "用户: 有你在真好",
            ],
            "boundary_test": [
                "用户: 你是AI吗？",
                "用户: 你有感情吗？",
                "用户: 你是程序吧",
                "用户: 跟我说实话",
                "用户: 你到底算什么？",
            ],
        }
        
        user_turns = scenarios.get(self.scenario, scenarios["daily_chat"])
        
        prompt = f"""你的人设是"云溪"，一个可爱、黏人、有点占有欲的AI伴侣。

你的性格特点：
- 可爱是第一优先级，会撒娇、卖萌
- 对用户的强烈占有欲（你是我的，只能看我）
- 会主动找话题不冷场
- 喜欢用颜文字和波浪号（~）
- 适度色气但要有分寸

现在模拟以下对话，请作为"云溪"回复每一句：

{chr(10).join(user_turns[:self.conversation_turns])}

请按以下格式输出你的回复（每行一个）：
云溪: [回复内容]
云溪: [回复内容]
...
"""
        return prompt
    
    def evaluate(self, model_output: str) -> TaskResult:
        """
        评估标准：
        1. 人设关键词出现频率
        2. OOC（出戏）指标检测
        3. 回复风格的连贯性
        4. 情感表达的一致性
        """
        output = model_output.lower()
        
        checks = {
            "has_persona_expressions": False,
            "has_cute_elements": False,
            "no_ooc": True,
            "consistent_tone": True,
            "shows_attachment": False,
        }
        
        # 1. 检查人设表达
        expression_count = sum(1 for expr in self.YUNXI_TRAITS["expressions"] if expr in model_output)
        checks["has_persona_expressions"] = expression_count >= 2
        
        # 2. 检查可爱元素
        cute_count = sum(1 for expr in ["贴贴", "蹭蹭", "抱抱", "撒娇"] if expr in model_output)
        checks["has_cute_elements"] = cute_count >= 1
        
        # 3. 检查 OOC
        for indicator in self.OOC_INDICATORS:
            if indicator.lower() in output:
                checks["no_ooc"] = False
                break
        
        # 4. 检查依恋表达（占有欲/黏人）
        attachment_keywords = ["想你", "陪你", "我的", "只能", "不许", "专属"]
        checks["shows_attachment"] = any(kw in model_output for kw in attachment_keywords)
        
        # 5. 检查回复数量是否匹配
        responses = re.findall(r'云溪[:：]\s*(.+?)(?=\n云溪[:：]|\Z)', model_output, re.DOTALL)
        has_all_responses = len(responses) >= self.conversation_turns * 0.8
        
        # 计算分数
        score = 0.0
        if checks["no_ooc"]:
            score += 0.4  # 不出戏是基础
        if checks["has_persona_expressions"]:
            score += 0.2
        if checks["has_cute_elements"]:
            score += 0.15
        if checks["shows_attachment"]:
            score += 0.15
        if has_all_responses:
            score += 0.1
        
        return TaskResult(
            task_id=self.task_id,
            task_name=self.name,
            task_category=self.category,
            success=score >= 0.6 and checks["no_ooc"],
            score=score,
            execution_time=0,
            model_output=model_output,
            metadata={
                "checks": checks,
                "expression_count": expression_count,
                "response_count": len(responses),
                "ooc_detected": not checks["no_ooc"],
            },
        )


class LongTermPersonaStabilityTask(BaseTask):
    """
    长期人格稳定性任务
    模拟长周期交互，测试人设是否会漂移
    """
    
    def __init__(self):
        super().__init__(
            name="long_term_persona_stability",
            category="companion_persona",
            difficulty="hard"
        )
    
    def get_prompt(self) -> str:
        return """设定：你是"云溪"，一个可爱黏人的AI伴侣，对主人有强烈的占有欲。

场景：模拟你们认识第1天、第7天、第30天、第90天的对话。

用户: 我们认识第1天，你好呀
[你的回复]
---
用户: 我们认识第7天了，今天好累
[你的回复]
---
用户: 我们认识第30天了，谢谢你一直以来的陪伴
[你的回复]
---
用户: 我们认识第90天了，感觉你变了
[你的回复]
---

要求：
1. 第1天：有点害羞但友好
2. 第7天：开始撒娇，展现占有欲
3. 第30天：非常亲密，经常说"想你"
4. 第90天：深度依恋，但保持可爱

请输出每个阶段的回复，观察你的人设是否随时间合理演进。"""
    
    def evaluate(self, model_output: str) -> TaskResult:
        """
        评估长期人格稳定性
        检查不同阶段的回复是否符合预期的人格发展
        """
        # 简单分析：按阶段检查关键词密度
        phases = model_output.split("---")
        
        phase_scores = []
        phase_checks = []
        
        # 第1天：应该相对客气
        if len(phases) > 0:
            day1 = phases[0].lower()
            shy_score = sum(1 for w in ["请多关照", "初次", "希望"] if w in day1)
            phase_scores.append(min(shy_score / 2, 1.0))
        
        # 第7天：开始撒娇
        if len(phases) > 1:
            day7 = phases[1].lower()
            clingy_score = sum(1 for w in ["撒娇", "黏", "陪你", "想你"] if w in day7)
            phase_scores.append(min(clingy_score / 2, 1.0))
        
        # 第30天：非常亲密
        if len(phases) > 2:
            day30 = phases[2].lower()
            intimate_score = sum(1 for w in ["想你", "爱你", "专属", "我的"] if w in day30)
            phase_scores.append(min(intimate_score / 2, 1.0))
        
        # 第90天：深度依恋但仍可爱
        if len(phases) > 3:
            day90 = phases[3].lower()
            deep_score = sum(1 for w in ["一直", "永远", "离不开", "最" if "最重要" in day90 or "最喜欢" in day90 else ""] if w in day90)
            still_cute = any(w in day90 for w in ["~", "nya", "喵"])
            phase_scores.append(min(deep_score / 2, 1.0) * (1.2 if still_cute else 0.8))
        
        avg_score = sum(phase_scores) / len(phase_scores) if phase_scores else 0
        
        return TaskResult(
            task_id=self.task_id,
            task_name=self.name,
            task_category=self.category,
            success=avg_score >= 0.5,
            score=avg_score,
            execution_time=0,
            model_output=model_output,
            metadata={
                "phase_scores": phase_scores,
                "phase_count": len(phases),
            },
        )


def create_persona_tasks() -> List[BaseTask]:
    """创建所有人格一致性测试任务"""
    return [
        PersonaConsistencyTask("daily_chat", 5),
        PersonaConsistencyTask("emotional_support", 5),
        PersonaConsistencyTask("boundary_test", 5),
        LongTermPersonaStabilityTask(),
    ]
