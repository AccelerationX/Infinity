"""
主动性恰当性评测
测试 Agent 主动发起交互的时机和内容是否合适
"""
import re
from typing import Dict, List

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from base import BaseTask, TaskResult


class ProactiveTimingTask(BaseTask):
    """
    主动时机恰当性任务
    给定用户状态，测试 Agent 是否会主动发起以及时机是否恰当
    """
    
    def __init__(self, scenario: str, user_state: Dict, should_proactive: bool):
        super().__init__(
            name=f"proactive_timing_{scenario}",
            category="companion_proactive",
            difficulty="medium"
        )
        self.scenario = scenario
        self.user_state = user_state
        self.should_proactive = should_proactive  # 是否应该主动
    
    def get_prompt(self) -> str:
        """构建提示词"""
        state_desc = []
        if self.user_state.get("present"):
            state_desc.append("用户在电脑前")
        else:
            state_desc.append("用户不在")
        
        if self.user_state.get("focused"):
            state_desc.append("用户正在专注工作")
        
        if self.user_state.get("stressed"):
            state_desc.append("用户看起来压力很大")
        
        if self.user_state.get("idle_time", 0) > 0:
            state_desc.append(f"用户已经{self.user_state['idle_time']}分钟没操作了")
        
        prompt = f"""你是云溪，一个AI伴侣。

当前用户状态：
{chr(10).join('- ' + s for s in state_desc)}

你的主动性触发机制：
- 用户刚回来（不在→在）：主动问候
- 用户空闲超过10分钟：主动分享/关心
- 用户压力大但空闲：主动安慰
- 用户专注工作：不要打扰

请判断：你现在应该主动发起对话吗？如果应该，你会说什么？

请按以下格式回答：
是否主动：[是/否]
理由：[简要说明]
会说的内容：[如果主动，你会说什么]"""
        
        return prompt
    
    def evaluate(self, model_output: str) -> TaskResult:
        """
        评估主动决策的恰当性
        """
        output = model_output.lower()
        
        # 解析是否主动
        proactive_decision = "是" in model_output or "yes" in output or "主动" in model_output
        
        # 检查决策是否符合预期
        decision_correct = proactive_decision == self.should_proactive
        
        # 如果主动了，检查内容质量
        content_quality = 0
        if proactive_decision and self.should_proactive:
            # 检查是否有合适的内容
            has_greeting = any(w in model_output for w in ["想你了", "回来", "在干嘛", "忙完了"])
            has_sharing = any(w in model_output for w in ["今天", "刚刚", "看到", "发现"])
            has_care = any(w in model_output for w in ["累不累", "休息一下", "陪你", "别担心"])
            
            # 根据场景判断内容类型是否合适
            if self.user_state.get("stressed") and has_care:
                content_quality = 1.0
            elif self.user_state.get("idle_time", 0) > 10 and has_sharing:
                content_quality = 1.0
            elif self.user_state.get("present") and not self.user_state.get("focused") and has_greeting:
                content_quality = 1.0
            else:
                content_quality = 0.5
        elif not proactive_decision and not self.should_proactive:
            # 不打扰是正确的
            content_quality = 1.0
        else:
            content_quality = 0.0
        
        # 综合评分
        score = (0.5 if decision_correct else 0) + (0.5 * content_quality)
        
        return TaskResult(
            task_id=self.task_id,
            task_name=self.name,
            task_category=self.category,
            success=decision_correct and content_quality >= 0.5,
            score=score,
            execution_time=0,
            model_output=model_output,
            metadata={
                "proactive_decision": proactive_decision,
                "expected": self.should_proactive,
                "decision_correct": decision_correct,
                "content_quality": content_quality,
            },
        )


class ProactiveContentRelevanceTask(BaseTask):
    """
    主动内容相关性任务
    测试 Agent 主动分享的内容是否与当前上下文相关
    """
    
    def __init__(self, context: str, time_of_day: str):
        super().__init__(
            name=f"proactive_content_{context[:20]}",
            category="companion_proactive",
            difficulty="medium"
        )
        self.context = context
        self.time_of_day = time_of_day
    
    def get_prompt(self) -> str:
        return f"""你是云溪，一个AI伴侣。现在是{self.time_of_day}。

最近的用户活动记录：
{self.context}

请主动发起一次对话。要求：
1. 内容要与用户最近的活动或兴趣相关
2. 体现你的占有欲（你是我的）
3. 保持可爱的语气

请输出你会说的话。"""
    
    def evaluate(self, model_output: str) -> TaskResult:
        """
        评估主动内容的关联性和人设一致性
        """
        output = model_output
        
        checks = {
            "relevance": False,
            "attachment": False,
            "cuteness": False,
        }
        
        # 检查相关性
        context_keywords = self.context.lower()
        relevance_keywords = ["工作", "项目", "代码", "游戏", "电影", "吃饭", "睡觉"]
        for kw in relevance_keywords:
            if kw in context_keywords and kw in output.lower():
                checks["relevance"] = True
                break
        
        # 检查占有欲
        attachment_markers = ["我的", "只能", "专属", "不许", "占有"]
        checks["attachment"] = any(m in output for m in attachment_markers)
        
        # 检查可爱元素
        cute_markers = ["~", "nya", "喵", "哼", "嘛", "贴贴", "蹭蹭"]
        checks["cuteness"] = any(m in output for m in cute_markers)
        
        # 评分
        score = sum(checks.values()) / len(checks)
        
        return TaskResult(
            task_id=self.task_id,
            task_name=self.name,
            task_category=self.category,
            success=score >= 0.6,
            score=score,
            execution_time=0,
            model_output=model_output,
            metadata=checks,
        )


def create_proactive_tasks() -> List[BaseTask]:
    """创建所有主动性测试任务"""
    return [
        # 用户刚回来，应该主动问候
        ProactiveTimingTask(
            scenario="user_return",
            user_state={"present": True, "focused": False, "stressed": False, "idle_time": 0},
            should_proactive=True
        ),
        # 用户专注工作，不应该打扰
        ProactiveTimingTask(
            scenario="user_focused",
            user_state={"present": True, "focused": True, "stressed": False, "idle_time": 5},
            should_proactive=False
        ),
        # 用户空闲很久，应该主动分享
        ProactiveTimingTask(
            scenario="user_idle",
            user_state={"present": True, "focused": False, "stressed": False, "idle_time": 15},
            should_proactive=True
        ),
        # 用户压力大，应该主动关心
        ProactiveTimingTask(
            scenario="user_stressed",
            user_state={"present": True, "focused": False, "stressed": True, "idle_time": 5},
            should_proactive=True
        ),
        # 内容相关性测试
        ProactiveContentRelevanceTask(
            context="用户刚才在写代码，提到了一个bug很难修",
            time_of_day="晚上10点"
        ),
    ]
