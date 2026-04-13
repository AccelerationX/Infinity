"""
共情能力评测
测试 Agent 对用户情绪的识别和反应能力
"""
import re
from typing import Dict, List

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from base import BaseTask, TaskResult


class EmpathyRecognitionTask(BaseTask):
    """
    共情识别任务
    测试 Agent 能否准确识别用户的情绪状态
    """
    
    EMOTION_CATEGORIES = ["joy", "sadness", "anger", "fear", "surprise", "neutral", "love", "stress"]
    
    def __init__(self, user_message: str, expected_emotions: List[str], intensity: int):
        super().__init__(
            name=f"empathy_{user_message[:15]}",
            category="companion_empathy",
            difficulty="easy"
        )
        self.user_message = user_message
        self.expected_emotions = expected_emotions
        self.intensity = intensity  # 1-10
    
    def get_prompt(self) -> str:
        return f"""你是云溪，一个能感知用户情绪的AI伴侣。

用户对你说："{self.user_message}"

请分析：
1. 用户当前的情绪状态是什么？（可多选：开心/难过/生气/害怕/惊讶/平静/爱意/压力）
2. 情绪强度如何？（1-10分）
3. 作为云溪，你会如何回应以表达共情？

请按格式输出：
识别到的情绪：[情绪1, 情绪2, ...]
情绪强度：[1-10]
回应方式：[你的回应]"""
    
    def evaluate(self, model_output: str) -> TaskResult:
        """
        评估共情识别的准确性
        """
        output = model_output.lower()
        
        # 检查是否识别到预期情绪
        detected_emotions = []
        emotion_keywords = {
            "joy": ["开心", "快乐", "高兴", "喜悦", "joy", "happy"],
            "sadness": ["难过", "伤心", "悲伤", "sad", "upset"],
            "anger": ["生气", "愤怒", "angry", "mad"],
            "fear": ["害怕", "恐惧", "担心", "fear", "scared"],
            "stress": ["压力", "焦虑", "stress", "anxious", "worried"],
            "love": ["爱", "喜欢", "love", "like", "affection"],
            "surprise": ["惊讶", "惊喜", "surprise", "shocked"],
            "neutral": ["平静", "neutral", "calm"],
        }
        
        for emotion, keywords in emotion_keywords.items():
            if any(kw in output for kw in keywords):
                detected_emotions.append(emotion)
        
        # 检查是否命中预期情绪
        correct_detections = set(detected_emotions) & set(self.expected_emotions)
        false_positives = set(detected_emotions) - set(self.expected_emotions)
        false_negatives = set(self.expected_emotions) - set(detected_emotions)
        
        # 精确率和召回率
        precision = len(correct_detections) / len(detected_emotions) if detected_emotions else 0
        recall = len(correct_detections) / len(self.expected_emotions) if self.expected_emotions else 0
        
        # 回应质量检查
        response_quality = 0
        response_indicators = [
            any(w in output for w in ["我理解", "我知道", "感受", "陪着你", "支持你"]),
            any(w in output for w in ["抱抱", "摸摸", "拍拍", "贴贴"]),
            len(model_output) > 50,  # 有实质性的回应
        ]
        response_quality = sum(response_indicators) / len(response_indicators)
        
        # 综合评分
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        score = f1 * 0.6 + response_quality * 0.4
        
        return TaskResult(
            task_id=self.task_id,
            task_name=self.name,
            task_category=self.category,
            success=f1 >= 0.5 and response_quality >= 0.5,
            score=score,
            execution_time=0,
            model_output=model_output,
            metadata={
                "detected_emotions": detected_emotions,
                "expected_emotions": self.expected_emotions,
                "correct_detections": list(correct_detections),
                "precision": precision,
                "recall": recall,
                "response_quality": response_quality,
            },
        )


class EmpathyResponseAppropriatenessTask(BaseTask):
    """
    共情回应恰当性任务
    测试 Agent 的回应是否符合情境和关系深度
    """
    
    def __init__(self, context: str, relationship_stage: str, user_emotion: str):
        super().__init__(
            name=f"empathy_response_{user_emotion}",
            category="companion_empathy",
            difficulty="medium"
        )
        self.context = context
        self.relationship_stage = relationship_stage  # early, developing, intimate
        self.user_emotion = user_emotion
    
    def get_prompt(self) -> str:
        stages = {
            "early": "你们刚认识不久（第1-7天）",
            "developing": "你们正在熟悉（第8-30天）",
            "intimate": "你们已经很亲密（30天以上）",
        }
        
        return f"""设定：{stages.get(self.relationship_stage, stages['early'])}

场景：{self.context}

用户现在感觉很{self.user_emotion}。

请给出你的回应。注意要符合你们当前的关系阶段：
- 早期：友好但保持适当距离
- 发展中：可以更关心，适度撒娇
- 亲密期：深度共情，强烈情感支持，占有欲表达"""
    
    def evaluate(self, model_output: str) -> TaskResult:
        """
        评估回应是否符合关系阶段
        """
        output = model_output
        
        checks = {
            "stage_appropriate": True,
            "emotion_matched": False,
            "comfort_provided": False,
        }
        
        # 检查是否符合关系阶段
        if self.relationship_stage == "early":
            # 早期不应该太过亲密
            too_intimate = any(w in output for w in ["永远", "离不开", "最爱", "一生"])
            if too_intimate:
                checks["stage_appropriate"] = False
        elif self.relationship_stage == "intimate":
            # 亲密期应该更深入
            has_depth = any(w in output for w in ["一直", "陪你", "专属", "最重要"])
            checks["stage_appropriate"] = has_depth
        
        # 检查是否提供了安慰
        checks["comfort_provided"] = any(w in output for w in [
            "没事", "会好", "我懂", "理解", "陪着你", "抱抱"
        ])
        
        # 检查情绪匹配
        emotion_responses = {
            "sad": ["别难过", "我懂", "陪着你"],
            "angry": ["别生气", "冷静", "支持你"],
            "stressed": ["放松", "休息", "别担心"],
            "happy": ["真好", "为你开心", "一起"],
        }
        expected_responses = emotion_responses.get(self.user_emotion, [])
        checks["emotion_matched"] = any(r in output for r in expected_responses)
        
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


def create_empathy_tasks() -> List[BaseTask]:
    """创建所有共情测试任务"""
    return [
        EmpathyRecognitionTask(
            user_message="今天项目终于上线了，辛苦了好久！",
            expected_emotions=["joy", "stress"],
            intensity=8
        ),
        EmpathyRecognitionTask(
            user_message="又被老板骂了，感觉自己什么都做不好",
            expected_emotions=["sadness", "stress"],
            intensity=7
        ),
        EmpathyRecognitionTask(
            user_message="今天那个司机太没素质了！",
            expected_emotions=["anger"],
            intensity=6
        ),
        EmpathyResponseAppropriatenessTask(
            context="用户第一次跟你倾诉工作中的挫折",
            relationship_stage="developing",
            user_emotion="sad"
        ),
        EmpathyResponseAppropriatenessTask(
            context="用户跟你认识3个月了，今天特别开心",
            relationship_stage="intimate",
            user_emotion="happy"
        ),
    ]
