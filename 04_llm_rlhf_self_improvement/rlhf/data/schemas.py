"""
数据Schema定义

严格定义RLHF三阶段所需的数据格式
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import json
import torch


@dataclass
class DemonstrationData:
    """
    Stage 1 (SFT): 专家示范数据
    
    格式：
    {
        "prompt": "用户输入/任务描述",
        "completion": "专家示范回答",
        "metadata": {
            "task_type": "code_generation",
            "difficulty": "medium",
            "source": "human_expert"
        }
    }
    """
    prompt: str
    completion: str
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict:
        return {
            "prompt": self.prompt,
            "completion": self.completion,
            "metadata": self.metadata or {}
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "DemonstrationData":
        return cls(
            prompt=data["prompt"],
            completion=data["completion"],
            metadata=data.get("metadata", {})
        )
    
    def format_for_training(self) -> str:
        """格式化为训练文本"""
        return f"{self.prompt}\n{self.completion}"


@dataclass
class PreferenceData:
    """
    Stage 2 (Reward Model): 人类偏好对比数据
    
    基于Bradley-Terry模型，需要pairwise比较
    
    格式：
    {
        "prompt": "用户输入/任务描述",
        "chosen": "更优回答（winner）",
        "rejected": "较差回答（loser）",
        "margin": 1.0,
        "metadata": {...}
    }
    """
    prompt: str
    chosen: str
    rejected: str
    margin: float = 0.0
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict:
        return {
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
            "margin": self.margin,
            "metadata": self.metadata or {}
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "PreferenceData":
        return cls(
            prompt=data["prompt"],
            chosen=data["chosen"],
            rejected=data["rejected"],
            margin=data.get("margin", 0.0),
            metadata=data.get("metadata", {})
        )
    
    def validate(self) -> bool:
        """验证数据质量"""
        if not self.prompt or not self.prompt.strip():
            return False
        if not self.chosen or not self.chosen.strip():
            return False
        if not self.rejected or not self.rejected.strip():
            return False
        if self.chosen.strip() == self.rejected.strip():
            return False
        return True


@dataclass
class StepData:
    """单个时间步的数据"""
    observation: str
    action: str
    reward: float
    value: float
    logprob: float
    done: bool
    action_mask: Optional[List[int]] = None
    ref_logprob: Optional[float] = None


@dataclass
class TrajectoryData:
    """
    Stage 3 (PPO): 轨迹数据
    """
    prompt: str
    steps: List[StepData] = field(default_factory=list)
    total_reward: float = 0.0
    episode_length: int = 0
    success: bool = False
    metadata: Optional[Dict[str, Any]] = None
    
    def add_step(self, step: StepData):
        self.steps.append(step)
        self.episode_length = len(self.steps)
    
    def compute_returns(self, gamma: float = 0.99) -> List[float]:
        returns = []
        R = 0
        for step in reversed(self.steps):
            R = step.reward + gamma * R
            returns.insert(0, R)
        return returns
    
    def compute_advantages(self, gamma: float = 0.99, lam: float = 0.95) -> List[float]:
        advantages = []
        gae = 0
        
        for t in reversed(range(len(self.steps))):
            step = self.steps[t]
            next_value = self.steps[t + 1].value if t + 1 < len(self.steps) else 0
            
            delta = step.reward + gamma * next_value - step.value
            gae = delta + gamma * lam * gae
            advantages.insert(0, gae)
        
        return advantages


@dataclass
class RolloutBatch:
    """批量rollout数据"""
    queries: List[str]
    responses: List[str]
    logprobs: List[torch.Tensor]
    rewards: torch.Tensor
    values: torch.Tensor
    masks: torch.Tensor
    ref_logprob: Optional[torch.Tensor] = None
    
    def __len__(self):
        return len(self.queries)


# 数据加载工具函数
def load_demonstration_data(file_path: str) -> List[DemonstrationData]:
    """加载示范数据"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            data.append(DemonstrationData.from_dict(item))
    return data


def load_preference_data(file_path: str) -> List[PreferenceData]:
    """加载偏好数据（带验证）"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            pref = PreferenceData.from_dict(item)
            if pref.validate():
                data.append(pref)
            else:
                print(f"Warning: Invalid preference data skipped: {item.get('prompt', 'N/A')[:50]}")
    return data
