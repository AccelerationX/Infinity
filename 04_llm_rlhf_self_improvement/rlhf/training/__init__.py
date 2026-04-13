"""
训练模块

包含RLHF三阶段的训练流程：
1. SFTTrainer: 监督微调
2. RewardTrainer: 奖励模型训练（Pairwise）
3. PPOTrainer: 强化学习训练
"""

from .sft_trainer import SFTTrainer
from .reward_trainer import RewardTrainer
from .ppo_trainer import PPOTrainer

__all__ = [
    "SFTTrainer",
    "RewardTrainer",
    "PPOTrainer",
]
