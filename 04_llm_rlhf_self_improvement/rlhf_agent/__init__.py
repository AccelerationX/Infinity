"""
RLHF Agent Self-Improvement System
基于人类反馈强化学习的Agent自我改进系统

核心组件:
1. Reward Model - 奖励模型，评估Agent输出质量
2. PPO Trainer - 策略优化器
3. Replay Buffer - 经验回放缓冲区
4. SelfImprovementLoop - 自我改进主循环
"""

__version__ = "1.0.0"

from .reward_model import RewardModel, RewardConfig
from .policy_model import PolicyModel, PolicyConfig
from .ppo_trainer import PPOTrainer, PPOConfig
from .replay_buffer import ReplayBuffer, Experience
from .self_improvement import SelfImprovementLoop, ImprovementConfig

__all__ = [
    "RewardModel",
    "RewardConfig",
    "PolicyModel",
    "PolicyConfig",
    "PPOTrainer",
    "PPOConfig",
    "ReplayBuffer",
    "Experience",
    "SelfImprovementLoop",
    "ImprovementConfig",
]
