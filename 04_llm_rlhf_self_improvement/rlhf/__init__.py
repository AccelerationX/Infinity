"""
严谨的RLHF Agent自我改进系统

标准RLHF三阶段流程：
1. SFT (Supervised Fine-Tuning): 监督微调
2. Reward Modeling: 基于Bradley-Terry模型的Pairwise训练
3. RL Fine-tuning: 标准Actor-Critic PPO

核心组件：
- data: 数据管理（demonstration + preference）
- models: 模型定义（SFT、Reward、Actor-Critic）
- training: 训练流程（三阶段）
- environment: 真实Agent环境（工具执行）
"""

__version__ = "2.0.0"
__author__ = "Research Team"

from .config import RLHFConfig, SFTConfig, RewardConfig, PPOConfig

__all__ = [
    "RLHFConfig",
    "SFTConfig", 
    "RewardConfig",
    "PPOConfig",
]
