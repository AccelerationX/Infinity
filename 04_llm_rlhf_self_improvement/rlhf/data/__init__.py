"""
数据管理模块

两种核心数据类型：
1. DemonstrationData: 专家示范数据（用于SFT）
2. PreferenceData: 人类偏好对比数据（用于Reward Model）
"""

from .schemas import DemonstrationData, PreferenceData, TrajectoryData
from .datasets import SFTDataset, PreferenceDataset, RolloutDataset
from .collators import SFTDataCollator, PreferenceDataCollator

__all__ = [
    "DemonstrationData",
    "PreferenceData",
    "TrajectoryData",
    "SFTDataset",
    "PreferenceDataset",
    "RolloutDataset",
    "SFTDataCollator",
    "PreferenceDataCollator",
]
