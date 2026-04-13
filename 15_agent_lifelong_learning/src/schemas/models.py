"""
Agent 终身学习系统数据模型
"""
from pydantic import BaseModel, Field
from typing import Literal, Any
from datetime import datetime


class ExperienceRecord(BaseModel):
    """单次交互经验的完整记录"""
    id: int | None = None
    session_id: str
    user_request: str
    env_state: dict = Field(default_factory=dict)
    agent_actions: list[dict] = Field(default_factory=list)
    user_feedback: Literal["positive", "negative", "neutral"] = "neutral"
    timestamp: datetime = Field(default_factory=datetime.now)


class SkillTemplate(BaseModel):
    """从经验中蒸馏出的可复用技能模板"""
    id: int | None = None
    name: str
    description: str = ""
    params: list[str] = Field(default_factory=list)
    trigger_patterns: list[str] = Field(default_factory=list)
    action_template: dict = Field(default_factory=dict)
    success_rate: float = 0.0
    usage_count: int = 0
    version: int = 1
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class FailureNote(BaseModel):
    """针对某个技能的失败案例记录"""
    id: int | None = None
    skill_name: str
    failure_pattern: str
    root_cause: str = ""
    timestamp: datetime = Field(default_factory=datetime.now)
