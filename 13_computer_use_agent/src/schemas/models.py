"""
Computer Use Agent 核心数据模型
"""
from pydantic import BaseModel, Field
from typing import Literal, Any
from datetime import datetime


class UIElement(BaseModel):
    """屏幕上的可交互元素"""
    element_id: str
    element_type: str  # button, input, text, link, checkbox, etc.
    text: str = ""
    bbox: list[int] = Field(default_factory=list)  # [x1, y1, x2, y2]
    confidence: float = 0.0


class ScreenObservation(BaseModel):
    """对当前屏幕状态的完整观察"""
    screenshot: Any = None  # PIL.Image.Image
    elements: list[UIElement] = Field(default_factory=list)
    raw_description: str = ""


class ActionPlan(BaseModel):
    """Planner 生成的下一步行动计划"""
    thought: str = ""
    action: Literal["click", "type", "hotkey", "scroll", "wait", "terminate", "noop"] = "noop"
    target_element_id: str | None = None
    value: str = ""  # 用于 type / hotkey / scroll
    expected_outcome: str = ""


class ExecutionRecord(BaseModel):
    """单步执行的完整记录"""
    step: int
    observation_before: ScreenObservation | None = None
    plan: ActionPlan | None = None
    observation_after: ScreenObservation | None = None
    verification_passed: bool = False
    verification_message: str = ""
    timestamp: datetime = Field(default_factory=datetime.now)


class SkillMacro(BaseModel):
    """从成功经验中提炼出的可复用 GUI 宏"""
    name: str
    trigger_description: str
    steps: list[ActionPlan] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
