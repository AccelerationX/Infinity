"""
MCP Tool Hub 核心数据模型
"""
from pydantic import BaseModel, Field
from typing import Literal, Any
from datetime import datetime


class RiskLevel:
    SAFE = "safe"
    CAUTION = "caution"
    DANGEROUS = "dangerous"


class ToolMetadata(BaseModel):
    """工具的元数据描述，用于 Hub 内部索引"""
    name: str
    description: str = ""
    input_schema: dict = Field(default_factory=dict)
    risk_level: Literal["safe", "caution", "dangerous"] = "safe"
    server_name: str = ""


class ToolChainStep(BaseModel):
    """工具链中的单个执行步骤"""
    step_id: str
    tool_name: str
    arguments: dict = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)
    reason: str = ""


class ToolChainPlan(BaseModel):
    """由 Planner 生成的完整执行计划"""
    steps: list[ToolChainStep]
    estimated_risk: Literal["safe", "caution", "dangerous"] = "safe"


class ExecutionResult(BaseModel):
    """单次工具调用的执行结果"""
    tool_name: str
    arguments: dict
    output: Any = None
    success: bool = True
    error_message: str | None = None


class AuditLogEntry(BaseModel):
    """审计日志条目，记录一次完整的意图-规划-执行链路"""
    timestamp: datetime
    session_id: str
    intent: str
    plan: ToolChainPlan | None = None
    executed_steps: list[ExecutionResult] = Field(default_factory=list)
    user_confirmed: bool = False
    passed_security: bool = True
