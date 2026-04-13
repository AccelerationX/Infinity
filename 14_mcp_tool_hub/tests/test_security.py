"""
测试 SecurityManager 的风险评估与权限校验
"""
from src.security import SecurityManager
from src.schemas.models import ToolMetadata


def test_assess_plan_risk():
    sm = SecurityManager()
    assert sm.assess_plan_risk([
        ToolMetadata(name="a", description="x", risk_level="safe"),
    ]) == "safe"

    assert sm.assess_plan_risk([
        ToolMetadata(name="a", description="x", risk_level="safe"),
        ToolMetadata(name="b", description="x", risk_level="caution"),
    ]) == "caution"

    assert sm.assess_plan_risk([
        ToolMetadata(name="a", description="x", risk_level="dangerous"),
        ToolMetadata(name="b", description="x", risk_level="safe"),
    ]) == "dangerous"


def test_check_permission_safe():
    sm = SecurityManager()
    allowed, msg = sm.check_tool_permission(
        ToolMetadata(name="read", description="x", risk_level="safe"), False
    )
    assert allowed is True
    assert msg == ""


def test_check_permission_dangerous_without_confirmation():
    sm = SecurityManager()
    allowed, msg = sm.check_tool_permission(
        ToolMetadata(name="rm", description="x", risk_level="dangerous"), False
    )
    assert allowed is False
    assert "requires explicit user confirmation" in msg


def test_check_permission_dangerous_with_confirmation():
    sm = SecurityManager()
    allowed, msg = sm.check_tool_permission(
        ToolMetadata(name="rm", description="x", risk_level="dangerous"), True
    )
    assert allowed is True
