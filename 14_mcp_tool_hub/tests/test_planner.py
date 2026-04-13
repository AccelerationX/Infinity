"""
测试 DAGPlanner 的规划逻辑
"""
from src.planner import DAGPlanner
from src.schemas.models import ToolMetadata


def test_planner_basic_ordering():
    planner = DAGPlanner()
    matched = [
        (ToolMetadata(name="calculate", description="do math", risk_level="safe"), 0.95),
        (ToolMetadata(name="read_file", description="read", risk_level="safe"), 0.90),
        (ToolMetadata(name="list_directory", description="list", risk_level="safe"), 0.85),
    ]
    plan = planner.plan("read something and calculate", matched)
    assert len(plan.steps) == 3
    # 按优先级 list_directory 应在第一位
    assert plan.steps[0].tool_name == "list_directory"
    assert plan.estimated_risk == "safe"


def test_planner_risk_assessment():
    planner = DAGPlanner()
    matched = [
        (ToolMetadata(name="write_file", description="write", risk_level="caution"), 0.9),
        (ToolMetadata(name="run_shell", description="shell", risk_level="dangerous"), 0.8),
    ]
    plan = planner.plan("dangerous ops", matched)
    assert plan.estimated_risk == "dangerous"
